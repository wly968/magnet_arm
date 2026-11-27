"""连续抓取演示：A 带最左端每隔几秒随机掉落一个方块，三模型接力确定性抓取。

运行：
    cd scene_magnet
    python demo.py                    # 打开窗口，连续抓取（默认 10 块轮转、1.5s 一块）
    python demo.py --headless 10      # 不开窗口，空跑 10 秒自检
    python demo.py --interval 2.0     # 只有想要回合间停顿时才加这个参数

流程：
    A 带最左端随机刷新方块 -> 手臂在 home 等它送到 -> 抓取 -> 搬运 -> 放料
    -> 回原位（方块交给 B 带自己滑进篮子）-> 下一个方块已排队，直接开始
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

# ---- 接力用的阈值（与 test.py 一致）----
TH_PHASE1 = (0.045, 0.950, 0)
TH_PHASE2 = (0.014, 0.992, 15)          # 抓取（真值模式）：与精调 model1 的训练目标一致
TH_PHASE2_VISION = (0.020, 0.980, 4)    # 抓取（视觉模式）：有 1~2 mm 估计噪声，验收略放宽
TH_PHASE35 = (0.034, 0.965, 2)
VISION_TOL_SCALE = 1.0    # 精调模型按 1.4cm/0.992/15 训练，无需放宽
TH_PHASE6 = (0.060, 0.940, 0)

# 方块在 A 带最左端的刷新区间（带 A 覆盖 x∈[-1.20, 0.30]）
SPAWN_X = (-1.180, -1.050)

CANDIDATES = {
    "model1": ["saves/phase1-2-vision.zip", "saves/phase1-2.zip"],  # 视觉鲁棒版优先
    "model2": ["saves/phase3-5.zip"],
    "model3": ["saves/phase6.zip"],
}


class ObsAdapter:
    """观测维度不一致时：多则裁剪，少则末尾补零。"""

    def __init__(self, model):
        self.model = model
        self.dim = int(model.observation_space.shape[0])

    def predict(self, obs, deterministic=True):
        o = np.asarray(obs, dtype=np.float32)
        if o.shape[-1] > self.dim:
            o = o[..., : self.dim]
        elif o.shape[-1] < self.dim:
            o = np.concatenate(
                [o, np.zeros(o.shape[:-1] + (self.dim - o.shape[-1],), dtype=np.float32)], axis=-1)
        return self.model.predict(o, deterministic=deterministic)


def find_model(name):
    for rel in CANDIDATES[name]:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            return p
    return None


def load_policies():
    models = []
    for name in ("model1", "model2", "model3"):
        p = find_model(name)
        if p is None:
            print(f"⚠️  找不到 {name}（{CANDIDATES[name]}）")
            models.append(None)
            continue
        m = PPO.load(p, device="cpu")
        print(f"✅ {name}: {os.path.basename(p)}  obs {m.observation_space.shape}")
        models.append(ObsAdapter(m))
    return models


def pick_policy(models, phase):
    m1, m2, m3 = models
    if phase < 3:
        return m1
    if phase < 6:
        return m2 if m2 is not None else m1
    return m3 if m3 is not None else (m2 if m2 is not None else m1)


def apply_thresholds(env, phase, entry=False):
    if phase <= 1:
        d, a, s = TH_PHASE1
        env.phase1_dist, env.phase1_angle = d, a
    elif phase == 2:
        d, a, s = TH_PHASE2
    elif phase < 6:
        d, a, s = TH_PHASE35
    else:
        d, a, s = TH_PHASE6
    # 视觉模式引入 1~2 mm 估计误差，抓取容差按 VISION_TOL_SCALE 放宽（phase 2 最敏感）
    if getattr(env, "use_vision", False) and phase == 2:
        d, a, s = TH_PHASE2_VISION          # 视觉模式换一套验收标准
        d = d * VISION_TOL_SCALE
    env._set_dist(d)
    env._set_angle1(a)
    env._set_still_count(s)
    if entry:
        env.phase_entry_steps = 6


class CameraView:
    """俯视相机：渲染 -> 交给视觉（同步管线或线程）-> 可选小窗口显示。"""

    def __init__(self, env, vision, every=2, show=False, scale=0.75, threaded=True):
        self.env, self.thread = env, vision
        self.threaded = threaded
        self.every, self.show, self.scale = max(1, every), show, scale
        self.renderer = mujoco.Renderer(env.model, 480, 640)
        self.n = 0
        self.last_img = None
        self.frames = 0

    def tick(self):
        self.n += 1
        if self.n % self.every:
            return None
        self.renderer.update_scene(self.env.data, camera="birdeye")
        img = self.renderer.render()
        self.last_img = img
        t_now = float(self.env.data.time)
        if self.threaded:
            self.thread.submit(img, t_now)
        else:
            self.thread.step(img, t_now)      # 同步：无队列、零延迟
        self.frames += 1
        if self.show:
            self.show_img(img, self.thread.snapshot())
        return img

    def show_img(self, img, st):
        import cv2
        vis = img.copy()
        for o in (st.objects if st else []):
            uv = o.get("uv")
            if not uv:
                continue
            cx, cy = int(uv[0]), int(uv[1])
            cv2.circle(vis, (cx, cy), 6, (0, 255, 0), 2)
            cv2.putText(vis, f"#{o['id']} x={o['xy'][0]:.3f}",
                        (cx + 8, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if self.env.vision_pos is not None and getattr(self.env, "use_vision", False):
            txt = f"vision ON | sigma {float(self.env.pos_std[0])*1000:.1f} mm"
        else:
            txt = "vision OFF"
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        h, w = vis.shape[:2]
        small = cv2.resize(vis, (int(w * self.scale), int(h * self.scale)))
        cv2.imshow("birdeye camera (q to quit)", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    def close(self):
        try:
            cv2.destroyAllWindows() if self.show else None
        except Exception:
            pass
        try:
            self.renderer.close()
        except Exception:
            pass


def run_episode(env, models, do_reset=True, cam=None):
    """跑一个完整回合，返回 (结束 phase, 步数)。do_reset=False 时续接上一个回合。"""
    if do_reset or env.n_cubes == 1:
        obs, _ = env.reset()
    else:
        env.next_cycle()
        obs = env._get_obs()
    apply_thresholds(env, env.phase)
    last_phase = env.phase
    steps = 0
    limit = env.max_steps + env.max_wait_steps + 10
    for steps in range(limit):
        if env.phase != last_phase:
            apply_thresholds(env, env.phase, entry=True)
        last_phase = env.phase
        policy = pick_policy(models, env.phase)
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if cam is not None:
            cam.tick()
        if terminated or truncated:
            break
    return env.phase, steps + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=float, default=0.0,
                    help="不开窗口，按给定秒数空跑自检")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="回合之间的停顿秒数（0 = 不停，连续跑）")
    ap.add_argument("--episodes", type=int, default=0,
                    help="跑多少个回合后退出（0 = 一直跑）")
    ap.add_argument("--cubes", type=int, default=10,
                    help="方块池大小（默认 10 个轮流投放；设 1 则每轮重新投一个）")
    ap.add_argument("--spawn-period", type=float, default=1.5,
                    help="每隔多少秒投放一个新方块（1.5 以下可保证队列里总有货，手臂不用等）")
    ap.add_argument("--no-vision", action="store_true",
                    help="关闭视觉（默认开启：用 YOLO 估计位置替代 mocat 真值）")
    ap.add_argument("--weights", default=os.path.join("runs", "segment", "runs", "segment",
                                                      "weights", "best.pt"),
                    help="YOLO-seg 权重")
    ap.add_argument("--calib", default=os.path.join("vision", "calib_yolo.npz"),
                    help="与该检测器配对的标定文件")
    ap.add_argument("--cam-every", type=int, default=2, help="每 N 个物理步渲染一帧")
    ap.add_argument("--vision-tol", type=float, default=1.0,
                    help="视觉模式下抓取容差的放宽倍数（默认 1.0=不放宽）")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="仿真倍速：每个渲染帧对应 60*speed 分之一秒的睡眠")
    args = ap.parse_args()

    global VISION_TOL_SCALE
    VISION_TOL_SCALE = args.vision_tol
    models = load_policies()
    if models[0] is None:
        raise SystemExit("缺少 model1（saves/phase1-2.zip）")

    env = ArmEnv()
    env.smooth_alpha = 0.5
    env.n_cubes = args.cubes            # 方块池：连续投放
    env.spawn_period = max(1, int(round(args.spawn_period / 0.01)))
    env.pool_spawn_x = SPAWN_X        # 池模式投放点
    env.spawn_x_range = SPAWN_X       # 单方块模式也从带左端出现
    print(f"方块池 {env.n_cubes} 个，投放间隔 {args.spawn_period:.1f}s，投放点 x∈[{SPAWN_X[0]:.2f}, {SPAWN_X[1]:.2f}]，y∈"
          f"[{env.spawn_y_range[0]:.2f}, {env.spawn_y_range[1]:.2f}]，"
          f"带速 {env.cube_speed:.2f} m/s")
    print(f"回原位后立即结束（不等入篮）：collect_after_place = {env.collect_after_place}")

    # ---------------- 视觉 ----------------
    cam = None
    vision_thread = None
    if not args.no_vision:
        try:
            from vision.detect import SegDetector
            from vision.vision_thread import VisionThread
            from vision.bridge import VisionBridge
            if not os.path.exists(args.weights) or not os.path.exists(args.calib):
                raise FileNotFoundError(f"缺少权重或标定：{args.weights} / {args.calib}")
            from vision.vision_thread import VisionPipeline
            detector = SegDetector(args.weights, calib_path=args.calib)
            use_thread = (args.headless == 0)      # 只有开窗口（实时节拍）才用线程
            if use_thread:
                vision_obj = VisionThread(detector, max_hz=15)
                vision_obj.start()
                print("视觉模式：异步线程（15 Hz）")
            else:
                vision_obj = VisionPipeline(detector)
                print("视觉模式：同步管线（零延迟）")
            vision_thread = vision_obj
            bridge = VisionBridge(vision_obj, x_max=-0.10)
            env.set_vision(bridge, enable=True)
            cam = CameraView(env, vision_obj, every=args.cam_every,
                             show=(args.headless == 0), threaded=use_thread)
            print(f"视觉已接入：YOLO-seg + 卡尔曼跟踪，相机每 {args.cam_every} 步一帧，最高 15 Hz")
        except Exception as exc:
            print(f"⚠️ 视觉不可用（{exc}），回退到 mocap 真值")
            env.use_vision = False
    else:
        print("视觉已关闭（--no-vision），使用 mocap 真值")

    if args.headless > 0:
        t0 = time.time()
        k = 0
        while time.time() - t0 < args.headless:
            phase, n = run_episode(env, models, do_reset=(k == 0), cam=cam)
            k += 1
            print(f"  [{k}] phase {phase} | {n} 步 | {time.time()-t0:.1f}s")
            if args.episodes and k >= args.episodes:
                break
            time.sleep(args.interval)
        print(f"自检完成：{k} 个回合 | 相机帧 {cam.frames if cam else 0}")
        if vision_thread is not None:
            vision_thread.stop()
        return

    with mujoco.viewer.launch_passive(env.model, env.data,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90

        obs, _ = env.reset()
        apply_thresholds(env, env.phase)
        last_phase = env.phase
        step_count = 0
        episode_reward = 0.0
        k = 0

        while viewer.is_running():
            if env.phase != last_phase:
                apply_thresholds(env, env.phase, entry=True)
            last_phase = env.phase
            policy = pick_policy(models, env.phase)
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            if cam is not None:
                cam.tick()
            viewer.sync()
            time.sleep(1.0 / (60.0 * max(0.1, args.speed)))

            if terminated or truncated:
                k += 1
                ok = "✅ 完成" if env.phase >= 7 else ("⏱️ 超时" if truncated else "⚠️ 结束")
                print(f"[{k}] {ok} | phase {env.phase} | {step_count} 步 | 奖励 {episode_reward:.1f}")
                # 停顿几秒（方块此刻还在 B 带上滑向篮子），然后掉下一个
                t_end = time.time() + args.interval
                while viewer.is_running() and time.time() < t_end:
                    if cam is not None:
                        cam.tick()
                    viewer.sync()
                    time.sleep(1 / 60)
                if args.episodes and k >= args.episodes:
                    break
                if env.n_cubes > 1:
                    env.next_cycle()        # 池模式：不重置方块，接着抓下一个
                    obs = env._get_obs()
                else:
                    obs, _ = env.reset()
                apply_thresholds(env, env.phase)
                last_phase = env.phase
                step_count = 0
                episode_reward = 0.0
        print(f"演示结束：{k} 个回合")


if __name__ == "__main__":
    main()
