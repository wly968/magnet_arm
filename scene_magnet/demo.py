"""连续抓取演示：A 带最左端每隔几秒随机掉落一个方块，三模型接力确定性抓取。

运行：
    cd scene_magnet
    python demo.py                    # 打开窗口，连续演示
    python demo.py --headless 10      # 不开窗口，空跑 10 秒自检
    python demo.py --interval 2.0     # 每个回合之间停顿 2 秒再掉下一个方块

流程：
    A 带最左端随机刷新方块 -> 手臂在 home 等它送到 -> 抓取 -> 搬运 -> 放料
    -> 回原位（方块交给 B 带自己滑进篮子）-> 停顿 interval 秒 -> 下一个
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
TH_PHASE2 = (0.032, 0.960, 1)
TH_PHASE35 = (0.034, 0.965, 2)
TH_PHASE6 = (0.060, 0.940, 0)

# 方块在 A 带最左端的刷新区间（带 A 覆盖 x∈[-1.20, 0.30]）
SPAWN_X = (-1.180, -1.050)

CANDIDATES = {
    "model1": ["saves/phase2.zip", "saves/phase1.zip"],
    "model2": ["saves/phase34.zip"],
    "model3": ["saves/phase6.zip", "saves/phase6_home.zip"],
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
    env._set_dist(d)
    env._set_angle1(a)
    env._set_still_count(s)
    if entry:
        env.phase_entry_steps = 6


def run_episode(env, models, do_reset=True):
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
    ap.add_argument("--cubes", type=int, default=1,
                    help="方块池大小：1=单方块（稳）；>1=连续投放（实验性，偶尔漏一个）")
    ap.add_argument("--spawn-period", type=float, default=1.5,
                    help="每隔多少秒投放一个新方块（1.5 以下可保证队列里总有货，手臂不用等）")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="仿真倍速：每个渲染帧对应 60*speed 分之一秒的睡眠")
    args = ap.parse_args()

    models = load_policies()
    if models[0] is None:
        raise SystemExit("缺少 model1（saves/phase2.zip）")

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

    if args.headless > 0:
        t0 = time.time()
        k = 0
        while time.time() - t0 < args.headless:
            phase, n = run_episode(env, models, do_reset=(k == 0))
            k += 1
            print(f"  [{k}] phase {phase} | {n} 步 | {time.time()-t0:.1f}s")
            if args.episodes and k >= args.episodes:
                break
            time.sleep(args.interval)
        print(f"自检完成：{k} 个回合")
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
            viewer.sync()
            time.sleep(1.0 / (60.0 * max(0.1, args.speed)))

            if terminated or truncated:
                k += 1
                ok = "✅ 完成" if env.phase >= 7 else ("⏱️ 超时" if truncated else "⚠️ 结束")
                print(f"[{k}] {ok} | phase {env.phase} | {step_count} 步 | 奖励 {episode_reward:.1f}")
                # 停顿几秒（方块此刻还在 B 带上滑向篮子），然后掉下一个
                t_end = time.time() + args.interval
                while viewer.is_running() and time.time() < t_end:
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
