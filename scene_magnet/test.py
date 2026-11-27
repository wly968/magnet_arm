"""可视化测试：根据已训练好的模型，自动选择能跑到哪一阶段。

三种模式（自动判断，也可用 --only 强制）：
  phase12 ：只有 model1             → 跑 phase 0/1/2，phase 2 成功后结束
  phase125：model1 + model2         → 跑到放料完成（phase 6）
  full    ：model1 + model2 + model3 → 全流程（回初始位，phase 7）

模型文件按候选列表自动查找；观测维度不一致时自动裁剪/补零并提示。
方块是 mocap 体：phase 2 成功后锁定到末端下方 6cm（保持竖直），phase 5 成功后解锁。

运行：
    python test.py                 # 自动模式
    python test.py --only 12       # 只测 phase 1/2
    python test.py --episodes 200  # 预跑局数
    python test.py --no-viewer     # 只跑统计，不开窗口
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from envs.arm_env import ArmEnv

# 候选文件（按顺序找第一个存在的）
CANDIDATES = {
    "model1": [os.path.join("saves", "phase1-2.zip")],
    "model2": [os.path.join("saves", "phase3-5.zip")],
    "model3": [os.path.join("saves", "phase6.zip")],
}

# 每个阶段用的 (dist, angle, still)
TH_PHASE1 = (0.050, 0.940, 0)    # 接近：固定宽松（与训练时 phase1 判据一致）
TH_PHASE2 = (0.014, 0.992, 15)   # 抓取：与精调 model1 的训练目标一致
TH_PHASE35 = (0.034, 0.980, 2)   # 搬运/放料：与 model2 训练的最后一档一致
TH_PHASE6 = (0.060, 0.940, 0)

PRE_RUN_EPISODES = 100


class ObsAdapter:
    """观测维度不一致时：多则裁剪，少则末尾补零。"""

    def __init__(self, model, dim):
        self.model = model
        self.dim = int(model.observation_space.shape[0])
        self.target_dim = dim

    def predict(self, obs, deterministic=True):
        o = np.asarray(obs, dtype=np.float32)
        if o.shape[-1] > self.dim:
            o = o[..., : self.dim]
        elif o.shape[-1] < self.dim:
            pad = np.zeros(o.shape[:-1] + (self.dim - o.shape[-1],), dtype=np.float32)
            o = np.concatenate([o, pad], axis=-1)
        return self.model.predict(o, deterministic=deterministic)


def find_model(name, override=None):
    if override is not None:
        p = override if os.path.isabs(override) else os.path.join(HERE, override)
        return p if os.path.exists(p) else None
    for rel in CANDIDATES[name]:
        p = rel if os.path.isabs(rel) else os.path.join(HERE, rel)
        if os.path.exists(p):
            return p
    return None


def load_model(path, env_dim, label):
    if path is None:
        return None
    try:
        m = PPO.load(path, device="cpu")
    except Exception as exc:
        print(f"⚠️  {label} 加载失败（{os.path.basename(path)}）：{exc}")
        return None
    dim = int(m.observation_space.shape[0])
    if dim != env_dim:
        how = "裁剪多余维度" if dim > env_dim else "末尾补零"
        print(f"⚠️  {label} 观测 {dim} 维，环境 {env_dim} 维 → 自动{how}")
        m = ObsAdapter(m, env_dim)
    print(f"✅ {label}: {os.path.basename(path)}")
    return m


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
        env.phase1_dist, env.phase1_angle = d, a      # phase 1 用固定阈值
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


def run_episode(env, models, stop_phase=None):
    obs, _ = env.reset()
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
        if stop_phase is not None and env.phase >= stop_phase:
            break                       # 达到当前模型能跑到的阶段就停
        if terminated or truncated:
            break
    return env.phase, steps + 1, int(env.wait_steps)


def pre_run(env, models, target_phase, n=PRE_RUN_EPISODES):
    print(f"\n===== 预跑 {n} 局（无渲染，目标阶段 = {target_phase}）=====")
    counts = {i: 0 for i in range(8)}
    steps = []
    waits = []
    t0 = time.time()
    for ep in range(n):
        phase, n_steps, n_wait = run_episode(env, models, stop_phase=target_phase)
        counts[phase] = counts.get(phase, 0) + 1
        steps.append(n_steps)
        waits.append(n_wait)

    print(f"\n===== {n} 局结果 =====")
    for p in range(8):
        print(f"  phase {p}: {counts.get(p, 0)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", default=None, help="model1 路径")
    ap.add_argument("--m2", default=None, help="model2 路径")
    ap.add_argument("--m3", default=None, help="model3 路径")
    ap.add_argument("--only", choices=["12", "125", "full"], default=None,
                    help="强制只跑到指定阶段")
    ap.add_argument("--episodes", type=int, default=PRE_RUN_EPISODES)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--target", type=int, default=None,
                    help="手动指定目标阶段（默认按可用模型自动）")
    args = ap.parse_args()

    p1 = find_model("model1", args.m1)
    p2 = None if args.only in ("12",) else find_model("model2", args.m2)
    p3 = None if args.only in ("12", "125") else find_model("model3", args.m3)

    if p1 is None:
        print("❌ 没有可用模型：需要 saves/phase1-2.zip")
        return

    if p2 is None:
        env = ArmEnv()
        target, mode = 3, "phase 1/2（只有 model1）"
    elif p3 is None:
        env = ArmEnv()
        target, mode = 6, "phase 1-5（model1 + model2）"
    else:
        env = ArmEnv()
        target, mode = 7, "全流程（三模型）"
    env.smooth_alpha = 0.5

    print(f"\n模式：{mode}")
    print(f"刷新范围：x∈[{env.spawn_x_range[0]:.2f}, {env.spawn_x_range[1]:.2f}] "
          f"y∈[{env.spawn_y_range[0]:.2f}, {env.spawn_y_range[1]:.2f}] | "
          f"带速 {env.cube_speed:.2f} m/s | 模式 {env.spawn_mode}")
    print(f"obs {env.observation_space.shape}  action {env.action_space.shape}\n")
    env_dim = env.observation_space.shape[0]
    models = [load_model(p1, env_dim, "model1"),
              load_model(p2, env_dim, "model2"),
              load_model(p3, env_dim, "model3")]

    if args.target is not None:
        target = args.target
    pre_run(env, models, target, args.episodes)
    if args.no_viewer:
        return

    with mujoco.viewer.launch_passive(env.model, env.data,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90

        obs, _ = env.reset()
        apply_thresholds(env, env.phase)
        last_phase = env.phase
        episode_reward = 0.0
        step_count = 0

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
            time.sleep(1 / 60)

            if terminated or truncated:
                if info.get("nan"):
                    status = "💥 物理爆炸 (NaN)"
                elif env.phase >= target:
                    status = "🎯 达到目标阶段"
                elif truncated:
                    status = "⏱️ 超时"
                else:
                    status = "结束"
                print(f"{status} | 步数 {step_count}, 奖励 {episode_reward:.2f}, phase {env.phase}")
                time.sleep(0.5)
                obs, _ = env.reset()
                apply_thresholds(env, env.phase)
                last_phase = env.phase
                episode_reward = 0.0
                step_count = 0


if __name__ == "__main__":
    main()
