"""统计 phase 2 中"末端与方块的相对速度"分布，用于确定阈值。

只读数据，不修改判定逻辑。默认统计条件：
  末端到抓取点距离 < dist、姿态角 > angle、末端高度在 [box_z+0.05, target_z+0.01] 内
在这些"位置已到位"的帧里，记录 |v_ee - v_cube|（世界系线速度，m/s）。

用法：
    python tools/rel_vel_threshold.py                     # 100 局
    python tools/rel_vel_threshold.py --episodes 200
    python tools/rel_vel_threshold.py --dist 0.03 --angle 0.98
"""
import argparse
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scene_magnet")
sys.path.insert(0, SCENE)

from stable_baselines3 import PPO  # noqa: E402

from envs.arm_env import ArmEnv  # noqa: E402

CANDIDATES = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
DT = 0.01          # 控制周期 = 5 × 0.002 s


def ee_linear_velocity(env):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_SITE,
                             env.ee_id, res, 0)
    return res[3:].copy()


def angle1_of(env):
    ee = env.data.site_xpos[env.ee_id]
    j3 = env.data.xanchor[env.arm_joints[3]]
    f = (ee - j3) / np.linalg.norm(ee - j3)
    return float(np.dot(f, [0.0, 0.0, -1.0]))


def run_episode(env, model, dist_thr, angle_thr, limit):
    obs, _ = env.reset()
    env._set_dist(dist_thr)
    env._set_angle1(angle_thr)
    frames = []          # 每帧的 (dist, angle1, rel_vel)
    for _ in range(limit):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if env.phase == 2:
            ee = env.data.site_xpos[env.ee_id].copy()
            target = env.box_pos + env.grasp_offset
            d = float(np.linalg.norm(ee - target))
            a1 = angle1_of(env)
            v_rel = float(np.linalg.norm(ee_linear_velocity(env) - env.box_vel))
            frames.append((d, a1, v_rel, float(ee[2]), float(env.box_pos[2]), float(target[2])))
        if env.phase >= 3 or terminated or truncated:
            break
    return frames, env.phase


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--model", default=None, help="默认沿用 test.py 的 model1")
    ap.add_argument("--dist", type=float, default=0.03)
    ap.add_argument("--angle", type=float, default=0.98)
    args = ap.parse_args()

    import test as t
    env = ArmEnv()
    path = t.find_model("model1", args.model)
    if path is None:
        print("找不到 model1（ppo_arm_reach.zip / saves/phase1.zip）")
        return
    print(f"model1: {os.path.basename(path)}")
    model = PPO.load(path, device="cpu")
    if int(model.observation_space.shape[0]) != env.observation_space.shape[0]:
        model = t.ObsAdapter(model, env.observation_space.shape[0])
        print("观测维度不一致，已自动适配")

    limit = env.max_steps + env.max_wait_steps + 10
    per_ep_min = []
    all_frames = []
    reached = 0

    for _ in range(args.episodes):
        frames, end_phase = run_episode(env, model, args.dist, args.angle, limit)
        if end_phase >= 2:
            reached += 1
        if not frames:
            continue
        ok = [f for f in frames
              if f[0] < args.dist and f[1] > args.angle
              and f[3] < f[5] + 0.01 and f[3] > f[4] + 0.05]
        if ok:
            per_ep_min.append(min(f[2] for f in ok))
            all_frames.extend(f[2] for f in ok)

    n = args.episodes
    print(f"\n局数 {n}，进入 phase 2 的 {reached} 局，"
          f"至少有一帧位置到位的 {len(per_ep_min)} 局")
    if not all_frames:
        print("没有采集到“位置到位”的帧，无法定阈值。"
              "可能是模型还没学会 phase 2，或 dist/angle 设得太严。")
        return

    print(f"\n位置到位帧数 {len(all_frames)}")
    print("  单帧 rel_vel  (m/s)："
          f"p10 {pct(all_frames,10):.3f}  p25 {pct(all_frames,25):.3f}  "
          f"中位 {pct(all_frames,50):.3f}  p75 {pct(all_frames,75):.3f}  "
          f"p90 {pct(all_frames,90):.3f}")
    print("  每局最小 rel_vel (m/s)："
          f"p10 {pct(per_ep_min,10):.3f}  p25 {pct(per_ep_min,25):.3f}  "
          f"中位 {pct(per_ep_min,50):.3f}  p75 {pct(per_ep_min,75):.3f}  "
          f"p90 {pct(per_ep_min,90):.3f}")

    print("\n阈值候选（成功率上限 = 该局最小 rel_vel 低于阈值的比例）")
    print("  阈值(m/s)   可达局数   比例     4 步内漂移(mm)")
    for c in CANDIDATES:
        k = sum(1 for v in per_ep_min if v < c)
        print(f"  {c:8.3f}   {k:6d}   {100.0*k/len(per_ep_min):5.1f}%   "
              f"{c*4*DT*1000:10.1f}")


if __name__ == "__main__":
    main()
