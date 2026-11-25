"""phase 1/2 精度微调：从 saves/phase1-2.zip 热启动，按 FINE_SCHEDULE 收紧 dwell。

    python train_fine.py                 # 默认 300 万步，22 档课程
    python train_fine.py --steps 400000  # 自定步数

产物：`ppo_arm_reach.zip`（微调后的 model1，可复制成 saves/phase1-2.zip 使用）
"""
import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.arm_env import WaitAwareMonitor
from envs.arm_fine import ArmFineEnv, FINE_SCHEDULE
from callbacks import (CurriculumCallback, PhaseTrackerCallback, DiagCallback,
                       TrainLogCallback)

SAVES = os.path.join(HERE, "saves")
N_ENVS = 24


def make_env():
    return WaitAwareMonitor(ArmFineEnv())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--start-stage", type=int, default=0, help="从课程第几档开始（0 起）")
    args = ap.parse_args()

    device = "cpu"
    env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env])

    # 微调课程：沿用同一套性能门控（确定性探测 30 局 + 连续 2 次达标）
    curriculum = CurriculumCallback(
        min_steps=100_000, success_threshold=0.6,
        probe_episodes=30, probe_every=100_000, confirm_passes=2,
        start_idx=args.start_stage,
    )
    curriculum.SCHEDULE = FINE_SCHEDULE          # 覆盖成精度档位
    train_log = TrainLogCallback(os.path.join(SAVES, "train_log_fine.txt"),
                                 every=20_000, curriculum=curriculum)

    ckpt = CheckpointCallback(save_freq=max(1, 200_000 // N_ENVS),
                              save_path=os.path.join(SAVES, "ckpt_fine"),
                              name_prefix="fine", save_vecnormalize=False)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVES, "best_fine"),
        log_path=os.path.join(SAVES, "eval_logs_fine"),
        eval_freq=max(5000, 200_000 // N_ENVS), n_eval_episodes=20,
        deterministic=True, verbose=0,
    )

    warm = os.path.join(SAVES, "phase1-2.zip")
    if os.path.exists(warm):
        model = PPO.load(warm, env=env, device=device, learning_rate=args.lr)
        print(f"热启动自 {os.path.basename(warm)}（lr={args.lr}）")
    else:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                    n_steps=512, batch_size=1024, n_epochs=5,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
                    device=device, policy_kwargs=dict(net_arch=[128, 128]),
                    target_kl=0.02, seed=0)
        print("⚠️ 没有找到 phase1-2.zip，从零开始训练")

    print(f"开始微调：{N_ENVS} 环境，{args.steps} 步 | 课程档位 {len(FINE_SCHEDULE)} 档")
    model.learn(total_timesteps=args.steps,
                callback=[curriculum, PhaseTrackerCallback(), train_log, ckpt, eval_cb])
    out = os.path.join(HERE, "ppo_arm_reach.zip")
    model.save(out)
    print("微调完成，已保存：", out, "（达标后可复制成 saves/phase1-2.zip）")


if __name__ == "__main__":
    main()
