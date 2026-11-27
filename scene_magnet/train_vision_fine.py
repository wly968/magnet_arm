"""视觉鲁棒性微调：从 saves/phase1-2.zip 热启动，在带视觉噪声的环境里继续训练。

    cd scene_magnet
    python train_vision_fine.py                 # 默认 80 万步
    python train_vision_fine.py --steps 1500000

产物：`ppo_arm_reach.zip`（达标后复制成 saves/phase1-2.zip）
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
from envs.arm_vision_fine import ArmVisionFineEnv, VISION_SCHEDULE
from callbacks import (CurriculumCallback, PhaseTrackerCallback, DiagCallback,
                       TrainLogCallback)

SAVES = os.path.join(HERE, "saves")
N_ENVS = 24


def make_env(seed=0, sigma=(0.5, 3.0)):
    return WaitAwareMonitor(ArmVisionFineEnv(sigma_mm=sigma, seed=seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--start-stage", type=int, default=0)
    ap.add_argument("--sigma-max", type=float, default=3.0, help="最大位置噪声（mm）")
    args = ap.parse_args()

    device = "cpu"
    env = DummyVecEnv([lambda i=i: make_env(i, (0.5, args.sigma_max)) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([lambda: make_env(0, (0.5, args.sigma_max))])

    curriculum = CurriculumCallback(
        min_steps=100_000, success_threshold=0.6,
        probe_episodes=30, probe_every=100_000, confirm_passes=2,
        start_idx=args.start_stage,
        probe_env_fn=lambda: ArmVisionFineEnv(sigma_mm=(0.5, args.sigma_max)),
    )
    curriculum.SCHEDULE = VISION_SCHEDULE
    train_log = TrainLogCallback(os.path.join(SAVES, "train_log_vision.txt"),
                                 every=20_000, curriculum=curriculum)
    ckpt = CheckpointCallback(save_freq=max(1, 200_000 // N_ENVS),
                              save_path=os.path.join(SAVES, "ckpt_vision"),
                              name_prefix="vision", save_vecnormalize=False)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVES, "best_vision"),
        log_path=os.path.join(SAVES, "eval_logs_vision"),
        eval_freq=max(5000, 200_000 // N_ENVS), n_eval_episodes=20,
        deterministic=True, verbose=0,
    )

    warm = os.path.join(SAVES, "phase1-2.zip")
    if os.path.exists(warm):
        model = PPO.load(warm, env=env, device=device, learning_rate=args.lr)
        print(f"热启动自 {os.path.basename(warm)}（lr={args.lr}）")
    else:
        raise SystemExit("缺少 saves/phase1-2.zip")

    print(f"开始视觉鲁棒微调：{N_ENVS} 环境，{args.steps} 步 | 噪声 0.5~{args.sigma_max} mm "
          f"| 课程 {len(VISION_SCHEDULE)} 档")
    model.learn(total_timesteps=args.steps,
                callback=[curriculum, PhaseTrackerCallback(), train_log, ckpt, eval_cb])
    out = os.path.join(HERE, "ppo_arm_reach.zip")
    model.save(out)
    print("完成，已保存：", out)


if __name__ == "__main__":
    main()
