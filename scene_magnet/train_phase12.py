"""训练 phase 1/2 的机械臂模型（ppo_arm_reach.zip）。

加速配置（实测 2461 -> ~7900 fps，约 3.2 倍）：
  - torch 单线程 + OMP_NUM_THREADS=1（20 线程对小 MLP 是负收益）
  - 24 个并行环境
  - n_epochs 5、batch_size 1024、net [128,128]
  - mj_step(nstep=5)（在 env 里）
评估环境固定为终点阈值，这样 best_model 才是“最准”的模型。
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.arm_env import (ArmEnv, CurriculumCallback, PhaseTrackerCallback,
                          WaitAwareMonitor, DiagCallback, TrainLogCallback)

N_ENVS = 24
TOTAL_STEPS = 2000_000   # 1.5M 只够到 0.06 档，3M 才能到 0.04 档
START_STAGE = 3          # 从第几档开始（0 起）：4 = 第 5 档，跳过已攻克的 1~4 档
LEARNING_RATE = 3e-4

# 终点阈值：eval 固定用它，保证 best_model 按“最难档”挑选
EVAL_DIST, EVAL_ANGLE, EVAL_STILL = 0.032, 0.96, 1


def make_env():
    env = ArmEnv()
    env.smooth_alpha = 0.6
    return WaitAwareMonitor(env)


if __name__ == "__main__":
    env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env])
    eval_env.env_method("_set_dist", EVAL_DIST)
    eval_env.env_method("_set_angle1", EVAL_ANGLE)
    eval_env.env_method("_set_still_count", EVAL_STILL)

    curriculum = CurriculumCallback(min_steps=300_000, success_threshold=0.6,
                                    probe_episodes=30, confirm_passes=2,
                                    start_idx=START_STAGE)
    train_log = TrainLogCallback(os.path.join(HERE, "saves", "train_log.txt"),
                                 every=20_000, curriculum=curriculum)

    ckpt_callback = CheckpointCallback(
        save_freq=max(1, 500_000 // N_ENVS),      # 每 50 万步存一个
        save_path=os.path.join(HERE, "saves", "ckpt"),
        name_prefix="ckpt",
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(HERE, "saves", "best_4dof"),
        log_path=os.path.join(HERE, "saves", "eval_logs"),
        eval_freq=max(20000, 1000_000 // N_ENVS),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = PPO.load("./saves/phase2.zip", env=env, device="cpu", learning_rate=1e-4)
    '''
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE,
        n_steps=512, batch_size=1024, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
        device="cpu", policy_kwargs=dict(net_arch=[128, 128]),
        seed=0, target_kl=0.02,
    )'''
    model.ent_coef = 0.0005

    print(f"开始训练 phase 1/2：{N_ENVS} 环境，{TOTAL_STEPS} 步")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[curriculum, PhaseTrackerCallback(), DiagCallback(every=20_000),
              train_log, ckpt_callback, eval_callback],
        progress_bar=False,
    )
    model.save(os.path.join(HERE, "ppo_arm_reach"))
    print("训练完成，模型已保存为 ppo_arm_reach.zip")
