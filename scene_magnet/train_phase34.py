"""训练 phase 3/4 的模型（model2）。

训练前先加载 phase2.zip（phase 1/2 的模型），每次 reset 都先跑它直到
phase>=3，把真实的交接姿态作为 phase 3/4 的起点，避免在分布外姿态上训练。
"""
import os
import sys
import functools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from envs.arm_env import ArmEnv
from envs.phase34 import Phase34Env

SAVES = os.path.join(HERE, "saves")
N_ENVS = 8
TOTAL_STEPS = 6_000_000


def make_env(model1):
    def _init():
        # 必须套 Monitor，否则 SB3 统计不到 ep_len / ep_rew
        return Monitor(Phase34Env(ArmEnv(), model1))
    return _init


if __name__ == "__main__":
    device = "cpu"
    model1 = PPO.load(os.path.join(SAVES, "phase2.zip"), device=device)
    print("已加载 phase2.zip 作为 phase 1/2 的交接模型")
    print("开始训练 phase 3/4（每次 reset 先跑 phase 1/2）...")

    env = DummyVecEnv([make_env(model1) for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env(model1)])
    model2 = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVES, "best_phase34"),
        log_path=os.path.join(SAVES, "eval_logs_phase34"),
        eval_freq=max(2000, 50000 // N_ENVS),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model2.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb)
    out = os.path.join(SAVES, "phase34.zip")
    model2.save(out)
    print("训练完成，模型已保存为", out)
