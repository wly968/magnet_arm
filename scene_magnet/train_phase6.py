"""训练 phase 6（回原位）的模型（model3）：输出 ppo_arm_reach.zip

reset 时先跑 model1（phase 1/2）到 phase 3，再跑 model2（phase 3-5）到 phase 6，
把真实的交接姿态作为 phase 6 的起点。

phase 6 的成功条件：手臂回到初始关节配置（|q-q_home| < 0.15 rad）
                     且 方块已滑入 B 带末端的收集篮。
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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.arm_env import ArmEnv, WaitAwareMonitor
from callbacks import PhaseTrackerCallback, CurriculumCallback, PhaseCurriculumCallback
from envs.phase6 import Phase6Env

SAVES = os.path.join(HERE, "saves")
N_ENVS = 16
TOTAL_STEPS = 1000_000
HANDOFF_STEPS = 600          # m1(约 230 步) + m2(约 150 步)，留余量

# phase 6 课程：只调"连续多少帧 q_err 在容差内"（0 = 1 帧，1 = 2 帧 ...）
P6_STAGES = [
    (0.060, 0.940, 0),   # 档1：先学会回原位
    (0.060, 0.940, 1),   # 档2：连续 2 帧
    (0.060, 0.940, 2),   # 档3：连续 3 帧
    (0.060, 0.940, 3),   # 档4：连续 4 帧（最终验收）
]
P6_START = 0

# 交接时固定使用的判据档位（0 起）
STAGE_M1 = 4                 # 第 5 档：model1（phase 1/2）
STAGE_M2 = 5                 # 第 6 档：model2（phase 3-5）


def make_env(model1, model2):
    def _init():
        env = Phase6Env(ArmEnv(), model1, model2,
                        max_handoff_steps=HANDOFF_STEPS,
                        stage1=CurriculumCallback.SCHEDULE[STAGE_M1],
                        stage2=CurriculumCallback.SCHEDULE[STAGE_M2])
        return WaitAwareMonitor(env)
    return _init


if __name__ == "__main__":
    device = "cpu"
    m1_path = os.path.join(SAVES, "phase1-2.zip")
    if not os.path.exists(m1_path):
        m1_path = os.path.join(HERE, "ppo_arm_reach.zip")
    m2_path = os.path.join(SAVES, "phase3-5.zip")
    if not os.path.exists(m2_path):
        m2_path = os.path.join(HERE, "ppo_arm_reach.zip")

    model1 = PPO.load(m1_path, device=device)
    print(f"model1: {os.path.basename(m1_path)} | obs {model1.observation_space.shape}")
    if not os.path.exists(m2_path):
        raise SystemExit("缺少 saves/phase3-5.zip，请先跑 train_phase3-5.py")
    model2 = PPO.load(m2_path, device=device)
    print(f"model2: {os.path.basename(m2_path)} | obs {model2.observation_space.shape}")

    env = DummyVecEnv([make_env(model1, model2) for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env(model1, model2)])

    ckpt = CheckpointCallback(
        save_freq=max(1, 500_000 // N_ENVS),
        save_path=os.path.join(SAVES, "ckpt_p6"),
        name_prefix="p6",
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVES, "best_p6"),
        log_path=os.path.join(SAVES, "eval_logs_p6"),
        eval_freq=max(5000, 200_000 // N_ENVS),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model3 = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4,
        n_steps=512, batch_size=1024, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
        device=device, policy_kwargs=dict(net_arch=[128, 128]),
        seed=0, target_kl=0.02,
    )
    model3 = PPO.load("./saves/phase6.zip", env=env, device="cpu", learning_rate=1e-4)
    print(f"开始训练 phase 6：{N_ENVS} 环境，{TOTAL_STEPS} 步")
    p6_curriculum = PhaseCurriculumCallback(
        make_env(model1, model2), P6_STAGES, target_phase=7,
        min_steps=200_000, success_threshold=0.6,
        probe_episodes=30, probe_every=200_000, confirm_passes=2,
        start_idx=P6_START, still_target="home_still",
    )
    model3.learn(total_timesteps=TOTAL_STEPS,
                 callback=[PhaseTrackerCallback(), p6_curriculum, ckpt, eval_cb])
    out = os.path.join(HERE, "ppo_arm_reach.zip")
    model3.save(out)
    print("训练完成，模型已保存为", out)
