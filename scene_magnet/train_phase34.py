"""训练 phase 3-5 的模型（model2）：输出 ppo_arm_reach.zip

每次 reset 先让 model1（phase 1/2）跑到 phase 3，拿到真实交接姿态再开训。
到达 phase 6（回原位）立即终止——phase 6 由 model3 负责。
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.arm_env import ArmEnv, WaitAwareMonitor
from callbacks import PhaseTrackerCallback, PhaseCurriculumCallback
from envs.phase34 import Phase34Env

SAVES = os.path.join(HERE, "saves")
N_ENVS = 16
TOTAL_STEPS = 2000_000
STOP_PHASE = 6                      # 到 phase 6 就结束（phase 6 交给 model3）
HANDOFF_STEPS = 400                 # phase0(~87)+phase1(~56)+phase2(~90)，留余量
# phase 5 课程：直接复用 phase 1/2 的 SCHEDULE（dist/angle/still/rel 六档）
from callbacks import CurriculumCallback
P5_STAGES = [tuple(s) for s in CurriculumCallback.SCHEDULE]
P5_START = 0

WARM_START = True                   # 从上一轮 model2 热启动（判据已修，热身更快）


class StopAtPhase(gym.Wrapper):
    """到达指定阶段即视为回合结束（成功）。"""

    def __init__(self, env, stop_phase=6):
        super().__init__(env)
        self.stop_phase = int(stop_phase)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.phase >= self.stop_phase:
            terminated = True
            info["final_phase"] = self.env.unwrapped.phase
        return obs, reward, terminated, truncated, info


def make_env(model1):
    def _init():
        env = StopAtPhase(Phase34Env(ArmEnv(), model1,
                                     max_handoff_steps=HANDOFF_STEPS),
                          stop_phase=STOP_PHASE)
        # phase 3-5 的判据（固定，不随课程变）：放料要求 2 帧静止
        env.unwrapped._set_dist(0.05)
        env.unwrapped._set_angle1(0.945)
        env.unwrapped._set_still_count(0)
        return WaitAwareMonitor(env)
    return _init


if __name__ == "__main__":
    device = "cpu"
    m1_path = os.path.join(SAVES, "phase2.zip")          # 优先用 phase2.zip
    if not os.path.exists(m1_path):
        m1_path = os.path.join(HERE, "ppo_arm_reach.zip")
    model1 = PPO.load(m1_path, device=device)
    print(f"model1: {os.path.basename(m1_path)} | obs {model1.observation_space.shape}")

    env = DummyVecEnv([make_env(model1) for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env(model1)])

    ckpt = CheckpointCallback(
        save_freq=max(1, 500_000 // N_ENVS),
        save_path=os.path.join(SAVES, "ckpt_p35"),
        name_prefix="p35",
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVES, "best_p35"),
        log_path=os.path.join(SAVES, "eval_logs_p35"),
        eval_freq=max(5000, 500_000 // N_ENVS),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    import glob
    cands = glob.glob(os.path.join(SAVES, "ckpt_p35", "*.zip"))
    cands += [os.path.join(SAVES, "phase34_retract.zip")]
    cands = [c for c in cands if os.path.exists(c)]
    cands.sort(key=os.path.getmtime)
    warm = cands[-1] if cands else ""
    if WARM_START and warm:
        model2 = PPO.load(warm, env=env, device=device, learning_rate=1e-4)
        print(f"热启动自 {os.path.basename(warm)}（lr=1e-4）")
    else:
        model2 = PPO(
            "MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=512, batch_size=1024, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.001,
            device=device, policy_kwargs=dict(net_arch=[128, 128]),
            seed=0, target_kl=0.02,
        )
    model2 = PPO.load("./saves/phase34", env=env, device="cpu", learning_rate=1e-4)
    p5_curriculum = PhaseCurriculumCallback(
        make_env(model1), P5_STAGES, target_phase=6,
        min_steps=100_000, success_threshold=0.6,
        probe_episodes=30, probe_every=100_000, confirm_passes=2,
        start_idx=P5_START,
    )
    print(f"开始训练 phase 3-5：{N_ENVS} 环境，{TOTAL_STEPS} 步")
    model2.learn(total_timesteps=TOTAL_STEPS, callback=[PhaseTrackerCallback(), p5_curriculum, ckpt, eval_cb])
    out = os.path.join(HERE, "ppo_arm_reach.zip")
    model2.save(out)
    print("训练完成，模型已保存为", out)
