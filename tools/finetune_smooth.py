import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback

# 压缩课程：从旧模型微调，快速走到终点
CurriculumCallback.SCHEDULE = [
    (0,       0.040, 0.950, 1),
    (100_000, 0.035, 0.955, 1),
    (200_000, 0.030, 0.960, 1),
    (300_000, 0.030, 0.960, 2),
]
env = DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(8)])
model = PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), env=env, device="cpu", learning_rate=1e-4)
model.learn(total_timesteps=400_000, callback=CurriculumCallback(verbose=1))
model.save(os.path.join(SCENE,"saves","ppo_arm_reach_ft.zip"))
print("FT DONE")
