import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback
N=8
env = DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(N)])
model = PPO.load(os.path.join(SCENE,"saves","arm5dof.zip"), env=env, device="cpu")
model.learn(total_timesteps=2_000_000, callback=CurriculumCallback(verbose=1), reset_num_timesteps=False)
model.save(os.path.join(SCENE,"saves","arm5dof_long.zip"))
print("LONG TRAIN DONE")
