import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, TrainLogCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(8)])
m=PPO.load("scene_magnet/saves/ckpt/ckpt_2499960_steps.zip", env=env, device="cpu")
cur=CurriculumCallback(min_steps=10**9, verbose=0)
log=TrainLogCallback("scene_magnet/saves/_test_log.txt", every=8000, curriculum=cur, verbose=0)
m.learn(total_timesteps=24000, callback=[cur, log])
print(open("scene_magnet/saves/_test_log.txt", encoding="utf-8").read())
