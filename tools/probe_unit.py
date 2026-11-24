import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(4)])
m=PPO.load("scene_magnet/saves/ckpt/ckpt_499992_steps.zip", env=env, device="cpu")
cur=CurriculumCallback(min_steps=1, probe_episodes=20, verbose=0)
cur.init_callback(m)
print("probe 档位1 =", cur._probe((0.05,0.94,0,0.20)))
print("probe 档位2 =", cur._probe((0.045,0.95,0,0.10)))
