import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(8)])
m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
cb=CurriculumCallback()
m.learn(total_timesteps=20000, callback=cb)
print("课程设定 rel =", CurriculumCallback.SCHEDULE[0][4])
print("环境实际 rel_threshold =", env.get_attr("rel_threshold"))
print("环境实际 dist/angle/still =", env.get_attr("dist_threshold"), env.get_attr("angle1_threshold"), env.get_attr("still_count_threshold"))
