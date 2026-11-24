import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, WaitAwareMonitor, DiagCallback
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(8)])
m=PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=4, device="cpu",
      policy_kwargs=dict(net_arch=[64,64]))
m.learn(4000, callback=DiagCallback(every=800), progress_bar=False)
print("smoke done")
