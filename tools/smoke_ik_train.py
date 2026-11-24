import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, TrainLogCallback, WaitAwareMonitor
N=16
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(N)])
m=PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=512,
      n_epochs=4, ent_coef=0.001, device="cpu", policy_kwargs=dict(net_arch=[128,128]))
cur=CurriculumCallback(min_steps=10**9, verbose=0)
log=TrainLogCallback("scene_magnet/saves/_ik_log.txt", every=8000, curriculum=cur, verbose=0)
t0=time.time(); m.learn(total_timesteps=16000, callback=[cur, log])
print("fps = %.0f" % (16000/(time.time()-t0)))
print(open("scene_magnet/saves/_ik_log.txt", encoding="utf-8").read())
