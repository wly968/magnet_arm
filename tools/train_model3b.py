import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback
from envs.phase6 import Phase6Env
m1=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
m2=PPO.load(os.path.join(SCENE,"saves","phase34_retract.zip"), device="cpu")
def make(): return Monitor(Phase6Env(ArmEnv(), m1, m2))
env=DummyVecEnv([make for _ in range(16)])
m3=PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=1024,
       n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, device="cpu",
       policy_kwargs=dict(net_arch=[128,128]), seed=0, target_kl=0.02)
t0=time.time(); m3.learn(total_timesteps=800_000); print("model3 %.0fs"%(time.time()-t0))
m3.save(os.path.join(SCENE,"saves","phase6_home.zip"))
