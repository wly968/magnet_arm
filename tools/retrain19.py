import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback
from envs.phase34 import Phase34Env
from envs.phase6 import Phase6Env

# ---- model1：phase 0/1/2，从零训练（obs 已是 19 维） ----
env1=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(24)])
m1=PPO("MlpPolicy", env1, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=1024,
       n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, device="cpu",
       policy_kwargs=dict(net_arch=[128,128]), seed=0, target_kl=0.02)
t0=time.time(); m1.learn(total_timesteps=1_500_000, callback=CurriculumCallback()); print("model1 %.0fs"%(time.time()-t0))
m1.save(os.path.join(SCENE,"ppo_arm_reach.zip"))

# ---- model2：phase 3-5 ----
def make2(): return Monitor(Phase34Env(ArmEnv(), m1))
env2=DummyVecEnv([make2 for _ in range(24)])
m2=PPO("MlpPolicy", env2, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=1024,
       n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, device="cpu",
       policy_kwargs=dict(net_arch=[128,128]), seed=0, target_kl=0.02)
CurriculumCallback.SCHEDULE=[(0,0.060,0.930,0),(500_000,0.050,0.945,0),(1_200_000,0.045,0.950,1)]
t0=time.time(); m2.learn(total_timesteps=2_000_000, callback=CurriculumCallback()); print("model2 %.0fs"%(time.time()-t0))
m2.save(os.path.join(SCENE,"saves","phase34_retract.zip"))

# ---- model3：phase 6 ----
def make3(): return Monitor(Phase6Env(ArmEnv(), m1, m2))
env3=DummyVecEnv([make3 for _ in range(16)])
m3=PPO("MlpPolicy", env3, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=1024,
       n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, device="cpu",
       policy_kwargs=dict(net_arch=[128,128]), seed=0, target_kl=0.02)
t0=time.time(); m3.learn(total_timesteps=800_000); print("model3 %.0fs"%(time.time()-t0))
m3.save(os.path.join(SCENE,"saves","phase6_home.zip"))
print("ALL DONE")
