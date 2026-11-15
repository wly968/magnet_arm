import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback

N=8; STEPS=1_500_000
env = DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(N)])
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=512, batch_size=512,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.005, device="cpu",
            policy_kwargs=dict(net_arch=[256,256]), seed=0, target_kl=0.02)
model.learn(total_timesteps=STEPS, callback=CurriculumCallback(verbose=1))
model.save(os.path.join(SCENE,"saves","ppo_arm_reach_smooth.zip"))
print("TRAIN DONE")
