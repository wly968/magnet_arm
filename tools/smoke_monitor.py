import os, sys
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train_phase34 import make_env, N_ENVS
m1 = PPO.load(os.path.join(SCENE,"saves","phase2.zip"), device="cpu")
env = DummyVecEnv([make_env(m1) for _ in range(4)])
model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=512, n_epochs=4,
            learning_rate=3e-4, device="cpu")
model.learn(total_timesteps=20480)
