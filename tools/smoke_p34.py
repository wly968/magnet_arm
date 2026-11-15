import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
from envs.phase34 import Phase34Env
m1 = PPO.load(os.path.join(SCENE,"saves","phase2.zip"), device="cpu")
env = Phase34Env(ArmEnv(), m1)
o,_ = env.reset()
print("交接后 phase", env.unwrapped.phase, "obs", o.shape)
o2,r,t,tr,info = env.step(np.zeros(4))
print("step ok r=%.2f phase=%d" % (r, env.unwrapped.phase))
