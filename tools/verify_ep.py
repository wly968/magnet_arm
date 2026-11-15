import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv

env = ArmEnv()
tot = 0.0
o,_ = env.reset(seed=0)
for t in range(env.max_steps):
    o,r,term,tr,info = env.step(np.zeros(4))
    tot += r
    if term or tr: break
print("zero-action episode total:", round(tot,2), "steps", t+1, "phase", env.phase)
