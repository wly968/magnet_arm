import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","curriculum_check.zip"), device="cpu")
ph={}; steps=[]
for ep in range(200):
    o,_=env.reset()
    for t in range(env.max_steps):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        if term or tr: break
    ph[env.phase]=ph.get(env.phase,0)+1; steps.append(t+1)
print("200 局阶段分布:", ph, "平均步数 %.1f"%np.mean(steps))
