import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), device="cpu")
m2=PPO.load(os.path.join(SCENE,"saves","phase34.zip"), device="cpu")
for ep in range(5):
    o,_=env.reset(); 
    for t in range(env.max_steps):
        pol = m1 if env.phase < 3 else m2
        a,_=pol.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        if term or tr: break
    print("ep%d phase=%d t=%d term=%s trunc=%s info=%s ee=%s" %
          (ep, env.phase, t+1, term, tr, {k:v for k,v in info.items()}, np.round(env.data.site_xpos[env.ee_id],2)))
