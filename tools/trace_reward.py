import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv(); obs,_=env.reset(seed=0)
tot=0.0
for t in range(150):
    a=np.zeros(4)
    obs,r,term,tr,info=env.step(a)
    tot+=r
    if t<8 or t%30==0:
        print("t",t,"reward",round(float(r),3),"total",round(tot,1),"phase",env.phase)
    if term or tr: break
print("episode total", round(tot,1), "steps", t+1)
