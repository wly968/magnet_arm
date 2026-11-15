import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
e=ArmEnv(); o,_=e.reset(seed=0)
for t in range(150):
    o,r,term,tr,info=e.step(np.zeros(4))
    if t<5 or t%15==0:
        print("t",t,"r",round(float(r),2),"box",np.round(e.box_pos,3),"phase",e.phase,"weld",int(e.data.eq_active[e.weld_id]) if e.weld_id is not None else -1)
    if term or tr: break
