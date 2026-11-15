import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv(); obs,_=env.reset(seed=0)
for t in range(6):
    a=np.zeros(4)
    obs,r,term,tr,info=env.step(a)
    d=env.data; ee=d.site_xpos[env.ee_id]
    tgt=np.array([env.box_pos[0],env.box_pos[1],env.box_pos[2]+0.15])
    dist=float(np.linalg.norm(ee-tgt))
    j3=d.xanchor[env.arm_joints[3]]
    fwd=(ee-j3)/np.linalg.norm(ee-j3)
    a1=float(np.dot(fwd,[0,0,-1]))
    print("t",t,"r",round(float(r),4),"dist",round(dist,4),"angle1",round(a1,4),"ee",np.round(ee,3),"j3",np.round(j3,3))
