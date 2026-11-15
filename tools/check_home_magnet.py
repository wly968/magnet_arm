import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv()
for ep in range(5):
    obs,_=env.reset(seed=ep)
    d=env.data; ee=d.site_xpos[env.ee_id]
    box=env.box_pos
    target=box+np.array([0,0,0.15])
    dist=np.linalg.norm(ee-target)
    j3=d.xanchor[env.arm_joints[3]]
    fwd=(ee-j3)/np.linalg.norm(ee-j3)
    a1=float(np.dot(fwd,[0,0,-1]))
    print("ep%d ee=%s box=%s dist=%.3f angle1=%.3f qpos=%s" % (ep, np.round(ee,3), np.round(box,3), dist, a1, np.round(d.qpos[:6],3)))
