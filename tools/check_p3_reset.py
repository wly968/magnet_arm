import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv()
obs,_=env.reset(option={"phase":3})
d=env.data; ee=d.site_xpos[env.ee_id]
rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
a2=abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0]))))
q=float(d.qpos[env.finger_qpos_adr])
print("phase", env.phase)
print("ee", np.round(ee,3), "box", np.round(env.box_pos,3), "dz", round(float(ee[2]-env.box_pos[2]),3))
print("angle2", round(a2,3), "q", round(q,4))
