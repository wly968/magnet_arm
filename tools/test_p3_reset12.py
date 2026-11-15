import os, sys, time, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE); os.chdir(SCENE)
from train_phase3 import Phase3Env
env=Phase3Env()
t0=time.time(); obs,info=env.reset(); t1=time.time()
d=env.env.data; ee=d.site_xpos[env.env.ee_id]
rot=d.geom_xmat[env.env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.env.palm_id].reshape(3,3)
a2=abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0]))))
print("reset time %.3fs phase %d ee %s dz %.3f q %.4f tool_roll %.3f angle2 %.3f" % (
    t1-t0, env.env.phase, np.round(ee,3), ee[2]-env.env.box_pos[2],
    float(d.qpos[env.env.finger_qpos_adr]), float(d.qpos[4]), a2))
# 多次 reset 计时
ts=[]
for i in range(20):
    t0=time.time(); env.reset(seed=i); ts.append(time.time()-t0)
print("20 resets: mean %.3fs max %.3fs" % (np.mean(ts), np.max(ts)))
