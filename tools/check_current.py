import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env); m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
# current reset phase3
env.reset(option={"phase":3}); d=env.data
ee=d.site_xpos[env.ee_id]; rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
a2r=abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0]))))
yaw=float(np.arctan2(env.box_pos[1],env.box_pos[0]))
print("current reset: yaw %.3f qroll %.3f angle2 %.3f" % (yaw, float(d.qpos[4]), a2r))
# phase2 handoff
rolls=[]; a2s=[]
for ep in range(40):
    obs,_=env.reset(seed=ep); term=False
    for t in range(env.max_steps):
        a,_=m1.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            dd=env.data; rot=dd.geom_xmat[env.box_geom_id].reshape(3,3); px=dd.geom_xmat[env.palm_id].reshape(3,3)
            a2s.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            rolls.append(float(dd.qpos[4])); break
        if term or trunc: break
print("phase2 handoff: qroll mean %.3f std %.3f  angle2 mean %.3f" % (np.mean(rolls), np.std(rolls), np.mean(a2s)))
# dual
N=60; succ=0; ends={}
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==4)
print("dual:",f"{succ}/{N}",ends)
