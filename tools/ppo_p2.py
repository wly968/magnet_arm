import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), env=env)
N=60; ph2=0; q=[]; a2=[]; st=[]; ds=[]; ends={}
for ep in range(N):
    obs,_=env.reset(); term=False; ms=0
    for t in range(env.max_steps):
        act,_=m.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(act)
        ms=max(ms,env.still_count)
        if env.phase==2:
            d=env.data; q.append(float(d.qpos[env.finger_qpos_adr]))
            rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
            a2.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            ee=d.site_xpos[env.ee_id]; ds.append(float(np.linalg.norm(ee-(env.box_pos+np.array([0,0,0.15])))))
        if term or trunc: break
    ph2+=int(env.phase>=2); ends[env.phase]=ends.get(env.phase,0)+1
print("phase2 entered",ph2,"/",N,"end",ends)
print("q",round(float(np.mean(q)),4) if q else None,"angle2",round(float(np.mean(a2)),3) if a2 else None,
      "dist",round(float(np.mean(ds)),3) if ds else None,"still",round(float(np.mean(st)),1) if st else None)
