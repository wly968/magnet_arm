import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env)
print("dist_thr",env.dist_threshold,"a1_thr",env.angle1_threshold,"a2_thr",env.angle2_threshold)
N=20; ph2=0; q2=[]; a2=[]; stills=[]; dists=[]; succ=0; endph={}
for ep in range(N):
    obs,_=env.reset(); term=False; maxph=1; ms=0; qs=[]; a2s=[]; ds=[]
    for t in range(env.max_steps):
        a,_=model.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        maxph=max(maxph,env.phase); ms=max(ms,env.still_count)
        if env.phase==2:
            d=env.data; q=float(d.qpos[env.finger_qpos_adr]); qs.append(q)
            rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
            a2s.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            ee=d.site_xpos[env.ee_id]; ds.append(float(np.linalg.norm(ee-(env.box_pos+np.array([0,0,0.15])))))
        if term or trunc: break
    endph[maxph]=endph.get(maxph,0)+1
    ph2+=int(maxph>=2)
    if qs: q2.append(np.mean(qs)); a2.append(np.mean(a2s)); dists.append(np.mean(ds))
    stills.append(ms); succ+=int(term and maxph>=3)
print("entry2",ph2,"/",N,"end phases",endph)
print("phase2 q mean",round(float(np.mean(q2)),4) if q2 else None,
      "angle2 mean",round(float(np.mean(a2)),3) if a2 else None,
      "dist mean",round(float(np.mean(dists)),3) if dists else None,
      "still mean",round(float(np.mean(stills)),1))
