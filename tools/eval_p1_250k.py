import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(ROOT,"tools","retrain","p1_250k"), env=env)
N=60; ends={}; succ=0; d=[]; a1=[]; a2=[]; q=[]; steps=[]
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==1:
            dd=env.data; ee=dd.site_xpos[env.ee_id]
            d.append(float(np.linalg.norm(ee-(env.box_pos+np.array([0,0,0.15])))))
            fwd=ee-dd.xanchor[env.arm_joints[3]]; fwd=fwd/np.linalg.norm(fwd)
            a1.append(float(np.dot(fwd,[0,0,-1])))
            rot=dd.geom_xmat[env.box_geom_id].reshape(3,3); px=dd.geom_xmat[env.palm_id].reshape(3,3)
            a2.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            q.append(float(dd.qpos[env.finger_qpos_adr]))
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(term); steps.append(t+1)
print(f"success {succ}/{N} ends {ends} steps {np.mean(steps):.0f}")
print("dist",round(float(np.mean(d)),3),"a1",round(float(np.mean(a1)),3),"a2",round(float(np.mean(a2)),3),"q",round(float(np.mean(q)),4))
