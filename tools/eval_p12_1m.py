import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(ROOT,"tools","retrain","p12_1m"), env=env)
N=60; ends={}; succ=0; steps=[]; q=[]; a1=[]; a2=[]
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==2:
            dd=env.data; rot=dd.geom_xmat[env.box_geom_id].reshape(3,3); px=dd.geom_xmat[env.palm_id].reshape(3,3)
            a2.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            q.append(float(dd.qpos[env.finger_qpos_adr]))
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==4); steps.append(t+1)
print(f"grasp {succ}/{N} ends {ends} steps {np.mean(steps):.0f}")
print("a2",round(float(np.mean(a2)),3) if a2 else None,"q",round(float(np.mean(q)),4) if q else None)
