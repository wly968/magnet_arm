import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
m2=PPO.load(os.path.join(ROOT,"tools","retrain","model2_p3"), env=env)
N=60; succ=0; ends={}; steps=[]; a2s=[]; qs=[]
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        a,_=(m1 if env.phase<3 else m2).predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            d=env.data; rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
            a2s.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            qs.append(float(d.qpos[env.finger_qpos_adr]))
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==4); steps.append(t+1)
print(f"DUAL grasp {succ}/{N} ends {ends} steps {np.mean(steps):.0f}")
print("p3 a2",round(float(np.mean(a2s)),3) if a2s else None,"q",round(float(np.mean(qs)),4) if qs else None)
