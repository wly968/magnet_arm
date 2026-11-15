import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"))
m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
m3=PPO.load(os.path.join(SCENE,"saves","model3_p4.zip"), env=env)
N=60; succ=0; ends={}; steps=[]; qs=[]; a2=[]
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        if env.phase<3: pol=m1
        elif env.phase==3: pol=m2
        else: pol=m3
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase in (3,4):
            qs.append(float(env.data.qpos[env.finger_qpos_adr]))
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==5); steps.append(t+1)
print("triple:",f"{succ}/{N}",ends,"steps",round(float(np.mean(steps)),0))
if qs: print("q mean %.4f min %.4f max %.4f" % (np.mean(qs),min(qs),max(qs)))
