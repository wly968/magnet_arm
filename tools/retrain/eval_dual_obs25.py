import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip")); m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
N=60; succ=0; ends={}; steps=[]; qs=[]; fs=[]
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        if env.phase<3:
            a,_=m1.predict(obs[:23],deterministic=True)
        else:
            a,_=m2.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            qs.append(float(env.data.qpos[env.finger_qpos_adr])); fs.append((env.left_force,env.right_force))
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==4); steps.append(t+1)
print("dual:",f"{succ}/{N}",ends,"steps",round(float(np.mean(steps)),0))
if qs: print("p3 q mean %.4f min %.4f max %.4f" % (np.mean(qs),min(qs),max(qs)))
if fs: print("p3 force mean %.2f max %.2f" % (np.mean([min(f) for f in fs]), max(max(f) for f in fs)))
