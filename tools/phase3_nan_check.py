import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
N=40; succ=0; nans=0; terms={}; steps=[]; qs=[]; fs=[]
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    if term and env.phase==4: succ+=1
    if info.get("nan"): nans+=1
    terms[env.phase]=terms.get(env.phase,0)+1
    steps.append(t+1)
    if env.phase==4:
        qs.append(float(env.data.qpos[env.finger_qpos_adr]))
    if t+1 < 150 and info.get("nan"):
        pass
print(f"phase3/4 success {succ}/{N}; nan {nans}/{N}; endphases {terms}; steps mean {np.mean(steps):.0f}")
if qs: print("success q", round(float(np.mean(qs)),4))
