import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(ROOT,"tools","retrain","residual_ft"), env=env)
N=40; succ=0; nans=0; ends={}; steps=[]; ees=[]; qs=[]
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            ees.append(float(env.data.site_xpos[env.ee_id][2])); qs.append(float(env.data.qpos[env.finger_qpos_adr]))
        if term or trunc: break
    if term and env.phase==4: succ+=1
    if info.get("nan"): nans+=1
    ends[env.phase]=ends.get(env.phase,0)+1; steps.append(t+1)
print(f"success {succ}/{N}; nan {nans}/{N}; ends {ends}; steps {np.mean(steps):.0f}")
if ees: print("ee_z", round(float(np.mean(ees)),3), "q", round(float(np.mean(qs)),4))
