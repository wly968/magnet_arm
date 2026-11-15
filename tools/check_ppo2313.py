import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
for name,path in (("ppo_2313",os.path.join(SCENE,"ppo_arm_reach.zip")),
                  ("phase2_new",os.path.join(SCENE,"saves","phase2.zip"))):
    m=PPO.load(path, env=env); N=100; ph={}; succ=0; steps=[]; fails={}
    for ep in range(N):
        obs,_=env.reset(); term=False
        for t in range(env.max_steps):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(a)
            if term or trunc: break
        ph[env.phase]=ph.get(env.phase,0)+1
        if env.phase==4: succ+=1
        steps.append(t+1)
    print(name, "success", f"{succ}/{N}", "end", ph, "steps", round(float(np.mean(steps)),0))
