import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
for name,path in (("phase2",os.path.join(SCENE,"saves","phase2.zip")),("scene_ppo",os.path.join(SCENE,"ppo_arm_reach.zip"))):
    model=PPO.load(path, env=env)
    N=60; succ=0; phases={}; steps=[]; qs=[]
    for ep in range(N):
        obs,_=env.reset()
        term=False
        for t in range(env.max_steps):
            a,_=model.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(a)
            if term or trunc: break
        phases[env.phase]=phases.get(env.phase,0)+1
        if term and env.phase==4:
            succ+=1; steps.append(t+1)
        qs.append(float(env.data.qpos[env.finger_qpos_adr]))
    print(name, f"success {succ}/{N} = {succ/N:.2f}", "phases", phases,
          "steps", (f"{np.mean(steps):.0f}" if steps else "-"))
