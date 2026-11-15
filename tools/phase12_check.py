import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
for name,path in (("phase1",os.path.join(SCENE,"saves","phase1.zip")),
                  ("phase2",os.path.join(SCENE,"saves","phase2.zip")),
                  ("ppo16",os.path.join(SCENE,"ppo_arm_reach.zip"))):
    model=PPO.load(path, env=env)
    N=30; phases={}; succ=0; ph2=0; steps=[]; qs=[]; a2s=[]; stills=[]
    for ep in range(N):
        obs,_=env.reset(); term=False; maxph=1; ms=0
        for t in range(env.max_steps):
            a,_=model.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(a)
            maxph=max(maxph,env.phase); ms=max(ms,env.still_count)
            if term or trunc: break
        phases[maxph]=phases.get(maxph,0)+1
        ph2 += int(maxph>=2)
        succ += int(term and env.phase==4)
        steps.append(t+1); qs.append(float(env.data.qpos[env.finger_qpos_adr])); stills.append(ms)
    print(name, f"phase2_entry {ph2}/{N}", f"grasp {succ}/{N}", "maxphases", phases,
          "steps", round(float(np.mean(steps)),0), "q_mean", round(float(np.mean(qs)),4), "still_mean", round(float(np.mean(stills)),1))
