import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
for det in (True,False):
    N=30; succ=0; stills=[]; qs=[]; a2s=[]; steps=[]
    for ep in range(N):
        obs,_=env.reset(); term=False; ms=0; q=[]; a2=[]
        for t in range(env.max_steps):
            a,_=m.predict(obs,deterministic=det)
            obs,r,term,trunc,info=env.step(a)
            ms=max(ms,env.still_count)
            if env.phase==2:
                d=env.data; q.append(float(d.qpos[env.finger_qpos_adr]))
                rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
                a2.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            if term or trunc: break
        succ+=int(term and env.phase==3); stills.append(ms); steps.append(t+1)
        if q: qs.append(np.mean(q)); a2s.append(np.mean(a2))
    print("det" if det else "stoch", f"phase2_success {succ}/{N}",
          "still", round(float(np.mean(stills)),1),
          "q", round(float(np.mean(qs)),4) if qs else None,
          "angle2", round(float(np.mean(a2s)),3) if a2s else None,
          "steps", round(float(np.mean(steps)),0))
