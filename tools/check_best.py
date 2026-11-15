import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
for name,path in (("best_model",os.path.join(SCENE,"saves","best_model.zip")),
                  ("ppo_0923",os.path.join(SCENE,"ppo_arm_reach.zip"))):
    m=PPO.load(path, env=env); N=60; ends={}; a1=[]; a2=[]; q=[]; d=[]; succ=0
    for ep in range(N):
        obs,_=env.reset(); term=False; ms=0; ep_a2=[]; ep_a1=[]; ep_q=[]; ep_d=[]
        for t in range(env.max_steps):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(act)
            ms=max(ms,env.still_count)
            if env.phase==1:
                d0=env.data; ee=d0.site_xpos[env.ee_id]
                ep_d.append(float(np.linalg.norm(ee-(env.box_pos+np.array([0,0,0.15])))))
                fwd=ee-d0.xanchor[env.arm_joints[3]]; fwd=fwd/np.linalg.norm(fwd)
                ep_a1.append(float(np.dot(fwd,[0,0,-1])))
            if env.phase==2:
                d0=env.data; ee=d0.site_xpos[env.ee_id]
                rot=d0.geom_xmat[env.box_geom_id].reshape(3,3); px=d0.geom_xmat[env.palm_id].reshape(3,3)
                ep_a2.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
                ep_q.append(float(d0.qpos[env.finger_qpos_adr]))
                ep_d.append(float(np.linalg.norm(ee-(env.box_pos+np.array([0,0,0.15])))))
            if term or trunc: break
        ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(term)
        a2 += ep_a2; q += ep_q; a1 += ep_a1
    print(name, f"grasp {succ}/{N}", "ends", ends,
          "| a1", round(float(np.mean(a1)),3) if a1 else None,
          "a2", round(float(np.mean(a2)),3) if a2 else None,
          "q", round(float(np.mean(q)),4) if q else None)
