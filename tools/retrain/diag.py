import os, sys, numpy as np
ROOT = r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE); os.chdir(SCENE)
from stable_baselines3 import PPO
from arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(ROOT,"tools","retrain","ppo_current"), env=env)
for ep in range(5):
    obs,_=env.reset(seed=100+ep)
    for t in range(env.max_steps):
        a,_=model.predict(obs, deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    d=env.data; q=d.qpos[:4]; w=d.xanchor[env.arm_joints[3]]; tgt=env.box_pos+np.array([0,0,0.25])
    ee=d.site_xpos[env.ee_id]; ang=float(np.dot((ee-w)/np.linalg.norm(ee-w),np.array([0,0,-1.])))
    print(f"ep{ep} target={np.round(tgt,2)} q={np.round(q,2)} wrist={np.round(w,2)} dist={np.linalg.norm(w-tgt):.2f} ang={ang:.3f}")
