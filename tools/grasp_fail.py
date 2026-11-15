import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
for ep in range(60):
    obs,_=env.reset()
    box=env.box_pos.copy(); term=False
    for t in range(env.max_steps):
        a,_=model.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    if env.phase==3:
        d=env.data; ee=d.site_xpos[env.ee_id]; q=d.qpos[env.finger_qpos_adr]
        print(f"ep{ep} box={np.round(box,2)} phase3_fail ee={np.round(ee,3)} q={q:.4f} t={t+1}")
