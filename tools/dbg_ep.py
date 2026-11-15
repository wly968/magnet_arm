import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"saves","magnet_5dim_s1.zip"), device="cpu")
for ep in range(2):
    o,_=env.reset(seed=ep); print("--- ep",ep,"box",np.round(env.box_pos,3))
    for t in range(12):
        a,_=model.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        ee=env.data.site_xpos[env.ee_id]
        tgt=np.array([env.box_pos[0],env.box_pos[1],env.box_pos[2]+0.15])
        print(" t",t,"a",np.round(a,2),"r",round(float(r),2),"dist %.3f"%np.linalg.norm(ee-tgt),
              "ee_z %.3f"%ee[2],"box_z %.3f"%env.box_pos[2],"info",{k:v for k,v in info.items() if k!="final_phase"})
        if term or tr: break
