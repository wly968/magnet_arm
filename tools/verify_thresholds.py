import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")

def measure(d_th, a_th, eps=15):
    env=ArmEnv(); env.max_steps=250
    env._set_dist(d_th); env._set_angle1(a_th); env._set_still_count(0)
    both=0; tot=0
    for ep in range(eps):
        o,_=env.reset()
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            if env.phase==2:
                tot+=1
                ee=env.data.site_xpos[env.ee_id]; tgt=env.box_pos+env.grasp_offset
                if np.linalg.norm(ee-tgt)<d_th: both+=int(env.phase>=2 and info.get("final_phase",0)>=3 or False)
            if term or tr: break
    return tot

for d_th,a_th in ((0.025,0.96),(0.03,0.96),(0.025,0.97),(0.04,0.97)):
    env=ArmEnv(); env.max_steps=250
    env._set_dist(d_th); env._set_angle1(a_th); env._set_still_count(0)
    ok=0
    for ep in range(60):
        o,_=env.reset()
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            if term or tr: break
        ok+=int(env.phase>=3)
    print("dist %.3f & angle %.2f -> phase2 完成 %d/60" % (d_th,a_th,ok))
