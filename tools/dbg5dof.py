import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","arm5dof_long.zip"), device="cpu")
env=ArmEnv(); env.max_steps=250
env._set_dist(0.04); env._set_angle1(0.95); env._set_still_count(1)
for ep in range(2):
    o,_=env.reset(seed=ep); 
    print("--- ep",ep,"cube",np.round(env.box_pos,2))
    for t in range(120):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        if t%20==0 or term:
            ee=env.data.site_xpos[env.ee_id]
            tgt=env.box_pos+(env.pick_offset if env.phase==1 else env.grasp_offset)
            j3=env.data.xanchor[env.arm_joints[3]]
            f=(ee-j3)/np.linalg.norm(ee-j3)
            print(" t%3d phase=%d dist=%.3f ang=%.3f still=%d q=%s" %
                  (t,env.phase,float(np.linalg.norm(ee-tgt)),float(np.dot(f,[0,0,-1])),env.still_count,np.round(env.data.qpos[env.arm_qpos_adr],2)))
        if term: break
