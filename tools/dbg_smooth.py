import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","ppo_arm_reach_smooth.zip"), device="cpu")
env=ArmEnv(); env.max_steps=250
env._set_dist(0.03); env._set_angle1(0.96); env._set_still_count(2)
for ep in range(3):
    o,_=env.reset(); best=9; hits=0
    for t in range(env.max_steps):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        if env.phase==1:
            tgt=env.box_pos+env.pick_offset
        elif env.phase==2:
            tgt=env.box_pos+env.grasp_offset
        else:
            break
        d=float(np.linalg.norm(env.data.site_xpos[env.ee_id]-tgt))
        best=min(best,d); hits+=int(env.cube_hit)
        if term or tr: break
    print("ep%d phase=%d best_dist=%.4f cube_hit_steps=%d ee=%s cube=%s"
          % (ep, env.phase, best, hits, np.round(env.data.site_xpos[env.ee_id],2), np.round(env.box_pos,2)))
