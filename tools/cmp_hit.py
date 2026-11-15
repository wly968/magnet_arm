import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("ppo_arm_reach.zip","saves/ppo_arm_reach_smooth.zip"):
    m=PPO.load(os.path.join(SCENE,name), device="cpu")
    env=ArmEnv(); env.max_steps=250
    env._set_dist(0.03); env._set_angle1(0.96); env._set_still_count(0)
    ok=0; hit_eps=0; hit_steps=0; tot=0
    for ep in range(80):
        o,_=env.reset(); had=False
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            tot+=1
            if env.cube_hit: hit_steps+=1; had=True
            if term or tr: break
        ok+=int(env.phase>=3); hit_eps+=int(had)
    print("%-34s 完成 %2d/80  有碰撞局 %2d  碰撞步 %d/%d"
          % (name, ok, hit_eps, hit_steps, tot))
