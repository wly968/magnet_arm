import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
for still in (1,2):
    env=ArmEnv(); env.max_steps=250
    env._set_dist(0.03); env._set_angle1(0.96); env._set_still_count(still)
    ok=0; mx=0
    for ep in range(60):
        o,_=env.reset()
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            mx=max(mx,env.still_count)
            if term or tr: break
        ok+=int(env.phase>=3)
    print("dist 0.03 & angle 0.96 & still %d -> %d/60, max still_count %d" % (still,ok,mx))
