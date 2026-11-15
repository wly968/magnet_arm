import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
for d_,a_,s_ in ((0.06,0.90,0),(0.03,0.96,2)):
    env=ArmEnv(); env.max_steps=250
    env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
    ok=0
    for ep in range(200):
        o,_=env.reset()
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            if term or tr: break
        ok+=int(env.phase>=3)
    print("新 ppo_arm_reach  阈值 %.2f/%.2f/%d -> %d/200 = %.1f%%" % (d_,a_,s_,ok,100*ok/200))
