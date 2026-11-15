import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","arm4dof_10m.zip"), device="cpu")
env=ArmEnv(); env.max_steps=250
env._set_dist(0.03); env._set_angle1(0.96); env._set_still_count(2)
print("thresholds", env.dist_threshold, env.angle1_threshold, env.still_count_threshold)
for ep in range(5):
    o,_=env.reset(); tot=0.0; done=False
    for t in range(env.max_steps):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a); tot+=r
        if term or tr: done=True; break
    print("ep%d phase=%d term=%s reward=%.1f steps=%d info=%s" % (ep, env.phase, done, tot, t+1, {k:v for k,v in info.items()}))
