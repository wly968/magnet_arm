import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load('scene_magnet/saves/phase2.zip', device='cpu')
e=ArmEnv()
for trial in range(3):
    obs,_=e.reset(); marks={}
    for t in range(260):
        a,_=m1.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase not in marks: marks[e.phase]=t+1
        if e.phase>=3 or term or tr: break
    print("trial%d: 各阶段首次到达步数 %s, 结束 phase=%d, 总步数 %d" % (trial, marks, e.phase, t+1))
