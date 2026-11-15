import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
for name in ("saves/phase1.zip","ppo_arm_reach_prev.zip","ppo_arm_reach.zip","saves/phase12_shaped.zip"):
    m=PPO.load(os.path.join(SCENE,name), device="cpu")
    ls=m.policy.log_std.detach().numpy()
    print("%-28s log_std %s  -> 动作标准差 %s" % (name, np.round(ls,2), np.round(np.exp(ls),2)))
