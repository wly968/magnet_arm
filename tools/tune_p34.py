import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
env=t.ArmEnv()
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_new.zip", env=env, device="cpu")
for th in ((0.05,0.945,0),(0.06,0.94,0),(0.07,0.93,0)):
    t.TH_PHASE34=th
    ok=0; best=9
    for ep in range(100):
        ph,succ,n=t.run_episode(env,m1,m2)
        ok+=int(ph>=5)
    print("phase3/4 阈值 %.3f/%.3f -> 全流程 %d/100" % (th[0],th[1],ok))
