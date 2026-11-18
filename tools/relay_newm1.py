import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
env=t.ArmEnv()
m1=t.PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu") if hasattr(t,"PPO") else None
from stable_baselines3 import PPO
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", env=env, device="cpu")
ok=0; hits=0; by={}
for ep in range(100):
    ph,succ,n = t.run_episode(env,m1,m2)
    ok+=int(ph>=6)
print("换 model1 后（未重训 model2）全流程 %d/100" % ok)
