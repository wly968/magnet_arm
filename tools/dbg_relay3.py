import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
env=t.ArmEnv()
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", env=env, device="cpu")
for ep in range(12):
    ph,succ,n = t.run_episode(env,m1,m2)
    print("ep%2d final_phase=%d success=%s steps=%d ee=%s target=%s" %
          (ep, ph, succ, n, np.round(env.data.site_xpos[env.ee_id],2), np.round(env._target(),2)))
