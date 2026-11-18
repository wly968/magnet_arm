import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
env=t.ArmEnv()
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", env=env, device="cpu")
m3=PPO.load(r"scene_magnet/saves/phase6_home.zip", env=env, device="cpu")
obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
for i in range(400):
    if env.phase!=last: t.apply_thresholds(env, env.phase)
    last=env.phase
    pol = m1 if env.phase<3 else (m2 if env.phase<6 else m3)
    a,_=pol.predict(obs,deterministic=True)
    obs,r,term,tr,info=env.step(a)
    if env.phase>=6 and i%15==0:
        print("i%3d phase=%d dist_home=%.3f ee=%s act=%s" % (i, env.phase,
              float(np.linalg.norm(env.data.site_xpos[env.ee_id]-env.home_ee)),
              np.round(env.data.site_xpos[env.ee_id],2), np.round(a,2)))
    if term or tr: break
