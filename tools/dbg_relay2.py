import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
env=t.ArmEnv(terminate_on_phase2=False)
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", env=env, device="cpu")
obs,_=env.reset(); t.apply_thresholds(env, env.phase)
last=env.phase
for i in range(250):
    if env.phase>=3 and last<3: t.apply_thresholds(env, env.phase)
    last=env.phase
    pol = m1 if env.phase<3 else m2
    a,_=pol.predict(obs,deterministic=True)
    obs,r,term,tr,info=env.step(a)
    if i%10==0 or env.phase>=4:
        print("i%3d phase=%d dist=%.3f ee=%s" % (i, env.phase,
              float(np.linalg.norm(env.data.site_xpos[env.ee_id]-env._target())),
              np.round(env.data.site_xpos[env.ee_id],2)))
    if term or tr: break
