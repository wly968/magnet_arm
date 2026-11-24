import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
e=ArmEnv(); e._set_dist(0.030); e._set_angle1(0.99); e._set_still_count(2); e._set_rel(0.05)
for ep in range(20):
    obs,_=e.reset()
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase>=3 or term or tr: break
d=e._diag(); n=max(1,d["n"])
print("stage6 阈值 0.030/0.99/still2 下 phase2 的逐项达标率（20 局）：")
print("  dist %.3f 达标 %5.1f%%" % (0.030, 100*d["dist"]/n))
print("  angle 0.99 达标 %5.1f%%" % (100*d["angle"]/n))
print("  band 达标 %5.1f%%" % (100*d["band"]/n))
print("  rel 0.05 达标 %5.1f%%" % (100*d["rel"]/n))
print("  位置(dist+angle+band)到位 %5.1f%% | 其中 rel 达标 %5.1f%% | 全满足 %.2f%%"
      % (100*d["n_pos"]/n, (100*d["rel_pos"]/d["n_pos"] if d["n_pos"] else 0), 100*d["all"]/n))
print("  均值 dist %.4f | angle %.4f | dz %+.4f | rel %.3f"
      % (d["sum_dist"]/n, d["sum_angle"]/n, d["sum_dz"]/n, d["sum_rel"]/n))
