import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("ckpt_1499976_steps.zip","ckpt_2999952_steps.zip"):
    p=os.path.join("scene_magnet/saves/ckpt",name)
    if not os.path.exists(p): continue
    m=PPO.load(p, device='cpu')
    e=ArmEnv(); e._set_dist(0.045); e._set_angle1(0.95); e._set_still_count(0); e._set_rel(0.10)
    for ep in range(25):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase>=3 or term or tr: break
    d=e._diag(); n=max(1,d["n"])
    print("%-24s 帧 %5d | dist %5.1f%% | angle %5.1f%% | rel %5.1f%% | 高度带 %5.1f%% | 位置到位 %5.1f%% | 其中rel达标 %5.1f%% | 全满足 %4.1f%% | mrel %.3f"
          % (name, d["n"], 100*d["dist"]/n, 100*d["angle"]/n, 100*d["rel"]/n, 100*d["band"]/n,
             100*d["n_pos"]/n, (100*d["rel_pos"]/d["n_pos"] if d["n_pos"] else 0),
             100*d["all"]/n, d["sum_rel"]/n))
