import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1)
e=ArmEnv()
def rate(y, n=30):
    e.spawn_y_range=(y-1e-4, y+1e-4); ok=0
    for ep in range(n):
        obs,_=e.reset()
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        last=e.phase
        for t in range(e.max_steps+e.max_wait_steps+10):
            if e.phase!=last:
                last=e.phase
                d,a,s=TH2 if e.phase==2 else TH1
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            act,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    return ok/n
for y in (0.725, 0.730, 0.735, 0.740, 0.745):
    print("  y=%.3f : %3.0f%%" % (y, 100*rate(y)))
for y in (0.545, 0.550, 0.560):
    print("  y=%.3f : %3.0f%%" % (y, 100*rate(y)))
