import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
def run(entry, n=40):
    e=ArmEnv(); ok=0
    for ep in range(n):
        obs,_=e.reset(); e.phase1_dist, e.phase1_angle = TH1[0], TH1[1]
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s)
        last=e.phase
        for t in range(e.max_steps+e.max_wait_steps+10):
            if e.phase!=last:
                last=e.phase
                d,a,s = TH2 if e.phase==2 else TH1
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s)
                e.phase1_dist, e.phase1_angle = TH1[0], TH1[1]
                if entry: e.phase_entry_steps=entry
            act,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    return ok
for entry in (0, 2, 6):
    print("phase_entry_steps=%d -> %d/40" % (entry, run(entry)))
