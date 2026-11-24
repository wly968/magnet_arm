import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
e=ArmEnv(); e.max_steps=400
for ep in range(12):
    obs,_=e.reset(); x0=float(e.box_pos[0])
    if x0 > -0.95: continue
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    last=e.phase
    for t in range(1400):
        if e.phase!=last:
            last=e.phase
            d,a,s = TH2 if e.phase==2 else TH1
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        act,_=m1.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if term or tr or e.phase>=3: break
    print("x0=%.2f y=%.2f -> 结束 phase=%d 步数=%d wait_steps=%d cube x=%.2f term=%s tr=%s"
          % (x0, e.box_pos[1], e.phase, t+1, e.wait_steps, e.box_pos[0], term, tr))
