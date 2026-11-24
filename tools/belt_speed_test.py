import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
for speed in (0.20, 0.12, 0.08, 0.05):
    e=ArmEnv(); e.max_steps=400; e.cube_speed=speed
    ok=0; bins={}
    for ep in range(30):
        obs,_=e.reset(); x0=float(e.box_pos[0]); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        for t in range(1200):
            if e.phase!=last:
                last=e.phase
                d,a,s=TH2 if e.phase==2 else TH1
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            act,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
        k=int((x0+0.475)/0.125); o,n=bins.get(k,(0,0)); bins[k]=(o+int(e.phase>=3), n+1)
    print("带速 %.2f m/s: 抓取成功 %d/30 | 右半段(x0>-0.30) %d/%d"
          % (speed, ok, sum(o for k,(o,n) in bins.items() if k>=1), sum(n for k,(o,n) in bins.items() if k>=1)))
