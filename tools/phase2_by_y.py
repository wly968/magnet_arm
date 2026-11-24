import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1)
e=ArmEnv()
bins={}
for ep in range(80):
    obs,_=e.reset()
    y0=float(e.box_pos[1])
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
    k=int((y0-0.44)/0.084)
    ok, n = bins.get(k,(0,0)); bins[k]=(ok+int(e.phase>=3), n+1)
print("model1 按方块初始 y 分桶的成功率（阈值 0.032/0.98/still1）:")
for k in sorted(bins):
    ok,n=bins[k]; lo=0.44+k*0.084
    print("  y %.2f~%.2f : %2d/%2d = %3.0f%%" % (lo,(lo+0.084) if k<4 else 0.86, ok,n,100*ok/n))
