import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
e=ArmEnv(); e.max_steps=400
grasp_x=[]; enter_x=[]; wait=[]
for ep in range(40):
    obs,_=e.reset(); last=e.phase; p1x=None
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    for t in range(1400):
        if e.phase!=last:
            last=e.phase
            if e.phase==1 and p1x is None: p1x=float(e.box_pos[0])
            d,a,s=TH2 if e.phase==2 else TH1
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        act,_=m1.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase>=3 or term or tr: break
    if e.phase>=3:
        grasp_x.append(float(e.box_pos[0])); enter_x.append(p1x); wait.append(e.wait_steps)
print("成功 %d/40 局" % len(grasp_x))
print("进入 phase1（方块到 arrive_x）时的 x : 均值 %.3f（应≈-0.475）" % np.mean(enter_x))
print("真正抓取（phase2 成功）时的 x : 中位 %.3f, 范围 %.3f ~ %.3f" % (np.median(grasp_x), min(grasp_x), max(grasp_x)))
print("phase0 等待步数 : 中位 %.0f" % np.median(wait))
