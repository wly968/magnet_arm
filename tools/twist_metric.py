import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1); TH35=(0.034,0.980,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
e=ArmEnv(); e.max_steps=400
traj=[]
obs,_=e.reset(); last=e.phase
d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
for t in range(400):
    if e.phase!=last:
        last=e.phase
        d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
        e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    pol = m1 if e.phase<3 else m2
    act,_=pol.predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(act)
    traj.append((e.phase, e.data.qpos[e.arm_qpos_adr].copy()))
    if term or tr: break
import collections
for ph in (3,4,5):
    qs=np.array([q for p,q in traj if p==ph])
    if len(qs)<3: continue
    rng=qs.max(0)-qs.min(0)
    # 方向反转次数（抖动的直接度量）
    dq=np.diff(qs,axis=0); rev=int(np.sum(np.sign(dq[1:])*np.sign(dq[:-1])<0))
    print("phase %d: %3d 步 | 各关节变化幅度 %s | 方向反转 %d 次" % (ph, len(qs), np.round(rng,2), rev))
