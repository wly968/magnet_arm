import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
e=ArmEnv(); e.max_steps=400; e.smooth_alpha=0.5
ok=0; peak={}
for ep in range(40):
    obs,_=e.reset(); last=e.phase
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    for t in range(450):
        if e.phase!=last:
            last=e.phase
            d,a,s = TH2 if e.phase==2 else TH35
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        act,_=(m1 if e.phase<3 else m2).predict(np.asarray(obs,dtype=np.float32),deterministic=True)
        obs,r,term,tr,info=e.step(act)
        ph=int(min(e.phase,5)); v=float(np.abs(e.data.qvel[e.arm_dof_adr]).max())
        peak[ph]=max(peak.get(ph,0.0), v)
        if term or tr or e.phase>=6: break
    ok+=int(e.phase>=6)
print("到 phase6: %d/40" % ok)
print("各阶段峰值关节速度:", {p: round(v,1) for p,v in sorted(peak.items())})
