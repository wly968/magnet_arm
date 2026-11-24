import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
e=ArmEnv(); e.max_steps=400
ok=0; bins={}; waits=[]
for ep in range(60):
    obs,_=e.reset(); x0=float(e.box_pos[0]); last=e.phase
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    for t in range(450):
        if e.phase!=last:
            last=e.phase
            d,a,s = TH2 if e.phase==2 else TH35
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        act,_=(m1 if e.phase<3 else m2).predict(np.asarray(obs,dtype=np.float32),deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if term or tr or e.phase>=6: break
    ok+=int(e.phase>=6); waits.append(e.wait_steps)
    k=round((x0+1.15)/0.2)
    o,n=bins.get(k,(0,0)); bins[k]=(o+int(e.phase>=6), n+1)
print("60 局: 到放料完成 %d | 平均 phase0 等待 %.0f 步" % (ok, np.mean(waits)))
for k in sorted(bins):
    o,n=bins[k]; print("  x0 %.2f~%.2f : %2d/%2d" % (-1.15+k*0.2, -1.15+(k+1)*0.2, o, n))
