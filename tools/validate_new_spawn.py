import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
e=ArmEnv(); e.max_steps=400; e.smooth_alpha=0.5
ok=0; waits=[]; p1=[]; cnt={}
for ep in range(100):
    obs,_=e.reset(); last=e.phase
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    t2=None
    for t in range(450):
        if e.phase!=last:
            last=e.phase
            if e.phase==2 and t2 is None: t2=t
            d,a,s = TH2 if e.phase==2 else TH35
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        act,_=(m1 if e.phase<3 else m2).predict(np.asarray(obs,dtype=np.float32),deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if term or tr or e.phase>=6: break
    cnt[int(e.phase)]=cnt.get(int(e.phase),0)+1
    ok+=int(e.phase>=6)
    if t2: p1.append(t2); waits.append(e.wait_steps)
print("100 局: 到放料完成 %d | 分布 %s" % (ok, dict(sorted(cnt.items()))))
print("phase1 用时 中位 %.0f 步 (范围 %d~%d) | phase0 等待 中位 %.0f 步"
      % (np.median(p1), min(p1), max(p1), np.median(waits)))
