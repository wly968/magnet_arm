import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
def adapt(o,m):
    d=m.observation_space.shape[0]; o=np.asarray(o,dtype=np.float32)
    return o[:d] if o.shape[-1]>d else (np.concatenate([o,np.zeros(d-o.shape[-1],dtype=np.float32)]) if o.shape[-1]<d else o)
e=ArmEnv(); e.max_steps=400; e._set_home_still(2)
cnt={}; lens=[]; t6=[]
for ep in range(60):
    obs,_=e.reset(); last=e.phase; start6=None
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    for t in range(1400):
        if e.phase!=last:
            last=e.phase
            if e.phase==6 and start6 is None: start6=t
            d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        m=pol(e.phase); act,_=m.predict(adapt(obs,m),deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if term or tr: break
    fp=int(e.phase); cnt[fp]=cnt.get(fp,0)+1; lens.append(t+1)
    if start6 is not None: t6.append((t+1)-start6)
print("全链路 60 局:", dict(sorted(cnt.items())))
print("平均总步数 %.0f | 其中 phase 6 用时 %.0f 步（原 91 步含等入篮）" % (np.mean(lens), np.mean(t6) if t6 else -1))
