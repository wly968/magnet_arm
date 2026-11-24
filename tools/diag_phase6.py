import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1); TH35=(0.034,0.980,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
def adapt(o,m):
    d=m.observation_space.shape[0]; o=np.asarray(o,dtype=np.float32)
    return o[:d] if o.shape[-1]>d else (np.concatenate([o,np.zeros(d-o.shape[-1],dtype=np.float32)]) if o.shape[-1]<d else o)
e=ArmEnv(); e.max_steps=600
rows=[]
for ep in range(15):
    obs,_=e.reset(); last=e.phase
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    minq=9; coll=False; t6=None
    for t in range(e.max_steps+e.max_wait_steps+10):
        if e.phase!=last:
            last=e.phase
            d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            if e.phase==6 and t6 is None: t6=t
        m=pol(e.phase); act,_=m.predict(adapt(obs,m),deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase>=6:
            q=float(np.linalg.norm(e.data.qpos[e.arm_qpos_adr]-e.home_qpos))
            minq=min(minq,q); coll=coll or e.collected
        if term or tr: break
    rows.append((int(e.phase), minq, coll, (t6 is not None)))
print("phase6 内最小 q_err 中位 %.3f（要求 <0.15） | collected %d/15 | 进入 phase6 %d/15"
      % (np.median([r[1] for r in rows]), sum(r[2] for r in rows), sum(r[3] for r in rows)))
print("结束 phase 分布:", {p: sum(1 for r in rows if r[0]==p) for p in sorted(set(r[0] for r in rows))})
