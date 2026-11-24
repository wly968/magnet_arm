import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1); TH35=(0.034,0.980,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
def run(p4_dist, n=6):
    e=ArmEnv(); e.max_steps=400; stats=[]
    for ep in range(n):
        obs,_=e.reset(); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        traj=[]
        for t in range(400):
            if e.phase!=last:
                last=e.phase
                if e.phase==4:
                    e._set_dist(p4_dist); e._set_angle1(0.6); e._set_still_count(0)
                else:
                    d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
                    e._set_dist(d); e._set_angle1(a); e._set_still_count(s)
                e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            pol = m1 if e.phase<3 else m2
            act,_=pol.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            traj.append((e.phase, e.data.qpos[e.arm_qpos_adr].copy()))
            if term or tr: break
        qs=np.array([q for p,q in traj if p==4])
        if len(qs)>3:
            dq=np.diff(qs,axis=0); rev=int(np.sum(np.sign(dq[1:])*np.sign(dq[:-1])<0))
            stats.append((len(qs), rev, float((qs.max(0)-qs.min(0))[0])))
    if stats:
        A=np.array(stats)
        print("phase4 dist 阈值 %.3f -> 平均 %4.1f 步 | 方向反转 %.1f 次 | yaw 摆动 %.2f rad"
              % (p4_dist, A[:,0].mean(), A[:,1].mean(), A[:,2].mean()))
for d in (0.05, 0.08, 0.12, 0.20):
    run(d)
