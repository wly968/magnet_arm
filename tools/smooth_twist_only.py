import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
def pol(ph): return m1 if ph<3 else m2
def run(alpha, n=12):
    e=ArmEnv(); e.max_steps=400; e.smooth_alpha=alpha
    stats=[]; vmax=[]
    for ep in range(n):
        obs,_=e.reset(); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        traj=[]
        for t in range(450):
            if e.phase!=last:
                last=e.phase
                d,a,s = TH2 if e.phase==2 else TH35
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            act,_=pol(e.phase).predict(np.asarray(obs,dtype=np.float32),deterministic=True)
            obs,r,term,tr,info=e.step(act)
            vmax.append(float(np.abs(e.data.qvel[e.arm_dof_adr]).max()))
            if e.phase==4: traj.append(e.data.qpos[e.arm_qpos_adr].copy())
            if term or tr or e.phase>=5: break
        if len(traj)>3:
            qs=np.array(traj); dq=np.diff(qs,axis=0)
            stats.append((len(qs), int(np.sum(np.sign(dq[1:])*np.sign(dq[:-1])<0)), float((qs.max(0)-qs.min(0))[0])))
    A=np.array(stats)
    print("smooth_alpha=%.2f | phase4 %.1f 步 / %.1f 次反转 / yaw %.2f rad | 关节速度 95分位 %.1f rad/s"
          % (alpha, A[:,0].mean(), A[:,1].mean(), A[:,2].mean(), np.percentile(vmax,95)))
for al in (0.0, 0.5, 0.7, 0.85):
    run(al)
