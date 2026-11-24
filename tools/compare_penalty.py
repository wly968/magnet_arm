import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
for tag,path in (("旧 model2(无速度惩罚)","scene_magnet/saves/phase34.zip"),
                 ("新 model2(0.02 速度惩罚)","scene_magnet/ppo_arm_reach.zip")):
    m2=PPO.load(path, device="cpu")
    e=ArmEnv(); e.max_steps=400; e.smooth_alpha=0.75
    stats=[]; vmax=[]; ok=0
    for ep in range(12):
        obs,_=e.reset(); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        traj=[]
        for t in range(450):
            if e.phase!=last:
                last=e.phase
                d,a,s = TH2 if e.phase==2 else TH35
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            pol = m1 if e.phase<3 else m2
            act,_=pol.predict(np.asarray(obs,dtype=np.float32),deterministic=True)
            obs,r,term,tr,info=e.step(act)
            vmax.append(float(np.abs(e.data.qvel[e.arm_dof_adr]).max()))
            if e.phase==4: traj.append(e.data.qpos[e.arm_qpos_adr].copy())
            if term or tr or e.phase>=6: break
        ok+=int(e.phase>=6)
        if len(traj)>3:
            qs=np.array(traj); dq=np.diff(qs,axis=0)
            stats.append((len(qs), int(np.sum(np.sign(dq[1:])*np.sign(dq[:-1])<0)), float((qs.max(0)-qs.min(0))[0])))
    A=np.array(stats)
    print("%-26s 到 phase6 %2d/12 | phase4 %.1f 步/%.1f 次反转/yaw %.2f rad | 关节速度 95分位 %.1f rad/s"
          % (tag, ok, A[:,0].mean(), A[:,1].mean(), A[:,2].mean(), np.percentile(vmax,95)))
