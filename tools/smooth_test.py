import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
def adapt(o,m):
    d=m.observation_space.shape[0]; o=np.asarray(o,dtype=np.float32)
    return o[:d] if o.shape[-1]>d else (np.concatenate([o,np.zeros(d-o.shape[-1],dtype=np.float32)]) if o.shape[-1]<d else o)
def run(alpha, n=15):
    e=ArmEnv(); e.max_steps=400; e._set_home_still(2); e.smooth_alpha=alpha
    ok=0; stats=[]; vmax=[]; lens=[]
    for ep in range(n):
        obs,_=e.reset(); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        traj=[]
        for t in range(450):
            if e.phase!=last:
                last=e.phase
                d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            m=pol(e.phase); act,_=m.predict(adapt(obs,m),deterministic=True)
            obs,r,term,tr,info=e.step(act)
            vmax.append(float(np.abs(e.data.qvel[e.arm_dof_adr]).max()))
            if e.phase==4: traj.append(e.data.qpos[e.arm_qpos_adr].copy())
            if term or tr: break
        ok+=int(e.phase>=7); lens.append(t+1)
        if len(traj)>3:
            qs=np.array(traj); dq=np.diff(qs,axis=0)
            stats.append((len(qs), int(np.sum(np.sign(dq[1:])*np.sign(dq[:-1])<0)), float((qs.max(0)-qs.min(0))[0])))
    A=np.array(stats) if stats else None
    print("smooth_alpha=%.2f | 成功 %2d/%d | 平均步数 %.0f | phase4 %s | 95分位关节速度 %.2f rad/s"
          % (alpha, ok, n, np.mean(lens),
             ("%.1f 步/%.1f 次反转/%.2f rad" % (A[:,0].mean(),A[:,1].mean(),A[:,2].mean())) if A is not None else "-",
             np.percentile(vmax,95)))
for al in (0.0, 0.5, 0.7, 0.85):
    run(al)
