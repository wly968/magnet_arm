import os, sys, numpy as np, torch, glob
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
cands=sorted(glob.glob("scene_magnet/saves/ckpt_p35/p35_*_steps.zip"), key=os.path.getmtime)
m2=PPO.load(cands[-1], device="cpu")
print("用 model2:", os.path.basename(cands[-1]))
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
def adapt(o,m):
    d=m.observation_space.shape[0]; o=np.asarray(o,dtype=np.float32)
    return o[:d] if o.shape[-1]>d else (np.concatenate([o,np.zeros(d-o.shape[-1],dtype=np.float32)]) if o.shape[-1]<d else o)
e=ArmEnv(); e.max_steps=400; e._set_home_still(2); cnt={}; stats=[]; lens=[]
for ep in range(30):
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
        if e.phase==4: traj.append(e.data.qpos[e.arm_qpos_adr].copy())
        if term or tr: break
    cnt[int(e.phase)]=cnt.get(int(e.phase),0)+1; lens.append(t+1)
    if len(traj)>3:
        qs=np.array(traj); dq=np.diff(qs,axis=0)
        stats.append((len(qs), int(np.sum(np.sign(dq[1:])*np.sign(dq[:-1])<0)), float((qs.max(0)-qs.min(0))[0])))
print("全链路(新转移点, 旧 model3) 30 局:", dict(sorted(cnt.items())), "| 平均步数 %.0f" % np.mean(lens))
A=np.array(stats)
print("phase4: 平均 %4.1f 步 | 方向反转 %.1f 次 | yaw 摆动 %.2f rad（旧转移点 16.0/5.4/1.39）"
      % (A[:,0].mean(), A[:,1].mean(), A[:,2].mean()))
