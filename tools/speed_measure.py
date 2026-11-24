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
e=ArmEnv(); e.max_steps=400; e._set_home_still(2); e.alpha=0.5
e.smooth_alpha=0.5
maxv=[]; tcp=[]
for ep in range(10):
    obs,_=e.reset(); last=e.phase
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    prev=None
    for t in range(450):
        if e.phase!=last:
            last=e.phase
            d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        m=pol(e.phase); act,_=m.predict(adapt(obs,m),deterministic=True)
        obs,r,term,tr,info=e.step(act)
        qv=np.abs(e.data.qvel[e.arm_dof_adr])
        maxv.append(qv.max())
        ee=e.data.site_xpos[e.ee_id].copy()
        if prev is not None: tcp.append(float(np.linalg.norm(ee-prev))/0.01)  # m/s
        prev=ee
        if term or tr: break
print("关节最大角速度: 中位 %.2f rad/s, 95分位 %.2f rad/s" % (np.median(maxv), np.percentile(maxv,95)))
print("末端线速度(TCP): 中位 %.2f m/s, 95分位 %.2f m/s, 峰值 %.2f m/s"
      % (np.median(tcp), np.percentile(tcp,95), np.max(tcp)))
print("工业协作臂参考: 关节 1~2 rad/s，TCP 0.5~1.0 m/s")
