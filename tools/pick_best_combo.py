import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
import glob
old=sorted([p for p in glob.glob("scene_magnet/saves/ckpt_p35/p35_*_steps.zip") if "2500000" in p], key=os.path.getmtime)[-1]
def test(m2path, alpha, n=30):
    m2=PPO.load(m2path, device="cpu")
    e=ArmEnv(); e.max_steps=400; e.smooth_alpha=alpha
    ok=0; vmax=[]
    for ep in range(n):
        obs,_=e.reset(); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        for t in range(450):
            if e.phase!=last:
                last=e.phase
                d,a,s = TH2 if e.phase==2 else TH35
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            act,_=(m1 if e.phase<3 else m2).predict(np.asarray(obs,dtype=np.float32),deterministic=True)
            obs,r,term,tr,info=e.step(act)
            vmax.append(float(np.abs(e.data.qvel[e.arm_dof_adr]).max()))
            if term or tr or e.phase>=6: break
        ok+=int(e.phase>=6)
    return ok, np.percentile(vmax,95)
for tag,m2p in (("无惩罚(250万检查点)",old),("0.02惩罚(新)",  "scene_magnet/saves/phase34.zip")):
    for al in (0.5, 0.75):
        ok,p95=test(m2p, al)
        print("%-22s alpha=%.2f -> 到 phase6 %2d/30 | 速度95分位 %.1f rad/s" % (tag, al, ok, p95))
