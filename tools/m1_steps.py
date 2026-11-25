import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
def run(d,a,s,rv,n=60,tag=""):
    e=ArmEnv(); e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(rv)
    wait=[]; p1=[]; p2=[]; ok=0
    for ep in range(n):
        obs,_=e.reset(); t1=t2=None
        for t in range(1400):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=1 and t1 is None: t1=t
            if e.phase>=2 and t2 is None: t2=t
            if e.phase>=3 or term or tr: break
        if e.phase>=3:
            ok+=1
            wait.append(t1 or 0)
            p1.append((t2 or t)-(t1 or 0))
            p2.append((t+1)-(t2 or 0))
    print("%s: 成功 %d/%d | phase0 等待 %.0f 步 | phase1 %.0f 步 | phase2 %.0f 步 | 任务步(1+2) %.0f | 原始总步数 %.0f"
          % (tag, ok, n, np.mean(wait), np.mean(p1), np.mean(p2), np.mean(p1)+np.mean(p2), np.mean(wait)+np.mean(p1)+np.mean(p2)))
run(0.032,0.960,1,0.05,60,"第6档")
run(0.050,0.940,0,0.20,60,"第1档")
