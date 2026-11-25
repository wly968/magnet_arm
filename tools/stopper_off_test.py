import os, sys, numpy as np, torch, collections
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
def run(stopper, period=150, steps=3000):
    e=ArmEnv(); e.n_cubes=10; e.spawn_period=period; e.max_steps=400
    e.stopper=stopper
    e.reset(); e._set_home_still(2)
    last=-1; obs=e._get_obs(); done=0; fails=collections.Counter(); t1x=[]
    for t in range(steps):
        ph=e.phase
        if ph!=last:
            if ph==1: t1x.append(float(e.box_pos[0]))
            last=ph
            d,a,s = (TH1 if ph<=1 else TH2 if ph==2 else (TH35 if ph<6 else TH6))
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            e.phase_entry_steps=6
        act,_=pol(ph).predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase>=7:
            done+=1; e.next_cycle(); obs=e._get_obs(); last=-1
        elif term or tr:
            fails[int(e.phase)]+=1; e.next_cycle(); obs=e._get_obs(); last=-1
    return done, dict(fails), (np.mean(t1x[1:]) if len(t1x)>1 else -1), (min(t1x[1:]) if len(t1x)>1 else -1), (max(t1x[1:]) if len(t1x)>1 else -1)
for st in (True, False):
    d,f,avg,mn,mx = run(st)
    print("stopper=%-5s -> 完成 %2d | 失败 %s | phase1 起点 cube x 平均 %.2f (%.2f~%.2f)"
          % (st, d, f, avg, mn, mx))
