import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
e=ArmEnv(); e.n_cubes=10; e.spawn_period=260; e.max_steps=400
e.reset(); e._set_home_still(2)
last=-1; obs=e._get_obs(); cyc=0
for t in range(2000):
    ph=e.phase
    if ph!=last:
        if ph==1:
            q=float(np.linalg.norm(e.data.qpos[e.arm_qpos_adr]-e.home_qpos))
            v=float(np.abs(e.data.qvel[e.arm_dof_adr]).max())
            print("cycle%d phase1 起点: 手臂偏差 %.2f | 最大关节速度 %.2f rad/s" % (cyc, q, v))
        if ph==7: cyc+=1
        last=ph
        d,a,s = (TH1 if ph<=1 else TH2 if ph==2 else (TH35 if ph<6 else TH6))
        e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        e.phase_entry_steps=6
    act,_=pol(ph).predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(act)
    if e.phase>=7 or term or tr:
        e.next_cycle(); obs=e._get_obs(); last=-1
