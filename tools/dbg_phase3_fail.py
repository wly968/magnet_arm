import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
def pol(ph): return m1 if ph<3 else m2
e=ArmEnv(); e.n_cubes=10; e.spawn_period=260; e.max_steps=400
e.reset(); e._set_home_still(2)
last=-1; obs=e._get_obs()
for t in range(900):
    ph=e.phase
    if ph!=last:
        if ph==3:
            print("进入 phase3: box_pos=%s carried_idx=%s mocap_id=%s attached=%s"
                  % (np.round(e.box_pos,2), e.carried_idx, e.mocap_id, e.attached))
        last=ph
        d,a,s = (TH1 if ph<=1 else TH2 if ph==2 else (TH35 if ph<6 else TH6))
        e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        e.phase_entry_steps=6
    act,_=pol(ph).predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(act)
    if term or tr:
        print("失败 @t%d phase=%d term=%s tr=%s info=%s" % (t, e.phase, term, tr, {k:v for k,v in info.items()}))
        break
