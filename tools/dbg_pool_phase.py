import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1); TH35=(0.034,0.965,2)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
e=ArmEnv(); e.n_cubes=10; e.spawn_period=200; e.max_steps=400
e.reset(); e._set_dist(TH1[0]); e._set_angle1(TH1[1]); e._set_still_count(0)
for t in range(600):
    act,_=m1.predict(e._get_obs(),deterministic=True)
    obs,r,term,tr,info=e.step(act)
    if t%60==0 or e.phase==2:
        print("t%3d phase=%d 目标x=%.2f belts=%s" % (t, e.phase, e.box_pos[0],
              [round(float(e.data.mocap_pos[e.cube_ids[i]][0]),2) for i,s in enumerate(e.cube_state) if s=="belt"]))
    if e.phase>=2: break
