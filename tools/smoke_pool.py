import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
TH1=(0.045,0.950,0); TH2=(0.032,0.960,1)
e=ArmEnv(); e.n_cubes=10; e.spawn_period=200
obs,_=e.reset()
print("reset 后: 状态", e.cube_state, "| 目标 x=%.2f" % e.box_pos[0])
e._set_dist(TH2[0]); e._set_angle1(TH2[1]); e._set_still_count(TH2[2])
for t in range(300):
    act,_=m1.predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(act)
    if t%50==0:
        belts=[round(float(e.data.mocap_pos[e.cube_ids[i]][0]),2) for i,st in enumerate(e.cube_state) if st=="belt"]
        print("  t%3d phase=%d 目标x=%.2f 带上方块=%s" % (t, e.phase, e.box_pos[0], belts))
    if term or tr or e.phase>=3: break
print("结果 phase=%d | 状态 %s" % (e.phase, e.cube_state))
