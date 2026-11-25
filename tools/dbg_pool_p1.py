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
last=-1; obs=e._get_obs(); n=0
for t in range(1500):
    ph=e.phase
    if ph!=last:
        if ph==1:
            qerr=float(np.linalg.norm(e.data.qpos[e.arm_qpos_adr]-e.home_qpos))
            print("phase1 起点: cube x=%.2f | 手臂偏差 %.2f | belts=%s" % (e.box_pos[0], qerr,
                  [round(float(e.data.mocap_pos[e.cube_ids[i]][0]),2) for i,s in enumerate(e.cube_state) if s=="belt"]))
        if ph==7:
            n+=1; print("  ==> 完成第 %d 次 @t=%d" % (n,t)); e.next_cycle(); obs=e._get_obs(); last=-1; continue
        last=ph
        d,a,s = (TH1 if ph<=1 else TH2 if ph==2 else (TH35 if ph<6 else TH6))
        e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        e.phase_entry_steps=6
    act,_=pol(ph).predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(act)
    if term or tr:
        print("  失败: phase=%d truncated=%s @t=%d" % (e.phase, tr, t)); e.next_cycle(); obs=e._get_obs(); last=-1
