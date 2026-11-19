import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load('scene_magnet/saves/phase2.zip', device='cpu')

def cond_frames(vthr):
    e=ArmEnv(); runs=[]
    for ep in range(20):
        obs,_=e.reset(); run=0; best=0
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase==2:
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                d=float(np.linalg.norm(ee-tgt))
                j3=e.data.xanchor[e.arm_joints[3]]
                f=(ee-j3)/np.linalg.norm(ee-j3)
                a1=float(np.dot(f,[0,0,-1]))
                vr=float(np.linalg.norm(e.ee_val - e.box_vel))
                good = (d<e.dist_threshold and a1>e.angle1_threshold and
                        ee[2]<tgt[2]+0.01 and ee[2]>e.box_pos[2]+0.05 and vr<vthr)
                run = run+1 if good else 0
                best=max(best,run)
            if e.phase>=3 or term or tr: break
        runs.append(best)
    return np.array(runs)

e0=ArmEnv(); e0._set_dist(0.03); e0._set_angle1(0.98)
for v in (0.03, 0.05, 0.10):
    r=cond_frames(v)
    print("rel_vel<%.2f: 每局最长连续满足帧数 中位 %d 最大 %d, >=4 帧的局数 %d/20"
          % (v, np.median(r), r.max(), int((r>=4).sum())))
