import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ('ppo_arm_reach.zip','saves/phase1.zip'):
    m=PPO.load('scene_magnet/'+name, device='cpu')
    e=ArmEnv(); e._set_dist(0.03); e._set_angle1(0.98); e._set_still_count(0)
    dmin=[]; amax_near=[]
    for ep in range(20):
        obs,_=e.reset(); md=9; best_a=-1
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase<2:
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.pick_offset
                d=float(np.linalg.norm(ee-tgt))
                j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
                ang=float(np.dot(f,[0,0,-1]))
                md=min(md,d)
                if d<0.035: best_a=max(best_a,ang)
            if e.phase>=2 or term or tr: break
        dmin.append(md); amax_near.append(best_a)
    print("%-22s 最小dist 中位 %.3f | dist<3.5cm 时最大 angle 中位 %.4f (需 0.98)" % (name, np.median(dmin), np.median(amax_near)))
