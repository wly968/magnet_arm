import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')
rows=[]
for ep in range(10):
    e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
    obs,_=e.reset()
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase==2:
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            d=float(np.linalg.norm(ee-tgt))
            vx=float(e.ee_val[0]); vr=float(np.linalg.norm(e.ee_val-e.box_vel))
            rows.append((d, vx, vr, float(ee[0]-tgt[0])))
        if e.phase>=3 or term or tr: break
R=np.array(rows)
near=R[R[:,0]<0.03]
print("phase2 总帧 %d, dist<3cm 的帧 %d" % (len(R), len(near)))
if len(near):
    print("dist<3cm 时: v_ee_x 中位 %+.3f (方块 +0.200), rel_vel 中位 %.3f, 领先量 ee_x-tgt_x 中位 %+.3f"
          % (np.median(near[:,1]), np.median(near[:,2]), np.median(near[:,3])))
print("全部 phase2 帧: v_ee_x 中位 %+.3f, rel 中位 %.3f" % (np.median(R[:,1]), np.median(R[:,2])))
print("v_ee_x 分布: p10 %+.3f p50 %+.3f p90 %+.3f" % tuple(np.percentile(R[:,1],[10,50,90])))
