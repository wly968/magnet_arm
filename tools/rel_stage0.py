import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/saves/phase1.zip', device='cpu')
e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
rel=[]; dist=[]; ang=[]
for ep in range(15):
    obs,_=e.reset()
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase==2:
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            dist.append(float(np.linalg.norm(ee-tgt)))
            rel.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
            j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            ang.append(float(np.dot(f,[0,0,-1])))
        if e.phase>=3 or term or tr: break
rel=np.array(rel)
print("phase2 帧 %d" % len(rel))
print("rel_vel (m/s): p5 %.3f p25 %.3f 中位 %.3f p75 %.3f" % tuple(np.percentile(rel,[5,25,50,75])))
print("dist: 中位 %.3f | angle: 中位 %.3f" % (np.median(dist), np.median(ang)))
for thr in (0.05,0.10,0.15,0.20,0.30,0.50):
    print("  rel<%.2f 的帧占比 %.1f%%" % (thr, 100*(rel<thr).mean()))
