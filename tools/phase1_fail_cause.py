import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/ppo_arm_reach.zip", device="cpu")
e=ArmEnv(); e._set_dist(0.032); e._set_angle1(0.985); e._set_still_count(1); e._set_rel(0.05)
fails=[]
for ep in range(30):
    obs,_=e.reset(); md=9; ma=-1
    for t in range(e.max_steps+e.max_wait_steps+10):
        act,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase==1:
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.pick_offset
            d=float(np.linalg.norm(ee-tgt))
            j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            md=min(md,d); ma=max(ma,float(np.dot(f,[0,0,-1])))
        if e.phase>=2 or term or tr: break
    if e.phase<2: fails.append((md,ma))
print("phase1 失败 %d/30 局" % len(fails))
if fails:
    F=np.array(fails)
    print("  失败局的 最小dist: p10 %.3f 中位 %.3f | 最大angle: p10 %.3f 中位 %.3f"
          % (np.percentile(F[:,0],10), np.median(F[:,0]), np.percentile(F[:,1],10), np.median(F[:,1])))
    print("  (要求 dist<0.032 且 angle>0.985)")
    print("  dist 达标但 angle 不达标: %d 局" % int(((F[:,0]<0.032)&(F[:,1]<0.985)).sum()))
    print("  angle 达标但 dist 不达标: %d 局" % int(((F[:,0]>=0.032)&(F[:,1]>=0.985)).sum()))
