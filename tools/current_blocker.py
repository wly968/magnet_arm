import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')
e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
ok=0; rows=[]
for ep in range(10):
    obs,_=e.reset(); best=None
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase==2:
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            d=float(np.linalg.norm(ee-tgt))
            j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            ang=float(np.dot(f,[0,0,-1]))
            vr=float(np.linalg.norm(e.ee_val-e.box_vel))
            band=(ee[2] < tgt[2]+0.03) and (ee[2] > e.box_pos[2]+0.03)
            if best is None or d<best[0]: best=(d,ang,vr,float(ee[2]),float(tgt[2]),band)
        if e.phase>=3 or term or tr: break
    ok+=int(e.phase>=3)
    if best: rows.append(best)
print("当前判据（band ±0.03）下: 成功 %d/10" % ok)
for i,r in enumerate(rows[:5]):
    print("  ep%d 最近点: dist %.3f | angle %.3f | rel %.3f | ee_z %.3f (需 <%.3f, >%.3f) | band %s"
          % (i, r[0], r[1], r[2], r[3], r[4]+0.03, r[3]-0.03, r[5]))
print("  angle 中位 %.3f (需 > 0.94), 满足角度的局数 %d/10"
      % (np.median([r[1] for r in rows]), sum(1 for r in rows if r[1]>0.94)))
