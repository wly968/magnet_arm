import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')

rows=[]
for ep in range(12):
    e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94)   # 放宽 phase1，保证能进 phase2
    e._set_still_count(2); e._set_rel(0.05)              # 用训练终点的 dwell/rel
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
            rec=(d,ang,vr,float(ee[2]),float(tgt[2]),float(e.box_pos[2]),e.still_count)
            if best is None or d<best[0]: best=rec
        if e.phase>=3 or term or tr: break
    if best: rows.append(best)

print("phase2 最近点统计（12 局）:")
P=np.array([[r[0],r[1],r[2],r[3]] for r in rows])
print("  dist 中位 %.3f | angle 中位 %.4f | rel_vel 中位 %.3f | ee_z 中位 %.3f" %
      (np.median(P[:,0]), np.median(P[:,1]), np.median(P[:,2]), np.median(P[:,3])))
print("  条件核对（需全部满足）:")
print("    dist < 0.05      : %d/12" % sum(1 for r in rows if r[0]<0.05))
print("    angle > 0.94     : %d/12" % sum(1 for r in rows if r[1]>0.94))
print("    rel_vel < 0.05   : %d/12" % sum(1 for r in rows if r[2]<0.05))
print("    ee_z < tgt+0.01  : %d/12" % sum(1 for r in rows if r[3] < r[4]+0.01))
print("    ee_z > box+0.05  : %d/12" % sum(1 for r in rows if r[3] > r[5]+0.05))
print("  ee_z 与允许带 [box+0.05, tgt+0.01] 对比: 中位 %.3f, 带 = [%.3f, %.3f]" %
      (np.median(P[:,3]), np.median([r[5]+0.05 for r in rows]), np.median([r[4]+0.01 for r in rows])))
