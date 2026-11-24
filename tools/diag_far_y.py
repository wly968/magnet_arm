import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1)
e=ArmEnv(); rows=[]
for ep in range(90):
    obs,_=e.reset(); y0=float(e.box_pos[1])
    d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
    last=e.phase; md=9; ma=-1; ok=False
    for t in range(e.max_steps+e.max_wait_steps+10):
        if e.phase!=last:
            last=e.phase
            d,a,s=TH2 if e.phase==2 else TH1
            e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        act,_=m1.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase==2:
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            md=min(md,float(np.linalg.norm(ee-tgt)))
            j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            ma=max(ma,float(np.dot(f,[0,0,-1])))
        if e.phase>=3 or term or tr: break
    rows.append((y0, int(e.phase)>=3, md, ma, int(e.phase)))
far=[r for r in rows if r[0]>0.78]
print("y>0.78 的 %d 局： 成功 %d | 到达的最小 dist 中位 %.3f | 最大 angle 中位 %.3f"
      % (len(far), sum(1 for r in far if r[1]), np.median([r[2] for r in far]), np.median([r[3] for r in far])))
print("  结束 phase 分布:", {p: sum(1 for r in far if r[4]==p) for p in set(r[4] for r in far)})
near=[r for r in rows if r[0]<=0.78]
print("y<=0.78 的 %d 局： 成功 %d" % (len(near), sum(1 for r in near if r[1])))
