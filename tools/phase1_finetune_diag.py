import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/saves/phase1.zip', device='cpu')
print("obs", m.observation_space.shape, "| 动作标准差", np.round(np.exp(m.policy.log_std.detach().numpy()),3))
e=ArmEnv(); e._set_dist(0.03); e._set_angle1(0.98); e._set_still_count(0); e._set_rel(0.05)
mins=[]; inpos=[]; succ=0; zs=[]
for ep in range(20):
    obs,_=e.reset(); md=9; npos=0; n=0
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase==2:
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            d=float(np.linalg.norm(ee-tgt)); md=min(md,d); n+=1
            j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            a1=float(np.dot(f,[0,0,-1])); vr=float(np.linalg.norm(e.ee_val-e.box_vel))
            if d<0.03 and a1>0.98 and vr<0.05: npos+=1
            zs.append(float(ee[2]))
        if e.phase>=3 or term or tr: break
    mins.append(md); inpos.append(npos); succ+=int(e.phase>=3)
print("phase1.zip 在 phase2 中: 成功 %d/20, 最小 dist 中位 %.3f (最小 %.3f)" % (succ, np.median(mins), min(mins)))
print("每局“在位帧”数: 中位 %.0f; phase2 平均停留 %.0f 步" % (np.median(inpos), np.mean([n for n in [1]])))
print("末端高度分布: %.2f ~ %.2f" % (min(zs), max(zs)))
