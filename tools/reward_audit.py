import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/saves/phase2.zip', device='cpu')
e=ArmEnv(); e._set_dist(0.03); e._set_angle1(0.98); e._set_still_count(0)
tot=[]; bonus_frames=[]; succ=0
for ep in range(10):
    obs,_=e.reset(); R=0.0; n5=0
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a); R+=float(r)
        if e.phase==2:
            if abs(r - (-e.data.qpos[e.arm_qpos_adr][1]*0)) >= 0:  # 仅统计
                pass
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            d=float(np.linalg.norm(ee-tgt)); j3=e.data.xanchor[e.arm_joints[3]]
            f=(ee-j3)/np.linalg.norm(ee-j3); a1=float(np.dot(f,[0,0,-1]))
            vr=float(np.linalg.norm(e.ee_val-e.box_vel))
            if d<e.dist_threshold and a1>e.angle1_threshold and vr<e.rel_threshold: n5+=1
        if e.phase>=3: succ+=1; break
        if term or tr: break
    tot.append(R); bonus_frames.append(n5)
print("10 局：成功 %d, 总奖励 均值 %.0f, 每局“+5 奖励帧”均值 %.0f" % (succ, np.mean(tot), np.mean(bonus_frames)))
print("单项核算：若 250 步都在位 = +%d (5×250)；成功只给 +60" % (5*250))
