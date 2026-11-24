import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
paths=['scene_magnet/saves/phase1.zip','scene_magnet/saves/phase2.zip','scene_magnet/saves/best_4dof/best_model.zip']
e=ArmEnv()
for p in paths:
    if not os.path.exists(p): print(p,"不存在"); continue
    m=PPO.load(p, device='cpu')
    e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.20)
    rel=[]; ok=0; pos=0; tot=0; p1=0
    for ep in range(20):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase==2:
                tot+=1
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                dd=float(np.linalg.norm(ee-tgt))
                j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
                aa=float(np.dot(f,[0,0,-1])); vv=float(np.linalg.norm(e.ee_val-e.box_vel))
                band=(ee[2]<tgt[2]+0.03) and (ee[2]>e.box_pos[2]+0.03)
                rel.append(vv)
                if dd<0.05 and aa>0.94 and band: pos+=1
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3); p1+=int(e.phase>=2)
    rel=np.array(rel) if rel else np.array([np.nan])
    print("%-46s 步数 %8s | 进phase2 %2d/20 | 成功 %2d/20 | 位置到位 %4.1f%% | rel 中位 %.3f (p25 %.3f, p75 %.3f)"
          % (os.path.basename(p), getattr(m,'num_timesteps',0), p1, ok, 100*pos/max(1,tot),
             np.median(rel), np.percentile(rel,25), np.percentile(rel,75)))
