import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
import glob
cands=['scene_magnet/saves/best_4dof/best_model.zip','scene_magnet/ppo_arm_reach.zip']
p=[c for c in cands if os.path.exists(c)][0]
m=PPO.load(p, device='cpu'); print("模型:", p)
e=ArmEnv(); e._set_dist(0.03); e._set_angle1(0.98); e._set_still_count(2); e._set_rel(0.05)
for ep in range(8):
    obs,_=e.reset(); n2=0; allframe=0
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        prev=e.phase
        obs,r,term,tr,info=e.step(a)
        if e.phase==2:
            n2+=1
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            d=float(np.linalg.norm(ee-tgt))
            j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            a1=float(np.dot(f,[0,0,-1])); vr=float(np.linalg.norm(e.ee_val-e.box_vel))
            band=(ee[2]<tgt[2]+0.03) and (ee[2]>e.box_pos[2]+0.03)
            if d<0.03 and a1>0.98 and vr<0.05 and band: allframe+=1
        if prev==2 and e.phase==3:
            print("  ep%d 进入 phase3：phase2 帧 %d，其中全满足 %d 帧" % (ep,n2,allframe)); break
        if term or tr: print("  ep%d 结束于 phase %d（phase2 帧 %d，全满足 %d）" % (ep,e.phase,n2,allframe)); break
