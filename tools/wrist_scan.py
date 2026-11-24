import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')
e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
obs,_=e.reset(); best=None
for t in range(e.max_steps+e.max_wait_steps+10):
    a,_=m.predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(a)
    if e.phase==2:
        ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
        d=float(np.linalg.norm(ee-tgt))
        j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        ang=float(np.dot(f,[0,0,-1]))
        if best is None or d<best[0]:
            best=(d,ang,e.data.qpos[e.arm_qpos_adr].copy(),tgt.copy())
    if e.phase>=3 or term or tr: break
d,ang,q,tgt=best
print("策略最近点: dist %.3f angle %.3f q=%s" % (d,ang,np.round(q,3)))
print("扫描腕关节 q[3]:")
lo,hi=e.model.jnt_range[e.arm_joints[3]]
for val in np.linspace(lo,hi,9):
    qq=q.copy(); qq[3]=val
    e.data.qpos[e.arm_qpos_adr]=qq; mujoco.mj_forward(e.model,e.data)
    ee=e.data.site_xpos[e.ee_id].copy()
    j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    a1=float(np.dot(f,[0,0,-1]))
    print("  q3=%+.3f -> dist %.3f, angle %.3f" % (val, float(np.linalg.norm(ee-tgt)), a1))
