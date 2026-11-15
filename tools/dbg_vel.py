import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","ppo_arm_reach_smooth.zip"), device="cpu")
env=ArmEnv(); env.max_steps=250
env._set_dist(0.05); env._set_angle1(0.94); env._set_still_count(0)
vs=[]; still=[0]
for ep in range(8):
    o,_=env.reset()
    for t in range(env.max_steps):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        if env.phase==2:
            ee=env.data.site_xpos[env.ee_id]; tgt=env.box_pos+env.grasp_offset
            d=float(np.linalg.norm(ee-tgt))
            j3=env.data.xanchor[env.arm_joints[3]]
            fwd=(ee-j3)/np.linalg.norm(ee-j3)
            ang=float(np.dot(fwd,[0,0,-1]))
            if d<0.05 and ang>0.94:
                vs.append(float(np.linalg.norm(env.data.qvel[env.arm_dof_adr])))
            still[0]=max(still[0],env.still_count)
        if term or tr: break
v=np.array(vs) if vs else np.array([np.nan])
print("到位区间样本 %d, qvel_norm: min %.3f p25 %.3f median %.3f p75 %.3f max %.3f"
      % (len(v), v.min(), np.percentile(v,25), np.median(v), np.percentile(v,75), v.max()))
print("max still_count", still[0])
