import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
x,y=0.20,0.60
obs,_=env.reset(seed=0)
env.box_pos=np.array([x,y,0.375]); env.data.mocap_pos[env.mocap_id]=env.box_pos
env.data.mocap_quat[env.mocap_id]=[1,0,0,0]; mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
rows=[]; term=False
for t in range(60):
    ph_before=env.phase
    a,_=model.predict(obs,deterministic=True)
    obs,r,term,trunc,info=env.step(a)
    d=env.data; w=d.xanchor[env.arm_joints[3]]; tgt=env.box_pos+np.array([0,0,0.25])
    dist=np.linalg.norm(w-tgt); ee=d.site_xpos[env.ee_id]; fwd=(ee-w)/np.linalg.norm(ee-w)
    a2=float(np.dot(fwd,np.array([0,0,-1.]))); op=float(d.qpos[env.finger_qpos_adr])
    rows.append((t,ph_before,env.phase,round(dist,4),round(a2,3),round(op,4),round(float(r),2),term))
    if term: break
print("t ph_before->after dist angle2 open reward term")
for r_ in rows: print(r_)
