import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); print("obs space", env.observation_space.shape)
model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
x,y=0.0,0.6
obs,_=env.reset(seed=0)
env.box_pos=np.array([x,y,0.375]); env.data.mocap_pos[env.mocap_id]=env.box_pos
env.data.mocap_quat[env.mocap_id]=[1,0,0,0]; mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
print("t phase dist angle1 angle2 speed open still")
term=False
for t in range(120):
    a,_=model.predict(obs,deterministic=True)
    obs,r,term,trunc,info=env.step(a)
    d=env.data; w=d.xanchor[env.arm_joints[3]]; tgt=env.box_pos+np.array([0,0,0.25])
    dist=float(np.linalg.norm(w-tgt)); ee=d.site_xpos[env.ee_id]; fwd=(ee-w)/np.linalg.norm(ee-w)
    a1=float(np.dot(fwd,np.array([0,0,-1.])))
    rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
    a2=float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))
    sp=float(np.linalg.norm(d.qvel[env.arm_dof_adr])); op=float(d.qpos[env.finger_qpos_adr])
    if t<5 or t%10==0 or env.phase>=2:
        print(f"{t:3d} {env.phase} {dist:.3f} {a1:.3f} {a2:.3f} {sp:.3f} {op:.4f} {env.still_count}")
    if term or trunc: break
