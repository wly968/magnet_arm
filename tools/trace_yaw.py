import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
x,y=0.20,0.60
obs,_=env.reset(seed=0)
a=env.box_qpos_adr; env.data.qpos[a:a+3]=[x,y,0.375]; env.data.qpos[a+3:a+7]=[1,0,0,0]
env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([x,y,0.375])
mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
for t in range(150):
    act,_=model.predict(obs,deterministic=True)
    obs,r,term,trunc,info=env.step(act)
    if env.phase==2 and t%20==0:
        d=env.data; px=d.geom_xmat[env.palm_id].reshape(3,3); rot=d.geom_xmat[env.box_geom_id].reshape(3,3)
        a2=float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))
        print(f"t={t} a2={a2:.4f} tool_roll={d.qpos[4]:+.3f} action[4]={act[4]:+.3f} ctrl4={d.ctrl[4]:+.3f} still={env.still_count}")
    if term: print("terminated t",t); break
else: print("timeout")
