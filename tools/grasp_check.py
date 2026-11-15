import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
def run(model, x=0.0,y=0.6):
    obs,_=env.reset(seed=0)
    a=env.box_qpos_adr; env.data.qpos[a:a+3]=[x,y,0.375]; env.data.qpos[a+3:a+7]=[1,0,0,0]
    env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([x,y,0.375])
    mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
    term=False; details=[]
    for t in range(150):
        act,_=model.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(act)
        if env.phase==3:
            ee=env.data.site_xpos[env.ee_id]; q=env.data.qpos[env.finger_qpos_adr]
            details.append((t,round(float(ee[2]),3),round(float(q),4),round(float(r),1)))
        if term or trunc: break
    return t+1,term,env.phase,details[-4:]
m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), env=env)
for p in [(0.0,0.6),(0.0,0.5),(-0.2,0.7),(0.2,0.6)]:
    print("scene ppo",p,run(m,*p))
m2=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
for p in [(0.0,0.6),(-0.2,0.7)]:
    print("phase2",p,run(m2,*p))
