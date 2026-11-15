import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
def run(model, det, x=0.0,y=0.6):
    obs,_=env.reset(seed=0)
    a=env.box_qpos_adr; env.data.qpos[a:a+3]=[x,y,0.375]; env.data.qpos[a+3:a+7]=[1,0,0,0]
    env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([x,y,0.375])
    mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
    maxph=1; maxstill=0; term=False
    for t in range(150):
        act,_=model.predict(obs,deterministic=det)
        obs,r,term,trunc,info=env.step(act)
        maxph=max(maxph,env.phase); maxstill=max(maxstill,env.still_count)
        if term or trunc: break
    q=float(env.data.qpos[env.finger_qpos_adr])
    return t+1, term, maxph, maxstill, 0.02+q, round(float(r),1)
for name in ("phase2","ppo_arm_reach"):
    m=PPO.load(os.path.join(SCENE,"saves",name) if name=="phase2" else os.path.join(SCENE,name), env=env)
    print(name)
    for det in (True,False):
        for i in range(3):
            print("  det" if det else "  sto", run(m,det))
