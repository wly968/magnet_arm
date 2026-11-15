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
    maxph=1; maxstill=0; term=False; eez=[]; qs=[]
    for t in range(150):
        act,_=model.predict(obs,deterministic=det)
        obs,r,term,trunc,info=env.step(act)
        maxph=max(maxph,env.phase); maxstill=max(maxstill,env.still_count)
        if env.phase==3:
            eez.append(float(env.data.site_xpos[env.ee_id][2])); qs.append(float(env.data.qpos[env.finger_qpos_adr]))
        if term or trunc: break
    return dict(steps=t+1,term=term,maxph=maxph,maxstill=maxstill,
                ee3=(round(float(np.mean(eez)),3) if eez else None),
                q3=(round(float(np.mean(qs)),4) if qs else None),reward=round(float(r),1))
for name,path in (("phase2",os.path.join(SCENE,"saves","phase2.zip")),("ppo",os.path.join(SCENE,"ppo_arm_reach.zip"))):
    m=PPO.load(path, env=env)
    print(name)
    for det in (True,False):
        print("  det" if det else "  sto", run(m,det))
        print("  det2" if det else "  sto2", run(m,det,0.2,0.6))
