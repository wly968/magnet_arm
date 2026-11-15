import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
def ep(det, x=0.0,y=0.6):
    obs,_=env.reset(seed=0)
    a=env.box_qpos_adr; env.data.qpos[a:a+3]=[x,y,0.375]; env.data.qpos[a+3:a+7]=[1,0,0,0]
    env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([x,y,0.375])
    mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
    maxstill=0; term=False
    for t in range(150):
        act,_=model.predict(obs,deterministic=det)
        obs,r,term,trunc,info=env.step(act)
        maxstill=max(maxstill, env.still_count)
        if term or trunc: break
    return t+1, term, env.phase, maxstill, float(r)
for det in (True, False):
    print("deterministic" if det else "stochastic")
    for i in range(5):
        steps,term,phase,ms,r=ep(det)
        print(f"  ep{i}: steps={steps} term={term} phase={phase} max_still={ms}")
