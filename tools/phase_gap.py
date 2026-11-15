import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
def run(x,y):
    obs,_=env.reset(seed=0)
    env.box_pos=np.array([x,y,0.375]); env.data.mocap_pos[env.mocap_id]=env.box_pos
    env.data.mocap_quat[env.mocap_id]=[1,0,0,0]; mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
    flips=0; first=None; term=False
    for t in range(env.max_steps):
        ph0=env.phase
        a,_=model.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase!=ph0:
            flips+=1
            if ph0==1 and first is None:
                ee=env.data.site_xpos[env.ee_id].copy(); w=env.data.xanchor[env.arm_joints[3]].copy()
                tgt=env.box_pos+np.array([0,0,0.25])
                first=(round(float(ee[2]),3), round(float(w[2]),3), round(float(np.linalg.norm(w-tgt)),3))
        if term or trunc: break
    return flips, first, term, env.phase
for (x,y) in [(0.20,0.60),(0.20,0.72),(0.00,0.60),(0.10,0.72),(0.20,0.85),(-0.20,0.60)]:
    f,first,term,ph=run(x,y)
    print(f"({x:+.2f},{y:+.2f}) flips={f} first_phase2(ee_z,wrist_z,wrist_dist)={first} term={term} phase={ph}")
