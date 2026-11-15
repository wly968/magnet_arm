import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
def run(x,y):
    obs,_=env.reset(seed=0)
    a=env.box_qpos_adr; env.data.qpos[a:a+3]=[x,y,0.375]; env.data.qpos[a+3:a+7]=[1,0,0,0]
    env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([x,y,0.375])
    mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
    term=False; last=None
    for t in range(150):
        act,_=model.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(act)
        d=env.data; w=d.xanchor[env.arm_joints[3]]; tgt=env.box_pos+np.array([0,0,0.25])
        dist=float(np.linalg.norm(w-tgt)); ee=d.site_xpos[env.ee_id]; fwd=(ee-w)/np.linalg.norm(ee-w)
        a1=float(np.dot(fwd,np.array([0,0,-1.])))
        rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
        a2=float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))
        sp=float(np.linalg.norm(d.qvel[env.arm_dof_adr])); op=float(d.qpos[env.finger_qpos_adr])
        last=(t,env.phase,dist,a1,a2,sp,op,env.still_count,r,term)
        if term: break
    print(f"({x:+.2f},{y:+.2f}) end t={last[0]} phase={last[1]} dist={last[2]:.4f} a1={last[3]:.3f} a2={last[4]:.3f} speed={last[5]:.3f} open={last[6]:.4f} still={last[7]} r={last[8]:.1f} term={last[9]}")
for p in [(0.0,0.6),(0.2,0.6),(0.0,0.5),(0.2,0.8),(-0.3,0.6)]:
    run(*p)
