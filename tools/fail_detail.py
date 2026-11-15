import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
for (x,y) in [(0.20,0.45),(0.20,0.58),(0.20,0.72),(0.20,0.85),(0.10,0.72),(0.20,0.30)]:
    obs,_=env.reset(seed=0)
    env.box_pos=np.array([x,y,0.375]); env.data.mocap_pos[env.mocap_id]=env.box_pos
    env.data.mocap_quat[env.mocap_id]=[1,0,0,0]; mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
    term=False; phases={1:0,2:0}
    for t in range(env.max_steps):
        phases[env.phase]=phases.get(env.phase,0)+1
        a,_=model.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    d=env.data; w=d.xanchor[env.arm_joints[3]]; tgt=env.box_pos+np.array([0,0,0.25])
    dist=np.linalg.norm(w-tgt); ee=d.site_xpos[env.ee_id]; fwd=(ee-w)/np.linalg.norm(ee-w)
    a2=float(np.dot(fwd,np.array([0,0,-1.])))
    q=d.qpos[:7]; roll=q[4]
    print(f"box({x:+.2f},{y:+.2f}) dist={dist:.3f} angle2={a2:.3f} phase={env.phase} term={term} steps={t+1} ph1={phases.get(1,0)} ph2={phases.get(2,0)} q={np.round(q,2)}")
