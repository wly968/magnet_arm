import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

env=ArmEnv(); model=PPO.load(os.path.join(SCENE,"ppo_arm_reach"), env=env)
xs=np.linspace(-0.45,0.20,7); ys=np.linspace(0.45,0.85,7)
res=np.zeros((len(ys),len(xs))); succ=np.zeros_like(res)
for j,y in enumerate(ys):
    for i,x in enumerate(xs):
        obs,_=env.reset(seed=0)
        env.box_pos=np.array([x,y,0.375]); env.data.mocap_pos[env.mocap_id]=env.box_pos
        env.data.mocap_quat[env.mocap_id]=[1,0,0,0]; mujoco.mj_forward(env.model,env.data)
        obs=env._get_obs()
        term=False
        for t in range(env.max_steps):
            a,_=model.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(a)
            if term or trunc: break
        d=env.data; wrist=d.xanchor[env.arm_joints[3]]; tgt=env.box_pos+np.array([0,0,0.25])
        dist=np.linalg.norm(wrist-tgt)
        res[j,i]=dist; succ[j,i]=int(term and env.phase==3)
print("dist (rows y %.2f..%.2f, cols x %.2f..%.2f)"%(ys[0],ys[-1],xs[0],xs[-1]))
for j in range(len(ys)-1,-1,-1):
    print("y=%.2f "%ys[j] + " ".join(f"{res[j,i]:.3f}" for i in range(len(xs))))
print("success map (1=phase3)")
for j in range(len(ys)-1,-1,-1):
    print("y=%.2f "%ys[j] + " ".join("  O  " if succ[j,i] else "  X  " for i in range(len(xs))))
print("total success", int(succ.sum()), "/", succ.size)
