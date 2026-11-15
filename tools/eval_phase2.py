import os, sys, numpy as np
ROOT = r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
sys.path.insert(0, SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

env = ArmEnv()
model = PPO.load(os.path.join(SCENE, "ppo_arm_reach"), env=env)

N=50; succ=0; a1s=[]; a2s=[]; ops=[]; rolls=[]; p2steps=[]; p1steps=[]; dists=[]
for ep in range(N):
    obs,_=env.reset(seed=200+ep)
    term=False; p2_start=None
    for t in range(env.max_steps):
        a,_=model.predict(obs, deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==2 and p2_start is None: p2_start=t+1
        if term or trunc: break
    d=env.data
    wrist=d.xanchor[env.arm_joints[3]]; tgt=env.box_pos+np.array([0,0,0.25])
    dist=np.linalg.norm(wrist-tgt)
    rot=d.geom_xmat[env.box_geom_id].reshape(3,3); boxx=rot[:2,0]
    palm=d.geom_xmat[env.palm_id].reshape(3,3); py=palm[:2,1]
    a1=float(np.dot(py/np.linalg.norm(py), boxx/np.linalg.norm(boxx)))
    ee=d.site_xpos[env.ee_id]; fwd=(ee-wrist)/np.linalg.norm(ee-wrist)
    a2=float(np.dot(fwd,np.array([0,0,-1.])))
    op=float(d.qpos[env.finger_qpos_adr]); roll=float(d.qpos[env.model.joint("joint_tool_roll").id] if False else d.qpos[env.model.jnt_qposadr[env.model.joint("joint_tool_roll").id]])
    a1s.append(a1); a2s.append(a2); ops.append(op); rolls.append(roll); dists.append(dist)
    p2steps.append((t+1-p2_start) if p2_start else 0)
    p1steps.append(p2_start if p2_start else t+1)
    succ += int(term and env.phase==3)
print(f"success(phase3 terminated): {succ}/{N}")
print(f"angle1(palmY·boxX) mean {np.mean(a1s):.3f}  min {np.min(a1s):.3f}  max {np.max(a1s):.3f}")
print(f"angle2(tool down)  mean {np.mean(a2s):.3f}  min {np.min(a2s):.3f}")
print(f"gripper open       mean {np.mean(ops):.4f}  min {np.min(ops):.4f}")
print(f"tool_roll q        mean {np.mean(rolls):.3f}  min {np.min(rolls):.3f}  max {np.max(rolls):.3f}")
print(f"phase1 steps mean {np.mean(p1steps):.0f}   phase2 steps mean {np.mean(p2steps):.1f}  max {np.max(p2steps)}")
print(f"dist mean {np.mean(dists):.4f}")
