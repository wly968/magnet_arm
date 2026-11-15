import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(magnet_action=False)
model=PPO.load(os.path.join(SCENE,"saves","phase1_4dim.zip"), device="cpu")
print("obs", env.observation_space.shape, "model obs", model.observation_space.shape)
for ep in range(2):
    o,_=env.reset(seed=ep); best=9
    for t in range(60):
        a,_=model.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        ee=env.data.site_xpos[env.ee_id]
        tgt=np.array([env.box_pos[0],env.box_pos[1],env.box_pos[2]+0.15])
        d=np.linalg.norm(ee-tgt)
        j3=env.data.xanchor[env.arm_joints[3]]
        fwd=(ee-j3)/np.linalg.norm(ee-j3); a1=float(np.dot(fwd,[0,0,-1]))
        best=min(best,d)
        if d<0.08 or term or tr:
            print(" ep%d t%d dist %.4f angle1 %.3f success=%s term=%s" % (ep,t,d,a1,info.get("success"),term))
        if term or tr: break
    print(" ep%d best dist %.4f" % (ep,best))
