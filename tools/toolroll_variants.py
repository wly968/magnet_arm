import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
def run(sign, off, tag):
    N=20; succ=0; a2s=[]
    for ep in range(N):
        obs,_=env.reset(seed=ep); term=False
        for t in range(env.max_steps):
            a,_=m.predict(obs,deterministic=True)
            # 让 phase2 成功进 phase3，然后覆盖 tool_roll 目标
            obs,r,term,trunc,info=env.step(a)
            if env.phase==3:
                d=env.data
                box_rot=d.geom_xmat[env.box_geom_id].reshape(3,3)
                edge=np.arctan2(box_rot[1,0],box_rot[0,0])
                yaw=np.arctan2(env.box_pos[1],env.box_pos[0])
                q=(edge + sign*yaw + off + np.pi)%(2*np.pi)-np.pi
                d.ctrl[4]=q
                rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
                a2s.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            if term or trunc: break
        succ+=int(env.phase==4)
    print(tag, f"success {succ}/{N}", "a2", round(float(np.mean(a2s)),3) if a2s else None)
for sign, off, tag in ((1,-np.pi/2,"edge+yaw-pi/2"),(-1,-np.pi/2,"edge-yaw-pi/2"),
                       (1,np.pi/2,"edge+yaw+pi/2"),(-1,np.pi/2,"edge-yaw+pi/2"),
                       (1,0,"edge+yaw"),(-1,0,"edge-yaw")):
    run(sign,off,tag)
