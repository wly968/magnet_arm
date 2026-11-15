import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
rows=[]
for ep in range(60):
    obs,_=env.reset(seed=ep); term=False
    for t in range(env.max_steps):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            d=env.data
            yaw=float(np.arctan2(env.box_pos[1],env.box_pos[0]))
            rot=d.geom_xmat[env.box_geom_id].reshape(3,3)
            edge=float(np.arctan2(rot[1,0],rot[0,0]))
            px=d.geom_xmat[env.palm_id].reshape(3,3)
            py=float(np.arctan2(px[1,1],px[0,1]))   # palm Y angle in world
            qroll=float(d.qpos[4])
            rows.append((ep,yaw,edge,py,qroll, (edge-py+np.pi)%(2*np.pi)-np.pi, (edge-yaw+np.pi)%(2*np.pi)-np.pi))
            break
        if term or trunc: break
print("ep  yaw    edge   palmY  qroll  edge-palmY  edge-yaw")
for r in rows[:15]:
    print("%2d %6.3f %6.3f %6.3f %6.3f %10.3f %9.3f"%r)
arr=np.array([[r[1],r[2],r[3],r[4]] for r in rows])
print("corr(qroll, edge-yaw):", np.corrcoef(arr[:,3], (arr[:,2]-arr[:,1]))[0,1])
print("corr(qroll, palmY):", np.corrcoef(arr[:,3], arr[:,2])[0,1])
print("qroll mean/std:", arr[:,3].mean(), arr[:,3].std())
print("edge-yaw mean/std:", (arr[:,2]-arr[:,1]).mean(), (arr[:,2]-arr[:,1]).std())
