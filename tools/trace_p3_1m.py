import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env); m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
N=30; succ=0; ends={}; rows=[]
for ep in range(N):
    obs,_=env.reset(); term=False; ms=[]
    for t in range(env.max_steps):
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            d=env.data; ee=d.site_xpos[env.ee_id]; q=float(d.qpos[env.finger_qpos_adr])
            ncon=0; forces=[]
            for i in range(d.ncon):
                c=d.contact[i]
                if (c.geom1==env.finger_L_id and c.geom2==env.box_geom_id) or (c.geom1==env.box_geom_id and c.geom2==env.finger_L_id) or \
                   (c.geom1==env.finger_R_id and c.geom2==env.box_geom_id) or (c.geom1==env.box_geom_id and c.geom2==env.finger_R_id):
                    f=np.zeros(6); import mujoco; mujoco.mj_contactForce(env.model,d,i,f); forces.append(abs(f[0]))
            ms.append((t,round(float(ee[2]),3),round(q,4),len(forces),round(max(forces),2) if forces else 0))
        if term or trunc: break
    if ms: rows.append((ep,env.phase,term,ms[-1]))
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==4)
print("dual:",f"{succ}/{N}",ends)
print("ep phase term  t     ee_z   q      ncon force")
for r in rows[:12]: print(r)
