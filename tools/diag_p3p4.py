import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"))
m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
N=40; rows=[]
for ep in range(N):
    obs,_=env.reset(); term=False; t3=None; t4=None
    for t in range(env.max_steps):
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            d=env.data; ee=d.site_xpos[env.ee_id]
            t3=(float(ee[0]-env.box_pos[0]), float(ee[1]-env.box_pos[1]),
                float(ee[2]-env.box_pos[2]), float(d.qpos[env.finger_qpos_adr]),
                env.left_touched if hasattr(env,'left_touched') else None)
        if env.phase==4:
            d=env.data
            t4=(float(d.qpos[env.finger_qpos_adr]), min(env.left_force, env.right_force),
                abs(env.left_force-env.right_force))
        if term or trunc: break
    rows.append((ep, env.phase, t3, t4))
print("ep phase  phase3_end(dx,dy,dz,q)            phase4_end(q,minF,dF)")
for r in rows[:20]:
    print(r[0], r[1], tuple(round(x,3) for x in (r[2][:4] if r[2] else [])), tuple(round(x,3) for x in (r[3] if r[3] else [])))
