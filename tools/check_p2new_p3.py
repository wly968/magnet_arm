import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env); m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
def phase2_end_stats():
    N=40; d=[]; a1=[]; a2=[]; q=[]; roll=[]; z=[]
    for ep in range(N):
        obs,_=env.reset(seed=ep); term=False
        for t in range(env.max_steps):
            a,_=m1.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(a)
            if env.phase==2:
                dd=env.data; rot=dd.geom_xmat[env.box_geom_id].reshape(3,3); px=dd.geom_xmat[env.palm_id].reshape(3,3)
                a2.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
                q.append(float(dd.qpos[env.finger_qpos_adr]))
                roll.append(float(dd.qpos[4]))
                ee=dd.site_xpos[env.ee_id]; d.append(float(np.linalg.norm(ee-(env.box_pos+np.array([0,0,0.15])))))
                z.append(float(ee[2]-env.box_pos[2]))
            if term or trunc: break
    return d,a2,q,roll,z
d,a2,q,roll,z=phase2_end_stats()
print("phase2 end: dist",round(float(np.mean(d)),4),"angle2",round(float(np.mean(a2)),3),
      "q",round(float(np.mean(q)),4),"tool_roll",round(float(np.mean(roll)),3),"ee_dz",round(float(np.mean(z)),4))
# phase3 reset target stats
env.reset(option={"phase":3}); dd=env.data
ee=dd.site_xpos[env.ee_id]; rot=dd.geom_xmat[env.box_geom_id].reshape(3,3); px=dd.geom_xmat[env.palm_id].reshape(3,3)
a2r=abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0]))))
print("phase3 reset: angle2",round(a2r,3),"q",round(float(dd.qpos[env.finger_qpos_adr]),4),"ee_dz",round(float(ee[2]-env.box_pos[2]),4))
# dual eval
N=60; succ=0; ends={}
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==4)
print("dual:",f"{succ}/{N}",ends)
