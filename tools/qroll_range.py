import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
N=120; rolls=[]; a2s=[]; qs=[]; dzs=[]
for ep in range(N):
    obs,_=env.reset(seed=ep); term=False
    for t in range(env.max_steps):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase==3:
            d=env.data; rolls.append(float(d.qpos[4]))
            rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
            a2s.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            qs.append(float(d.qpos[env.finger_qpos_adr]))
            ee=d.site_xpos[env.ee_id]; dzs.append(float(ee[2]-env.box_pos[2]))
            break
        if term or trunc: break
rolls=np.array(rolls); a2s=np.array(a2s); qs=np.array(qs); dzs=np.array(dzs)
print("entries", len(rolls), "/", N)
print("tool_roll: min %.4f  p5 %.4f  mean %.4f  std %.4f  p95 %.4f  max %.4f" % (
    rolls.min(), np.percentile(rolls,5), rolls.mean(), rolls.std(), np.percentile(rolls,95), rolls.max()))
print("angle2: min %.3f mean %.3f max %.3f" % (a2s.min(), a2s.mean(), a2s.max()))
print("q: min %.4f mean %.4f max %.4f" % (qs.min(), qs.mean(), qs.max()))
print("ee_dz: min %.4f mean %.4f max %.4f" % (dzs.min(), dzs.mean(), dzs.max()))
hist,edges=np.histogram(rolls,bins=10)
for c,e0,e1 in zip(hist,edges[:-1],edges[1:]):
    print(f"  [{e0:+.3f},{e1:+.3f}): {c}")
