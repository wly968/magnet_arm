import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"))
m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
# 物块质量/摩擦
bi=env.box_body_id; gi=env.box_geom_id
print("cube mass %.3f kg, weight %.3f N" % (env.model.body_mass[bi], env.model.body_mass[bi]*9.81))
print("cube friction", env.model.geom_friction[gi])
# 跑 40 局，记录 phase5 时的最小/最大接触力
mins=[]; maxs=[]; success=0
for ep in range(40):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if env.phase>=4 and (env.left_force>0 or env.right_force>0):
            mins.append(min(env.left_force,env.right_force)); maxs.append(max(env.left_force,env.right_force))
        if term or trunc: break
    success+=int(env.phase==5)
print("success %d/40" % success)
if mins:
    print("grip normal force: min-side mean %.2f N, max-side mean %.2f N" % (np.mean(mins), np.mean(maxs)))
    print("min-side range %.2f ~ %.2f N" % (min(mins), max(mins)))
mu=env.model.geom_friction[gi][0]
m=env.model.body_mass[bi]
print("lift condition: 2*mu*F > m*g -> F > %.3f N (mu=%.2f, m=%.3f kg)" % (m*9.81/(2*mu), mu, m))
