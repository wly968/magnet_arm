import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv(); rng=np.random.default_rng(0)
bad=0
for ep in range(5):
    o,_=env.reset(seed=ep); tot=0.0
    for t in range(env.max_steps):
        o,r,term,tr,info=env.step(rng.uniform(-1,1,4)); tot+=r
        if not np.isfinite(o).all(): bad+=1; break
        if term or tr: break
    print("ep%d steps %d ret %.1f phase %d open %s attached %s" % (ep, t+1, tot, env.phase, env.open, env.magnet_attached))
print("non-finite obs:", bad)
