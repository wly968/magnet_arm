import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","phase34.zip"), device="cpu")
e=ArmEnv(); e.max_steps=250
e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0)
for ep in range(3):
    e.reset(); e.phase=3
    for t in range(e.max_steps):
        o=e._get_obs(); a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=e.step(a)
        if term or tr: break
    print("ep%d phase=%d steps=%d term=%s trunc=%s info=%s ee=%s" %
          (ep, e.phase, t+1, term, tr, info, np.round(e.data.site_xpos[e.ee_id],2)))
