import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("saves/phase345_smooth2.zip","saves/phase34_retract.zip"):
    m2=PPO.load(os.path.join(r"scene_magnet",name), device="cpu")
    for th in ((0.06,0.93),(0.05,0.945)):
        e=ArmEnv(terminate_on_phase2=False, max_phase=6); e.max_steps=300
        e._set_dist(th[0]); e._set_angle1(th[1]); e._set_still_count(0)
        ph={}
        for ep in range(40):
            e.reset(); e.phase=3
            for t in range(e.max_steps):
                o=e._get_obs(); a,_=m2.predict(o,deterministic=True)
                o,r,term,tr,info=e.step(a)
                if term or tr: break
            ph[e.phase]=ph.get(e.phase,0)+1
        print("%-32s %.3f/%.3f -> %s" % (name,th[0],th[1],ph))
