import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("saves/phase12_004.zip","saves/phase1.zip"):
    p=os.path.join(SCENE,name)
    if not os.path.exists(p): continue
    m=PPO.load(p, device="cpu")
    for d_,a_,s_ in ((0.06,0.90,0),(0.04,0.96,1)):
        e=ArmEnv(); e.max_steps=250
        e._set_dist(d_); e._set_angle1(a_); e._set_still_count(s_)
        ok=0
        for ep in range(80):
            o,_=e.reset()
            for t in range(e.max_steps):
                act,_=m.predict(o,deterministic=True)
                o,r,term,tr,info=e.step(act)
                if term or tr: break
            ok+=int(e.phase>=3)
        print("%-22s %.2f/%.2f/%d -> %2d/80" % (name,d_,a_,s_,ok))
