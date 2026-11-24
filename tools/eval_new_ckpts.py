import os, sys, numpy as np, glob
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
files=sorted(glob.glob("scene_magnet/saves/ckpt/*.zip"))
for p in files:
    try:
        m=PPO.load(p, device='cpu')
    except Exception as e:
        continue
    if int(m.observation_space.shape[0])!=23: continue
    e=ArmEnv()
    res=[]
    for (d,a,s,rv) in ((0.034,0.98,2,0.05),(0.030,0.99,2,0.05)):
        e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(rv)
        ok=0
        for ep in range(15):
            obs,_=e.reset()
            for t in range(e.max_steps+e.max_wait_steps+10):
                act,_=m.predict(obs,deterministic=True)
                obs,r,term,tr,info=e.step(act)
                if e.phase>=3 or term or tr: break
            ok+=int(e.phase>=3)
        res.append(ok)
    print("%-30s 第5档 %2d/15 | 第6档 %2d/15" % (os.path.basename(p), res[0], res[1]))
