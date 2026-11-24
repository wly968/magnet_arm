import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from train_phase6 import make_env
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
for thr in (0,2,5,10):
    res={}
    for det in (True, False):
        e=make_env(m1,m2)(); e.unwrapped._set_home_still(thr); ok=0
        for ep in range(15):
            o,_=e.reset()
            for t in range(240):
                a,_=m3.predict(o, deterministic=det)
                o,r,term,tr,info=e.step(a)
                if term or tr: break
            ok+=int(e.unwrapped.phase>=7)
        res[det]=ok
    print("home_still=%2d | 确定性 %2d/15 | 随机 %2d/15" % (thr, res[True], res[False]))
