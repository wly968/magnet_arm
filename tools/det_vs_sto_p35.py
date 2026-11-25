import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
import train_phase3-5 as T
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
import glob
files=sorted(glob.glob("scene_magnet/saves/ckpt_p35/*.zip"), key=os.path.getmtime)[-2:]
for p in files:
    m2=PPO.load(p, device="cpu")
    e=T.make_env(m1)()
    out={}
    for det in (True, False):
        cnt={}
        for ep in range(20):
            obs,_=e.reset()
            for t in range(300):
                a,_=m2.predict(obs, deterministic=det)
                obs,r,term,tr,info=e.step(a)
                if term or tr: break
            fp=int(e.unwrapped.phase); cnt[fp]=cnt.get(fp,0)+1
        out[det]=dict(sorted(cnt.items()))
    print("%-22s 确定性 %s | 随机 %s" % (os.path.basename(p), out[True], out[False]))
