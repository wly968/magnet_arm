import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from train_phase6 import make_env
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
e=make_env(m1,m2)(); e.unwrapped._set_home_still(2)
ok=0; durs=[]; coll=0
for ep in range(20):
    o,_=e.reset(); u=e.unwrapped; t0=None
    for t in range(300):
        a,_=m3.predict(o, deterministic=True)
        o,r,term,tr,info=e.step(a)
        if t0 is None: t0=t
        if term or tr: break
    ok+=int(u.phase>=7); durs.append(t+1); coll+=int(u.collected)
print("phase6 单独测试 20 局: 成功 %d/20 | phase6 用时 %.0f 步（原含入篮约 91 步）| 期间 collected %d/20"
      % (ok, np.mean(durs), coll))
