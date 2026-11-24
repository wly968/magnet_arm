import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
import train_phase34 as T
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
for tag in ("p35_500000_steps",):
    p=f"scene_magnet/saves/ckpt_p35/{tag}.zip"
    m2=PPO.load(p, device="cpu")
    e=T.make_env(m1)(); cnt={}; lens=[]
    for ep in range(20):
        obs,_=e.reset(); u=e.unwrapped
        p0=u.phase
        for t in range(300):
            a,_=m2.predict(obs, deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if term or tr: break
        fp=int(u.phase); cnt[fp]=cnt.get(fp,0)+1; lens.append(t+1)
        if ep==0: print("reset 后 phase=%d, 首回合走了 %d 步, 结束 phase=%d" % (p0, t+1, fp))
    print("%s 分布 %s | 平均长度 %.1f" % (tag, dict(sorted(cnt.items())), np.mean(lens)))
