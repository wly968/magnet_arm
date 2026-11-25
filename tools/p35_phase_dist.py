import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from train_phase3-5 import make_env
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
for tag in ("p35_500000_steps","p35_1000000_steps"):
    p=f"scene_magnet/saves/ckpt_p35/{tag}.zip"
    if not os.path.exists(p): continue
    m2=PPO.load(p, device="cpu")
    env=make_env(m1)(); env.model1 if False else None
    # 直接驱动这个 env（它内部已带 model1 交接）
    import train_phase3-5 as T
    e=T.make_env(m1)()
    cnt={}
    for ep in range(20):
        obs,_=e.reset()
        for t in range(400):
            a,_=m2.predict(obs, deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if term or tr: break
        fp=int(e.unwrapped.phase); cnt[fp]=cnt.get(fp,0)+1
    print("%-22s 结束 phase 分布 %s" % (tag, dict(sorted(cnt.items()))))
