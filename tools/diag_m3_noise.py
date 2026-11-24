import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from train_phase6 import make_env
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
e=make_env(m1,m2)()
for det in (True, False):
    ok=0; minqs=[]; lens=[]
    for ep in range(20):
        o,_=e.reset(); u=e.unwrapped; mq=9
        for t in range(220):
            a,_=m3.predict(o, deterministic=det)
            o,r,term,tr,info=e.step(a)
            mq=min(mq, float(np.linalg.norm(u.data.qpos[u.arm_qpos_adr]-u.home_qpos)))
            if term or tr: break
        ok+=int(u.phase>=7); minqs.append(mq); lens.append(t+1)
    print("%s: 成功 %2d/20 | 最小 q_err 中位 %.3f | 平均步数 %.0f"
          % ("确定性" if det else "随机(训练时)", ok, np.median(minqs), np.mean(lens)))
