import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
import train_phase3-5 as T
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34_retract.zip", device="cpu")
e=T.make_env(m1)(); cnt={}; info=[]
for ep in range(30):
    obs,_=e.reset()
    u=e.unwrapped
    for t in range(300):
        a,_=m2.predict(obs, deterministic=True)
        obs,r,term,tr,info_=e.step(a)
        if term or tr: break
    fp=int(u.phase); cnt[fp]=cnt.get(fp,0)+1
    # 结束时到当前 target 的偏差
    ee=u.data.site_xpos[u.ee_id]; tgt=u._target()
    j3=u.data.xanchor[u.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    info.append((fp, float(np.linalg.norm(ee-tgt)), float(np.dot(f,[0,0,-1]))))
print("phase 分布:", dict(sorted(cnt.items())))
import collections
for fp in sorted(set(i[0] for i in info)):
    sub=[i for i in info if i[0]==fp]
    print("  停在 P%d: %d 局 | dist 中位 %.3f | angle 中位 %.3f (阈值 0.05 / 0.945)"
          % (fp, len(sub), np.median([s[1] for s in sub]), np.median([s[2] for s in sub])))
