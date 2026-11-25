import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
import train_phase3-5 as T
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/ppo_arm_reach.zip", device="cpu")
print("model2 步数", getattr(m2,'num_timesteps',0), "| obs", m2.observation_space.shape)
e=T.make_env(m1)(); cnt={}; lens=[]; env_thr=None
for ep in range(30):
    obs,_=e.reset()
    for t in range(300):
        a,_=m2.predict(obs, deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if term or tr: break
    fp=int(e.unwrapped.phase); cnt[fp]=cnt.get(fp,0)+1; lens.append(t+1)
print("30 局 phase 分布:", dict(sorted(cnt.items())), "| 平均步数 %.1f" % np.mean(lens))
# 随机采样对照
cnt2={}
for ep in range(30):
    obs,_=e.reset()
    for t in range(300):
        a,_=m2.predict(obs, deterministic=False)
        obs,r,term,tr,info=e.step(a)
        if term or tr: break
    fp=int(e.unwrapped.phase); cnt2[fp]=cnt2.get(fp,0)+1
print("随机采样分布:", dict(sorted(cnt2.items())))
