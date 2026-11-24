import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(24)])
m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
m.learn(total_timesteps=60000, callback=CurriculumCallback())   # 训练到 dist0.05/angle0.94/still0/rel0.2

e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.2)
tot2=0; frames=[]; ok=0
for ep in range(30):
    obs,_=e.reset(); n2=0
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=False)     # 训练时的随机采样
        obs,r,term,tr,info=e.step(a)
        if e.phase==2: n2+=1
        if e.phase>=3 or term or tr: break
    tot2+=n2; frames.append(n2); ok+=int(e.phase>=3)
d=e._diag()
print("30 局: 进入 phase3 的 %d 局" % ok)
print("phase2 帧总数 %d, 每局中位 %.0f" % (tot2, np.median(frames)))
print("env diag（只覆盖最近 30 局）: n=%d, 各条件 rel=%d, all=%d" % (d["n"], d["rel"], d["all"]))
print("=> 每帧全满足率 %.2f%%, 每局成功率 %.1f%%" % (100*d["all"]/max(1,d["n"]), 100*ok/30))
