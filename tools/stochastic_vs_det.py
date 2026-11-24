import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(8)])
m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
m.learn(total_timesteps=20000, callback=CurriculumCallback())

def run(det, n=10):
    e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.2)
    rels=[]; ok=0
    for ep in range(n):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=det)
            obs,r,term,tr,info=e.step(a)
            if e.phase==2: rels.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    rel=np.array(rels)
    return ok, np.median(rel), 100*(rel<0.2).mean()

for det,name in ((True,"确定性"),(False,"随机(训练时)")):
    ok,med,frac = run(det)
    print("%-12s: 成功 %d/10, rel 中位 %.3f, rel<0.2 占比 %.1f%%" % (name, ok, med, frac))
