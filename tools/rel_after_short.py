import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, DiagCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(24)])
m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
m.learn(total_timesteps=120_000, callback=[CurriculumCallback(), DiagCallback(every=40_000)])
m.save("/tmp/short120k.zip".replace("/tmp","scene_magnet/saves/_short120k"))

e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
rel=[]
for ep in range(15):
    obs,_=e.reset()
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase==2: rel.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
        if e.phase>=3 or term or tr: break
rel=np.array(rel)
print("训练后 rel 分布: p5 %.3f p25 %.3f 中位 %.3f p75 %.3f | rel<0.05 占比 %.1f%%"
      % (*np.percentile(rel,[5,25,50,75]), 100*(rel<0.05).mean()))
