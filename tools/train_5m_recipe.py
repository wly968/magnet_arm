import os, sys, time, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
import torch; torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback

env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(24)])
m=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env, device="cpu", learning_rate=1e-4)
t0=time.time(); m.learn(total_timesteps=3_000_000, callback=CurriculumCallback()); print("粗训 3M 完成 %.0fs" % (time.time()-t0))

# 窄课程精修
CurriculumCallback.SCHEDULE=[(0,0.035,0.955,1),(800_000,0.030,0.960,2)]
m.learn(total_timesteps=2_000_000, callback=CurriculumCallback(), reset_num_timesteps=False)
m.save(os.path.join(SCENE,"saves","phase12_5m.zip"))
print("精修完成")
for d_,a_,s_ in ((0.06,0.90,0),(0.04,0.95,1),(0.03,0.96,2),(0.025,0.97,2)):
    ev=ArmEnv(); ev.max_steps=250
    ev._set_dist(d_); ev._set_angle1(a_); ev._set_still_count(s_)
    ok=0
    for ep in range(200):
        o,_=ev.reset()
        for t in range(ev.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=ev.step(a)
            if term or tr: break
        ok+=int(ev.phase>=3)
    print("  %.3f/%.2f/%d -> %3d/200 = %.1f%%" % (d_,a_,s_,ok,100*ok/200))
