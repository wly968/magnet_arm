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
t0=time.time(); m.learn(total_timesteps=3_000_000, callback=CurriculumCallback()); print("3M 训练 %.0fs"%(time.time()-t0))
m.save(os.path.join(SCENE,"saves","phase12_004_096.zip"))
for d_,a_,s_ in ((0.04,0.96,1),(0.04,0.97,1),(0.035,0.96,1)):
    e=ArmEnv(); e.max_steps=250
    e._set_dist(d_); e._set_angle1(a_); e._set_still_count(s_)
    ok=0
    for ep in range(150):
        o,_=e.reset()
        for t in range(e.max_steps):
            act,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(act)
            if term or tr: break
        ok+=int(e.phase>=3)
    print("  %.3f/%.2f/%d -> %3d/150 = %5.1f%%"%(d_,a_,s_,ok,100*ok/150))
