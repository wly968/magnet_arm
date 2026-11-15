import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback
env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(24)])
m=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env, device="cpu", learning_rate=1e-4)
t0=time.time(); m.learn(total_timesteps=1_500_000, callback=CurriculumCallback()); print("从 phase1 训练 %.0fs"%(time.time()-t0))
m.save(os.path.join(SCENE,"saves","phase12_mocap.zip"))
e=ArmEnv(); e.max_steps=250
e._set_dist(0.03); e._set_angle1(0.98); e._set_still_count(1)
ok=0
for ep in range(150):
    o,_=e.reset()
    for t in range(e.max_steps):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=e.step(a)
        if term or tr: break
    ok+=int(e.phase>=3)
print("mocap 场景下从 phase1 训练: 0.03/0.98 -> %d/150" % ok)
