import os, sys, time, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
import torch; torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback

def evaluate(m, tag, n=150):
    e=ArmEnv(); e.max_steps=250
    e._set_dist(0.04); e._set_angle1(0.95); e._set_still_count(1)
    ok=0; steps=[]
    for ep in range(n):
        o,_=e.reset()
        for t in range(e.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(a)
            if term or tr: break
        ok+=int(e.phase>=3); steps.append(t+1)
    print("%s -> 0.04/0.95/1: %3d/%d = %5.1f%%, 平均 %.1f 步" % (tag, ok, n, 100*ok/n, np.mean(steps)))

m0=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
evaluate(m0, "当前 ppo_arm_reach")

env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(24)])
m=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env, device="cpu", learning_rate=1e-4)
t0=time.time(); m.learn(total_timesteps=2_000_000, callback=CurriculumCallback()); print("新课程 2M 训练 %.0fs" % (time.time()-t0))
m.save(os.path.join(SCENE,"saves","phase12_004.zip"))
evaluate(m, "新课程 2M 模型")
