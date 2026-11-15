import os, sys, time, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
import torch; torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback

def run(steps, lr, tag):
    env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(24)])
    if lr is None:
        m=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env, device="cpu")
    else:
        m=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env, device="cpu", learning_rate=lr)
    t0=time.time(); m.learn(total_timesteps=steps, callback=CurriculumCallback()); dt=time.time()-t0
    res={}
    for d_,a_,s_ in ((0.06,0.90,0),(0.04,0.95,1),(0.03,0.96,2)):
        ev=ArmEnv(); ev.max_steps=250
        ev._set_dist(d_); ev._set_angle1(a_); ev._set_still_count(s_)
        ok=0
        for ep in range(100):
            o,_=ev.reset()
            for t in range(ev.max_steps):
                a,_=m.predict(o,deterministic=True)
                o,r,term,tr,info=ev.step(a)
                if term or tr: break
            ok+=int(ev.phase>=3)
        res["%.2f/%.2f/%d"%(d_,a_,s_)]=ok
    print("%s (steps=%d, lr=%s, %.0fs): %s" % (tag, steps, lr, dt, res))
    return m

run(1_500_000, None, "复现你的配置")
run(3_000_000, 1e-4, "修正: 3M + lr1e-4")
