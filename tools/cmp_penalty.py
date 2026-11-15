import os, sys, time, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
import torch; torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback

def make_env(pen):
    def _f():
        e=ArmEnv(); e.collision_penalty_value=pen; return Monitor(e)
    return _f

def try_penalty(pen, steps=400_000):
    env=DummyVecEnv([make_env(pen) for _ in range(24)])
    m=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env, device="cpu")
    t0=time.time()
    m.learn(total_steps=steps, callback=CurriculumCallback()) if False else m.learn(total_timesteps=steps, callback=CurriculumCallback())
    dt=time.time()-t0
    ev=ArmEnv(); ev.max_steps=250
    ev._set_dist(0.06); ev._set_angle1(0.90); ev._set_still_count(0)
    ok=0; hit=0
    for ep in range(100):
        o,_=ev.reset(); had=False
        for t in range(ev.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=ev.step(a)
            had = had or ev.cube_hit
            if term or tr: break
        ok+=int(ev.phase>=3); hit+=int(had)
    return ok, hit, dt

for pen in (10.0, 0.0):
    ok,hit,dt = try_penalty(pen)
    print("惩罚 %.0f -> 400k 训练后 phase2 完成 %3d/100, 碰撞局 %d, 用时 %.0fs" % (pen, ok, hit, dt))
