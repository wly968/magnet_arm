import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
p=os.path.join(SCENE,"saves","phase34.zip")
if not os.path.exists(p):
    print("phase34.zip 不存在"); raise SystemExit
m=PPO.load(p, device="cpu")
print("phase34.zip obs", m.observation_space.shape, "act", m.action_space.shape)
# 直接从 phase 3 起跑，看能否走到 phase 4/5
e=ArmEnv(); e.max_steps=250
e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0)
ph={}; steps=[]
for ep in range(60):
    e.reset(); e.phase=3
    for t in range(e.max_steps):
        o=e._get_obs()
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=e.step(a)
        if term or tr: break
    ph[e.phase]=ph.get(e.phase,0)+1; steps.append(t+1)
print("从 phase3 起跑 60 局:", ph, "平均步数 %.1f"%np.mean(steps))
