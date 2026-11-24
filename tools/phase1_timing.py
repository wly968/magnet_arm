import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/ppo_arm_reach.zip", device="cpu")
e=ArmEnv(); e._set_dist(0.032); e._set_angle1(0.985); e._set_still_count(1); e._set_rel(0.05)
durs=[]; ends={}
for ep in range(30):
    obs,_=e.reset(); t1=None
    for t in range(e.max_steps+e.max_wait_steps+10):
        act,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase==2 and t1 is None: t1=e.steps
        if e.phase>=3 or term or tr:
            k="success" if e.phase>=3 else ("phase1超时" if e.phase==1 else "phase2超时")
            ends[k]=ends.get(k,0)+1; break
    if t1: durs.append(t1)
print("结束原因:", ends)
if durs: print("进入 phase2 的回合里，phase1 用了 %d~%d 步（中位 %d），max_steps=%d"
               % (min(durs), max(durs), int(np.median(durs)), e.max_steps))
