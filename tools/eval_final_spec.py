import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/ppo_arm_reach.zip", device="cpu")
e=ArmEnv(); e._set_dist(0.032); e._set_angle1(0.98); e._set_still_count(1); e._set_rel(0.05)
ok=0; p1=0
for ep in range(60):
    obs,_=e.reset()
    for t in range(e.max_steps+e.max_wait_steps+10):
        act,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase>=3 or term or tr: break
    ok+=int(e.phase>=3); p1+=int(e.phase>=2)
print("新终点档 0.032/0.98/still1 : 进phase2 %d/60 | 成功 %d/60 = %.0f%%" % (p1, ok, 100*ok/60))
