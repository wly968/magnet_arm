import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(r"scene_magnet/saves/phase1.zip", device="cpu")

def run(spawn, hold, n=30):
    e=ArmEnv()
    e.spawn_x_range=spawn; e.phase0_hold_tol=hold
    e._set_dist(0.06); e._set_angle1(0.90); e._set_still_count(0)
    ok=0
    for ep in range(n):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    return ok

OLD=(-1.160,-1.060); NEW=(-0.875,-0.775)
for name,sp,hd in (("旧生成点+无hold",OLD,0.0),("旧生成点+hold",OLD,0.06),
                   ("新生成点+无hold",NEW,0.0),("新生成点+hold",NEW,0.06)):
    print("%-18s -> %2d/30" % (name, run(sp,hd)))
