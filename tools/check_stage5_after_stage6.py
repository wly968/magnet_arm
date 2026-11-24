import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("ckpt_2999952_steps.zip","ckpt_3499944_steps.zip"):
    p=os.path.join("scene_magnet/saves/ckpt",name)
    if not os.path.exists(p): continue
    m=PPO.load(p, device='cpu')
    e=ArmEnv(); e._set_dist(0.034); e._set_angle1(0.98); e._set_still_count(2); e._set_rel(0.05)
    ok=0
    for ep in range(20):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    print("%-24s 第5档(0.034/0.98/still2) 成功 %2d/20" % (name, ok))
