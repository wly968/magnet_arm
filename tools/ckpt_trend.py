import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
files=[("scene_magnet/saves/ckpt/ckpt_499992_steps.zip","50万"),
       ("scene_magnet/saves/ckpt/ckpt_999984_steps.zip","100万"),
       ("scene_magnet/saves/ckpt/ckpt_1499976_steps.zip","150万"),
       ("scene_magnet/saves/ckpt/ckpt_1999968_steps.zip","200万"),
       ("scene_magnet/ppo_arm_reach.zip","最终")]
e=ArmEnv()
for p,tag in files:
    if not os.path.exists(p): print(tag,"缺"); continue
    m=PPO.load(p, device="cpu")
    if int(m.observation_space.shape[0])!=23: print(tag,"维度不符"); continue
    ok=0
    for (d,a,s,rv) in [(0.032,0.985,1,0.05)]:
        e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(rv)
        for ep in range(20):
            obs,_=e.reset()
            for t in range(e.max_steps+e.max_wait_steps+10):
                act,_=m.predict(obs,deterministic=True)
                obs,r,term,tr,info=e.step(act)
                if e.phase>=3 or term or tr: break
            ok+=int(e.phase>=3)
    print("%-6s 第6档 0.032/0.985/still1 -> %2d/20 (%3.0f%%)" % (tag, ok, 100*ok/20))
