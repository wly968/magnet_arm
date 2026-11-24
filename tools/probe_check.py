import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/saves/ckpt/ckpt_499992_steps.zip", device="cpu")
e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.20)
for det in (True, False):
    ok=0; ends={}; lens=[]; rews=[]
    for ep in range(20):
        obs,_=e.reset(); R=0.0
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=det)
            obs,r,term,tr,info=e.step(a); R+=float(r)
            if e.phase>=3 or term or tr:
                k="success" if e.phase>=3 else ("nan" if info.get("nan") else "timeout")
                ends[k]=ends.get(k,0)+1; lens.append(t+1); rews.append(R); break
        ok+=int(e.phase>=3)
    print("%s: %s | 平均原始步数 %.0f | 平均回报 %.1f" %
          ("确定性" if det else "随机", ends, np.mean(lens), np.mean(rews)))
