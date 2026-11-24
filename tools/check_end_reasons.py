import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')
for det in (True, False):
    e=ArmEnv(); e._set_dist(0.045); e._set_angle1(0.95); e._set_still_count(0); e._set_rel(0.10)
    ends={}; lens=[]
    for ep in range(20):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=det)
            obs,r,term,tr,info=e.step(a)
            if e.phase>=3 or term or tr:
                k="success" if e.phase>=3 else ("nan" if info.get("nan") else "timeout")
                ends[k]=ends.get(k,0)+1; lens.append(t+1); break
    print("%s: %s | 平均原始步数 %.0f | 平均有效步数 %.0f" %
          ("确定性" if det else "随机(训练)", ends, np.mean(lens), np.mean(lens)-87))
