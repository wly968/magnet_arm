import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for p in ('scene_magnet/ppo_arm_reach.zip','scene_magnet/saves/phase2.zip'):
    if not os.path.exists(p): continue
    m=PPO.load(p, device='cpu')
    e=ArmEnv(); e._set_dist(0.045); e._set_angle1(0.95); e._set_still_count(0); e._set_rel(0.10)
    ok=0; ends={}
    for ep in range(30):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase>=3 or term or tr:
                k = "success" if e.phase>=3 else ("nan" if info.get("nan") else "timeout")
                ends[k]=ends.get(k,0)+1
                break
        ok+=int(e.phase>=3)
    print("%-34s 步数 %8s | 成功 %2d/30 | 结束原因 %s" % (os.path.basename(p), getattr(m,'num_timesteps',0), ok, ends))
