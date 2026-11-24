import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')
print("测试 10M 模型在各 (still, rel) 档位的成功率：")
for rel in (0.2, 0.1, 0.05):
    row=[]
    for still in (0,1,2):
        e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(still); e._set_rel(rel)
        ok=0
        for ep in range(20):
            obs,_=e.reset()
            for t in range(e.max_steps+e.max_wait_steps+10):
                a,_=m.predict(obs,deterministic=True)
                obs,r,term,tr,info=e.step(a)
                if e.phase>=3 or term or tr: break
            ok+=int(e.phase>=3)
        row.append(ok)
    print("  rel<%.2f: still=0 %2d/20 | still=1 %2d/20 | still=2 %2d/20" % (rel,row[0],row[1],row[2]))
