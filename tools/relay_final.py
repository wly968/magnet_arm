import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
env=t.ArmEnv()
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", env=env, device="cpu")
ok=0; hits=0; by={}; steps=[]
for ep in range(200):
    obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
    for i in range(250):
        if env.phase>=3 and last<3: t.apply_thresholds(env, env.phase)
        last=env.phase
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,tr,info=env.step(a)
        if env.cube_hit: hits+=1; by[env.phase]=by.get(env.phase,0)+1
        if term or tr: break
    ok+=int(env.phase>=6); steps.append(i+1)
print("200 局: 成功 %d/200 (%.1f%%), 碰撞步 %d (按阶段 %s), 平均 %.1f 步"
      % (ok, 100*ok/200, hits, by, np.mean(steps)))
