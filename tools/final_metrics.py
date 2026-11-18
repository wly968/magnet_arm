import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
env=t.ArmEnv()
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", env=env, device="cpu")
ok=0; hits=0; by={}; zs_all=[]; steps=[]
for ep in range(100):
    obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase; zs=[]
    for i in range(250):
        if env.phase>=3 and last<3: t.apply_thresholds(env, env.phase)
        last=env.phase
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,tr,info=env.step(a)
        if env.cube_hit: hits+=1; by[env.phase]=by.get(env.phase,0)+1
        if env.phase==3: zs.append(float(env.data.site_xpos[env.ee_id][2]))
        if term or tr: break
    ok+=int(env.phase>=6); steps.append(i+1); zs_all.append((min(zs), max(zs)) if zs else (np.nan,np.nan))
print("100 局: 成功 %d, 碰撞步 %d (按阶段 %s)" % (ok, hits, by))
print("抬升段: 起点 z %.3f, 终点 z %.3f, 平均步数 %.1f" %
      (np.nanmean([a for a,b in zs_all]), np.nanmean([b for a,b in zs_all]),
       np.mean([len([1 for _ in range(1)])])))
print("平均总步数 %.1f" % np.mean(steps))
