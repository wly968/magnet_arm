import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
env=t.ArmEnv(terminate_on_phase2=False)
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", env=env, device="cpu")
hits=0; min_z=[]; lift_steps=[]
for ep in range(100):
    obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
    z_grasp=None; zs=[]
    for i in range(250):
        if env.phase>=3 and last<3: t.apply_thresholds(env, env.phase)
        last=env.phase
        pol=m1 if env.phase<3 else m2
        a,_=pol.predict(obs,deterministic=True)
        obs,r,term,tr,info=env.step(a)
        if env.phase==2: z_grasp=float(env.data.site_xpos[env.ee_id][2])
        if env.phase==3: zs.append(float(env.data.site_xpos[env.ee_id][2]))
        if env.cube_hit: hits+=1
        if term or tr: break
    min_z.append(min(zs) if zs else np.nan)
    lift_steps.append(len(zs))
print("100 局: 碰撞步 %d, 抬升段平均步数 %.1f, 抬升起点最低 z %.3f, 终点最高 z %.3f"
      % (hits, np.nanmean(lift_steps), np.nanmin(min_z), np.nanmax(min_z)))
