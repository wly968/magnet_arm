import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), device="cpu")
env=ArmEnv(); env.max_steps=250
env._set_dist(0.06); env._set_angle1(0.90); env._set_still_count(0)
hit_steps=0; total=0; eps_with_hit=0; ok=0
for ep in range(100):
    o,_=env.reset(); had=False
    for t in range(env.max_steps):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        total+=1
        if env.cube_hit: hit_steps+=1; had=True
        if term or tr: break
    ok+=int(env.phase>=3); eps_with_hit+=int(had)
print("100 局: phase2完成 %d, 有碰撞的局 %d, 碰撞步占比 %.2f%%"
      % (ok, eps_with_hit, 100.0*hit_steps/max(total,1)))
