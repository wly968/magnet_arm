import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("ppo_arm_reach.zip","saves/ppo_arm_reach_ft.zip"):
    m=PPO.load(os.path.join(SCENE,name), device="cpu")
    env=ArmEnv(); env.max_steps=250
    env._set_dist(0.03); env._set_angle1(0.96); env._set_still_count(2)
    ok=0; hit=0; steps=[]
    for ep in range(200):
        o,_=env.reset(); had=False
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            had = had or env.cube_hit
            if term or tr: break
        ok+=int(env.phase>=3); steps.append(t+1); hit+=int(had)
    print("%-30s -> 完成 %3d/200  平均步数 %.1f  有碰撞局 %d" % (name, ok, np.mean(steps), hit))
