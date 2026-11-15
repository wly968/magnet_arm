import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","ppo_arm_reach_smooth.zip"), device="cpu")
for name,(d_,a_,s_) in {"1M档(旧终点 0.025/0.98/2)":(0.025,0.98,2),
                        "新终点(0.03/0.96/2)":(0.03,0.96,2)}.items():
    env=ArmEnv(); env.max_steps=250
    env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
    ok=0; steps=[]; hit=0
    for ep in range(200):
        o,_=env.reset(); had=False
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            had = had or env.cube_hit
            if term or tr: break
        ok+=int(env.phase>=3); steps.append(t+1); hit+=int(had)
    print("%s -> phase2 %d/200, 平均步数 %.1f, 有碰撞局 %d" % (name, ok, np.mean(steps), hit))
