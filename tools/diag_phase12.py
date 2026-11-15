import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

for name in ("saves/phase1.zip","ppo_arm_reach.zip"):
    path=os.path.join(SCENE,name)
    m=PPO.load(path, device="cpu")
    print("%s  obs %s act %s" % (name, m.observation_space.shape, m.action_space.shape))
    for d_,a_,s_ in ((0.06,0.90,0),(0.03,0.96,2)):
        env=ArmEnv(); env.max_steps=250
        env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
        ok1=ok2=0
        for ep in range(200):
            o,_=env.reset()
            for t in range(env.max_steps):
                a,_=m.predict(o,deterministic=True)
                o,r,term,tr,info=env.step(a)
                if term or tr: break
            ok1+=int(env.phase>=2); ok2+=int(env.phase>=3)
        print("   阈值 %.2f/%.2f/%d -> phase1完成 %3d/200, phase2完成 %3d/200" % (d_,a_,s_,ok1,ok2))
