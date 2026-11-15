import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
models={"旧 ppo_arm_reach":os.path.join(SCENE,"ppo_arm_reach.zip"),
        "微调 arm4dof_ft":os.path.join(SCENE,"saves","arm4dof_ft.zip"),
        "微调 best":os.path.join(SCENE,"saves","best_4dof_ft","best_model.zip")}
ladder=[(0.06,0.90,0),(0.04,0.95,1),(0.03,0.96,2)]
for name,path in models.items():
    if not os.path.exists(path): print("missing",path); continue
    m=PPO.load(path, device="cpu")
    print(name)
    for d_,a_,s_ in ladder:
        env=ArmEnv(); env.max_steps=250
        env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
        ok=0; steps=[]; hit=0
        for ep in range(300):
            o,_=env.reset(); had=False
            for t in range(env.max_steps):
                a,_=m.predict(o,deterministic=True)
                o,r,term,tr,info=env.step(a)
                had = had or env.cube_hit
                if term or tr: break
            ok+=int(env.phase>=3); steps.append(t+1); hit+=int(had)
        print("   %.3f/%.2f/still%d -> %3d/300 = %5.1f%%, 平均 %.1f 步, 碰撞局 %d"
              % (d_,a_,s_,ok,100*ok/300,np.mean(steps),hit))
