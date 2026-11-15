import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

models = {"arm5dof_10m.zip": os.path.join(SCENE,"saves","arm5dof_10m.zip"),
          "best_5dof/best_model.zip": os.path.join(SCENE,"saves","best_5dof","best_model.zip")}
for name,path in models.items():
    if not os.path.exists(path):
        print("missing", path); continue
    m=PPO.load(path, device="cpu")
    for tag,(d_,a_,s_) in {"终点(0.03/0.96/2)":(0.03,0.96,2),
                           "严格(0.025/0.98/2)":(0.025,0.98,2)}.items():
        env=ArmEnv(); env.max_steps=250
        env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
        ok=0; steps=[]; hit=0
        for ep in range(150):
            o,_=env.reset(); had=False
            for t in range(env.max_steps):
                a,_=m.predict(o,deterministic=True)
                o,r,term,tr,info=env.step(a)
                had = had or env.cube_hit
                if term or tr: break
            ok+=int(env.phase>=3); steps.append(t+1); hit+=int(had)
        print("%-26s %-18s -> phase2 %3d/150, 平均步数 %.1f, 碰撞局 %d"
              % (name, tag, ok, np.mean(steps), hit))
