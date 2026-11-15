import os, sys, time, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

models = {
 "旧 ppo_arm_reach": os.path.join(SCENE,"ppo_arm_reach.zip"),
 "新 arm4dof_10m": os.path.join(SCENE,"saves","arm4dof_10m.zip"),
 "best_4dof": os.path.join(SCENE,"saves","best_4dof","best_model.zip"),
}
ladder = [(0.06,0.90,0),(0.04,0.95,1),(0.03,0.96,2),(0.025,0.98,2)]
for name,path in models.items():
    if not os.path.exists(path): print("missing",path); continue
    m=PPO.load(path, device="cpu")
    row=[]
    for d_,a_,s_ in ladder:
        env=ArmEnv(); env.max_steps=250
        env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
        ok=0; hit=0; steps=[]
        for ep in range(200):
            o,_=env.reset(); had=False
            for t in range(env.max_steps):
                a,_=m.predict(o,deterministic=True)
                o,r,term,tr,info=env.step(a)
                had = had or env.cube_hit
                if term or tr: break
            ok+=int(env.phase>=3); steps.append(t+1); hit+=int(had)
        row.append((ok,len(steps),np.mean(steps)))
    print(name)
    for (d_,a_,s_),(ok,n,ms) in zip(ladder,row):
        print("   %.3f/%.2f/still%d -> %3d/%d = %5.1f%%, 平均 %.1f 步" % (d_,a_,s_,ok,n,100*ok/n,ms))
