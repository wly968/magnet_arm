import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","arm5dof_10m.zip"), device="cpu")
for d_,a_,s_ in ((0.06,0.90,0),(0.05,0.93,0),(0.04,0.95,1),(0.03,0.96,2)):
    env=ArmEnv(); env.max_steps=250
    env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
    ok=0; best=9; steps=[]
    for ep in range(100):
        o,_=env.reset()
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            if env.phase==2:
                ee=env.data.site_xpos[env.ee_id]; tgt=env.box_pos+env.grasp_offset
                best=min(best,float(np.linalg.norm(ee-tgt)))
            if term or tr: break
        ok+=int(env.phase>=3); steps.append(t+1)
    print("阈值 %.3f/%.2f/%d -> 完成 %3d/100, best_dist %.4f, 平均步数 %.1f" % (d_,a_,s_,ok,best,np.mean(steps)))
