import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
from envs.phase2 import Phase2Env
m1=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), device="cpu")
for name in ("saves/phase2_only.zip","ppo_arm_reach_prev.zip"):
    m=PPO.load(os.path.join(SCENE,name), device="cpu")
    e=ArmEnv(); e.max_steps=250
    e._set_dist(0.035); e._set_angle1(0.96); e._set_still_count(1)
    env=Phase2Env(e, m1)          # 保证从 phase 2 开始，公平比较
    ds=[]; angs=[]; hits=0; first_d=[]
    for ep in range(30):
        o,_=env.reset()
        for t in range(e.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(a)
            if e.phase==2:
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                ds.append(float(np.linalg.norm(ee-tgt)))
                j3=e.data.xanchor[e.arm_joints[3]]
                f=(ee-j3)/np.linalg.norm(ee-j3); angs.append(float(np.dot(f,[0,0,-1])))
            hits+=int(e.cube_hit)
            if term or tr: break
    print("%-28s 最小 dist %.4f, 中位 %.4f, angle 中位 %.3f, 碰撞步 %d"
          % (name, min(ds), np.median(ds), np.median(angs), hits))
