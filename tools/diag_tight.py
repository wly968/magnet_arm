import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("saves/phase12_tight.zip","ppo_arm_reach.zip"):
    m=PPO.load(os.path.join(SCENE,name), device="cpu")
    e=ArmEnv(); e.max_steps=250
    e._set_dist(0.06); e._set_angle1(0.90); e._set_still_count(0)
    ok1=0; best=9; ang=[]
    for ep in range(60):
        o,_=e.reset()
        for t in range(e.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(a)
            if e.phase==2:
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                d=float(np.linalg.norm(ee-tgt)); best=min(best,d)
                j3=e.data.xanchor[e.arm_joints[3]]
                f=(ee-j3)/np.linalg.norm(ee-j3); ang.append(float(np.dot(f,[0,0,-1])))
            if term or tr: break
        ok1+=int(e.phase>=2)
    print("%-28s phase1 %2d/60, phase2 内 best_dist %.4f, angle 中位 %.3f" %
          (name, ok1, best, np.median(ang) if ang else -1))
