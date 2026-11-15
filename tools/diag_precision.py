import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("saves/phase12_5m.zip","ppo_arm_reach_prev.zip","ppo_arm_reach.zip"):
    p=os.path.join(SCENE,name)
    if not os.path.exists(p): continue
    m=PPO.load(p, device="cpu")
    ev=ArmEnv(); ev.max_steps=250
    ev._set_dist(0.03); ev._set_angle1(0.96); ev._set_still_count(2)
    best_d=9; best_combo=9; ang_when_close=[]; still_max=0; ok=0
    for ep in range(100):
        o,_=ev.reset()
        for t in range(ev.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=ev.step(a)
            if ev.phase==2:
                ee=ev.data.site_xpos[ev.ee_id]; tgt=ev.box_pos+ev.grasp_offset
                d=float(np.linalg.norm(ee-tgt))
                j3=ev.data.xanchor[ev.arm_joints[3]]
                f=(ee-j3)/np.linalg.norm(ee-j3); ang=float(np.dot(f,[0,0,-1]))
                best_d=min(best_d,d); still_max=max(still_max,ev.still_count)
                if d<0.035: ang_when_close.append(ang)
            if term or tr: break
        ok+=int(ev.phase>=3)
    a=np.array(ang_when_close) if ang_when_close else np.array([np.nan])
    print("%-28s ok %3d/100, best_dist %.4f, dist<0.035 时 angle 中位数 %.3f, max still %d"
          % (name, ok, best_d, np.nanmedian(a), still_max))
