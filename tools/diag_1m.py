import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), device="cpu")

for name,(d_,a_,s_) in {
  "500k档 (0.06/0.90/0)":(0.06,0.90,0),
  "1M档   (0.025/0.98/2)":(0.025,0.98,2),
}.items():
    env=ArmEnv(); env.max_steps=250
    env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
    ok=0; still_max=0; best_dist=9; best_angle=0; steps=[]
    for ep in range(200):
        o,_=env.reset()
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            if env.phase==2:
                ee=env.data.site_xpos[env.ee_id]; tgt=env.box_pos+env.grasp_offset
                best_dist=min(best_dist,float(np.linalg.norm(ee-tgt)))
                j3=env.data.xanchor[env.arm_joints[3]]
                fwd=(ee-j3)/np.linalg.norm(ee-j3)
                best_angle=max(best_angle,float(np.dot(fwd,[0,0,-1])))
                still_max=max(still_max,env.still_count)
            if term or tr: break
        ok += int(env.phase>=3); steps.append(t+1)
    print("%s -> phase2完成 %d/200  max still %d  best dist %.4f  best angle %.4f  mean_steps %.1f"
          % (name, ok, still_max, best_dist, best_angle, np.mean(steps)))
