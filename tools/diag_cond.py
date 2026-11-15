import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
env=ArmEnv(); env.max_steps=250
env._set_dist(0.025); env._set_angle1(0.98); env._set_still_count(2)
cnt_d=cnt_a=cnt_both=0; max_still=0; tot=0; angle_at_close=[]
for ep in range(50):
    o,_=env.reset()
    for t in range(env.max_steps):
        a,_=m.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        if env.phase!=2: 
            if term or tr: break
            continue
        ee=env.data.site_xpos[env.ee_id]; tgt=env.box_pos+env.grasp_offset
        dist=float(np.linalg.norm(ee-tgt))
        j3=env.data.xanchor[env.arm_joints[3]]
        fwd=(ee-j3)/np.linalg.norm(ee-j3)
        ang=float(np.dot(fwd,[0,0,-1]))
        tot+=1
        cnt_d+=int(dist<0.025); cnt_a+=int(ang>0.98); cnt_both+=int(dist<0.025 and ang>0.98)
        max_still=max(max_still,env.still_count)
        if dist<0.025: angle_at_close.append(ang)
        if term or tr: break
print("phase2 总步数 %d" % tot)
print("满足 dist<0.025: %.1f%%" % (100*cnt_d/tot))
print("满足 angle1>0.98: %.1f%%" % (100*cnt_a/tot))
print("同时满足: %.1f%%   max still_count %d" % (100*cnt_both/tot, max_still))
if angle_at_close:
    a=np.array(angle_at_close)
    print("dist<0.025 时 angle1: min %.4f mean %.4f max %.4f, >0.98 的比例 %.1f%%"
          % (a.min(), a.mean(), a.max(), 100*(a>0.98).mean()))
