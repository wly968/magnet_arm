import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
print("模型 ppo_arm_reach（96.7%）在不同 angle1 下的表现：")
for d_, a_, s_ in ((0.04,0.95,1),(0.04,0.96,1),(0.04,0.97,1),(0.04,0.98,1),
                   (0.035,0.96,1),(0.03,0.96,1)):
    e=ArmEnv(); e.max_steps=250
    e._set_dist(d_); e._set_angle1(a_); e._set_still_count(s_)
    ok=0; steps=[]; best=9
    for ep in range(200):
        o,_=e.reset()
        for t in range(e.max_steps):
            act,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(act)
            if e.phase==2:
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                best=min(best,float(np.linalg.norm(ee-tgt)))
            if term or tr: break
        ok+=int(e.phase>=3); steps.append(t+1)
    print("  %.3f/%.2f/still%d -> %3d/200 = %5.1f%%, 平均 %.1f 步, best_dist %.4f"
          % (d_,a_,s_,ok,100*ok/200,np.mean(steps),best))
