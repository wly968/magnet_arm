import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"saves","phase12_legacy.zip"), device="cpu")
for d_,a_,s_ in ((0.035,0.96,1),(0.030,0.96,1),(0.035,0.97,1),(0.030,0.98,1)):
    e=ArmEnv(); e.max_steps=250
    e._set_dist(d_); e._set_angle1(a_); e._set_still_count(s_)
    ok=0; steps=[]; hit=0
    for ep in range(300):
        o,_=e.reset(); had=False
        for t in range(e.max_steps):
            act,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(act)
            had = had or e.cube_hit
            if term or tr: break
        ok+=int(e.phase>=3); steps.append(t+1); hit+=int(had)
    print("%.3f/%.2f/%d -> %3d/300 = %5.1f%%, 平均 %.1f 步, 碰撞局 %d"
          %(d_,a_,s_,ok,100*ok/300,np.mean(steps),hit))
