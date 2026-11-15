import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
for d_,a_,s_ in ((0.030,0.98,1),(0.035,0.96,1)):
    e=ArmEnv(); e.max_steps=250
    e._set_dist(d_); e._set_angle1(a_); e._set_still_count(s_)
    ok=0; steps=[]
    for ep in range(300):
        o,_=e.reset()
        for t in range(e.max_steps):
            act,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(act)
            if term or tr: break
        ok+=int(e.phase>=3); steps.append(t+1)
    print("新 ppo_arm_reach  %.3f/%.2f/%d -> %3d/300 = %5.1f%%, 平均 %.1f 步"%(d_,a_,s_,ok,100*ok/300,np.mean(steps)))
