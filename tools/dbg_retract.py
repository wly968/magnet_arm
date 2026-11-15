import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
from envs.phase34 import Phase34Env
m1=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
m2=PPO.load(os.path.join(SCENE,"saves","phase34_retract.zip"), device="cpu")
e=ArmEnv(terminate_on_phase2=False); e.max_steps=250
e._set_dist(0.05); e._set_angle1(0.945); e._set_still_count(0)
env=Phase34Env(e, m1)
o,_=env.reset()
print("交接后 phase", e.phase, "ee", np.round(e.data.site_xpos[e.ee_id],3), "target", np.round(e._target(),3))
for t in range(40):
    a,_=m2.predict(o,deterministic=True)
    o,r,term,tr,info=e.step(a)
    if t%8==0:
        print("t%2d phase=%d dist=%.3f ee=%s act=%s" % (t, e.phase,
              float(np.linalg.norm(e.data.site_xpos[e.ee_id]-e._target())),
              np.round(e.data.site_xpos[e.ee_id],2), np.round(a,2)))
    if term or tr: break
