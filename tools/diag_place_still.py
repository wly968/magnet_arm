import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m2=PPO.load(os.path.join(SCENE,"saves","phase345.zip"), device="cpu")
e=ArmEnv(terminate_on_phase2=False, max_phase=6); e.max_steps=300
e._set_dist(0.05); e._set_angle1(0.945); e._set_still_count(0)
vs=[]; d=[]
for ep in range(20):
    e.reset(); e.phase=5
    for t in range(120):
        o=e._get_obs(); a,_=m2.predict(o,deterministic=True)
        o,r,term,tr,info=e.step(a)
        if e.phase==5:
            vs.append(float(np.linalg.norm(e.data.qvel[e.arm_dof_adr])))
            d.append(float(np.linalg.norm(e.data.site_xpos[e.ee_id]-e._target())))
        if term or tr: break
v=np.array(vs); dd=np.array(d)
print("phase5: 样本 %d, dist 中位 %.4f, qvel 中位 %.3f (p25 %.3f p75 %.3f), qvel<0.15 占比 %.0f%%"
      % (len(v), np.median(dd), np.median(v), np.percentile(v,25), np.percentile(v,75), 100*(v<0.15).mean()))
