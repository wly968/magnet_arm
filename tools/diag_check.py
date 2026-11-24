import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv, WaitAwareMonitor
m=PPO.load('scene_magnet/saves/_short120k.zip', device='cpu')
e=WaitAwareMonitor(ArmEnv())
e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
obs,_=e.reset()
print("rel_threshold =", e.unwrapped.rel_threshold)
vals=[]
for t in range(400):
    a,_=m.predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(a)
    if e.unwrapped.phase==2:
        vals.append(float(np.linalg.norm(e.unwrapped.ee_val - e.unwrapped.box_vel)))
    if e.unwrapped.phase>=3 or term or tr: break
print("手动统计 phase2 帧 %d, rel 中位 %.3f, rel<阈值 占比 %.1f%%" % (len(vals), np.median(vals), 100*np.mean(np.array(vals)<e.unwrapped.rel_threshold)))
print("env 自己的 diag:", {k:v for k,v in e.unwrapped._diag().items() if k!='sum_dist'})
