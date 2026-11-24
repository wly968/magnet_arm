import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
e.phase=2
man=[]
for t in range(30):
    o,r,term,tr,info=e.step(np.zeros(4))
    man.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
d=e._diag()
print("rel_threshold =", e.rel_threshold)
print("diag: n=%d rel=%d  -> %.1f%%" % (d["n"], d["rel"], 100*d["rel"]/max(1,d["n"])))
print("手动: 中位 %.3f, <阈值占比 %.1f%%" % (np.median(man), 100*np.mean(np.array(man)<e.rel_threshold)))
print("ee_val 样例:", np.round(e.ee_val,3), " box_vel:", np.round(e.box_vel,3))
