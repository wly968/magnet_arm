import sys, numpy as np
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(terminate_on_phase1=False, terminate_on_phase2=False); e.reset()
# 直接模拟“已放料”：释放到 B 带
e.phase=6; e.attached=False; e.released=True; e.collected=False
ys=[]
for t in range(200):
    o,r,term,tr,info=e.step(np.zeros(4))
    ys.append(float(e.data.mocap_pos[e.mocap_id][1]))
    if e.collected: break
print('B带输送: y %.3f -> %.3f, 用时 %d 步, collected=%s, 最终位置 %s'
      % (ys[0], ys[-1], t+1, e.collected, np.round(e.data.mocap_pos[e.mocap_id],3)))
