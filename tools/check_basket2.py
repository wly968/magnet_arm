import sys, numpy as np
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(terminate_on_phase1=False, terminate_on_phase2=False); e.reset()
# 模拟“已经放到放料点”：把方块摆到 drop_pos，然后释放
e.data.mocap_pos[e.mocap_id] = [0.65, -0.45, 0.375]
e.phase=6; e.attached=False; e.released=True; e.collected=False
ys=[]
for t in range(300):
    o,r,term,tr,info=e.step(np.zeros(4))
    ys.append(float(e.data.mocap_pos[e.mocap_id][1]))
    if e.collected: break
print('从放料点 y=-0.45 出发: 用时 %d 步 -> y=%.3f, collected=%s, 最终 %s'
      % (t+1, ys[-1], e.collected, np.round(e.data.mocap_pos[e.mocap_id],3)))
