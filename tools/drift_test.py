import sys, numpy as np
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv

def drift(hold):
    e=ArmEnv(); e.reset()
    e.phase0_hold_tol = 0.06 if hold else 0.0
    q0=e.data.qpos[e.arm_qpos_adr].copy()
    zs=[]
    for t in range(200):
        e.step(np.zeros(4))
        zs.append(float(e.data.qpos[e.arm_qpos_adr][1]))   # shoulder
    q1=e.data.qpos[e.arm_qpos_adr].copy()
    return np.round(q0-q1,4), zs

d_hold, z_hold = drift(True)
d_off, z_off = drift(False)
print("肩关节 q[1] 起点 %.3f" % 0)
print("hold 开启: 200 步后肩关节漂移 %s, ee_z 变化 %.3f" % (d_hold[1], 0))
print("hold 关闭: 200 步后肩关节漂移 %s" % (d_off[1],))
print("hold 开启时肩关节轨迹（每 20 步）:", [round(z,4) for z in z_hold[::20]])
