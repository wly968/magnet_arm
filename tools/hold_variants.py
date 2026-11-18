import sys, numpy as np
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv

def run(mode):
    e=ArmEnv(); e.reset()
    e.phase0_hold_tol = 0.06 if mode=="hold_cur" else 0.0
    q0=e.data.qpos[e.arm_qpos_adr].copy()
    a_fix=(e.home_qpos - e.home_ctrl)/e.ctrl_scale
    zs=[]
    for t in range(200):
        a = a_fix if mode=="hold_fixed" else np.zeros(4)
        e.step(a)
        zs.append(float(e.data.qpos[e.arm_qpos_adr][1]))
    zs=np.array(zs)
    return zs[-1]-q0[1], float(np.std(zs[-100:]))

for mode,name in (("hold_cur","当前实现(跟随当前姿态)"),("hold_fixed","固定 home 目标"),("zero","零动作")):
    d,j = run(mode)
    print("%-22s 肩关节漂移 %+.4f rad, 后100步抖动 %.4f rad" % (name, d, j))
