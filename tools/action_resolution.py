import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
m,d=e.model,e.data
qadr,dadr=e.arm_qpos_adr,e.arm_dof_adr
# 用一个典型抓取姿态
q=np.array([1.88,-0.879,1.973,1.327])
d.qpos[qadr]=q; mujoco.mj_forward(m,d)
jp=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,None,e.ee_id)
J=jp[:,dadr]
print("控制步 dt = 0.01 s，各关节的单位动作对应的末端位移：")
for i,name in enumerate(("yaw","shoulder","elbow","wrist")):
    mm=float(np.linalg.norm(J[:,i])*e.ctrl_scale[i])
    print("  %-8s ctrl_scale=%.1f -> 每单位动作末端移动 %.2f m" % (name, e.ctrl_scale[i], mm))
print()
print("要以 0.2 m/s 跟随，每步（10 ms）末端只需移动 2 mm")
print("=> yaw 的 2 mm 对应 action 变化量：%.5f" % (0.002/ (np.linalg.norm(J[:,0])*e.ctrl_scale[0])))
print("策略动作标准差约 0.08，比这个精度大 %.0f 倍" % (0.08/(0.002/(np.linalg.norm(J[:,0])*e.ctrl_scale[0]))))
