import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
tgt=np.array([0.42,0.27,0.85]); best=(9,None)
rng=np.random.default_rng(0)
lo,hi=e.jnt_lo,e.jnt_hi
for _ in range(8000):
    q=rng.uniform(lo,hi)
    e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
    d=float(np.linalg.norm(e.data.site_xpos[e.ee_id]-tgt))
    if d<best[0]: best=(d,q.copy())
print("随机搜索最接近 phase4 目标: dist %.4f" % best[0])
print("  关节角", np.round(best[1],2))
e.data.qpos[e.arm_qpos_adr]=best[1]; mujoco.mj_forward(e.model,e.data)
ee=e.data.site_xpos[e.ee_id]
j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
print("  ee", np.round(ee,3), "| angle %.3f" % float(np.dot(f,[0,0,-1])))
# 对照：phase5 放料点
for name,t in (("抓取点",None),("phase5 放料",np.array([0.65,-0.45,0.435])),("home",e.home_qpos)):
    pass
for name,t in (("phase5 放料",np.array([0.65,-0.45,0.435])),("phase1 抓取上方",np.array([-0.475,0.65,0.525]))):
    best=(9,None)
    for _ in range(4000):
        q=rng.uniform(lo,hi)
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
        d=float(np.linalg.norm(e.data.site_xpos[e.ee_id]-t))
        if d<best[0]: best=(d,q.copy())
    print("%-12s 最近距离 %.4f" % (name, best[0]))
