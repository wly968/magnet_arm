import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
rng=np.random.default_rng(1); lo,hi=e.jnt_lo,e.jnt_hi
def best_at(tgt, require_down=False):
    best=9.0
    for _ in range(6000):
        q=rng.uniform(lo,hi)
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
        ee=e.data.site_xpos[e.ee_id]
        if require_down:
            j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            if float(np.dot(f,[0,0,-1])) < 0.945: continue
        d=float(np.linalg.norm(ee-tgt))
        best=min(best,d)
    return best
for z in (0.85,0.80,0.75,0.70,0.65):
    tgt=np.array([0.42,0.27,z])
    print("transfer z=%.2f | 任意姿态最近 %.3f | 工具朝下最近 %.3f"
          % (z, best_at(tgt), best_at(tgt, True)))
print("放料点 0.65,-0.45,0.435 | 工具朝下最近 %.3f" % best_at(np.array([0.65,-0.45,0.435]), True))
