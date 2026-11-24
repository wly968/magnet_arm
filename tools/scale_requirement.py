import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
m,d=e.model,e.data
qadr,dadr=e.arm_qpos_adr,e.arm_dof_adr
lo,hi=m.jnt_range[e.arm_joints].T
jp=np.zeros((3,m.nv)); rng=np.random.default_rng(0)

def ik(target, q0, iters=600, lr=0.6):
    q=q0.copy()
    for _ in range(iters):
        d.qpos[qadr]=q; mujoco.mj_forward(m,d)
        ee=d.site_xpos[e.ee_id]; err=target-ee
        if np.linalg.norm(err)<1e-3: break
        jp[:]=0; mujoco.mj_jacSite(m,d,jp,None,e.ee_id)
        q=np.clip(q+lr*(jp[:,dadr].T@err), lo, hi)
    d.qpos[qadr]=q; mujoco.mj_forward(m,d)
    return q, float(np.linalg.norm(d.site_xpos[e.ee_id]-target))

# 关键点：取料区（3 个 y × 2 个高度）、中间姿态、放料点、home
targets = {
 "取料(y=0.45,z=0.435)": [-0.475,0.45,0.435],
 "取料(y=0.65,z=0.435)": [-0.475,0.65,0.435],
 "取料(y=0.85,z=0.435)": [-0.475,0.85,0.435],
 "取料(y=0.65,z=0.525)": [-0.475,0.65,0.525],
 "中间姿态":             [ 0.42, 0.27,0.85],
 "放料点":               [ 0.65,-0.45,0.435],
 "初始末端":             e.home_ee.tolist(),
}
print("关键点的 IK 解（多起点取最接近 home 的解）:")
sols={}
for name,t in targets.items():
    t=np.array(t,dtype=float); best=None
    for _ in range(60):
        q,err=ik(t, rng.uniform(lo,hi))
        if err<0.02:
            if best is None or np.linalg.norm(q-e.home_qpos)<np.linalg.norm(best-e.home_qpos):
                best=q
    if best is None:
        print("  %-22s 无解 (err>2cm)" % name); continue
    sols[name]=best
    print("  %-22s q = %s  |Δhome|=%s" % (name, np.round(best,2), np.round(best-e.home_qpos,2)))
D=np.array([sols[k]-e.home_qpos for k in sols])
need=np.max(np.abs(D),axis=0)
print("\n各关节所需的最小 ctrl_scale（= max|Δq|）:", np.round(need,2))
print("当前 ctrl_scale:", e.ctrl_scale)
