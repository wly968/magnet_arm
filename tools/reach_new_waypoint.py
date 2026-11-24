import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); lo,hi=e.jnt_lo,e.jnt_hi
def reach(tgt, min_angle=0.6, starts=40, iters=60):
    rng=np.random.default_rng(9); best=9.0
    for _ in range(starts):
        q=rng.uniform(lo,hi)
        for _ in range(iters):
            e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
            err=tgt-e.data.site_xpos[e.ee_id]
            jp=np.zeros((3,e.model.nv)); mujoco.mj_jacSite(e.model,e.data,jp,None,e.ee_id)
            dq=jp[:,e.arm_dof_adr].T@err; n=float(np.linalg.norm(dq))
            if n>1e-9: dq*=min(1.0,0.25/n)
            q=np.clip(q+0.9*dq,lo,hi)
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
        ee=e.data.site_xpos[e.ee_id]
        j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        if float(np.dot(f,[0,0,-1]))<min_angle: continue
        best=min(best,float(np.linalg.norm(ee-tgt)))
    return best
for name,t in (("旧 (0.42,0.27,0.78)",np.array([0.42,0.27,0.78])),
               ("新 (0.25,0.25,0.78)",np.array([0.25,0.25,0.78])),
               ("新 (0.25,0.25,0.72)",np.array([0.25,0.25,0.72])),
               ("新 (0.10,0.35,0.78)",np.array([0.10,0.35,0.78]))):
    d=reach(t)
    print("%-22s 最近可达 dist %s" % (name, ("%.4f"%d) if d<9 else "无解"))
