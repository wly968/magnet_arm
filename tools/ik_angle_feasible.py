import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv

e=ArmEnv(); e.reset()
m,d=e.model,e.data
lo,hi=m.jnt_range[e.arm_joints].T
qadr=e.arm_qpos_adr; dadr=e.arm_dof_adr

def fk(q):
    d.qpos[qadr]=q; mujoco.mj_forward(m,d)
    ee=d.site_xpos[e.ee_id].copy()
    j3=d.xanchor[e.arm_joints[3]]
    f=(ee-j3)/np.linalg.norm(ee-j3)
    return ee, float(np.dot(f,[0,0,-1]))

def ik(target, q0, iters=400, lr=0.02):
    q=q0.copy(); jp=np.zeros((3,m.nv))
    for _ in range(iters):
        ee,_=fk(q)
        err=target-ee
        if np.linalg.norm(err)<1e-3: break
        jp[:]=0; mujoco.mj_jacSite(m,d,jp,None,e.ee_id)
        dq=jp[:,dadr].T@err
        q=np.clip(q+lr*dq, lo, hi)
    ee,ang=fk(q)
    return q, float(np.linalg.norm(ee-target)), ang

best=[]
rng=np.random.default_rng(0)
for y in (0.50, 0.60, 0.65, 0.70, 0.80):
    target=np.array([e.arrive_x, y, 0.375])+e.grasp_offset
    cands=[]
    for _ in range(25):
        q0=rng.uniform(lo,hi)
        q,err,ang=ik(target,q0)
        if err<0.01: cands.append(ang)
    print("y=%.2f 目标 %s: 可行解 %2d/25, angle 最大 %.3f 中位 %.3f"
          % (y, np.round(target,3), len(cands), max(cands) if cands else -1, np.median(cands) if cands else -1))
