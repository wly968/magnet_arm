import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); lo,hi=e.jnt_lo,e.jnt_hi
def ik_down(tgt, starts=60, iters=60):
    rng=np.random.default_rng(3); best=9.0
    for _ in range(starts):
        q=rng.uniform(lo,hi)
        for _ in range(iters):
            e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
            err=tgt-e.data.site_xpos[e.ee_id]
            jp=np.zeros((3,e.model.nv)); mujoco.mj_jacSite(e.model,e.data,jp,None,e.ee_id)
            J=jp[:,e.arm_dof_adr]; dq=J.T@err
            n=float(np.linalg.norm(dq))
            if n>1e-9: dq*=min(1.0,0.3/n)
            q=np.clip(q+0.8*dq,lo,hi)
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
        ee=e.data.site_xpos[e.ee_id]
        j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        if float(np.dot(f,[0,0,-1]))<0.94: continue
        best=min(best,float(np.linalg.norm(ee-tgt)))
    return best
for y in (0.60,0.70,0.78,0.82,0.86):
    tgt=np.array([-0.475, y, 0.525])
    d=ik_down(tgt)
    print("抓取点 y=%.2f：工具朝下最近可达距离 %s" % (y, ("%.4f"%d) if d<9 else "无解"))
