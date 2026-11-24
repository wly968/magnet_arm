import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
lo,hi=e.jnt_lo,e.jnt_hi
def ik_multi(tgt, starts=40, iters=60):
    rng=np.random.default_rng(7); best=(9,None,None)
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
        d=float(np.linalg.norm(ee-tgt)); a=float(np.dot(f,[0,0,-1]))
        if d<best[0]: best=(d,a,q.copy())
    return best
for name,tgt in (("transfer 0.85",np.array([0.42,0.27,0.85])),
                 ("transfer 0.78",np.array([0.42,0.27,0.78])),
                 ("transfer 0.72",np.array([0.42,0.27,0.72])),
                 ("放料点",np.array([0.65,-0.45,0.435])),
                 ("抓取上方",np.array([-0.475,0.65,0.525]))):
    d,a,q=ik_multi(tgt)
    print("%-14s 最优 dist %.4f | 该解 angle %.3f | q %s" % (name,d,a,np.round(q,2)))
