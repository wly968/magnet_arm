import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); lo,hi=e.jnt_lo,e.jnt_hi
def ik_scan(tgt, starts=80, iters=60):
    rng=np.random.default_rng(11); sols=[]
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
        sols.append((float(np.linalg.norm(ee-tgt)), float(np.dot(f,[0,0,-1]))))
    down=[d for d,a in sols if a>0.945 and d<0.10]
    return (min(down) if down else float('inf'), sum(1 for d,a in sols if a>0.945))
for name,tgt in (("transfer 0.85",np.array([0.42,0.27,0.85])),
                 ("transfer 0.78",np.array([0.42,0.27,0.78])),
                 ("transfer 0.72",np.array([0.42,0.27,0.72])),
                 ("放料点",np.array([0.65,-0.45,0.435])),
                 ("抓取上方 y=0.65",np.array([-0.475,0.65,0.525]))):
    best,ndown=ik_scan(tgt)
    print("%-16s 工具朝下解中最近 dist %s（朝下解 %d/80）" %
          (name, ("%.4f"%best) if best<9 else "无", ndown))
