import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); lo,hi=e.jnt_lo,e.jnt_hi
def feasible(tgt, need_angle):
    rng=np.random.default_rng(5); best=9.0
    for _ in range(50):
        q=rng.uniform(lo,hi)
        for _ in range(70):
            e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
            err=tgt-e.data.site_xpos[e.ee_id]
            jp=np.zeros((3,e.model.nv)); mujoco.mj_jacSite(e.model,e.data,jp,None,e.ee_id)
            J=jp[:,e.arm_dof_adr]; dq=J.T@err
            n=float(np.linalg.norm(dq))
            if n>1e-9: dq*=min(1.0,0.25/n)
            q=np.clip(q+0.9*dq,lo,hi)
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
        ee=e.data.site_xpos[e.ee_id]
        j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        if float(np.dot(f,[0,0,-1])) < need_angle: continue
        best=min(best,float(np.linalg.norm(ee-tgt)))
    return best
print("抓取点可达性（要求 angle>0.98，dist<0.032 才算可以抓）")
for y in np.arange(0.53,0.79,0.03):
    tgt=np.array([-0.475, float(y), 0.525])
    d=feasible(tgt, 0.98)
    print("  y=%.2f : 最近 dist %s -> %s" % (y, ("%.4f"%d) if d<9 else "无解", "OK" if d<0.032 else "抓不到"))
print("\n接近点（phase1，只要求 angle>0.94、dist<0.05）")
for y in np.arange(0.53,0.79,0.03):
    tgt=np.array([-0.475, float(y), 0.675])
    d=feasible(tgt, 0.94)
    print("  y=%.2f : 最近 dist %s -> %s" % (y, ("%.4f"%d) if d<9 else "无解", "OK" if d<0.05 else "到不了"))
