import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
pick=np.array([-0.475,0.65]); drop=np.array([0.65,-0.45])
az_pick=np.degrees(np.arctan2(pick[1],pick[0])); az_drop=np.degrees(np.arctan2(drop[1],drop[0]))
az=(az_pick+az_drop)/2.0
r=0.45
pt=np.array([r*np.cos(np.radians(az)), r*np.sin(np.radians(az)), 0.78])
print("抓取方位 {:.1f} 度, 放料方位 {:.1f} 度, 均衡方位 {:.1f} 度".format(az_pick, az_drop, az))
print("候选转移点 ({:.3f}, {:.3f}, {:.2f}) 两段 yaw 各 {:.1f} 度".format(pt[0],pt[1],pt[2], abs(az_pick-az)))
# 可达性
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); lo,hi=e.jnt_lo,e.jnt_hi
rng=np.random.default_rng(4); best=9.0
for _ in range(50):
    q=rng.uniform(lo,hi)
    for _ in range(70):
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
        err=pt-e.data.site_xpos[e.ee_id]
        jp=np.zeros((3,e.model.nv)); mujoco.mj_jacSite(e.model,e.data,jp,None,e.ee_id)
        dq=jp[:,e.arm_dof_adr].T@err; n=float(np.linalg.norm(dq))
        if n>1e-9: dq*=min(1.0,0.25/n)
        q=np.clip(q+0.9*dq,lo,hi)
    e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
    ee=e.data.site_xpos[e.ee_id]
    j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    if float(np.dot(f,[0,0,-1]))<0.6: continue
    best=min(best,float(np.linalg.norm(ee-pt)))
print("可达距离 {:.4f}".format(best))
