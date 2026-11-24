import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); lo,hi=e.jnt_lo,e.jnt_hi
def ik(q0, tgt, iters=200):
    q=np.array(q0, dtype=float)
    for _ in range(iters):
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
        err=tgt-e.data.site_xpos[e.ee_id]
        jp=np.zeros((3,e.model.nv)); mujoco.mj_jacSite(e.model,e.data,jp,None,e.ee_id)
        J=jp[:,e.arm_dof_adr]; dq=J.T@err
        n=float(np.linalg.norm(dq))
        if n<1e-6: break
        dq*=min(1.0,0.1/n)
        q=np.clip(q+1.0*dq,lo,hi)
    e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
    ee=e.data.site_xpos[e.ee_id]
    j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    return q, float(np.linalg.norm(ee-tgt)), float(np.dot(f,[0,0,-1]))
# 先用之前的配置做起点
q=np.array([2.2,0.97,-1.85,-0.35])   # 之前 IK 得到的抓取解附近
for name,z,ang in (("抓取点 z=0.525",0.525,0.98), ("接近点 z=0.675",0.675,0.94)):
    print(name)
    q_cur,_,_=ik(q, np.array([-0.475,0.65,z]))
    for y in list(np.arange(0.65,0.79,0.01))+list(np.arange(0.64,0.52,-0.01)):
        q_cur,d,a=ik(q_cur, np.array([-0.475,float(y),z]))
        flag = "OK" if (d<0.032 and a>ang) else "FAIL"
        if flag=="FAIL" or abs(y-0.65)<1e-9 or abs(y-0.77)<1e-9 or abs(y-0.53)<1e-9:
            print("   y=%.2f dist %.4f angle %.3f %s" % (y,d,a,flag))
