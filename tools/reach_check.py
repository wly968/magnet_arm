import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
def ik_from(env,target,q0,iters=9000,tol=0.002):
    lo=env.model.jnt_range[env.arm_joints][:,0]; hi=env.model.jnt_range[env.arm_joints][:,1]
    q=np.array(q0,float).copy()
    jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
    for it in range(iters):
        env.data.qpos[env.arm_qpos_adr]=np.clip(q,lo,hi)
        mujoco.mj_forward(env.model,env.data)
        err=target-env.data.site_xpos[env.ee_id]
        if np.linalg.norm(err)<tol: break
        jp[:]=0; mujoco.mj_jacSite(env.model,env.data,jp,jr,env.ee_id)
        q=np.clip(q+0.05*np.clip(jp[:,env.arm_dof_adr].T@err,-1,1),lo,hi)
    env.data.qpos[env.arm_qpos_adr]=np.clip(q,lo,hi); env.data.qvel[:]=0
    mujoco.mj_forward(env.model,env.data)
    return np.clip(q,lo,hi), float(np.linalg.norm(target-env.data.site_xpos[env.ee_id]))
env=ArmEnv(); errs1=[]; errs2=[]
q_home = env.data.qpos[env.arm_qpos_adr].copy()
for sd in range(12):
    o,_=env.reset(seed=sd)
    p1=env.box_pos+env.pick_offset; p2=env.box_pos+env.grasp_offset
    _,e1=ik_from(env,p1,q_home); _,e2=ik_from(env,p2,q_home)
    errs1.append(e1); errs2.append(e2)
print("phase1 目标 IK 误差 max %.4f mean %.4f" % (max(errs1), np.mean(errs1)))
print("phase2 目标 IK 误差 max %.4f mean %.4f" % (max(errs2), np.mean(errs2)))
