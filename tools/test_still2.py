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
    mj=mujoco; mj.mj_forward(env.model,env.data); return np.clip(q,lo,hi)

env=ArmEnv(); env.max_steps=400; o,_=env.reset(seed=0)
env._set_dist(0.025); env._set_angle1(0.98); env._set_still_count(2)
target = env.box_pos + env.grasp_offset
# 模拟一个会补偿下垂的策略：目标略高 4cm
q = ik_from(env, target + np.array([0,0,0.04]), env.data.qpos[env.arm_qpos_adr].copy())
a = np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
env.phase = 2
for t in range(40):
    o,r,term,tr,info=env.step(a)
    ee=env.data.site_xpos[env.ee_id]
    d=float(np.linalg.norm(ee-target))
    if t%5==0 or term:
        print("t%2d dist %.4f still %d phase %d term %s" % (t,d,env.still_count,env.phase,term))
    if term: break
