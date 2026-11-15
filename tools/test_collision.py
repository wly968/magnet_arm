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
    mujoco.mj_forward(env.model,env.data); return np.clip(q,lo,hi)

env=ArmEnv(); o,_=env.reset(seed=0)
cube=env.box_pos.copy()
q=ik_from(env,cube+np.array([0,0,0.02]),env.data.qpos[env.arm_qpos_adr].copy())  # 直接压向方块
env.data.qpos[env.arm_qpos_adr]=q; env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
a=np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
print("arm_geom_ids", sorted(env.arm_geom_ids))
for t in range(8):
    o,r,term,tr,info=env.step(a)
    print("t%d cube_hit=%s reward=%.2f box_z=%.3f ncon=%d" % (t, env.cube_hit, r, env.box_pos[2], env.data.ncon))
