import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
def ik_lim(env,target,yaw0=None,iters=8000,tol=0.002):
    lo=env.model.jnt_range[env.arm_joints][:,0]; hi=env.model.jnt_range[env.arm_joints][:,1]
    q=env.data.qpos[env.arm_qpos_adr].copy()
    if yaw0 is not None: q[0]=yaw0
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
env=ArmEnv(); o,_=env.reset(seed=3); bx,by,bz=env.box_pos
env._set_dist(0.05); env._set_angle1(0.95); env.phase=2
q=ik_lim(env,np.array([bx,by,bz+0.060])); a=np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
env.step(a); print("attach",env.magnet_attached,"offset",np.round(env.magnet_offset,3))
q2=ik_lim(env,env.drop_pos,yaw0=-0.6); a2=np.clip((q2-env.home_ctrl)/env.ctrl_scale,-1,1)
for t in range(12):
    env.step(a2)
    print("t",t,"att",env.magnet_attached,"dxy %.3f"%np.hypot(env.data.site_xpos[env.ee_id][0]-env.box_pos[0],
          env.data.site_xpos[env.ee_id][1]-env.box_pos[1]),
          "ee",np.round(env.data.site_xpos[env.ee_id],2),"cube",np.round(env.box_pos,2),"off",np.round(env.magnet_offset,2) if env.magnet_offset is not None else None)
