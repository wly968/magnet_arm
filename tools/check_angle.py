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
env=ArmEnv(); env.reset(seed=0)
for name, off in (("pick", env.pick_offset), ("grasp", env.grasp_offset)):
    target = env.box_pos + off
    q = ik_from(env, target, env.data.qpos[env.arm_qpos_adr].copy())
    ee=env.data.site_xpos[env.ee_id]
    j3=env.data.xanchor[env.arm_joints[3]]
    fwd=(ee-j3)/np.linalg.norm(ee-j3)
    print("%s target %s -> q %s angle1 %.4f dist %.4f" %
          (name, np.round(target,3), np.round(q,2), float(np.dot(fwd,[0,0,-1])), float(np.linalg.norm(ee-target))))
