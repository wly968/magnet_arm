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
    return np.clip(q,lo,hi)

env=ArmEnv(); o,_=env.reset(seed=3); bx,by,bz=env.box_pos
env._set_dist(0.05); env._set_angle1(0.95); env.phase=2
q_grasp = ik_lim(env,np.array([bx,by,bz+0.060]))
env.data.qpos[env.arm_qpos_adr]=q_grasp; env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
a=np.clip((q_grasp-env.home_ctrl)/env.ctrl_scale,-1,1)
env.step(a); print("attach",env.magnet_attached)
q_grasp = env.data.qpos[env.arm_qpos_adr].copy()      # 记录吸附时的实际姿态
q_drop = ik_lim(env, env.drop_pos, yaw0=-0.6)
# 恢复手臂到吸附姿态，避免 IK 瞬移
env.data.qpos[env.arm_qpos_adr]=q_grasp; env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
a_drop=np.clip((q_drop-env.home_ctrl)/env.ctrl_scale,-1,1)
for t in range(120):
    o,r,term,tr,info=env.step(a_drop)
    if term or tr: break
print("到达 ee", np.round(env.data.site_xpos[env.ee_id],3), "目标", np.round(env.drop_pos,3), "att", env.magnet_attached)
print("搬运中方块", np.round(env.box_pos,3))
env.open=False
for t in range(80): env.step(a_drop)
print("释放后方块", np.round(env.box_pos,3), "（期望 0.65,-0.45,0.375）")
