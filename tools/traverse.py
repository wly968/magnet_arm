import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv

def ik_lim(env, target, yaw0=None, iters=9000, tol=0.002):
    lo = env.model.jnt_range[env.arm_joints][:,0]; hi = env.model.jnt_range[env.arm_joints][:,1]
    q = env.data.qpos[env.arm_qpos_adr].copy()
    if yaw0 is not None: q[0] = yaw0
    jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
    for it in range(iters):
        env.data.qpos[env.arm_qpos_adr]=np.clip(q,lo,hi)
        mujoco.mj_forward(env.model,env.data)
        err = target - env.data.site_xpos[env.ee_id]
        if np.linalg.norm(err)<tol: break
        jp[:]=0; mujoco.mj_jacSite(env.model,env.data,jp,jr,env.ee_id)
        q = np.clip(q + 0.05*np.clip(jp[:,env.arm_dof_adr].T@err,-1,1), lo, hi)
    env.data.qpos[env.arm_qpos_adr]=np.clip(q,lo,hi); env.data.qvel[:]=0
    mujoco.mj_forward(env.model,env.data)
    return np.clip(q,lo,hi), float(np.linalg.norm(target-env.data.site_xpos[env.ee_id]))

env=ArmEnv(); o,_=env.reset(seed=0)
env._set_dist(0.03); env._set_angle1(0.95)
for idx, wp in enumerate(env.waypoints):
    q_cur = env.data.qpos[env.arm_qpos_adr].copy()
    q_tgt, err = ik_lim(env, wp)
    # 恢复当前姿态，避免瞬移
    env.data.qpos[env.arm_qpos_adr]=q_cur; env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
    a = np.clip((q_tgt-env.home_ctrl)/env.ctrl_scale,-1,1)
    best=9.0
    for t in range(250):
        o,r,term,tr,info=env.step(a)
        d=float(np.linalg.norm(env.data.site_xpos[env.ee_id]-wp)); best=min(best,d)
        if term or tr or env.phase>idx+1: break
    ee=env.data.site_xpos[env.ee_id]
    print("WP%d 目标 %s  到达 %s  距离 %.3f  phase=%d  steps=%d" %
          (idx+1, np.round(wp,2), np.round(ee,2), float(np.linalg.norm(ee-wp)), env.phase, t+1))
