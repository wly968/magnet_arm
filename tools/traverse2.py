import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv

def ik_from(env, target, q0, iters=9000, tol=0.002):
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
    return np.clip(q,lo,hi)

env=ArmEnv(); env.max_steps=800; o,_=env.reset(seed=0)
env._set_dist(0.03); env._set_angle1(0.95)

env.dist_threshold = 0.03
q_cur = env.data.qpos[env.arm_qpos_adr].copy()
for idx, wp in enumerate(env.waypoints):
    q_tgt = ik_from(env, wp, q_cur)
    env.data.qpos[env.arm_qpos_adr]=q_cur; env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
    a0 = np.clip((q_cur-env.home_ctrl)/env.ctrl_scale,-1,1)
    a1 = np.clip((q_tgt-env.home_ctrl)/env.ctrl_scale,-1,1)
    arrived=False
    for t in range(300):
        s=min(1.0, (t+1)/50.0)
        o,r,term,tr,info=env.step(a0+(a1-a0)*s)
        d=float(np.linalg.norm(env.data.site_xpos[env.ee_id]-wp))
        if d<0.03: arrived=True; break
        if term or tr: break
    q_cur = env.data.qpos[env.arm_qpos_adr].copy()
    print("WP%d 到达 %s 距离 %.3f phase=%d steps=%d nan=%s" %
          (idx+1, np.round(env.data.site_xpos[env.ee_id],2), d, env.phase, t+1, info.get("nan")))
