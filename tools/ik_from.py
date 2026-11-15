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
    return np.clip(q,lo,hi), float(np.linalg.norm(target-env.data.site_xpos[env.ee_id]))
env=ArmEnv(); env.reset(seed=0)
q_wp2 = np.array([0.57,-1.70,2.10,0.65])
cands = [(0.65,-0.45,0.80),(0.65,-0.45,0.75),(0.60,-0.40,0.85),(0.55,-0.35,0.90),(0.65,-0.45,0.68)]
for c in cands:
    q,err = ik_from(env, np.array(c,float), q_wp2)
    print("above-place", c, "err %.4f"%err, "q", np.round(q,2))
