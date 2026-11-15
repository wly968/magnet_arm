import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv()
for seed in range(4):
    o,_=env.reset(seed=seed)
    bx,by,bz=env.box_pos
    target=np.array([bx,by,bz+0.15])
    q=env.data.qpos[env.arm_qpos_adr].copy()
    jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
    for it in range(3000):
        mujoco.mj_forward(env.model, env.data)
        err=target-env.data.site_xpos[env.ee_id]
        if np.linalg.norm(err)<0.002: break
        jp[:]=0; mujoco.mj_jacSite(env.model, env.data, jp, jr, env.ee_id)
        q=q+0.05*np.clip(jp[:,env.arm_dof_adr].T@err,-1,1)
        env.data.qpos[env.arm_qpos_adr]=q
    env.data.qvel[:]=0; mujoco.mj_forward(env.model, env.data)
    ee=env.data.site_xpos[env.ee_id]
    j3=env.data.xanchor[env.arm_joints[3]]
    fwd=(ee-j3)/np.linalg.norm(ee-j3)
    a1=float(np.dot(fwd,[0,0,-1]))
    print("seed",seed,"dist %.4f"%np.linalg.norm(ee-target),"angle1 %.3f"%a1,
          "q",np.round(q,2),"home",env.home_ctrl,"delta",np.round(q-env.home_ctrl,2))
