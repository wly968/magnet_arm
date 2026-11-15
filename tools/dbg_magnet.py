import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env = ArmEnv(); o,_ = env.reset(seed=1)
bx,by,bz = env.box_pos
target = np.array([bx,by,bz+0.048])
q = env.data.qpos[env.arm_qpos_adr].copy()
jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
for it in range(3000):
    mujoco.mj_forward(env.model, env.data)
    err = target - env.data.site_xpos[env.ee_id]
    if np.linalg.norm(err)<0.002: break
    jp[:]=0; mujoco.mj_jacSite(env.model, env.data, jp, jr, env.ee_id)
    q = q + 0.05*np.clip(jp[:,env.arm_dof_adr].T@err, -1, 1)
    env.data.qpos[env.arm_qpos_adr]=q
env.data.qvel[:]=0; mujoco.mj_forward(env.model, env.data)
base=(q-env.home_ctrl)/env.ctrl_scale
a=np.array([*np.clip(base,-1,1), 1.0])
for t in range(10):
    o,r,term,tr,info = env.step(a)
    print("t",t,"box_z %.4f"%env.box_pos[2],"ee %.4f"%env.data.site_xpos[env.ee_id][2],
          "gap %.4f"%env.magnet_gap,"attached",env.magnet_attached,
          "F",np.round(env.data.xfrc_applied[env.box_body_id,:3],2),
          "vz %.3f"%env.data.qvel[env.box_qvel_adr+2])
