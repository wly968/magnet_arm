import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
def ik(env,target,iters=4000,tol=0.002):
    q=env.data.qpos[env.arm_qpos_adr].copy()
    jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
    for it in range(iters):
        mujoco.mj_forward(env.model,env.data)
        err=target-env.data.site_xpos[env.ee_id]
        if np.linalg.norm(err)<tol: break
        jp[:]=0; mujoco.mj_jacSite(env.model,env.data,jp,jr,env.ee_id)
        q=q+0.05*np.clip(jp[:,env.arm_dof_adr].T@err,-1,1)
        env.data.qpos[env.arm_qpos_adr]=q
    env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data); return q
env=ArmEnv(); o,_=env.reset(seed=2); bx,by,bz=env.box_pos
env._set_dist(0.05); env._set_angle1(0.95); env.phase=2
q=ik(env,np.array([bx,by,bz+0.060])); a=np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
env.step(a)
for t in range(6):
    env.step(a)
    d=env.data
    cq=d.qpos[env.box_qpos_adr:env.box_qpos_adr+3]
    print("t",t,"att",env.magnet_attached,"open",env.open,
          "disc_low %.4f"%env._disc_lowest_z(), "cube_qpos_z %.4f"%cq[2],
          "box_pos_z %.4f"%env.box_pos[2], "gap %.4f"%(env._disc_lowest_z()-(cq[2]+0.025)),
          "off",np.round(env.magnet_offset,3) if env.magnet_offset is not None else None)
