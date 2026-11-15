import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
def ik_lim(env,target,yaw0=None,iters=8000,tol=0.003):
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
qg=ik_lim(env,np.array([bx,by,bz+0.060]))
env.data.qpos[env.arm_qpos_adr]=qg; env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
env.step(np.clip((qg-env.home_ctrl)/env.ctrl_scale,-1,1))
print("attach",env.magnet_attached)
qg = env.data.qpos[env.arm_qpos_adr].copy()
# 两段式：先抬高，再平移下降
for name, tgt in (("lift", [bx,by,0.78]), ("move", [0.65,-0.45,0.78]), ("drop", [0.65,-0.45,0.42])):
    q = ik_lim(env, np.array(tgt,float), yaw0=(-0.6 if name!="lift" else None))
    env.data.qpos[env.arm_qpos_adr]=qg; env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
    a = np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
    for t in range(120):
        o,r,term,tr,info=env.step(a)
        if term or tr: break
    qg = env.data.qpos[env.arm_qpos_adr].copy()
    print(name, "ee", np.round(env.data.site_xpos[env.ee_id],3), "att", env.magnet_attached, "cube", np.round(env.box_pos,3))
env.open=False
for t in range(80): env.step(a)
print("释放后方块", np.round(env.box_pos,3), "（期望 0.65,-0.45,0.375）")
