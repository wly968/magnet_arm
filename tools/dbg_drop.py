import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
def ik(env,target,iters=6000,tol=0.002):
    q=env.data.qpos[env.arm_qpos_adr].copy()
    jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
    for it in range(iters):
        mujoco.mj_forward(env.model,env.data)
        err=target-env.data.site_xpos[env.ee_id]
        if np.linalg.norm(err)<tol: break
        jp[:]=0; mujoco.mj_jacSite(env.model,env.data,jp,jr,env.ee_id)
        q=q+0.05*np.clip(jp[:,env.arm_dof_adr].T@err,-1,1)
        env.data.qpos[env.arm_qpos_adr]=q
    env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data)
    return q, float(np.linalg.norm(target-env.data.site_xpos[env.ee_id]))
env=ArmEnv(); o,_=env.reset(seed=3)
q2,err=ik(env, env.drop_pos)
print("drop_pos", env.drop_pos, "ik err %.4f"%err)
print("q2      ", np.round(q2,3))
print("home    ", env.home_ctrl)
print("q2-home ", np.round(q2-env.home_ctrl,3), "  超出 ±1 的关节:", np.where(np.abs(q2-env.home_ctrl)>1.0)[0])
a2=np.clip((q2-env.home_ctrl)/env.ctrl_scale,-1,1)
print("action  ", np.round(a2,3))
env.step(a2)
print("1 step ee", np.round(env.data.site_xpos[env.ee_id],3))
for t in range(60): env.step(a2)
print("final ee ", np.round(env.data.site_xpos[env.ee_id],3), "target", env.drop_pos)
