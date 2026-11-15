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
    env.data.qvel[:]=0; mujoco.mj_forward(env.model,env.data); return q
env=ArmEnv(); o,_=env.reset(seed=3); bx,by,bz=env.box_pos
env._set_dist(0.05); env._set_angle1(0.95); env.phase=2
q=ik(env,np.array([bx,by,bz+0.060])); a=np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
env.step(a); print("attach",env.magnet_attached,"phase",env.phase)
q2=ik(env, env.drop_pos); a2=np.clip((q2-env.home_ctrl)/env.ctrl_scale,-1,1)
for t in range(60):
    o,r,term,tr,info=env.step(a2)
    if term or tr: break
print("到达 ee", np.round(env.data.site_xpos[env.ee_id],3), "target", env.drop_pos, "attached", env.magnet_attached)
print("搬运中方块", np.round(env.box_pos,3))
env.open=False
for t in range(50): env.step(a2)
print("释放后方块", np.round(env.box_pos,3), "（期望 0.65, -0.45, 0.375）")
