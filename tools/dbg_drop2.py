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
env.step(a)
q2=ik(env, env.drop_pos); a2=np.clip((q2-env.home_ctrl)/env.ctrl_scale,-1,1)
print("q2", np.round(q2,3), "a2", np.round(a2,3))
ctrl_target = env.home_ctrl + a2*env.ctrl_scale
print("ctrl target", np.round(ctrl_target,3))
for t in range(6):
    o,r,term,tr,info=env.step(a2)
    print("t",t,"ctrl",np.round(env.data.ctrl[:4],2),"qpos",np.round(env.data.qpos[env.arm_qpos_adr],2),
          "ee",np.round(env.data.site_xpos[env.ee_id],3),"att",env.magnet_attached)
