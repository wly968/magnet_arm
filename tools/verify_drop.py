import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
def ik(env,target,iters=5000,tol=0.002):
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
bx,by,bz=env.box_pos
env._set_dist(0.05); env._set_angle1(0.95); env.phase=2
q,_=ik(env,np.array([bx,by,bz+0.060])); a=np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
env.step(a)
print("attach", env.magnet_attached, "phase", env.phase)

# 送到放料点
q2,err=ik(env, env.drop_pos)
a2=np.clip((q2-env.home_ctrl)/env.ctrl_scale,-1,1)
zs=[]
for t in range(40):
    o,r,term,tr,info=env.step(a2)
    zs.append(round(float(env.box_pos[2]),3))
    if term or tr: break
ee=env.data.site_xpos[env.ee_id]
print("IK err %.4f  ee %s  cube %s" % (err, np.round(ee,3), np.round(env.box_pos,3)))
print("carry cube z:", zs[::8], "attached", env.magnet_attached)
env.open=False
for t in range(40): env.step(a2)
print("after release cube:", np.round(env.box_pos,3), "attached", env.magnet_attached)
print("beltB surface z=0.350, 期望方块中心 (0.65,-0.45,0.375)")
