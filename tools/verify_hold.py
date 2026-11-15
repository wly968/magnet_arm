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
q=ik(env,np.array([bx,by,bz+0.026]))
a=np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
o,r,term,tr,info=env.step(a)
print("attached",env.magnet_attached,"phase",env.phase)
zs=[];gaps=[];disc_contacts=0
gid=env.model.geom("magnet_disc").id
for t in range(20):
    o,r,term,tr,info=env.step(a)   # 保持同一指令
    ee=env.data.site_xpos[env.ee_id]
    zs.append(round(float(ee[2]),4)); gaps.append(round(float((ee[2]-0.008)-(env.box_pos[2]+0.025)),4))
    for i in range(env.data.ncon):
        c=env.data.contact[i]
        if c.geom1==gid or c.geom2==gid: disc_contacts+=1
print("hold ee_z:", zs[::5], " 漂移 %.4f" % (zs[-1]-zs[0]))
print("gap:", gaps[::5], "圆盘接触次数", disc_contacts)
