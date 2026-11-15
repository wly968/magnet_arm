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
gid=env.model.geom("magnet_disc").id
drift=[];gaps=[];disc_c=0;ee0=None
for t in range(20):
    env.step(a)
    ee=env.data.site_xpos[env.ee_id]
    if ee0 is None: ee0=float(ee[2])
    drift.append(float(ee[2])-ee0)
    gaps.append(round(env._disc_lowest_z() - (env.box_pos[2]+0.025),4))
    for i in range(env.data.ncon):
        c=env.data.contact[i]
        if (c.geom1==gid or c.geom2==gid) and (env.model.geom(c.geom1).name=="cube_0_geom" or env.model.geom(c.geom2).name=="cube_0_geom"):
            disc_c+=1
print("arm drift %.4f m"%drift[-1])
print("gap(圆盘最低点-方块顶面):", gaps[::5])
print("圆盘-方块接触次数:", disc_c)
