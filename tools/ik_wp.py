import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
def ik(env,target,yaw0=None,iters=8000,tol=0.002):
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
    env.data.qpos[env.arm_qpos_adr]=np.clip(q,lo,hi); env.data.qvel[:]=0
    mujoco.mj_forward(env.model,env.data)
    return np.clip(q,lo,hi), float(np.linalg.norm(target-env.data.site_xpos[env.ee_id]))
env=ArmEnv(); env.reset(seed=0)
cands = {
 "pick(-0.125,0.65,0.50)": [-0.125,0.65,0.50],
 "pick(-0.125,0.65,0.55)": [-0.125,0.65,0.55],
 "transfer(0.42,0.27,0.85)": [0.42,0.27,0.85],
 "transfer(0.30,0.10,0.90)": [0.30,0.10,0.90],
 "transfer(0.25,0.10,0.85)": [0.25,0.10,0.85],
 "place(0.65,-0.45,0.42)": [0.65,-0.45,0.42],
 "place(0.65,-0.45,0.55)": [0.65,-0.45,0.55],
}
for name,t in cands.items():
    best=None
    for y0 in (None,-0.6,0.6,1.2,2.0,-1.5,3.0):
        q,err=ik(env,np.array(t,float),y0)
        if best is None or err<best[1]: best=(q,err)
    print("%-28s err %.4f  q %s" % (name,best[1],np.round(best[0],2)))
