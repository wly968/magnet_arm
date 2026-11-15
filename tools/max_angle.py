import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv(); env.reset(seed=0)
target = env.box_pos + env.grasp_offset
lo=env.model.jnt_range[env.arm_joints][:,0]; hi=env.model.jnt_range[env.arm_joints][:,1]
best=(-1,None,None)
jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
for w0 in np.linspace(-1.5,1.5,25):
    q=np.array([1.62,-1.0,2.1,w0])
    for it in range(6000):
        env.data.qpos[env.arm_qpos_adr]=np.clip(q,lo,hi)
        mujoco.mj_forward(env.model,env.data)
        err=target-env.data.site_xpos[env.ee_id]
        if np.linalg.norm(err)<0.002: break
        jp[:]=0; mujoco.mj_jacSite(env.model,env.data,jp,jr,env.ee_id)
        q=np.clip(q+0.05*np.clip(jp[:,env.arm_dof_adr].T@err,-1,1),lo,hi)
    env.data.qpos[env.arm_qpos_adr]=np.clip(q,lo,hi); env.data.qvel[:]=0
    mujoco.mj_forward(env.model,env.data)
    ee=env.data.site_xpos[env.ee_id]
    d=np.linalg.norm(ee-target)
    if d>0.004: continue
    j3=env.data.xanchor[env.arm_joints[3]]
    fwd=(ee-j3)/np.linalg.norm(ee-j3)
    a1=float(np.dot(fwd,[0,0,-1]))
    if a1>best[0]: best=(a1, q.copy(), d)
print("grasp 姿态下可达的最大 angle1 = %.4f  q=%s dist=%.4f" % (best[0], np.round(best[1],2), best[2]))
