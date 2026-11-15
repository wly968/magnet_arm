import os, sys, numpy as np, mujoco, time
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv

def ik(env, target, iters=3000, tol=0.002):
    q = env.data.qpos[env.arm_qpos_adr].copy()
    jp=np.zeros((3,env.model.nv)); jr=np.zeros((3,env.model.nv))
    for it in range(iters):
        mujoco.mj_forward(env.model, env.data)
        err = target - env.data.site_xpos[env.ee_id]
        if np.linalg.norm(err)<tol: break
        jp[:]=0; mujoco.mj_jacSite(env.model, env.data, jp, jr, env.ee_id)
        q = q + 0.05*np.clip(jp[:,env.arm_dof_adr].T@err, -1, 1)
        env.data.qpos[env.arm_qpos_adr]=q
    env.data.qvel[:]=0; mujoco.mj_forward(env.model, env.data)
    return q

env = ArmEnv(); o,_ = env.reset(seed=2)
bx,by,bz = env.box_pos
env._set_dist(0.05); env._set_angle1(0.95); env.phase = 2
q = ik(env, np.array([bx,by,bz+0.026]))
a = np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
o,r,term,tr,info = env.step(a)
print("phase", env.phase, "open", env.open, "attached", env.magnet_attached)

# 抬升 10cm
q2 = ik(env, np.array([bx,by,bz+0.126]))
a2 = np.clip((q2-env.home_ctrl)/env.ctrl_scale,-1,1)
zs=[]
for t in range(20):
    o,r,term,tr,info = env.step(a2)
    zs.append(round(float(env.box_pos[2]),3))
print("lift cube z:", zs[::4], "attached", env.magnet_attached)

# 关磁吸
env.open = False
zs2=[]
for t in range(20):
    o,r,term,tr,info = env.step(a2)
    zs2.append(round(float(env.box_pos[2]),3))
print("release cube z:", zs2[::4], "attached", env.magnet_attached)
