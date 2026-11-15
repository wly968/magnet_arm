import os, sys, numpy as np, mujoco
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

def act(env, q, magnet):
    return np.clip(np.array([*((q-env.home_ctrl)/env.ctrl_scale), magnet]), -1, 1)

env = ArmEnv(); o,_ = env.reset(seed=1)
bx,by,bz = env.box_pos
low = np.array([bx,by,bz+0.048])
q_low = ik(env, low)
print("approach: ee_z %.4f box_z %.4f" % (env.data.site_xpos[env.ee_id][2], env.box_pos[2]))

for t in range(8):
    o,r,term,tr,info = env.step(act(env, q_low, 1.0))
print("attached:", env.magnet_attached, "box_z %.3f" % env.box_pos[2])

# 竖直抬升 10cm
high = low + np.array([0,0,0.10])
q_high = ik(env, high)
zs=[]; ees=[]
for t in range(25):
    o,r,term,tr,info = env.step(act(env, q_high, 1.0))
    zs.append(round(float(env.box_pos[2]),3)); ees.append(round(float(env.data.site_xpos[env.ee_id][2]),3))
print("lift  ee_z:", ees[::5])
print("lift  cube_z:", zs[::5], "attached", env.magnet_attached)

zs2=[]
for t in range(25):
    o,r,term,tr,info = env.step(act(env, q_high, -1.0))
    zs2.append(round(float(env.box_pos[2]),3))
print("release cube_z:", zs2[::5], "attached", env.magnet_attached)
