import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv

env = ArmEnv()
o,_ = env.reset(seed=1)
bx, by, bz = env.box_pos
target = np.array([bx, by, bz + 0.048])

q = env.data.qpos[env.arm_qpos_adr].copy()
jp = np.zeros((3, env.model.nv)); jr = np.zeros((3, env.model.nv))
for it in range(3000):
    mujoco.mj_forward(env.model, env.data)
    err = target - env.data.site_xpos[env.ee_id]
    if np.linalg.norm(err) < 0.002: break
    jp[:] = 0
    mujoco.mj_jacSite(env.model, env.data, jp, jr, env.ee_id)
    g = jp[:, env.arm_dof_adr].T @ err
    q = q + 0.05 * np.clip(g, -1, 1)
    env.data.qpos[env.arm_qpos_adr] = q
env.data.qpos[env.arm_qpos_adr] = q
env.data.qvel[:] = 0
mujoco.mj_forward(env.model, env.data)
print("IK iters", it, "err", round(float(np.linalg.norm(target - env.data.site_xpos[env.ee_id])),4))
print("ee", np.round(env.data.site_xpos[env.ee_id],3), "box", np.round(env.box_pos,3))

base = (q - env.home_ctrl) / env.ctrl_scale
for label, m in (("ON", 1.0), ("OFF", -1.0)):
    a = np.array([*base, m]); zs=[]
    for t in range(15):
        o,r,term,tr,info = env.step(np.clip(a,-1,1))
        zs.append(round(float(env.box_pos[2]),3))
    print("magnet", label, "cube z:", zs, "weld", int(env.data.eq_active[env.weld_id]))
