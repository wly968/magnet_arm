import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv

def ik(env, target, iters=4000, tol=0.002):
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
    return q, float(np.linalg.norm(target - env.data.site_xpos[env.ee_id]))

env = ArmEnv(); o,_ = env.reset(seed=2)
bx,by,bz = env.box_pos
env._set_dist(0.05); env._set_angle1(0.95); env.phase = 2
q = ik(env, np.array([bx,by,bz+0.026]))[0]
a = np.clip((q-env.home_ctrl)/env.ctrl_scale,-1,1)
o,r,term,tr,info = env.step(a)
print("phase", env.phase, "open", env.open, "attached", env.magnet_attached, "offset", np.round(env.magnet_offset,3))

# 继续下压 3cm：检查是否穿模
q2 = ik(env, np.array([bx,by,bz-0.004]))[0]
a2 = np.clip((q2-env.home_ctrl)/env.ctrl_scale,-1,1)
for t in range(15):
    o,r,term,tr,info = env.step(a2)
    ee = env.data.site_xpos[env.ee_id]
    gap = (ee[2]-0.008) - (env.box_pos[2]+0.025)
    if t%5==0: print(" press t%d gap %.4f attached %s contact %d" % (t, gap, env.magnet_attached, env.data.ncon))

# 抬升 12cm
q3 = ik(env, np.array([bx,by,bz+0.146]))[0]
a3 = np.clip((q3-env.home_ctrl)/env.ctrl_scale,-1,1)
zs=[]
for t in range(20):
    o,r,term,tr,info = env.step(a3)
    zs.append(round(float(env.box_pos[2]),3))
print("lift cube z:", zs[::4], "attached", env.magnet_attached)

env.open = False
zs2=[]
for t in range(15):
    o,r,term,tr,info = env.step(a3)
    zs2.append(round(float(env.box_pos[2]),3))
print("release cube z:", zs2[::3], "attached", env.magnet_attached)

# 传送带 B 目标是否可达
env2 = ArmEnv(); env2.reset(seed=0)
for z in (0.525, 0.405):
    q, err = ik(env2, np.array([0.65,-0.45,z]))
    print("beltB target", (0.65,-0.45,z), "IK err %.4f" % err)
