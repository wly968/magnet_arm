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

env = ArmEnv()
print("obs", env.observation_space.shape, "act", env.action_space.shape, "open(init)", env.open)
o,_ = env.reset(seed=2)
bx,by,bz = env.box_pos
env.phase = 2                                  # 直接进 phase 2 验证开关
q = ik(env, np.array([bx, by, bz + 0.03 - 0.004]))
a = np.clip((q - env.home_ctrl)/env.ctrl_scale, -1, 1)
env._set_dist(0.05); env._set_angle1(0.95)
o,r,term,tr,info = env.step(a)
ee=env.data.site_xpos[env.ee_id]
tgt=env.box_pos+np.array([0,0,0.03])
j3=env.data.xanchor[env.arm_joints[3]]
fwd=(ee-j3)/np.linalg.norm(ee-j3)
print("dist %.4f angle1 %.3f ee_z %.3f target_z %.3f" % (np.linalg.norm(ee-tgt), float(np.dot(fwd,[0,0,-1])), ee[2], tgt[2]))
print("after step: phase", env.phase, "open", env.open, "attached", env.magnet_attached,
      "reward %.1f"%r, "success", info.get("success"))
zs=[]
for t in range(10):
    o,r,term,tr,info = env.step(a)
    zs.append(round(float(env.box_pos[2]),3))
print("cube z after open:", zs, "attached", env.magnet_attached)
