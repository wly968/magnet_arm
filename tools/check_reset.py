import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
m = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","train_scene.xml"))
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m,d,kid)
mujoco.mj_forward(m,d)
print("qpos:", np.round(d.qpos[:7],3), " ctrl:", np.round(d.ctrl,3))
arm={m.body(n).id for n in ("turret","shoulder","elbow","wrist","gripper","finger_L","finger_R")}
print("elbow:", np.round(d.body("elbow").xpos,3), " wrist:", np.round(d.body("wrist").xpos,3), " ee:", np.round(d.site("ee_site").xpos,3))
con=[]
for i in range(d.ncon):
    c=d.contact[i]; b1=int(m.geom_bodyid[c.geom1]); b2=int(m.geom_bodyid[c.geom2])
    if b1 in arm or b2 in arm:
        con.append((m.geom(c.geom1).name or m.body(b1).name, m.geom(c.geom2).name or m.body(b2).name, round(float(c.dist),3)))
print("arm contacts:", con[:10])
# 用 0 动作走 0.5s 看看初始瞬态
for k in range(250):
    d.ctrl[:5]=0; d.ctrl[5]=0.03; d.ctrl[6]=-0.03
    mujoco.mj_step(m,d)
print("after 0.5s action=0: qpos", np.round(d.qpos[:7],3), " wrist", np.round(d.body("wrist").xpos,3), " qvelmax", round(float(np.max(np.abs(d.qvel))),2))
