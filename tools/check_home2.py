import os, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
m = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","train_scene.xml"))
d = mujoco.MjData(m)
print("nq",m.nq,"nv",m.nv,"nu",m.nu,"nmocap",m.nmocap)
for i in range(m.njnt): print(" ",m.joint(i).name, int(m.jnt_qposadr[i]))
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m,d,kid); mujoco.mj_forward(m,d)
print("current keyframe qpos len", len(d.qpos), "qpos", np.round(d.qpos,2), "ctrl", np.round(d.ctrl,2))
arm={m.body(n).id for n in ("turret","shoulder","elbow","wrist","gripper","finger_L","finger_R")}
def check(name, q4):
    d.qpos[:]=0; d.qpos[0:4]=q4; d.qpos[4]=0; d.qpos[5]=0.03; d.qpos[6]=-0.03
    d.ctrl[:4]=q4; d.ctrl[4]=0; d.ctrl[5]=0.03; d.ctrl[6]=-0.03
    mujoco.mj_forward(m,d)
    con=[]
    for i in range(d.ncon):
        c=d.contact[i]; b1=int(m.geom_bodyid[c.geom1]); b2=int(m.geom_bodyid[c.geom2])
        if b1 in arm or b2 in arm:
            con.append((m.geom(c.geom1).name or m.body(b1).name, m.geom(c.geom2).name or m.body(b2).name, round(float(c.dist),3)))
    print(f"{name}: elbow={np.round(d.body('elbow').xpos,2)} wrist={np.round(d.body('wrist').xpos,2)} contacts={con[:6]}")
check("F 朝+Y 前折", [1.57,-1.0,1.4,0.0])
check("E 朝+X 前折", [0.0,-1.0,1.4,0.0])
check("H 朝NE 前折", [0.785,-0.9,1.2,0.0])
check("A 朝SW 前折", [-2.2,-0.9,1.3,0.0])
