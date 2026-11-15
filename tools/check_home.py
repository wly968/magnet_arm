import os, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
m = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","train_scene.xml"))
d = mujoco.MjData(m)
arm={m.body(n).id for n in ("turret","shoulder","elbow","wrist","gripper","finger_L","finger_R")}
def check(name, q):
    d.qpos[:7]=q; d.qpos[7:]=0; d.ctrl[:7]=q; mujoco.mj_forward(m,d)
    con=[]
    for i in range(d.ncon):
        c=d.contact[i]; b1=int(m.geom_bodyid[c.geom1]); b2=int(m.geom_bodyid[c.geom2])
        if b1 in arm or b2 in arm:
            con.append((m.geom(c.geom1).name or m.body(b1).name, m.geom(c.geom2).name or m.body(b2).name, round(float(c.dist),3)))
    el=d.body("elbow").xpos; wr=d.body("wrist").xpos
    print(f"{name}: elbow z={el[2]:.3f} wrist={np.round(wr,2)} contacts={con[:6]}")
check("你现在的 home", [1.57,1.0,-2.0,-0.57,0.0,0.03,-0.03])
check("旧 home(朝-Y 折上)", [-1.2,0.20,-1.90,1.20,0.0,0.03,-0.03])
check("qpos 全 0(伸直 +X)", [0,0,0,0,0,0.03,-0.03])
check("朝+Y 折上", [1.57,0.20,-1.90,1.20,0.0,0.03,-0.03])
check("朝+Y 折上(肩0)", [1.57,0.0,-1.90,1.20,0.0,0.03,-0.03])
