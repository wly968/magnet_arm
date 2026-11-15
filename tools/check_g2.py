import os, numpy as np, mujoco
from PIL import Image
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
# 1) train 模型验证 G2 无碰撞
m = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","train_scene.xml"))
d = mujoco.MjData(m)
arm={m.body(n).id for n in ("turret","shoulder","elbow","wrist","gripper","finger_L","finger_R")}
q=[1.57,-1.15,2.0,0.725,0.0,0.03,-0.03]
d.qpos[:]=q; d.ctrl[:]=q; mujoco.mj_forward(m,d)
con=[]
for i in range(d.ncon):
    c=d.contact[i]; b1=int(m.geom_bodyid[c.geom1]); b2=int(m.geom_bodyid[c.geom2])
    if b1 in arm or b2 in arm:
        con.append((m.geom(c.geom1).name or m.body(b1).name, m.geom(c.geom2).name or m.body(b2).name, round(float(c.dist),3)))
print("G2 train contacts:", con)
print("elbow", np.round(d.body("elbow").xpos,3), "wrist", np.round(d.body("wrist").xpos,3), "ee", np.round(d.site_xpos[m.site("ee_site").id],3))
# 2) 渲染 G2 两视图
ms = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","show_scene.xml"))
ds = mujoco.MjData(ms); r = mujoco.Renderer(ms, 900, 900)
def cam(look, dist, az, el):
    c=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c); c.lookat[:]=look; c.distance=dist; c.azimuth=az; c.elevation=el; return c
views=[cam([0.0,0.45,0.5],2.5,-60,-12), cam([0.0,0.45,0.55],2.2,-110,-8), cam([0.0,0.45,0.55],2.2,-20,-25)]
imgs=[]
for c in views:
    mujoco.mj_resetDataKeyframe(ms,ds,0)
    ds.qpos[:4]=q[:4]; ds.qpos[4]=0; ds.qpos[5]=0.03; ds.qpos[6]=-0.03
    ds.ctrl[:4]=q[:4]; ds.ctrl[4]=0.03; ds.ctrl[5]=-0.03
    mujoco.mj_forward(ms,ds)
    r.update_scene(ds,c); imgs.append(r.render().copy())
Image.fromarray(np.concatenate(imgs,axis=1)).save(os.path.join(ROOT,"tools","home_g2.png"))
print("rendered")
