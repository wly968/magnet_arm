import os, numpy as np, mujoco
from PIL import Image
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
m = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","show_scene.xml"))
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 800, 800)
c=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c)
c.lookat[:]=[0.0,0.45,0.45]; c.distance=2.5; c.azimuth=-50; c.elevation=-10
cands = {
 "F_yaw+Y_crane":   [1.57,-1.0,1.4,0.0],
 "G_yaw+Y_tool49":  [1.57,-1.154,2.0,0.0],
 "G2_yaw+Y_vert":   [1.57,-1.154,2.0,0.725],
 "G3_yaw+X_vert":   [0.0,-1.154,2.0,0.725],
}
imgs=[]
for name,q in cands.items():
    mujoco.mj_resetDataKeyframe(m,d,0)
    d.qpos[:4]=q; d.qpos[4]=0; d.qpos[5]=0.03; d.qpos[6]=-0.03
    d.ctrl[:4]=q; d.ctrl[4]=0.03; d.ctrl[5]=-0.03
    mujoco.mj_forward(m,d)
    r.update_scene(d,c)
    imgs.append(r.render().copy())
    print(name, "elbow", np.round(d.body("elbow").xpos,2), "wrist", np.round(d.body("wrist").xpos,2), "ee", np.round(d.site_xpos[m.site("ee_site").id],2))
Image.fromarray(np.concatenate(imgs,axis=1)).save(os.path.join(ROOT,"tools","home_candidates3.png"))
print("saved")
