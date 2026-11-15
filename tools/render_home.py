import os, numpy as np, mujoco
from PIL import Image
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
m = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","show_scene.xml"))
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 800, 800)
def cam(lookat, dist, az, el):
    c=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c)
    c.lookat[:]=lookat; c.distance=dist; c.azimuth=az; c.elevation=el; return c
cams = {
 "A_crane_SW": ([-2.2,-0.9,1.3,0.0], cam([-0.1,-0.2,0.5],2.4,-45,-8)),
 "B_high_SW":  ([-2.2,-1.2,1.6,0.0], cam([-0.1,-0.2,0.5],2.4,-45,-8)),
 "C_crane_S":  ([-1.57,-1.0,1.4,0.0], cam([-0.1,-0.2,0.5],2.4,-45,-8)),
 "D_reach_SW": ([-2.4,-0.5,0.9,0.0], cam([-0.1,-0.2,0.5],2.4,-45,-8)),
}
imgs=[]
for name,(q,c) in cams.items():
    mujoco.mj_resetDataKeyframe(m,d,0)
    d.qpos[:4]=q; d.qpos[4]=0; d.qpos[5]=0.03; d.qpos[6]=-0.03
    d.ctrl[:4]=q; d.ctrl[4]=0.03; d.ctrl[5]=-0.03
    mujoco.mj_forward(m,d)
    r.update_scene(d,c)
    imgs.append(r.render().copy())
    print(name, "elbow", np.round(d.body("elbow").xpos,2), "ee", np.round(d.site_xpos[m.site("ee_site").id],2))
sheet=np.concatenate(imgs,axis=1)
Image.fromarray(sheet).save(os.path.join(ROOT,"tools","home_candidates.png"))
print("saved", sheet.shape)
