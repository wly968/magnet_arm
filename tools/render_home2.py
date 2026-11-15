import os, numpy as np, mujoco
from PIL import Image
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
m = mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","show_scene.xml"))
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 800, 800)
c=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c)
c.lookat[:]=[0.05,0.20,0.45]; c.distance=2.9; c.azimuth=-45; c.elevation=-12
cands = {
 "E_crane_+X": [0.0,-1.0,1.4,0.0],
 "F_crane_+Y": [1.57,-1.0,1.4,0.0],
 "H_crane_NE": [0.785,-0.9,1.2,0.0],
 "A_crane_SW": [-2.2,-0.9,1.3,0.0],
}
imgs=[]
for name,q in cands.items():
    mujoco.mj_resetDataKeyframe(m,d,0)
    d.qpos[:4]=q; d.qpos[4]=0; d.qpos[5]=0.03; d.qpos[6]=-0.03
    d.ctrl[:4]=q; d.ctrl[4]=0.03; d.ctrl[5]=-0.03
    mujoco.mj_forward(m,d)
    r.update_scene(d,c)
    imgs.append(r.render().copy())
    print(name, "elbow", np.round(d.body("elbow").xpos,2), "ee", np.round(d.site_xpos[m.site("ee_site").id],2))
Image.fromarray(np.concatenate(imgs,axis=1)).save(os.path.join(ROOT,"tools","home_candidates2.png"))
print("saved")
