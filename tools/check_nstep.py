import numpy as np, mujoco, os
ROOT=r"D:\programming\python\mujoco_xml"
xml=os.path.join(ROOT,"scene_magnet","assets","models","train_scene.xml")
m1=mujoco.MjModel.from_xml_path(xml); d1=mujoco.MjData(m1)
m2=mujoco.MjModel.from_xml_path(xml); d2=mujoco.MjData(m2)
rng=np.random.default_rng(0)
for m,d in ((m1,d1),(m2,d2)): mujoco.mj_forward(m,d)
for i in range(200):
    ctrl=rng.uniform(-1,1,m1.nu)
    d1.ctrl[:]=ctrl; d2.ctrl[:]=ctrl
    for _ in range(5): mujoco.mj_step(m1,d1)
    mujoco.mj_step(m2,d2,5)
print("qpos max diff %.3e" % np.max(np.abs(d1.qpos-d2.qpos)))
print("qvel max diff %.3e" % np.max(np.abs(d1.qvel-d2.qvel)))
