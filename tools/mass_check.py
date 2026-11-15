import os, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
m=mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","train_scene.xml"))
print("nbody",m.nbody)
tot=0
for i in range(m.nbody):
    name=m.body(i).name; mass=float(m.body_mass[i])
    if mass>0:
        print(f"  {name:14s} mass={mass:.4f} kg  inertia={np.round(m.body_inertia[i],5)}")
        tot+=mass
print("total mass",round(tot,3))
# arm geoms volumes/masses
for i in range(m.ngeom):
    b=m.body(m.geom_bodyid[i]).name
    if b in ("turret","shoulder","elbow","wrist","gripper","finger_L","finger_R"):
        print(f"  geom {m.geom(i).name or '?':12s} body={b:9s} type={int(m.geom_type[i])} size={np.round(m.geom_size[i],4)} mass={float(m.geom_mass[i]) if hasattr(m,'geom_mass') else '?'}")
