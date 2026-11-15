import os, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
for name in ("train_scene.xml","show_scene.xml"):
    m=mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models",name))
    arm=0.0
    for b in ("turret","shoulder","elbow","wrist","gripper","finger_L","finger_R"):
        arm += float(m.body_mass[m.body(b).id])
    print(name, "nu", m.nu, "arm moving mass", round(arm,2), "kg")
    for nm in ("yaw_pos","shoulder_pos","elbow_pos","wrist_pos"):
        i=m.actuator(nm).id
        print(f"  {nm}: kp={m.actuator_gainprm[i,0]:.0f} forcerange={m.actuator_forcerange[i]}")
