import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene")
for name in ("train_scene.xml","show_scene.xml"):
    m=mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models",name)); d=mujoco.MjData(m)
    kid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,"home")
    mujoco.mj_resetDataKeyframe(m,d,kid); mujoco.mj_forward(m,d)
    arm={m.body(n).id for n in ("turret","shoulder","elbow","wrist","gripper","finger_L","finger_R")}
    con=[]
    for i in range(d.ncon):
        c=d.contact[i]; b1=int(m.geom_bodyid[c.geom1]); b2=int(m.geom_bodyid[c.geom2])
        if b1 in arm or b2 in arm:
            con.append((m.geom(c.geom1).name or m.body(b1).name, m.geom(c.geom2).name or m.body(b2).name, round(float(c.dist),3)))
    print(name, "nq",m.nq,"nu",m.nu,"qpos",np.round(d.qpos[:7],3))
    print("  ee",np.round(d.site_xpos[m.site("ee_site").id],3),"contacts",con)
# env reset/step
sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv(); obs,_=env.reset(seed=0)
print("env obs shape",obs.shape,"qpos",np.round(env.data.qpos[:7],3))
a=np.zeros(6,dtype=np.float32); obs,r,term,trunc,info=env.step(a)
print("after 1 step: phase",env.phase,"r",round(float(r),3),"qpos",np.round(env.data.qpos[:7],3),"nan",bool(np.isnan(env.data.qacc).any()))
