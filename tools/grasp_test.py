import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv(); m=env.model; d=env.data
mujoco.mj_resetDataKeyframe(m,d,0)
a=env.box_qpos_adr; d.qpos[a:a+3]=[0,0.6,0.375]; d.qpos[a+3:a+7]=[1,0,0,0]
d.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([0,0.6,0.375])
q=np.array([1.57,-0.95,2.16,0.36,-3.14,0.03,-0.03])
d.qpos[:7]=q; d.ctrl[:7]=q; d.ctrl[5]=0.0; d.ctrl[6]=0.0
mujoco.mj_forward(m,d)
print("start ee",np.round(d.site_xpos[env.ee_id],3),"finger q",round(float(d.qpos[env.finger_qpos_adr]),4))
for i in range(120):
    mujoco.mj_step(m,d)
    d.ctrl[:5]=q[:5]; d.ctrl[5]=0.0; d.ctrl[6]=0.0
print("after close: finger q",round(float(d.qpos[env.finger_qpos_adr]),4),"ee",np.round(d.site_xpos[env.ee_id],3))
# contacts
for i in range(d.ncon):
    c=d.contact[i]
    n1=m.geom(c.geom1).name or "?"; n2=m.geom(c.geom2).name or "?"
    if "finger" in (n1+n2) or "cube" in (n1+n2):
        f=np.zeros(6); mujoco.mj_contactForce(m,d,i,f)
        print("contact",n1,n2,"dist",round(float(c.dist),4),"force",np.round(f[:3],2))
