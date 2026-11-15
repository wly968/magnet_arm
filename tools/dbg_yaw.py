import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
env=ArmEnv(); o,_=env.reset(seed=3)
env.phase=3
# 直接命令 yaw 到 -0.6，其余保持 home
a=np.zeros(4); a[0]=(-0.6-env.home_ctrl[0])/env.ctrl_scale[0]
print("action",a,"ctrl will be",env.home_ctrl+a*env.ctrl_scale)
for t in range(10):
    o,r,term,tr,info=env.step(a)
    print("t",t,"ctrl0 %.2f"%env.data.ctrl[0],"yaw %.3f"%env.data.qpos[env.arm_qpos_adr[0]],
          "ee",np.round(env.data.site_xpos[env.ee_id],2),"ncon",env.data.ncon)
