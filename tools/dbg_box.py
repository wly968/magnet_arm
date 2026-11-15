import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
e=ArmEnv()
o,_=e.reset(seed=0)
print("after reset: box_pos(attr)=", np.round(e.box_pos,4))
print("data.xpos[cube]=", np.round(e.data.xpos[e.box_body_id],4), "body ", e.model.body(e.box_body_id).name)
print("cube qpos=", np.round(e.data.qpos[e.box_qpos_adr:e.box_qpos_adr+3],4))
print("ee=", np.round(e.data.site_xpos[e.ee_id],4))
# 手动复算
tgt=np.array([e.box_pos[0],e.box_pos[1],e.box_pos[2]+0.15])
print("target(attr)=", np.round(tgt,4), "dist=", round(float(np.linalg.norm(e.data.site_xpos[e.ee_id]-tgt)),4))
o,r,term,tr,info=e.step(np.zeros(4))
print("after 1 step: reward=", round(float(r),4))
print("box_pos(attr)=", np.round(e.box_pos,4))
tgt2=np.array([e.box_pos[0],e.box_pos[1],e.box_pos[2]+0.15])
print("target=", np.round(tgt2,4), "ee=", np.round(e.data.site_xpos[e.ee_id],4))
print("dist=", round(float(np.linalg.norm(e.data.site_xpos[e.ee_id]-tgt2)),4))
