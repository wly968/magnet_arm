import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
# 让手臂动起来
a=np.array([0.3,0.3,0.0,0.0])
for _ in range(5): e.step(a)
ee1=e.data.site_xpos[e.ee_id].copy()
v_site=e.data.site_xvelp[e.ee_id].copy()
for _ in range(1): e.step(a)
ee2=e.data.site_xpos[e.ee_id].copy()
v_fd=(ee2-ee1)/0.01
res=np.zeros(6)
mujoco.mj_objectVelocity(e.model, e.data, mujoco.mjtObj.mjOBJ_SITE, e.ee_id, res, 0)
print("site_xvelp      ", np.round(v_site,3))
print("有限差分(1步)    ", np.round(v_fd,3))
print("mj_objectVelocity", np.round(res,3), "  # 前3=角速度, 后3=线速度")
print("site_xvelp 字段存在:", hasattr(e.data,'site_xvelp'), "| site_xvelr:", hasattr(e.data,'site_xvelr'))
