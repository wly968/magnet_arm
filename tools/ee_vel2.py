import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
a=np.array([0.3,0.3,0.0,0.0])
for _ in range(5): e.step(a)
ee1=e.data.site_xpos[e.ee_id].copy()
res=np.zeros(6)
mujoco.mj_objectVelocity(e.model, e.data, mujoco.mjtObj.mjOBJ_SITE, e.ee_id, res, 0)
e.step(a)
ee2=e.data.site_xpos[e.ee_id].copy()
v_fd=(ee2-ee1)/0.01
print("mj_objectVelocity res =", np.round(res,3))
print("有限差分(1控制步)     =", np.round(v_fd,3))
print("res[3:] 与差分差", np.round(res[3:]-v_fd,3))
print("data.cvel 形状", e.data.cvel.shape, "| 有 site_xvelp?", hasattr(e.data,'site_xvelp'))
