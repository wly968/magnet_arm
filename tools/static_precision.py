import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); e.ik_iters=20
print("=== 静态目标：把 ctrl 直接设成 IK 解，看伺服最终停在哪 ===")
for x,y in ((-0.45,0.65),(-0.40,0.60),(-0.40,0.70)):
    tgt=np.array([x,y,0.525])
    q=e._ik_target_q(tgt)
    e.data.qpos[e.arm_qpos_adr]=e.home_qpos; e.data.qvel[:]=0; mujoco.mj_forward(e.model,e.data)
    for _ in range(200):
        e.data.ctrl[:]=q
        mujoco.mj_step(e.model,e.data,5)
    ee=e.data.site_xpos[e.ee_id]
    j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    print("  目标(%.2f,%.2f) -> 稳态误差 %.2f mm | 工具角 %.4f"
          % (x,y,1000*float(np.linalg.norm(ee-tgt)), float(np.dot(f,[0,0,-1]))))
