import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset(); e.ik_iters=20
kp=e.model.actuator_gainprm[:,0].copy()
print("执行器 kp:", kp)
def static_error(tgt, ff):
    q=e._ik_target_q(tgt)
    e.data.qpos[e.arm_qpos_adr]=q; e.data.qvel[:]=0; mujoco.mj_forward(e.model,e.data)
    tau=e.data.qfrc_bias[e.arm_dof_adr].copy()
    ctrl = q + (tau/kp if ff else 0.0)
    e.data.qpos[e.arm_qpos_adr]=e.home_qpos; e.data.qvel[:]=0; mujoco.mj_forward(e.model,e.data)
    for _ in range(300):
        e.data.ctrl[:]=ctrl
        mujoco.mj_step(e.model,e.data,5)
    return 1000*float(np.linalg.norm(e.data.site_xpos[e.ee_id]-tgt))
for x,y in ((-0.45,0.65),(-0.40,0.60),(-0.40,0.70)):
    t=np.array([x,y,0.525])
    print("  目标(%.2f,%.2f): 无补偿 %.1f mm | 重力前馈 %s mm" %
          (x,y,static_error(t,False), ("%.2f"%static_error(t,True))))
