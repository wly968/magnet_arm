import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_fine import ArmFineEnv
e=ArmFineEnv(); e.reset(); e.use_gravity_ff=True; e.ik_iters=20
kp=e.actuator_kp
def static(tgt):
    q=e._ik_target_q(tgt)
    qc=e._gravity_ff(q)
    e.data.qpos[e.arm_qpos_adr]=e.home_qpos; e.data.qvel[:]=0; mujoco.mj_forward(e.model,e.data)
    for _ in range(400):
        e.data.ctrl[:]=qc
        mujoco.mj_step(e.model,e.data,5)
    ee=e.data.site_xpos[e.ee_id]
    j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    return 1000*float(np.linalg.norm(ee-tgt)), float(np.dot(f,[0,0,-1]))
print("抓取点（方块上方 6 cm，z=0.435）静态可达性:")
for y in (0.58, 0.65, 0.72):
    for z in (0.435, 0.452):
        d,a=static(np.array([-0.40,y,z]))
        print("  y=%.2f z=%.3f -> 静差 %.2f mm | angle %.3f" % (y,z,d,a))
