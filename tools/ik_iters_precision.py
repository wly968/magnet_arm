import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
tgt=np.array([-0.40,0.65,0.525])
for iters in (5,10,20,40):
    e.ik_iters=iters
    ds=[]
    for y in np.arange(0.55,0.72,0.02):
        t=np.array([-0.40,float(y),0.525])
        q=e._ik_target_q(t)
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
        ds.append(float(np.linalg.norm(e.data.site_xpos[e.ee_id]-t)))
    print("ik_iters=%2d -> IK 残差 中位 %.2f mm, 最大 %.2f mm" % (iters, 1000*np.median(ds), 1000*max(ds)))
