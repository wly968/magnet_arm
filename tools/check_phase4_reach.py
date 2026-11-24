import sys, numpy as np, mujoco
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); e.reset()
# phase 4 目标可达性检查（用 env 自带的 IK）
e.phase=4
tgt=e._target()
q=e._ik_target_q(tgt)
print("phase4 目标", np.round(tgt,3))
e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
ee=e.data.site_xpos[e.ee_id].copy()
j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
print("IK 后 ee", np.round(ee,3), "| dist %.4f | angle %.3f"
      % (float(np.linalg.norm(ee-tgt)), float(np.dot(f,[0,0,-1]))))
print("关节角", np.round(q,3), "| 限位", np.round(e.jnt_lo,2), np.round(e.jnt_hi,2))
