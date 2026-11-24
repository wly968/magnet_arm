import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')
e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
obs,_=e.reset(); best=None
for t in range(e.max_steps+e.max_wait_steps+10):
    a,_=m.predict(obs,deterministic=True)
    obs,r,term,tr,info=e.step(a)
    if e.phase==2:
        ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
        d=float(np.linalg.norm(ee-tgt))
        if best is None or d<best[0]: best=(d, e.data.qpos[e.arm_qpos_adr].copy(), tgt.copy())
    if e.phase>=3 or term or tr: break
d,q0,tgt=best
md=e.model; dd=e.data
lo,hi=md.jnt_range[e.arm_joints].T

def fk(q):
    dd.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(md,dd)
    ee=dd.site_xpos[e.ee_id].copy(); j3=dd.xanchor[e.arm_joints[3]]
    f=(ee-j3)/np.linalg.norm(ee-j3)
    return ee, float(np.dot(f,[0,0,-1]))

def refine(q, free, iters=300, lr=0.5):
    jp=np.zeros((3,md.nv)); q=q.copy()
    for _ in range(iters):
        ee,_=fk(q); err=tgt-ee
        jp[:]=0; mujoco.mj_jacSite(md,dd,jp,None,e.ee_id)
        dq=jp[:,e.arm_dof_adr].T@err
        q[free]=np.clip(q[free]+lr*dq[free], lo[free], hi[free])
    ee,ang=fk(q)
    return q, float(np.linalg.norm(ee-tgt)), ang

print("从策略最近点出发，固定腕关节值、用其余 3 个关节重新对位置：")
for q3 in (0.5, 0.8, 1.0, 1.2, 1.5):
    q=q0.copy(); q[3]=q3
    q2,d2,a2=refine(q, [0,1,2])
    print("  q3=%+.2f -> dist %.3f, angle %.3f" % (q3, d2, a2))
