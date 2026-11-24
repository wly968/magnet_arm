import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); m,d=e.model,e.data
qadr,dadr=e.arm_qpos_adr,e.arm_dof_adr
lo,hi=m.jnt_range[e.arm_joints].T
jp=np.zeros((3,m.nv)); dt=0.01

def track(k, steps=600):
    e.reset(); e._set_dist(0.03); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.05)
    rels=[]; dists=[]; end=0
    for t in range(steps):
        if e.phase==0:
            a=(e.home_qpos-e.home_ctrl)/e.ctrl_scale
        else:
            target=e._target(); ee=d.site_xpos[e.ee_id].copy(); err=target-ee
            v_des=(e.box_vel + k*err) if e.phase>=2 else 6.0*err
            jp[:]=0; mujoco.mj_jacSite(m,d,jp,None,e.ee_id)
            dq=jp[:,dadr].T@(v_des*dt)
            q_des=np.clip(d.qpos[qadr]+dq, lo, hi)
            a=(q_des-e.home_ctrl)/e.ctrl_scale
        o,r,term,tr,info=e.step(np.clip(a,-1,1))
        if e.phase==2:
            rels.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
            dists.append(float(np.linalg.norm(d.site_xpos[e.ee_id]-(e.box_pos+e.grasp_offset))))
        if e.phase>=3 or term or tr: end=e.phase; break
    return np.array(rels), np.array(dists), (end or e.phase)

for k in (4.0, 8.0, 16.0):
    rels,dists,ph=track(k)
    if len(rels):
        print("k=%4.1f: phase2 帧 %3d | rel 中位 %.3f | rel<0.05 %4.1f%% | dist 中位 %.3f | 结束 phase %d"
              % (k,len(rels),np.median(rels),100*(rels<0.05).mean(),np.median(dists),ph))
    else:
        print("k=%4.1f: 未进入 phase2, 结束 phase %d" % (k,ph))
