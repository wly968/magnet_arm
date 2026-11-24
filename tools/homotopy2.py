import os, sys, numpy as np, mujoco, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
e=ArmEnv()
# 1) 强制 y≈0.65，让 model1 抓到，记录关节角（这是"工具朝下"的真实构型）
e.spawn_y_range=(0.645,0.655)
for _ in range(20):
    obs,_=e.reset()
    e._set_dist(0.032); e._set_angle1(0.98); e._set_still_count(1); e._set_rel(0.05)
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m1.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase>=3 or term or tr: break
    if e.phase>=3:
        q_succ=e.data.qpos[e.arm_qpos_adr].copy()
        print("抓到成功构型 y=%.3f q=%s" % (e.box_pos[1], np.round(q_succ,2)))
        break
else:
    raise SystemExit("没抓到")
# 2) 以该构型为种子做 y 方向 homotopy
lo,hi=e.jnt_lo,e.jnt_hi
def ik(q0,tgt,iters=150):
    q=np.array(q0,float)
    for _ in range(iters):
        e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_kinematics(e.model,e.data)
        err=tgt-e.data.site_xpos[e.ee_id]
        jp=np.zeros((3,e.model.nv)); mujoco.mj_jacSite(e.model,e.data,jp,None,e.ee_id)
        dq=jp[:,e.arm_dof_adr].T@err; n=float(np.linalg.norm(dq))
        if n<1e-7: break
        dq*=min(1.0,0.08/n)
        q=np.clip(q+dq,lo,hi)
    e.data.qpos[e.arm_qpos_adr]=q; mujoco.mj_forward(e.model,e.data)
    ee=e.data.site_xpos[e.ee_id]
    j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    return q,float(np.linalg.norm(ee-tgt)),float(np.dot(f,[0,0,-1]))
print("\n沿 y 连续跟踪（从 0.65 出发，每步 0.01）:")
for direction in (1,-1):
    q_cur=q_succ.copy()
    ys=np.arange(0.65, 0.79, 0.01) if direction>0 else np.arange(0.65,0.50,-0.01)
    for y in ys:
        q_cur,d,a=ik(q_cur, np.array([-0.475,float(y),0.525]))
        bad = (d>0.032 or a<0.98)
        if bad or abs(y-ys[-1])<1e-9:
            print("   y=%.2f dist %.4f angle %.3f %s" % (y,d,a,"FAIL" if bad else "OK"))
