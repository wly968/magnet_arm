import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet")
m=mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","train_scene.xml"))
d=mujoco.MjData(m)
jn=("joint_yaw","joint_shoulder","joint_elbow","joint_wrist")
jids=[m.joint(n).id for n in jn]; qa=m.jnt_qposadr[jids]; da=m.jnt_dofadr[jids]
lo,hi=m.jnt_range[jids].T; sid=m.site("ee_site").id
mujoco.mj_forward(m,d)

rng=np.random.default_rng(0)
bx=rng.uniform(-0.5,0.25); by=rng.uniform(0.45,0.85)
target=np.array([bx,by,0.435])
jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
best=(-1,None); rows=[]
for w in np.linspace(lo[3],hi[3],41):
    q=np.array([np.arctan2(by,bx),-1.2,2.0,w])
    for it in range(4000):
        d.qpos[qa]=np.clip(q,lo,hi); mujoco.mj_forward(m,d)
        err=target-d.site_xpos[sid]
        if np.linalg.norm(err)<0.002: break
        jp[:]=0; mujoco.mj_jacSite(m,d,jp,jr,sid)
        J=jp[:,da].copy(); J[:,3]=0.0            # 固定腕关节，只解前三个
        q[:3]=np.clip(q[:3]+0.05*np.clip(J[:,:3].T@err,-1,1),lo[:3],hi[:3])
    d.qpos[qa]=np.clip(q,lo,hi); d.qvel[:]=0; mujoco.mj_forward(m,d)
    ee=d.site_xpos[sid]; dd=float(np.linalg.norm(ee-target))
    j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    a1=float(np.dot(f,[0,0,-1]))
    if dd<0.005:
        rows.append((w,a1,dd))
        if a1>best[0]: best=(a1,(w,dd,q.copy()))
print("固定腕关节扫描：可行点数 %d/%d" % (len(rows),41))
for w,a1,dd in rows[::max(1,len(rows)//8)]:
    print("  wrist=%.2f angle1=%.4f dist=%.4f" % (w,a1,dd))
print("最佳 angle1 %.4f (wrist=%.2f, dist=%.4f)" % (best[0], best[1][0], best[1][1]))
