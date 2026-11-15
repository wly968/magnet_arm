import os, sys, numpy as np, mujoco, tempfile
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet")
base=open(os.path.join(SCENE,"assets","models","train_scene.xml"),encoding="utf-8").read()
patched=base.replace('<body name="gripper" pos="0.10 0 0">',
 '<body name="gripper" pos="0.10 0 0">\n                <joint name="joint_wrist2" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.1"/>')
p=os.path.join(tempfile.gettempdir(),"arm5b.xml"); open(p,"w",encoding="utf-8").write(patched)

m=mujoco.MjModel.from_xml_path(p); d=mujoco.MjData(m)
jn=("joint_yaw","joint_shoulder","joint_elbow","joint_wrist","joint_wrist2")
jids=[m.joint(n).id for n in jn]; qa=m.jnt_qposadr[jids]; da=m.jnt_dofadr[jids]
lo,hi=m.jnt_range[jids].T; sid=m.site("ee_site").id
rng=np.random.default_rng(0); bx=rng.uniform(-0.5,0.25); by=rng.uniform(0.45,0.85)
target=np.array([bx,by,0.435]); jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
good=0; tot=0; best=0
for w1 in np.linspace(lo[3],hi[3],9):
    for w2 in np.linspace(lo[4],hi[4],9):
        q=np.array([np.arctan2(by,bx),-1.2,2.0,w1,w2])
        for it in range(3000):
            d.qpos[qa]=np.clip(q,lo,hi); mujoco.mj_forward(m,d)
            err=target-d.site_xpos[sid]
            if np.linalg.norm(err)<0.002: break
            jp[:]=0; mujoco.mj_jacSite(m,d,jp,jr,sid)
            J=jp[:,da].copy(); J[:,3]=0; J[:,4]=0
            q[:3]=np.clip(q[:3]+0.05*np.clip(J[:,:3].T@err,-1,1),lo[:3],hi[:3])
        d.qpos[qa]=np.clip(q,lo,hi); d.qvel[:]=0; mujoco.mj_forward(m,d)
        ee=d.site_xpos[sid]
        if np.linalg.norm(ee-target)>0.005: continue
        tot+=1
        j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        a1=float(np.dot(f,[0,0,-1])); best=max(best,a1)
        if a1>0.98: good+=1
print("5 自由度 2D 扫描: 满足 angle>0.98 的采样 %d/%d (%.0f%%), 最高 %.4f" % (good,tot,100*good/max(tot,1),best))
