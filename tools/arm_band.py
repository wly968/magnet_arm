import os, sys, numpy as np, mujoco, tempfile
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet")
base=open(os.path.join(SCENE,"assets","models","train_scene.xml"),encoding="utf-8").read()

def make(L1,L2):
    s=base
    s=s.replace('fromto="0 0 0 0.70 0 0"','fromto="0 0 0 %.2f 0 0"'%L1)
    s=s.replace('fromto="0 0 0 0.55 0 0"','fromto="0 0 0 %.2f 0 0"'%L2)
    s=s.replace('<body name="forearm" pos="0.70 0 0">','<body name="forearm" pos="%.2f 0 0">'%L1)
    s=s.replace('<body name="wrist" pos="0.55 0 0">','<body name="wrist" pos="%.2f 0 0">'%L2)
    p=os.path.join(tempfile.gettempdir(),"a_%d_%d.xml"%(int(L1*100),int(L2*100)))
    open(p,"w",encoding="utf-8").write(s); return p

def band(L1,L2):
    m=mujoco.MjModel.from_xml_path(make(L1,L2)); d=mujoco.MjData(m)
    jn=("joint_yaw","joint_shoulder","joint_elbow","joint_wrist")
    jids=[m.joint(n).id for n in jn]; qa=m.jnt_qposadr[jids]; da=m.jnt_dofadr[jids]
    lo,hi=m.jnt_range[jids].T; sid=m.site("ee_site").id
    rng=np.random.default_rng(0); bx=rng.uniform(-0.5,0.25); by=rng.uniform(0.45,0.85)
    target=np.array([bx,by,0.435])
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
    ws=np.linspace(lo[3],hi[3],81); good=[]; best=0
    for w in ws:
        q=np.array([np.arctan2(by,bx),-1.2,2.0,w])
        for it in range(3000):
            d.qpos[qa]=np.clip(q,lo,hi); mujoco.mj_forward(m,d)
            err=target-d.site_xpos[sid]
            if np.linalg.norm(err)<0.002: break
            jp[:]=0; mujoco.mj_jacSite(m,d,jp,jr,sid)
            J=jp[:,da].copy(); J[:,3]=0
            q[:3]=np.clip(q[:3]+0.05*np.clip(J[:,:3].T@err,-1,1),lo[:3],hi[:3])
        d.qpos[qa]=np.clip(q,lo,hi); d.qvel[:]=0; mujoco.mj_forward(m,d)
        ee=d.site_xpos[sid]
        if np.linalg.norm(ee-target)>0.005: continue
        j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        a1=float(np.dot(f,[0,0,-1])); best=max(best,a1)
        if a1>0.98: good.append(w)
    width=(max(good)-min(good)) if good else 0.0
    return len(good), 81, width, best

print("L1    L2   | angle>0.98 的腕关节采样点 | 带宽(rad) | 最高 angle1")
for L1,L2 in ((0.70,0.55),(0.65,0.50),(0.60,0.50),(0.75,0.50),(0.65,0.60),(0.60,0.55),(0.55,0.45),(0.80,0.60)):
    n,tot,w,b=band(L1,L2)
    print("%.2f  %.2f | %2d/%2d | %.3f | %.4f" % (L1,L2,n,tot,w,b))
