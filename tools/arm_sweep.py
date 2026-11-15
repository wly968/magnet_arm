import os, sys, re, numpy as np, mujoco, tempfile
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet")
SRC=os.path.join(SCENE,"assets","models","train_scene.xml")
base=open(SRC,encoding="utf-8").read()

def variant(L1,L2):
    s=base
    s=s.replace('fromto="0 0 0 0.70 0 0"','fromto="0 0 0 %.2f 0 0"'%L1)
    s=s.replace('fromto="0 0 0 0.55 0 0"','fromto="0 0 0 %.2f 0 0"'%L2)
    s=s.replace('<body name="forearm" pos="0.70 0 0">','<body name="forearm" pos="%.2f 0 0">'%L1)
    s=s.replace('<body name="wrist" pos="0.55 0 0">','<body name="wrist" pos="%.2f 0 0">'%L2)
    path=os.path.join(tempfile.gettempdir(),"arm_%d_%d.xml"%(int(L1*100),int(L2*100)))
    open(path,"w",encoding="utf-8").write(s)
    return path

def eval_variant(path, trials=24):
    m=mujoco.MjModel.from_xml_path(path); d=mujoco.MjData(m)
    jn=("joint_yaw","joint_shoulder","joint_elbow","joint_wrist")
    jids=[m.joint(n).id for n in jn]
    qa=m.jnt_qposadr[jids]; da=m.jnt_dofadr[jids]
    lo,hi=m.jnt_range[jids].T
    sid=m.site("ee_site").id
    mujoco.mj_forward(m,d)
    # 目标：A 带随机方块上方的抓取点
    rng=np.random.default_rng(0)
    bx=rng.uniform(-0.5,0.25); by=rng.uniform(0.45,0.85)
    target=np.array([bx,by,0.435])
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
    good=0; best=0
    for t in range(trials):
        q=rng.uniform(lo,hi)
        for it in range(4000):
            d.qpos[qa]=np.clip(q,lo,hi); mujoco.mj_forward(m,d)
            err=target-d.site_xpos[sid]
            if np.linalg.norm(err)<0.003: break
            jp[:]=0; mujoco.mj_jacSite(m,d,jp,jr,sid)
            q=np.clip(q+0.05*np.clip(jp[:,da].T@err,-1,1),lo,hi)
        d.qpos[qa]=np.clip(q,lo,hi); d.qvel[:]=0; mujoco.mj_forward(m,d)
        ee=d.site_xpos[sid]
        if np.linalg.norm(ee-target)>0.006: continue
        j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        a1=float(np.dot(f,[0,0,-1])); best=max(best,a1)
        if a1>0.98: good+=1
    return good, trials, best

print("L1    L2   | 同时满足(dist<3mm & angle>0.98) 次数 | 最高 angle1")
for L1,L2 in ((0.70,0.55),(0.65,0.50),(0.60,0.50),(0.75,0.50),(0.65,0.60),(0.60,0.55),(0.55,0.45)):
    g,n,b=eval_variant(variant(L1,L2))
    print("%.2f  %.2f | %2d/%2d | %.4f" % (L1,L2,g,n,b))
