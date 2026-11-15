import os, sys, numpy as np, mujoco, tempfile
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet")
SRC=os.path.join(SCENE,"assets","models","train_scene.xml")
base=open(SRC,encoding="utf-8").read()
four = base.replace('range="-1.2 1.2" damping="0.1"/>', 'range="0 0" damping="0.1"/>',1)
p4=os.path.join(tempfile.gettempdir(),"arm4dof.xml"); open(p4,"w",encoding="utf-8").write(four)

def trial(path, trials=24, iters=400):
    m=mujoco.MjModel.from_xml_path(path); d=mujoco.MjData(m)
    jids=[m.joint(n).id for n in ("joint_yaw","joint_shoulder","joint_elbow","joint_wrist","joint_tool_pitch")]
    qa=m.jnt_qposadr[jids]; da=m.jnt_dofadr[jids]; lo,hi=m.jnt_range[jids].T
    sid=m.site("ee_site").id; mujoco.mj_forward(m,d)
    rng=np.random.default_rng(0); bx=rng.uniform(-0.5,0.25); by=rng.uniform(0.45,0.85)
    target=np.array([bx,by,0.435]); jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
    good=0
    for t in range(trials):
        q=rng.uniform(lo,hi)
        for it in range(iters):
            d.qpos[qa]=np.clip(q,lo,hi); mujoco.mj_forward(m,d)
            ee=d.site_xpos[sid]; err=target-ee
            j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
            a1=float(np.dot(f,[0,0,-1])); ang_err=1.0-a1
            if np.linalg.norm(err)<0.003 and a1>0.98: break
            jp[:]=0; mujoco.mj_jacSite(m,d,jp,jr,sid)
            g=jp[:,da].T@err
            # 数值梯度 of angle1
            ga=np.zeros(len(da))
            for k,adr in enumerate(da):
                qq=np.clip(q,lo,hi).copy(); qq[k]+=1e-4
                d.qpos[qa]=np.clip(qq,lo,hi); mujoco.mj_forward(m,d)
                ee2=d.site_xpos[sid]; f2=(ee2-j3)/np.linalg.norm(ee2-j3)
                ga[k]=(float(np.dot(f2,[0,0,-1]))-a1)/1e-4
            g=g+6.0*ang_err*ga
            q=np.clip(q+0.02*np.clip(g,-5,5),lo,hi)
        d.qpos[qa]=np.clip(q,lo,hi); d.qvel[:]=0; mujoco.mj_forward(m,d)
        ee=d.site_xpos[sid]; j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        a1=float(np.dot(f,[0,0,-1]))
        if np.linalg.norm(ee-target)<0.005 and a1>0.98: good+=1
    return good,trials

for name,path in (("4 自由度(冻结 tool_pitch)",p4),("5 自由度(新增 tool_pitch)",SRC)):
    g,n=trial(path)
    print("%s -> 联合 IK 成功 %d/%d" % (name,g,n))
