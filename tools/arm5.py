import os, sys, numpy as np, mujoco, tempfile
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet")
SRC=os.path.join(SCENE,"assets","models","train_scene.xml")
base=open(SRC,encoding="utf-8").read()

# 在 gripper body 里加一个腕部俯仰关节（仅用于可行性验证，不进环境）
patched = base.replace('<body name="gripper" pos="0.10 0 0">',
    '<body name="gripper" pos="0.10 0 0">\n                <joint name="joint_wrist2" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.1"/>')
path=os.path.join(tempfile.gettempdir(),"arm5.xml")
open(path,"w",encoding="utf-8").write(patched)

def eval5(path, trials=24):
    m=mujoco.MjModel.from_xml_path(path); d=mujoco.MjData(m)
    jn=("joint_yaw","joint_shoulder","joint_elbow","joint_wrist","joint_wrist2")
    jids=[m.joint(n).id for n in jn]
    qa=m.jnt_qposadr[jids]; da=m.jnt_dofadr[jids]; lo,hi=m.jnt_range[jids].T
    sid=m.site("ee_site").id; mujoco.mj_forward(m,d)
    rng=np.random.default_rng(0); bx=rng.uniform(-0.5,0.25); by=rng.uniform(0.45,0.85)
    target=np.array([bx,by,0.435]); jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
    good=0; best=0; dists=[]
    for t in range(trials):
        q=rng.uniform(lo,hi)
        for it in range(4000):
            d.qpos[qa]=np.clip(q,lo,hi); mujoco.mj_forward(m,d)
            err=target-d.site_xpos[sid]
            if np.linalg.norm(err)<0.003: break
            jp[:]=0; mujoco.mj_jacSite(m,d,jp,jr,sid)
            q=np.clip(q+0.05*np.clip(jp[:,da].T@err,-1,1),lo,hi)
        d.qpos[qa]=np.clip(q,lo,hi); d.qvel[:]=0; mujoco.mj_forward(m,d)
        ee=d.site_xpos[sid]; dd=float(np.linalg.norm(ee-target))
        if dd>0.006: continue
        j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
        a1=float(np.dot(f,[0,0,-1])); best=max(best,a1); dists.append(dd)
        if a1>0.98: good+=1
    return good,trials,best,np.mean(dists) if dists else -1
g,n,b,md=eval5(path)
print("5 自由度(增加 wrist2): 同时满足 %d/%d, 最高 angle1 %.4f, 平均 dist %.4f" % (g,n,b,md))
