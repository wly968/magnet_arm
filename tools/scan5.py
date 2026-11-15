import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet")
p=os.path.join(SCENE,"assets","models","train_scene.xml")
m=mujoco.MjModel.from_xml_path(p); d=mujoco.MjData(m)
jn=("joint_yaw","joint_shoulder","joint_elbow","joint_wrist","joint_tool_pitch")
jids=[m.joint(n).id for n in jn]; qa=m.jnt_qposadr[jids]; da=m.jnt_dofadr[jids]
lo,hi=m.jnt_range[jids].T; sid=m.site("ee_site").id
rng=np.random.default_rng(0); bx=rng.uniform(-0.5,0.25); by=rng.uniform(0.45,0.85)
target=np.array([bx,by,0.435]); jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
mujoco.mj_forward(m,d)

def solve(q):
    for it in range(1500):
        d.qpos[qa]=np.clip(q,lo,hi); mujoco.mj_forward(m,d)
        err=target-d.site_xpos[sid]
        if np.linalg.norm(err)<0.002: break
        jp[:]=0; mujoco.mj_jacSite(m,d,jp,jr,sid)
        J=jp[:,da].copy(); J[:,3]=0; J[:,4]=0
        q[:3]=np.clip(q[:3]+0.05*np.clip(J[:,:3].T@err,-1,1),lo[:3],hi[:3])
    d.qpos[qa]=np.clip(q,lo,hi); d.qvel[:]=0; mujoco.mj_forward(m,d)
    ee=d.site_xpos[sid]; dd=float(np.linalg.norm(ee-target))
    j3=d.xanchor[jids[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
    return dd, float(np.dot(f,[0,0,-1]))

ws=np.linspace(lo[3],hi[3],21)
feasible_wrist=[]; best_overall=0
for w in ws:
    ok=False
    for pitch in np.linspace(lo[4],hi[4],11):
        q=np.array([np.arctan2(by,bx),-1.2,2.0,w,pitch])
        dd,a1=solve(q)
        if dd<0.005:
            best_overall=max(best_overall,a1)
            if a1>0.98: ok=True
    if ok: feasible_wrist.append(w)
print("5 自由度: 有解的腕关节采样 %d/%d" % (len(feasible_wrist), len(ws)))
print("可满足的腕关节范围:", (round(min(feasible_wrist),2), round(max(feasible_wrist),2)) if feasible_wrist else None)
print("扫描中最高 angle1 %.4f" % best_overall)
