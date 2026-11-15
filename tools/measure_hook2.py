import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO

def measure(m2, n=20):
    env=t.ArmEnv(terminate_on_phase2=False)
    m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", env=env, device="cpu")
    devs=[]; jerks=[]; ok=0
    for ep in range(n):
        obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
        pts=[]; qv=[]
        for i in range(400):
            if env.phase!=last: t.apply_thresholds(env, env.phase)
            last=env.phase
            pol=m1 if env.phase<3 else m2
            a,_=pol.predict(obs,deterministic=True)
            obs,r,term,tr,info=env.step(a)
            if env.phase==4:
                pts.append(env.data.site_xpos[env.ee_id].copy())
                qv.append(env.data.qvel[env.arm_dof_adr].copy())
            if term or tr: break
        ok+=int(env.phase>=6)
        if len(pts)>=5:
            P=np.array(pts); A,B=P[0],P[-1]; AB=B-A; L=np.linalg.norm(AB)
            if L>1e-6:
                nrm=np.cross(AB,P-A)/L
                devs.append(float(np.max(np.linalg.norm(nrm,axis=1))))
            Q=np.array(qv)
            if len(Q)>=3:
                acc=np.diff(Q,axis=0)/0.01
                jerks.append(float(np.max(np.linalg.norm(acc,axis=1))))
    return ok,(np.mean(devs) if devs else -1),(np.mean(jerks) if jerks else -1)

for name in ("saves/phase34_retract.zip","saves/phase345_smooth.zip"):
    m2=PPO.load(os.path.join(r"scene_magnet",name), device="cpu")
    ok,dev,jerk=measure(m2)
    print("%-32s 成功 %2d/20, 偏离直线 %.3f m, 最大关节加速度 %.0f" % (name,ok,dev,jerk))
