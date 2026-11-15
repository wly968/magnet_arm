import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", device="cpu")
m3=PPO.load(r"scene_magnet/saves/phase6_home.zip", device="cpu")
for alpha in (0.0, 0.4, 0.6, 0.75):
    env=t.ArmEnv(terminate_on_phase2=False); env.smooth_alpha=alpha
    ok=0; devs=[]; jerks=[]
    for ep in range(25):
        obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
        pts=[]; qv=[]
        for i in range(400):
            if env.phase!=last: t.apply_thresholds(env, env.phase)
            last=env.phase
            pol=t.pick_policy([m1,m2,m3], env.phase)
            a,_=pol.predict(obs,deterministic=True)
            obs,r,term,tr,info=env.step(a)
            if env.phase==4:
                pts.append(env.data.site_xpos[env.ee_id].copy()); qv.append(env.data.qvel[env.arm_dof_adr].copy())
            if term or tr: break
        ok+=int(env.phase>=7)
        if len(pts)>=5:
            P=np.array(pts); A,B=P[0],P[-1]; AB=B-A; L=np.linalg.norm(AB)
            if L>1e-6:
                devs.append(float(np.max(np.linalg.norm(np.cross(AB,P-A)/L,axis=1))))
            Q=np.array(qv); acc=np.diff(Q,axis=0)/0.01
            jerks.append(float(np.max(np.linalg.norm(acc,axis=1))))
    print("alpha=%.2f: 成功 %2d/25, 偏离 %.3f m, 最大关节加速度 %.0f" % (alpha,ok,np.mean(devs) if devs else -1,np.mean(jerks) if jerks else -1))
