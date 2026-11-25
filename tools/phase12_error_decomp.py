import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/saves/phase1-2.zip", device="cpu")
e=ArmEnv(); e._set_dist(0.032); e._set_angle1(0.96); e._set_still_count(1); e._set_rel(0.05)
errs=[]; rels=[]; steps=[]
for ep in range(25):
    obs,_=e.reset(); t1=t2=None
    for t in range(800):
        act,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(act)
        if e.phase==1 and t1 is None: t1=t
        if e.phase==2:
            if t2 is None: t2=t
            ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
            errs.append(tgt-ee)
        if e.phase>=3 or term or tr: break
    if t2 is not None: steps.append((t1 or 0, t2-(t1 or 0), t-(t2 or 0)))
E=np.array(errs); S=np.array(steps)
print("phase1 平均 %.0f 步 | phase2 平均 %.0f 步" % (S[:,1].mean(), S[:,2].mean()))
print("phase2 内的位置误差（目标-末端，均值）: x %+.4f  y %+.4f  z %+.4f | 模长 %.4f"
      % (E[:,0].mean(), E[:,1].mean(), E[:,2].mean(), np.linalg.norm(E,axis=1).mean()))
print("误差的 95 分位: %.4f | x 方向占比 %.0f%%" % (np.percentile(np.linalg.norm(E,axis=1),95),
      100*abs(E[:,0].mean())/max(1e-9,np.linalg.norm(E,axis=1).mean())))
