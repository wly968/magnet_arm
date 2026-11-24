import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from train_phase6 import make_env
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
e=make_env(m1,m2)(); o,_=e.reset(); u=e.unwrapped
print("phase", u.phase, "| 起始 q_err %.3f" % float(np.linalg.norm(u.data.qpos[u.arm_qpos_adr]-u.home_qpos)))
acts=[]
for t in range(200):
    a,_=m3.predict(o, deterministic=True); acts.append(np.abs(a).mean())
    o,r,term,tr,info=e.step(a)
    if t%40==0 or term or tr:
        print("  t%3d q_err %.3f |a|%.2f collected=%s" % (t, float(np.linalg.norm(u.data.qpos[u.arm_qpos_adr]-u.home_qpos)), np.mean(acts[-1:]), u.collected))
    if term or tr: break
print("结束 phase", u.phase, "| 平均 |a| %.2f" % np.mean(acts))
# 对照：零动作
o,_=e.reset(); u=e.unwrapped
for t in range(200):
    o,r,term,tr,info=e.step(np.zeros(4))
    if term or tr: break
print("零动作 200 步后: q_err %.3f | phase %d" % (float(np.linalg.norm(u.data.qpos[u.arm_qpos_adr]-u.home_qpos)), u.phase))
