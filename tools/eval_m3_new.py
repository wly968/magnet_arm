import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from train_phase6 import make_env
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
import glob
newest=sorted(glob.glob("scene_magnet/saves/ckpt_p6/*.zip")+["scene_magnet/ppo_arm_reach.zip"], key=os.path.getmtime)[-1]
m3=PPO.load(newest, device="cpu")
print("model3:", os.path.basename(newest), "| 步数", getattr(m3,'num_timesteps',0))
print("obs", m3.observation_space.shape)
for thr in (0,1,2,3,5):
    row={}
    for det in (True, False):
        e=make_env(m1,m2)(); e.unwrapped._set_home_still(thr); ok=0; mq=9
        for ep in range(15):
            o,_=e.reset(); u=e.unwrapped
            for t in range(240):
                a,_=m3.predict(o, deterministic=det)
                o,r,term,tr,info=e.step(a)
                if u.phase>=6:
                    mq=min(mq, float(np.linalg.norm(u.data.qpos[u.arm_qpos_adr]-u.home_qpos)))
                if term or tr: break
            ok+=int(u.phase>=7)
        row[det]=(ok,mq)
    print("home_still=%d | 确定性 %2d/15 (min q_err %.3f) | 随机 %2d/15" % (thr,row[True][0],row[True][1],row[False][0]))
