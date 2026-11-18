import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
from stable_baselines3 import PPO
m1=PPO.load(r"scene_magnet/ppo_arm_reach.zip", device="cpu")
m2=PPO.load(r"scene_magnet/saves/phase34_retract.zip", device="cpu")
m3=PPO.load(r"scene_magnet/saves/phase6_home.zip", device="cpu")
for alpha in (0.0, 0.5):
    env=t.ArmEnv(); env.smooth_alpha=alpha
    dips=[]; ok=0
    for ep in range(25):
        obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
        zs=[]
        for i in range(400):
            if env.phase!=last:
                t.apply_thresholds(env, env.phase); env.phase_entry_steps=6
            last=env.phase
            pol=t.pick_policy([m1,m2,m3], env.phase)
            a,_=pol.predict(obs,deterministic=True)
            obs,r,term,tr,info=env.step(a)
            if env.phase==4: zs.append(float(env.data.site_xpos[env.ee_id][2]))
            if term or tr: break
        ok+=int(env.phase>=7)
        if len(zs)>=5:
            dips.append(float(min(zs) - min(zs[0], zs[-1])))   # <0 表示中途下探
    print("alpha=%.2f: 成功 %d/25, phase4 中途下探量 %.3f m (负=往下勾)" % (alpha, ok, np.mean(dips)))
