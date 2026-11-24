import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(8)])
m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
m.learn(total_timesteps=20000, callback=CurriculumCallback())

e=ArmEnv(); e._set_dist(0.05); e._set_angle1(0.94); e._set_still_count(0); e._set_rel(0.2)
rels=[]; lens=[]; ends={}
for ep in range(10):
    obs,_=e.reset()
    for t in range(e.max_steps+e.max_wait_steps+10):
        a,_=m.predict(obs,deterministic=True)
        obs,r,term,tr,info=e.step(a)
        if e.phase==2: rels.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
        if e.phase>=3 or term or tr:
            ends["success" if e.phase>=3 else ("nan" if info.get("nan") else "timeout")] = ends.get("success" if e.phase>=3 else ("nan" if info.get("nan") else "timeout"),0)+1
            break
    lens.append(t+1)
rel=np.array(rels)
print("20k 微调后: 相位2 帧 %d, rel 中位 %.3f" % (len(rel), np.median(rel)))
print("  rel<0.2 占比 %.1f%% | <0.1 %.1f%% | <0.05 %.1f%%" % (100*(rel<0.2).mean(), 100*(rel<0.1).mean(), 100*(rel<0.05).mean()))
print("  结束原因:", ends, "| 平均步数 %.0f" % np.mean(lens))
