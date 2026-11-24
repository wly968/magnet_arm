import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(4)])
m=PPO.load("scene_magnet/saves/ckpt/ckpt_2499960_steps.zip", env=env, device="cpu")
env.env_method("_set_dist",0.045); env.env_method("_set_angle1",0.95)
env.env_method("_set_still_count",0); env.env_method("_set_rel",0.10)
obs=env.reset(); eps=[]
for t in range(4000):
    a,_=m.predict(obs, deterministic=False)
    obs,r,done,infos=env.step(a)
    for info,d in zip(infos,done):
        if d:
            ep=info.get("episode",{})
            eps.append((ep.get("r"), ep.get("l"), info.get("final_phase")))
print("完成回合数:", len(eps))
for e in eps[:8]: print("  r=%s l=%s final_phase=%s" % e)
if eps:
    print("平均 r %.1f, 平均 l %.1f | final_phase 分布 %s"
          % (np.mean([e[0] for e in eps]), np.mean([e[1] for e in eps]),
             {p: sum(1 for e in eps if e[2]==p) for p in set(e[2] for e in eps)}))
