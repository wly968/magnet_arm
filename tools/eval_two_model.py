import os, sys, numpy as np, time
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

N = 300
env = ArmEnv()
m1 = PPO.load(os.path.join(SCENE,"saves","phase2.zip"), device="cpu")
m2 = PPO.load(os.path.join(SCENE,"saves","phase34.zip"), device="cpu")

ok = 0; steps=[]; phases={}; t0=time.time()
for ep in range(N):
    o,_ = env.reset()
    success = False
    for t in range(env.max_steps):
        pol = m1 if env.phase < 3 else m2
        a,_ = pol.predict(o, deterministic=True)
        o,r,term,tr,info = env.step(a)
        success = success or bool(info.get("success"))
        if term or tr: break
    phases[env.phase]=phases.get(env.phase,0)+1
    ok += int(success)
    steps.append(t+1)
print("两模型接力 %d 局: success %d (%.1f%%)  phases %s  mean_len %.1f  time %.0fs"
      % (N, ok, 100.0*ok/N, phases, np.mean(steps), time.time()-t0))
