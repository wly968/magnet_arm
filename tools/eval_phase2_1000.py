import os, sys, numpy as np, time
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

N_EP = 1000
env = ArmEnv(); env.max_steps = 250
model = PPO.load(os.path.join(SCENE,"saves","phase2.zip"), device="cpu")

ok = 0; steps_ok = []; fails = {}
t0 = time.time()
for ep in range(N_EP):
    o,_ = env.reset()
    reached = False
    for t in range(env.max_steps):
        a,_ = model.predict(o, deterministic=True)
        o, r, term, tr, info = env.step(a)
        if env.phase >= 3:
            reached = True; steps_ok.append(t+1); break
        if term or tr: break
    if reached: ok += 1
    else:
        key = "trunc" if tr else ("nan" if info.get("nan") else "term")
        fails[key] = fails.get(key, 0) + 1
print("episodes %d  success %d (%.1f%%)  mean steps %.1f  fails %s  time %.0fs"
      % (N_EP, ok, 100.0*ok/N_EP, np.mean(steps_ok) if steps_ok else -1, fails, time.time()-t0))
