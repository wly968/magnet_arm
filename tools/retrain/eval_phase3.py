import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "scene")
sys.path.insert(0, SCENE)
os.chdir(SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv

HERE = os.path.dirname(os.path.abspath(__file__))
env = ArmEnv()
model = PPO.load(os.path.join(HERE, "phase3_ft"), env=env)

N = 60
succ = 0
steps_ok, phases = [], {}
qs, halfs, forces = [], [], []
for ep in range(N):
    obs, _ = env.reset()
    term = False
    for t in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    ph = env.phase
    phases[ph] = phases.get(ph, 0) + 1
    if term and ph == 4:
        succ += 1
        steps_ok.append(t + 1)
    q = float(env.data.qpos[env.finger_qpos_adr])
    qs.append(q); halfs.append(0.02 + q)

def stat(x):
    x = np.array(x)
    return f"mean {x.mean():.3f} min {x.min():.3f} max {x.max():.3f}"

print(f"phase3 success: {succ}/{N} = {succ/N:.2f}")
print("final phase distribution:", phases)
if steps_ok:
    print(f"success steps: {stat(steps_ok)}")
print(f"final q: {stat(qs)}")
print(f"final half_open: {stat(halfs)}")
