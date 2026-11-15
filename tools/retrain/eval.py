import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "scene")
sys.path.insert(0, SCENE)
os.chdir(SCENE)

from stable_baselines3 import PPO
from arm_env import ArmEnv

HERE = os.path.dirname(os.path.abspath(__file__))
env = ArmEnv()
model = PPO.load(os.path.join(HERE, "ppo_current"), env=env)

N = 80
ok = 0
ok03 = 0
ok99 = 0
both03 = 0
dists, angs, steps, elb, wristz, eez = [], [], [], [], [], []
for ep in range(N):
    obs, _ = env.reset()
    term = trunc = False
    for t in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    d = env.data
    wrist = d.xanchor[env.arm_joints[3]]
    tgt = env.box_pos + np.array([0, 0, 0.25])
    dist = float(np.linalg.norm(wrist - tgt))
    ee = d.site_xpos[env.ee_id]
    fwd = (ee - wrist) / np.linalg.norm(ee - wrist)
    ang = float(np.dot(fwd, np.array([0, 0, -1.])))
    elbz = float(d.xpos[env.model.body("elbow").id][2])
    dists.append(dist); angs.append(ang); steps.append(t + 1)
    elb.append(elbz); wristz.append(wrist[2]); eez.append(ee[2])
    ok += int(term)
    ok03 += int(dist < 0.03); ok99 += int(ang > 0.99)
    both03 += int(dist < 0.03 and ang > 0.99)

print(f"env 判定(0.05/0.9)成功率: {ok}/{N} = {ok/N:.2f}")
print(f"dist  mean {np.mean(dists):.4f}  median {np.median(dists):.4f}  5/95% {np.percentile(dists,[5,95]).round(4)}")
print(f"angle mean {np.mean(angs):.4f}  min {np.min(angs):.4f}  5/95% {np.percentile(angs,[5,95]).round(4)}")
print(f"dist<0.03: {ok03}/{N} = {ok03/N:.2f}   angle>0.99: {ok99}/{N} = {ok99/N:.2f}   both: {both03}/{N} = {both03/N:.2f}")
print(f"steps mean {np.mean(steps):.0f}   elbow_z mean {np.mean(elb):.3f}   wrist_z mean {np.mean(wristz):.3f}   ee_z mean {np.mean(eez):.3f}")
