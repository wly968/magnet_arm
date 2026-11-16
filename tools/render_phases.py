"""生成 README 用的场景图与各阶段截图（取每个阶段结束前的最后一帧）。

输出：docs/scene.png、docs/phase1.png ... phase6.png、docs/train_scene.png
用法：python tools/render_phases.py
"""
import os
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scene_magnet")
DOCS = os.path.join(ROOT, "docs")
sys.path.insert(0, SCENE)

from stable_baselines3 import PPO  # noqa: E402

from envs.arm_env import ArmEnv  # noqa: E402

W, H = 900, 620
TH = {1: (0.03, 0.98, 1), 2: (0.03, 0.98, 1), 3: (0.05, 0.945, 1),
      4: (0.05, 0.945, 1), 5: (0.05, 0.945, 1), 6: (0.06, 0.94, 0)}
# 每个阶段偏好的机位（azim, elev, distance）
# 每阶段机位：lookat 相对末端的偏移 (dx,dy,dz), distance, azim, elev
VIEW = {
    1: ((0.0, 0.0, -0.10), 1.9, 70, -20),
    2: ((0.0, 0.0, -0.05), 1.1, 65, -12),
    3: ((0.0, 0.0, -0.05), 1.3, 65, -12),
    4: ((0.05, 0.05, -0.25), 2.9, -35, -22),
    5: ((0.02, 0.02, -0.12), 2.3, -65, -28),
    6: ((0.0, 0.10, -0.15), 2.7, -115, -22),
}


def cam(lookat, dist, azim, elev):
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = lookat
    c.distance = dist
    c.azimuth = azim
    c.elevation = elev
    return c


def overview():
    m = mujoco.MjModel.from_xml_path(os.path.join(SCENE, "assets", "models", "show_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(200):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, cam([-0.05, 0.10, 0.40], 3.0, -50, -22))
    Image.fromarray(r.render()).save(os.path.join(DOCS, "scene.png"))


def phases():
    env = ArmEnv(terminate_on_phase2=False)
    env.smooth_alpha = 0.5
    m1 = PPO.load(os.path.join(SCENE, "ppo_arm_reach.zip"), env=env, device="cpu")
    m2 = PPO.load(os.path.join(SCENE, "saves", "phase34_retract.zip"), env=env, device="cpu")
    m3 = PPO.load(os.path.join(SCENE, "saves", "phase6_home.zip"), env=env, device="cpu")

    renderer = mujoco.Renderer(env.model, H, W)

    def apply(phase):
        d, a, s = TH[min(phase, 6)]
        env._set_dist(d)
        env._set_angle1(a)
        env._set_still_count(s)

    def shot(phase):
        ee = env.data.site_xpos[env.ee_id]
        off, dist, azim, elev = VIEW[phase]
        lookat = np.array([float(ee[0]) + off[0], float(ee[1]) + off[1], float(ee[2]) + off[2]])
        renderer.update_scene(env.data, cam(lookat, dist, azim, elev))
        return renderer.render().copy()

    obs, _ = env.reset()
    apply(env.phase)
    last = env.phase
    pending = {1: shot(1)}          # 每个阶段最近一帧
    saved = {}

    for _ in range(env.max_steps * 2):
        if env.phase != last:
            apply(env.phase)
            env.phase_entry_steps = 6
            last = env.phase
            pending.setdefault(env.phase, None)

        pol = m1 if env.phase < 3 else (m2 if env.phase < 6 else m3)
        action, _ = pol.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if env.phase != last:                       # 刚完成上一个阶段
            img = pending.pop(last, None)
            if img is not None and last not in saved:
                Image.fromarray(img).save(os.path.join(DOCS, f"phase{last}.png"))
                saved[last] = True
                print("saved phase", last)
            apply(env.phase)
            env.phase_entry_steps = 6
            last = env.phase
            pending.setdefault(env.phase, None)

        if env.phase <= 6:
            pending[env.phase] = shot(env.phase)

        if terminated or truncated or len(saved) >= 6:
            break

    # phase 6 结束时补拍
    if 6 not in saved:
        Image.fromarray(pending[6]).save(os.path.join(DOCS, "phase6.png"))
        print("saved phase 6 (final)")


def train_scene():
    m = mujoco.MjModel.from_xml_path(os.path.join(SCENE, "assets", "models", "train_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(200):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, cam([0.0, 0.0, 0.30], 2.8, 62, -26))
    Image.fromarray(r.render()).save(os.path.join(DOCS, "train_scene.png"))


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    phases()
    overview()
    train_scene()
    print("done")
