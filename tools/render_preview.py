"""生成展示图 / 训练版对比图（过程文件，放在 scene 外面）。

用法：python tools/render_preview.py
输出：tools/preview_show.png, tools/preview_train.png
"""

import os

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scene")
TOOLS = os.path.join(ROOT, "tools")


def cam_at(lookat, dist, azim, elev):
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = lookat
    c.distance = dist
    c.azimuth = azim
    c.elevation = elev
    return c


def render_show():
    m = mujoco.MjModel.from_xml_path(os.path.join(SCENE, "show_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(300):
        mujoco.mj_step(m, d)

    r = mujoco.Renderer(m, 900, 900)
    opt = mujoco.MjvOption()

    def shot(cam=None):
        r.update_scene(d, cam, opt)
        return r.render().copy()

    # 1) belt_cam：A 带正上方俯视
    img_top = shot(m.camera("belt_cam").id)
    # 2) hero
    img_hero = shot(cam_at([0.0, 0.10, 0.45], 2.6, -52, -14))
    # 3) 全景
    img_overview = shot(cam_at([-0.05, 0.0, 0.35], 3.1, -45, -24))
    # 4) 抓取特写：先摆姿态，再把方块放到夹爪下
    d.ctrl[:] = [1.418, -0.907, 2.054, 0.424, 0.03, -0.03]
    for _ in range(900):
        mujoco.mj_step(m, d)
    j = m.body("cube_0").jntadr[0]
    a, b = m.jnt_qposadr[j], m.jnt_dofadr[j]
    d.qpos[a:a + 3] = [0.10, 0.65, 0.375]
    d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    d.qvel[b:b + 6] = 0
    mujoco.mj_forward(m, d)
    img_detail = shot(cam_at([0.15, 0.50, 0.58], 1.35, 35, -8))

    sheet = np.concatenate([img_top, img_hero, img_overview, img_detail], axis=1)
    Image.fromarray(sheet).save(os.path.join(TOOLS, "preview_show.png"))
    print("show saved", sheet.shape)


def render_train():
    m = mujoco.MjModel.from_xml_path(os.path.join(SCENE, "train_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(300):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, 900, 900)
    c = cam_at([0.0, 0.0, 0.30], 2.8, 62, -26)
    r.update_scene(d, c)
    Image.fromarray(r.render()).save(os.path.join(TOOLS, "preview_train.png"))
    print("train saved")


if __name__ == "__main__":
    render_show()
    render_train()
