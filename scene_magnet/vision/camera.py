"""相机与渲染：俯视相机 birdeye 的离屏渲染 + 像素↔带面坐标映射。

约定：
    - 世界系：带面 z = 0.350，方块中心 z = 0.375（用 0.375 做投影平面）
    - 像素系：左上角原点，x 向右，y 向下
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "assets", "models", "vision_scene.xml")
CAM_NAME = "birdeye"
PLANE_Z = 0.375          # 投影平面（方块中心高度）

DEFAULT_W, DEFAULT_H = 640, 480


def load_model(path=SCENE):
    return mujoco.MjModel.from_xml_path(os.path.abspath(path))


class Camera:
    """离屏渲染 + 单应矩阵（像素 ↔ 带面坐标）。"""

    def __init__(self, width=DEFAULT_W, height=DEFAULT_H, scene=SCENE):
        self.model = load_model(scene)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.width, self.height = int(width), int(height)
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.cam_id = self.model.camera(CAM_NAME).id
        self.H = None                     # 3x3，像素 -> 世界
        self.H_inv = None                 # 世界 -> 像素

    # ---------- 渲染 ----------
    def render(self):
        """返回 (H, W, 3) uint8，当前 data 状态的俯视图。"""
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    # ---------- 单应矩阵 ----------
    def fit_homography(self, pixels, worlds):
        """由 N>=4 组对应点拟合 像素->世界 的单应矩阵（DLT，不依赖 cv2）。"""
        P = np.asarray(pixels, dtype=np.float64)
        W = np.asarray(worlds, dtype=np.float64)[:, :2]
        assert len(P) >= 4 and len(P) == len(W)
        A = []
        for (u, v), (x, y) in zip(P, W):
            A.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
            A.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
        _, _, Vt = np.linalg.svd(np.asarray(A))
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]
        self.H, self.H_inv = H, np.linalg.inv(H)
        return H

    def pixel_to_world(self, uv):
        if self.H is None:
            raise RuntimeError("先调用 fit_homography()")
        u, v = float(uv[0]), float(uv[1])
        p = self.H @ np.array([u, v, 1.0])
        return p[:2] / p[2]

    def world_to_pixel(self, xy):
        if self.H_inv is None:
            raise RuntimeError("先调用 fit_homography()")
        x, y = float(xy[0]), float(xy[1])
        p = self.H_inv @ np.array([x, y, 1.0])
        return p[:2] / p[2]

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass
