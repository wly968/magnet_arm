"""相机与渲染：birdeye 俯视相机、解析投影、单应矩阵。

坐标约定
--------
- 世界系：带面 z = 0.350，方块中心 z = 0.375（`PLANE_Z`），顶面 z = 0.400
- 像素系：左上角原点，u 向右，v 向下
- 相机：MuJoCo 相机沿其局部 -z 方向看

注意：本机 LAPACK 会在部分调用下硬崩（0xc06d007f），
所以这里**不使用 np.linalg.svd / inv / solve**，全部用纯算术实现。
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "assets", "models", "vision_scene.xml")
CAM_NAME = "birdeye"
PLANE_Z = 0.375          # 投影平面：方块中心高度
TOP_Z = 0.400            # 方块顶面（掩码质心更接近这一层）

DEFAULT_W, DEFAULT_H = 640, 480


def load_model(path=SCENE):
    return mujoco.MjModel.from_xml_path(os.path.abspath(path))


def solve_linear(A, b):
    """高斯消元（部分主元），避免 LAPACK。A: n×n, b: n。"""
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    n = len(b)
    for col in range(n):
        piv = col + int(np.argmax(np.abs(A[col:, col])))
        if abs(A[piv, col]) < 1e-12:
            raise ValueError("矩阵奇异")
        if piv != col:
            A[[col, piv]] = A[[piv, col]]
            b[[col, piv]] = b[[piv, col]]
        for row in range(col + 1, n):
            f = A[row, col] / A[col, col]
            if f != 0.0:
                A[row, col:] -= f * A[col, col:]
                b[row] -= f * b[col]
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        s_rest = float(np.einsum("i,i->", A[row, row + 1:], x[row + 1:])) if row + 1 < n else 0.0
        x[row] = (b[row] - s_rest) / A[row, row]
    return x


def inv3(M):
    """3×3 求逆（伴随矩阵），避免 LAPACK。"""
    M = np.asarray(M, dtype=np.float64)
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("单应矩阵不可逆")
    return np.array([
        [(e * i - f * h), -(b * i - c * h), (b * f - c * e)],
        [-(d * i - f * g), (a * i - c * g), -(a * f - c * d)],
        [(d * h - e * g), -(a * h - b * g), (a * e - b * d)],
    ]) / det


class Camera:
    """离屏渲染 + 像素/世界坐标互转。"""

    def __init__(self, width=DEFAULT_W, height=DEFAULT_H, scene=SCENE):
        self.model = load_model(scene)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.width, self.height = int(width), int(height)
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.cam_id = self.model.camera(CAM_NAME).id
        self.H = None       # 像素 -> 世界
        self.H_inv = None   # 世界 -> 像素

    # ---------------- 渲染 ----------------
    def render(self):
        """返回当前 data 状态的俯视图 (H, W, 3) uint8。"""
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    # ---------------- 解析投影 ----------------
    def project_point(self, world_xyz):
        """世界点 -> 像素坐标（不依赖检测）。"""
        m = self.model
        pos = np.asarray(m.cam_pos[self.cam_id], dtype=np.float64)
        R = np.asarray(m.cam_mat0[self.cam_id], dtype=np.float64).reshape(3, 3)
        p = np.einsum("ji,j->i", R, np.asarray(world_xyz, dtype=np.float64) - pos)
        z = -p[2]
        if z <= 1e-6:
            return None
        f = 1.0 / np.tan(np.radians(float(m.cam_fovy[self.cam_id])) * 0.5)
        aspect = self.width / self.height
        u = ((f * p[0] / z) / aspect + 1.0) * 0.5 * self.width
        v = (1.0 - (f * p[1] / z)) * 0.5 * self.height
        return np.array([u, v])

    def project_many(self, pts):
        return np.array([self.project_point(p) for p in pts])

    # ---------------- 单应矩阵 ----------------
    def fit_homography(self, pixels, worlds):
        """最小二乘拟合 像素->世界 的单应矩阵（h33 固定为 1，n×n 高斯消元）。"""
        P = np.asarray(pixels, dtype=np.float64)
        W = np.asarray(worlds, dtype=np.float64)[:, :2]
        assert len(P) >= 4 and len(P) == len(W)
        M, rhs = [], []
        for (u, v), (x, y) in zip(P, W):
            M.append([u, v, 1, 0, 0, 0, -u * x, -v * x]); rhs.append(x)
            M.append([0, 0, 0, u, v, 1, -u * y, -v * y]); rhs.append(y)
        M = np.asarray(M)
        rhs = np.asarray(rhs)
        # 注意：本机 BLAS 的 matmul 会硬崩，用 einsum 代替
        h = solve_linear(np.einsum("ij,ik->jk", M, M),
                         np.einsum("ij,i->j", M, rhs))     # 正规方程，8×8
        H = np.array([[h[0], h[1], h[2]],
                      [h[3], h[4], h[5]],
                      [h[6], h[7], 1.0]])
        self.H = H
        self.H_inv = inv3(H)
        return H

    def pixel_to_world(self, uv):
        if self.H is None:
            raise RuntimeError("先调用 fit_homography()")
        p = self.H @ np.array([float(uv[0]), float(uv[1]), 1.0])
        return p[:2] / p[2]

    def world_to_pixel(self, xy):
        if self.H_inv is None:
            raise RuntimeError("先调用 fit_homography()")
        p = self.H_inv @ np.array([float(xy[0]), float(xy[1]), 1.0])
        return p[:2] / p[2]

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass
