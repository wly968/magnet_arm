"""标定：像素 ↔ 带面坐标。

先用 MuJoCo 相机参数做解析投影（精确），再用渲染出来的红色方块质心做**交叉验证**，
两者一致才说明相机模型/图像方向没问题。标定结果存到 `vision/calib.npz`。

    python -m vision.calibrate          # 在 scene_magnet 目录下
"""
import os

import mujoco
import numpy as np

from .camera import Camera, PLANE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "calib.npz")


def project_point(cam, world_xyz):
    """把世界点投影到像素坐标（不依赖检测，直接由相机参数算）。"""
    m = cam.model
    cid = cam.cam_id
    pos = np.array(m.cam_pos[cid], dtype=np.float64)
    # cam_mat 是 3x3，把相机局部坐标映射到世界坐标；它的转置就是 world->camera
    R = np.array(m.cam_mat0[cid], dtype=np.float64).reshape(3, 3)
    p_cam = R.T @ (np.asarray(world_xyz, dtype=np.float64) - pos)
    fovy = float(m.cam_fovy[cid])
    f = 1.0 / np.tan(np.radians(fovy) * 0.5)
    z = -p_cam[2]                      # MuJoCo 相机沿局部 -z 看
    if z <= 1e-6:
        return None
    x_ndc = f * p_cam[0] / z / (cam.width / cam.height)
    y_ndc = f * p_cam[1] / z
    u = (x_ndc + 1.0) * 0.5 * cam.width
    v = (1.0 - y_ndc) * 0.5 * cam.height
    return np.array([u, v])


def detect_red_centroid(img):
    """在渲染图里找红色块质心（仅用于标定交叉验证）。"""
    rgb = img.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 90) & (r > 1.6 * g) & (r > 1.6 * b)
    if mask.sum() < 20:
        return None, 0
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]), int(mask.sum())


def calibrate(width=640, height=480, verbose=True):
    cam = Camera(width=width, height=height)
    m, d = cam.model, cam.data

    # 1) 把一个方块放到网格点上，渲染 -> 检测质心 -> 与解析投影对比
    root = m.body("cube_0").rootid
    mid = int(m.body_mocapid[m.body("cube_0").id])
    for i in range(1, 10):
        try:
            j = int(m.body_mocapid[m.body(f"cube_{i}").id])
        except Exception:
            continue
        d.mocap_pos[j] = [0, 0, -5]
    mujoco.mj_forward(m, d)

    xs = [-1.05, -0.75, -0.45, -0.15]
    ys = [0.50, 0.65, 0.80]
    pairs, residual = [], []
    for x in xs:
        for y in ys:
            d.mocap_pos[mid] = [x, y, PLANE_Z]
            mujoco.mj_forward(m, d)
            img = cam.render()
            uv_det, area = detect_red_centroid(img)
            uv_proj = project_point(cam, [x, y, PLANE_Z])
            if uv_det is None or uv_proj is None or area < 20:
                continue
            pairs.append((uv_proj, (x, y)))
            residual.append(np.linalg.norm(uv_det - uv_proj))
    d.mocap_pos[mid] = [0, 0, -5]
    mujoco.mj_forward(m, d)

    if len(pairs) < 4:
        raise RuntimeError("标定失败：有效点不足 4 个")

    # 2) 用解析投影点拟合单应矩阵（精确、无检测噪声）
    pixels = np.array([p for p, _ in pairs])
    worlds = np.array([w for _, w in pairs])
    cam.fit_homography(pixels, worlds)

    # 3) 残差检查：世界坐标回投误差
    errs = []
    for (u, v), (x, y) in zip(pixels, worlds):
        xy = cam.pixel_to_world([u, v])
        errs.append(np.linalg.norm(xy - [x, y]))

    np.savez(CALIB_PATH, H=cam.H, H_inv=cam.H_inv,
             width=width, height=height, plane_z=PLANE_Z)
    if verbose:
        print(f"[calibrate] 采样点 {len(pairs)} 个")
        print(f"[calibrate] 解析投影 vs 检测质心偏差：中位 {np.median(residual):.2f} px, "
              f"最大 {np.max(residual):.2f} px")
        print(f"[calibrate] 单应矩阵往返误差：中位 {1000*np.median(errs):.2f} mm, "
              f"最大 {1000*np.max(errs):.2f} mm")
        print(f"[calibrate] 已保存 {CALIB_PATH}")
    return cam


def load_calibration():
    """读回标定结果，返回 (H, H_inv, meta)。"""
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError(f"没有标定文件：{CALIB_PATH}，先跑 python -m vision.calibrate")
    z = np.load(CALIB_PATH)
    return z["H"], z["H_inv"], {"width": int(z["width"]), "height": int(z["height"]),
                                "plane_z": float(z["plane_z"])}


if __name__ == "__main__":
    calibrate()
