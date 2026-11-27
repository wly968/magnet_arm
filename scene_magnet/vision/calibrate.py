"""标定：像素 ↔ 带面坐标。

做法
----
1. 用相机参数做解析投影（作为诊断量），真正用于拟合的是**检测到的质心**；
   这是经验标定：把"轮廓质心"相对"方块中心"的透视偏移一起吸收进单应矩阵
2. 把这些点放成红色方块渲染，检测红色质心，与解析投影对比（验证相机模型/图像方向）；
3. 用解析投影点拟合单应矩阵（解析点无检测噪声），保存 `calib.npz`；
4. 用"世界→像素→世界"往返误差检查系统偏差。

    cd scene_magnet
    python -m vision.calibrate
"""
import os

import mujoco
import numpy as np

from .camera import Camera, PLANE_Z, TOP_Z

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "calib.npz")


def detect_red_centroid(img):
    """红色掩码质心（仅用于标定交叉验证）。返回 (uv, area)。"""
    rgb = img.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 90) & (r > 1.6 * g) & (r > 1.6 * b)
    if mask.sum() < 20:
        return None, 0
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]), int(mask.sum())


def calibrate(width=640, height=480, verbose=True, save=True, detector=None, out=None):
    cam = Camera(width=width, height=height)
    m, d = cam.model, cam.data

    # 只留 cube_0 用于标定，其余藏起来
    mid = int(m.body_mocapid[m.body("cube_0").id])
    for i in range(1, 64):
        try:
            j = int(m.body_mocapid[m.body(f"cube_{i}").id])
        except Exception:
            break
        d.mocap_pos[j] = [0.0, 0.0, -5.0]
    mujoco.mj_forward(m, d)

    xs = [-1.05, -0.75, -0.45, -0.15]
    ys = [0.50, 0.65, 0.80]
    pixels, worlds, residual = [], [], []
    for x in xs:
        for y in ys:
            d.mocap_pos[mid] = [x, y, PLANE_Z]
            mujoco.mj_forward(m, d)
            uv_proj = cam.project_point([x, y, PLANE_Z])
            frame = cam.render()
            if detector is None:
                uv_det, area = detect_red_centroid(frame)
            else:
                ds = detector.detect(frame)
                if not ds:
                    continue
                uv_det, area = ds[0].uv, ds[0].area
            if uv_proj is None or uv_det is None or area < 20:
                continue
            pixels.append(uv_det)   # 用检测质心拟合，吸收轮廓偏移
            worlds.append((x, y))
            residual.append(float(np.linalg.norm(uv_det - uv_proj)))

    d.mocap_pos[mid] = [0.0, 0.0, -5.0]
    mujoco.mj_forward(m, d)
    if len(pixels) < 4:
        raise RuntimeError("标定失败：有效点不足 4 个")

    cam.fit_homography(pixels, worlds)

    errs = []
    for (u, v), (x, y) in zip(pixels, worlds):
        xy = cam.pixel_to_world([u, v])
        errs.append(float(np.linalg.norm(xy - [x, y])))

    if save:
        np.savez(out or CALIB_PATH, H=cam.H, H_inv=cam.H_inv,
                 width=width, height=height, plane_z=PLANE_Z)
    if verbose:
        print(f"[calibrate] 有效采样点 {len(pixels)} 个")
        print(f"[calibrate] 诊断：解析投影 vs 红色质心 中位 {np.median(residual):.2f} px，"
              f"最大 {np.max(residual):.2f} px")
        print(f"[calibrate] 单应往返误差：中位 {1000*np.median(errs):.3f} mm，"
              f"最大 {1000*np.max(errs):.3f} mm")
        if save:
            print(f"[calibrate] 已保存 {out or CALIB_PATH}")
    return cam, {"pixel_residual": residual, "world_error_mm": np.array(errs) * 1000.0}


def load_calibration(path=None):
    """读回标定：返回 (H, H_inv, meta)。path 可指定不同检测器的标定文件。"""
    path = path or CALIB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"没有标定文件：{path}，先跑 python -m vision.calibrate")
    z = np.load(path)
    return z["H"], z["H_inv"], {
        "width": int(z["width"]), "height": int(z["height"]),
        "plane_z": float(z["plane_z"]),
    }


if __name__ == "__main__":
    calibrate()
