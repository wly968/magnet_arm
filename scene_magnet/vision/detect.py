"""检测器：YOLOv8-seg 与颜色基线，统一输出 Detection 列表。

Detection.xy 是**带面世界坐标**（经标定单应矩阵映射），sigma_mm 是位置不确定度。
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibrate import load_calibration


@dataclass
class Detection:
    xy: np.ndarray                     # 世界坐标 (x, y)
    uv: np.ndarray                     # 像素坐标 (u, v)
    conf: float = 1.0
    area: int = 0
    sigma_mm: float = 1.0
    mask: object = field(default=None, repr=False)


def mm_per_px(H, uv=(320.0, 240.0), eps=4.0):
    """由单应矩阵估计局部像素当量（mm/px）。"""
    p0 = H @ np.array([uv[0], uv[1], 1.0]); p0 = p0[:2] / p0[2]
    p1 = H @ np.array([uv[0] + eps, uv[1], 1.0]); p1 = p1[:2] / p1[2]
    return float(np.linalg.norm(p1 - p0) / eps * 1000.0)


class _Calib:
    """标定包装：像素 -> 世界。"""

    def __init__(self, calib=None):
        H, H_inv, meta = load_calibration() if calib is None else calib
        self.H, self.H_inv, self.meta = H, H_inv, meta
        self.mm_per_px = mm_per_px(H)

    def to_world(self, uv):
        p = self.H @ np.array([float(uv[0]), float(uv[1]), 1.0])
        return p[:2] / p[2]


def _sigma_from_area(area, mmpp):
    """面积越大质心越稳：sigma ≈ 0.6 * mm/px / sqrt(area/100)。"""
    area = max(int(area), 20)
    return float(max(0.2, 0.6 * mmpp * np.sqrt(100.0 / area)))


class ColorDetector:
    """HSV/颜色阈值 + 连通域基线：不训练也能跑通全链路。"""

    def __init__(self, calib=None, min_area=40, max_area=6000):
        self.cal = _Calib(calib)
        self.min_area, self.max_area = int(min_area), int(max_area)

    def detect(self, img):
        rgb = img.astype(np.float64)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = ((r > 90) & (r > 1.6 * g) & (r > 1.6 * b)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue
            uv = np.array(cents[i], dtype=np.float64)
            out.append(Detection(xy=self.cal.to_world(uv), uv=uv, conf=1.0, area=area,
                                 sigma_mm=_sigma_from_area(area, self.cal.mm_per_px),
                                 mask=(lab == i)))
        return out


class SegDetector:
    """YOLOv8-seg：实例掩码 -> 质心 -> 世界坐标（取质心，不用框中心）。"""

    def __init__(self, weights, calib=None, conf=0.25, device="cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.device = conf, device
        self.cal = _Calib(calib)

    def detect(self, img):
        res = self.model.predict(img, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        if res.masks is None:
            return out
        masks = res.masks.data.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else [1.0] * len(masks)
        h, w = img.shape[:2]
        for k, m in enumerate(masks):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(m > 0.5)
            if len(xs) < 20:
                continue
            uv = np.array([xs.mean(), ys.mean()])
            out.append(Detection(xy=self.cal.to_world(uv), uv=uv, conf=float(confs[k]),
                                 area=int(len(xs)),
                                 sigma_mm=_sigma_from_area(len(xs), self.cal.mm_per_px),
                                 mask=m))
        return out
