"""检测器：YOLOv8-seg 推理 → 实例掩码 → 亚像素质心 → 世界坐标。

职责
----
- SegDetector：加载 best.pt 与标定，输出每个方块的 Detection
- ColorDetector：HSV + 连通域基线（训练前即可跑通全链路）
- σ 由掩码面积/形状给出（不是网络置信度）

计划接口
--------
@dataclass Detection: xy_world, conf, area_px, sigma_mm, mask
SegDetector.detect(img) -> [Detection]
ColorDetector.detect(img) -> [Detection]
mask_centroid(mask) -> (u, v)
mask_sigma(mask) -> sigma_mm

依赖：cv2, ultralytics, vision.calibrate
"""

# TODO: 待实现
