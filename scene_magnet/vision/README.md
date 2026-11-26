# vision/ — 视觉识别模块

用俯视相机识别 A 带上的方块，输出世界坐标、速度与不确定度，供 RL 策略使用。
完整设计见 `../docs/vision_design.pdf`（架构、函数清单、卡尔曼等知识清单）。

## 目录

| 文件 | 状态 | 职责 |
|---|---|---|
| `camera.py` | 已实现 | 离屏渲染、世界↔像素、单应矩阵 |
| `calibrate.py` | 已实现（待验证） | 标定 + 残差检查，产出 `calib.npz` |
| `scene.py` | 占位 | 场景封装、方块摆放、光照随机化 |
| `dataset.py` | 占位 | YOLO-seg 数据集（多边形标签 + data.yaml） |
| `train_yolo.py` | 占位 | YOLOv8-seg 训练入口 |
| `detect.py` | 占位 | 检测封装（YOLO-seg / 颜色基线） |
| `tracker.py` | 占位 | 卡尔曼多目标跟踪（关联/遮挡续命） |
| `vision_thread.py` | 占位 | 视觉线程 + 快照发布 |
| `bridge.py` | 占位 | 与环境对接（obs 替换） |
| `eval_vision.py` | 占位 | 精度与鲁棒性评测 |
| `data.yaml` | 模板 | ultralytics 数据集配置 |

## 依赖

```bash
pip install ultralytics opencv-python      # 已装：cv2 5.0、ultralytics 8.4
```

## 使用顺序

```bash
cd scene_magnet

# 1) 标定（像素 ↔ 带面坐标）
python -m vision.calibrate

# 2) 生成数据集（需要 scene.py / dataset.py 实现后）
python -m vision.dataset --images 5000 --out vision/dataset

# 3) 训练
python -m vision.train_yolo --data vision/data.yaml --model yolov8n-seg.pt --epochs 100

# 4) 评测
python -m vision.eval_vision
```

## 约定

- 投影平面取方块中心 `z = 0.375`（`camera.PLANE_Z`）；
- 检测取**掩码质心**（亚像素），不要用框中心（差 3~8 mm）；
- 不确定度来自卡尔曼协方差，不用网络置信度；
- 渲染只能在主线程（GL 上下文），子线程只做推理。
