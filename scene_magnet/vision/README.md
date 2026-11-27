# vision/ — 视觉识别模块

用俯视相机（`birdeye`）识别 A 带上的方块，输出世界坐标、速度与不确定度，供 RL 策略使用。
设计文档见 `../docs/vision_design.pdf`。

## 实测结果（本机已跑通）

| 环节 | 指标 | 数值 |
|---|---|---|
| 标定 | 独立验证点位置误差 | **中位 0.82 mm / 最大 1.03 mm** |
| **YOLOv8-seg** | 位置误差（光照 0 档） | **中位 0.86 mm / 95% 1.70 mm** |
| **YOLOv8-seg** | 位置误差（光照 3 档） | **中位 0.80 mm / 95% 1.76 mm** |
| YOLOv8-seg | 召回 / 数量准确 | 100% / 98~100% |
| 颜色基线 | 位置误差（光照 0/3 档） | 中位 1.2~1.3 mm / 95% 4.2 mm |
| **卡尔曼跟踪（YOLO）** | 含 0.3 s 遮挡的误差 | **中位 0.61 mm / 最大 1.06 mm** |
| 视觉线程 | 频率与快照 | 12 Hz，快照含 (xy, vel, sigma_mm) |

模型训练：`YOLOv8n-seg`，1200 张合成图（960 训练 / 240 验证），60 epochs，imgsz 640，GPU
（RTX 5060 Laptop）约 21 分钟；最终 mask mAP50 = **0.994**、mAP50-95 = 0.476。

> **关键经验：标定必须用"部署时那个检测器"的质心来拟合。**
> YOLO 掩码质心与颜色阈值质心相差约 1.5 px，混用会让误差从 0.8 mm 变成 4.5 mm。
> 因此有两个标定文件：`calib_color.npz` 与 `calib_yolo.npz`。

## 文件

| 文件 | 状态 | 职责 |
|---|---|---|
| `camera.py` | ✅ | 离屏渲染、解析投影、单应矩阵（纯算术，不用 LAPACK） |
| `calibrate.py` | ✅ | 标定：检测质心拟合 + 独立验证，产出 `calib.npz` |
| `scene.py` | ✅ | 场景、方块摆放、光照随机化（0~3 档） |
| `dataset.py` | ✅ | YOLO-seg 数据集（顶面四角多边形标签 + `data.yaml` + 预览） |
| `detect.py` | ✅ | `ColorDetector`（基线）+ `SegDetector`（YOLOv8-seg） |
| `tracker.py` | ✅ | 恒速卡尔曼 + 门限匹配 + 遮挡续命（矩阵运算用 einsum） |
| `vision_thread.py` | ✅ | 视觉线程、带时间戳快照 |
| `bridge.py` | ✅ | 快照 → 目标 (xy, vel, σ)，按延迟外推 |
| `eval_vision.py` | ✅ | 位置误差/召回/数量/光照/遮挡评测 |
| `train_yolo.py` | ✅ | YOLOv8-seg 训练入口 |
| `data.yaml` | 模板 | ultralytics 数据集配置 |

## 使用

```bash
cd scene_magnet

# 1) 标定（会打印像素/毫米级误差）
python -m vision.calibrate

# 2) 生成数据集（建议 ≥2000 张）
python -m vision.dataset --images 2000 --out vision/dataset --preview

# 3) 训练 YOLO-seg（在**你自己的终端**里跑，见下方说明）
python -m vision.train_yolo --data vision/dataset/data.yaml --epochs 60

# 4) 评测
python -m vision.eval_vision                                          # 颜色基线
python -m vision.eval_vision --weights runs/segment/weights/best.pt   # YOLO
```

## 三个必须知道的坑

**① 本机 LAPACK/BLAS 会硬崩（`0xc06d007f`）**

实测崩溃的调用：`np.linalg.svd`、`np.linalg.solve`、`np.linalg.pinv`、以及**矩阵乘法 `A @ B`**（4×4 就会崩）。
因此：

- 单应矩阵用**高斯消元**解正规方程（`camera.solve_linear`）；
- 3×3 求逆用伴随矩阵（`camera.inv3`）；
- 卡尔曼里的矩阵乘全部用 `einsum`（`tracker.mm/mv`）；
- 矩阵×向量（`H @ [u,v,1]`）是安全的。

**② YOLO 训练要在你自己的终端跑**

ultralytics 的训练会开 multiprocessing 管道，而本工具的运行沙箱禁止管道（`WinError 5`）。
`workers=0` 也挡不住它内部的池，所以训练请直接在 PowerShell 里执行（不走沙箱）。

**③ 光照随机化目前偏温和**

现在只改光源位置/强度/色温与 headlight。若要做真正的鲁棒性压力测试，建议再加：
有色光源（把红光洗淡）、投阴影、运动模糊、传感器噪声。

## 输出约定

- 位置：带面世界坐标（投影平面 z=0.375，即方块中心）；
- **检测器与标定文件要配对**：`ColorDetector(calib_path="vision/calib_color.npz")`、
  `SegDetector(weights, calib_path="vision/calib_yolo.npz")`；
- 检测取**掩码质心**（亚像素），不用框中心（差 3~8 mm）；
- σ：由掩码面积给出，经卡尔曼收敛后变小；遮挡时随 `P` 增长；
- 时间戳：快照带 `t_capture`，控制端统一按 `now - t_capture` 外推。
