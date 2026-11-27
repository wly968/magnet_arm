"""YOLOv8-seg 训练入口。

    cd scene_magnet
    python -m vision.train_yolo --data vision/dataset/data.yaml --epochs 60
"""
import argparse
import os


class _SerialPool:
    """替代 ultralytics 的 ThreadPool：沙箱禁止命名管道时，退化为串行。"""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap(self, func=None, iterable=(), **kw):
        return map(func, iterable)


def _patch_pool():
    """两个本机兼容补丁：

    1. ultralytics 用 ThreadPool 扫标签，沙箱禁止命名管道（WinError 5）→ 改串行；
    2. 几何增强 RandomPerspective 里有矩阵乘法，本机 BLAS 会硬崩（0xc06d007f）
       → 关掉它（合成数据集本身已有足够变化）。
    """
    try:
        import ultralytics.data.dataset as _ds
        _ds.ThreadPool = _SerialPool
    except Exception:
        pass
    # 注意：ultralytics 的几何增强里还有多段矩阵乘（apply_bboxes/apply_segments），
    # 本机 numpy 的 BLAS 会硬崩（0xc06d007f），无法逐一打补丁 —— 训练前必须先修好 numpy。


def _pick_device(device=None):
    """device=None 时自动选：有 GPU 用 GPU(0)，否则 CPU。"""
    if device not in (None, "auto"):
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train(data, model="yolov8n-seg.pt", epochs=60, imgsz=640, batch=8,
          device=None, project="runs", name="segment"):
    device = _pick_device(device)
    print(f"[train_yolo] 设备: {device}")
    _patch_pool()
    from ultralytics import YOLO
    try:
        net = YOLO(model)                 # 预训练权重（需要联网下载一次）
    except Exception as exc:
        print(f"[train_yolo] 预训练权重不可用（{exc}），改为从结构文件训练")
        net = YOLO(model.replace(".pt", ".yaml"))
    # workers=0：本机沙箱禁止子进程管道（WinError 5），数据加载必须留在主进程
    res = net.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch,
                    device=device, workers=0, project=project, name=name,
                    exist_ok=True, plots=False, mosaic=0.0)
    best = os.path.join(project, name, "weights", "best.pt")
    print("[train_yolo] 完成，权重：", best)
    return res, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="vision/dataset/data.yaml")
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="None=自动选 GPU，或 cpu / 0")
    a = ap.parse_args()
    train(a.data, a.model, a.epochs, a.imgsz, a.batch, a.device)
