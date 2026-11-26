"""YOLOv8-seg 训练入口。

    cd scene_magnet
    python -m vision.train_yolo --data vision/dataset/data.yaml --epochs 60
"""
import argparse
import os


def train(data, model="yolov8n-seg.pt", epochs=60, imgsz=640, batch=8,
          device="cpu", project="runs", name="segment"):
    from ultralytics import YOLO
    try:
        net = YOLO(model)                 # 预训练权重（需要联网下载一次）
    except Exception as exc:
        print(f"[train_yolo] 预训练权重不可用（{exc}），改为从结构文件训练")
        net = YOLO(model.replace(".pt", ".yaml"))
    # workers=0：本机沙箱禁止子进程管道（WinError 5），数据加载必须留在主进程
    res = net.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch,
                    device=device, workers=0, project=project, name=name,
                    exist_ok=True, plots=False)
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
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    train(a.data, a.model, a.epochs, a.imgsz, a.batch, a.device)
