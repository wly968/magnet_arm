"""YOLOv8-seg 训练入口（ultralytics）。

职责
----
- 封装 YOLO(...).train(...)，暴露 CLI
- 训练目标：单类别 cube 的实例分割

计划接口
--------
train(data_yaml, weights="yolov8n-seg.pt", epochs=100, imgsz=640, batch=16)
main()          # argparse：--data --model --epochs --imgsz --batch --project

产物
----
runs/segment/train*/weights/best.pt

依赖：ultralytics（已装 8.4.x）
"""

# TODO: 待实现
