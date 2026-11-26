"""视觉线程：取帧 → 检测 → 跟踪 → 发布带时间戳的快照。

职责
----
- 主线程渲染（GL 上下文限制），把 ndarray 通过 queue 交给本线程
- 本线程按固定频率（10~15 Hz）做 检测 + 跟踪，发布不可变快照
- 快照带 t_capture / t_processed，供控制端按延迟外推

计划接口
--------
@dataclass VisionState: t_capture, t_processed, objects[]
class VisionThread(threading.Thread):
    submit_frame(img, t_capture)
    snapshot() -> VisionState
    run()

说明：渲染必须在主线程调用 mujoco.Renderer，本线程只做 CPU 推理。
"""

# TODO: 待实现
