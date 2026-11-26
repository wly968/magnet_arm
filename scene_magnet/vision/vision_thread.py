"""视觉线程：取帧 -> 检测 -> 跟踪 -> 发布带时间戳的快照。

渲染必须在主线程（GL 上下文），所以本线程只从队列取图做 CPU 推理。
"""
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .tracker import MultiTracker


@dataclass
class VisionState:
    t_capture: float = 0.0
    t_processed: float = 0.0
    objects: list = field(default_factory=list)   # [{id, xy, vel, sigma_mm, conf}]

    def front(self, x_max=None):
        """带面上最靠前（x 最大）的目标。"""
        objs = self.objects
        if x_max is not None:
            objs = [o for o in objs if o["xy"][0] <= x_max]
        if not objs:
            return None
        return max(objs, key=lambda o: o["xy"][0])


class VisionThread(threading.Thread):
    """10~15 Hz 视觉流水线。用 submit() 投帧，snapshot() 取最新结果。"""

    def __init__(self, detector, max_hz=12.0, tracker=None, daemon=True):
        super().__init__(daemon=daemon)
        self.detector = detector
        self.tracker = tracker or MultiTracker()
        self.min_dt = 1.0 / float(max_hz)
        self._q = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._state = VisionState()
        self._stop_evt = threading.Event()   # 别叫 _stop：Thread 自带同名方法
        self.n_frames = 0
        self.n_dets = 0

    # 主线程调用
    def submit(self, img, t_capture):
        try:
            self._q.put_nowait((np.asarray(img), float(t_capture)))
        except queue.Full:
            pass                       # 丢旧帧，保证实时

    def snapshot(self):
        with self._lock:
            return self._state

    def process(self, img, t_capture):
        """同步处理一帧（测试用，也可被 run() 调用）。"""
        t0 = time.time()
        dets = self.detector.detect(img)
        self.tracker.step(dets, t_capture)
        objs = [{"id": tr.id, "xy": tr.pos.tolist(), "vel": tr.vel.tolist(),
                 "sigma_mm": tr.sigma_mm, "conf": 1.0}
                for tr in self.tracker.active()]
        st = VisionState(t_capture=t_capture, t_processed=time.time(),
                         objects=objs)
        with self._lock:
            self._state = st
        self.n_frames += 1
        self.n_dets += len(dets)
        return st, (time.time() - t0)

    def run(self):
        last = 0.0
        while not self._stop_evt.is_set():
            try:
                img, t = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if t - last < self.min_dt:
                continue               # 限频
            last = t
            self.process(img, t)

    def stop(self):
        self._stop_evt.set()
