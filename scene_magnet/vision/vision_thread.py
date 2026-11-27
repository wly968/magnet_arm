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


class VisionPipeline:
    """同步视觉管线：检测 -> 跟踪 -> 快照。没有线程、没有队列，零延迟。

    适合"仿真比实时快"的场合（headless 自检）：时间戳直接用仿真时间，
    不存在队列积压导致的估计滞后。
    """

    def __init__(self, detector, tracker=None):
        self.detector = detector
        self.tracker = tracker or MultiTracker()
        self._state = VisionState()
        self.n_frames = 0
        self.n_dets = 0

    def step(self, img, t_capture):
        t0 = time.time()
        dets = self.detector.detect(img)
        self.tracker.step(dets, float(t_capture))
        objs = [{"id": tr.id, "xy": tr.pos.tolist(), "vel": tr.vel.tolist(),
                 "sigma_mm": tr.sigma_mm, "conf": 1.0,
                 "uv": None if getattr(tr, "uv", None) is None else tr.uv.tolist()}
                for tr in self.tracker.active()]
        self._state = VisionState(t_capture=float(t_capture), t_processed=time.time(),
                                  objects=objs)
        self.n_frames += 1
        self.n_dets += len(dets)
        return self._state, (time.time() - t0)

    def snapshot(self):
        return self._state


class VisionThread(threading.Thread):
    """异步包装：主线程 submit() 投帧，本线程跑 VisionPipeline。"""

    def __init__(self, detector, max_hz=12.0, tracker=None, daemon=True):
        super().__init__(daemon=daemon)
        self.pipeline = VisionPipeline(detector, tracker)
        self.detector = detector
        self.tracker = self.pipeline.tracker
        self.min_dt = 1.0 / float(max_hz)
        self._q = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._state = VisionState()
        self._stop_evt = threading.Event()   # 别叫 _stop：Thread 自带同名方法
        self.n_frames = 0
        self.n_dets = 0

    # 主线程调用
    def submit(self, img, t_capture):
        """投递一帧。队列满时**丢弃最旧的**，保证线程处理的永远是最新画面。"""
        item = (np.asarray(img), float(t_capture))
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()   # 扔掉最旧的一帧
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def snapshot(self):
        with self._lock:
            return self._state

    def process(self, img, t_capture):
        """同步处理一帧（测试用，也可被 run() 调用）。"""
        st, dt = self.pipeline.step(img, t_capture)
        with self._lock:
            self._state = st
        self.n_frames = self.pipeline.n_frames
        self.n_dets = self.pipeline.n_dets
        return st, dt

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
