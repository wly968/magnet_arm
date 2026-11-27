"""与环境对接：视觉快照 -> 观测里的目标位置 / 速度 / σ。

用法（在 demo/test 里）：
    bridge = VisionBridge(thread, x_max=-0.10)
    ...
    thread.submit(img, now)
    st = bridge.query(now)          # 取最新快照，按延迟外推
    if st: 用 st["xy"] / st["vel"] / st["sigma_mm"]
"""
import time

import numpy as np


class VisionBridge:
    def __init__(self, thread, x_max=-0.10, max_age=0.5):
        self.thread = thread
        self.x_max = x_max      # 只考虑这条 x 左侧的目标（可达范围内）
        self.max_age = max_age  # 快照太旧就不外推
        self.locked_id = None   # 锁定某个 track（进入 phase 1 时锁定，抓完解锁）

    def lock_front(self, t_now=None):
        """锁定当前最前方的目标，之后 query() 一直返回它（避免方块排队时串目标）。"""
        obj = self._pick(t_now)
        self.locked_id = None if obj is None else obj["id"]
        return self.locked_id

    def lock_nearest(self, xy, tol=0.08):
        """按已知位置锁定最近的 track（用于与环境"当前要抓的方块"对齐）。"""
        st = self.thread.snapshot()
        if st is None or not st.objects:
            self.locked_id = None
            return None
        xy = np.asarray(xy, float)[:2]
        best, bid = tol, None
        for o in st.objects:
            d = float(np.hypot(o["xy"][0] - xy[0], o["xy"][1] - xy[1]))
            if d < best:
                best, bid = d, o["id"]
        self.locked_id = bid
        return bid

    def unlock(self):
        self.locked_id = None

    def _pick(self, t_now=None):
        st = self.thread.snapshot()
        if st is None or not st.objects:
            return None
        if self.locked_id is not None:
            for o in st.objects:
                if o["id"] == self.locked_id:
                    return o
            return None                  # 锁定的目标丢了，交给上层决定
        objs = [o for o in st.objects if o["xy"][0] <= self.x_max]
        if not objs:
            return None
        return max(objs, key=lambda o: o["xy"][0])

    def mark_carried(self, xy, tol=0.05):
        """方块被抓走后调用：对应 track 停止预测，不再参与"最前方"的选择。"""
        tr = getattr(self.thread, "tracker", None)
        if tr is not None:
            tr.mark_carried(np.asarray(xy, float)[:2], tol=tol)

    def query(self, t_now=None, dt_extra=0.0):
        """返回 dict(xy, vel, sigma_mm, age) 或 None。"""
        st = self.thread.snapshot()
        if st is None or not st.objects:
            return None
        t_now = time.time() if t_now is None else t_now
        age = max(0.0, t_now - st.t_capture) + dt_extra
        if age > self.max_age:
            return None
        obj = self._pick(t_now)
        if obj is None:
            return None
        xy = [obj["xy"][0] + obj["vel"][0] * age,
              obj["xy"][1] + obj["vel"][1] * age]
        # σ 随外推时间增长：每 100 ms 增加 2 mm
        sigma = float(obj["sigma_mm"]) + 2.0 * (age / 0.1)
        return {"xy": xy, "vel": list(obj["vel"]), "sigma_mm": sigma,
                "age": age, "t_capture": st.t_capture, "id": obj["id"]}
