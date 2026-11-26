"""与环境对接：视觉快照 -> 观测里的目标位置 / 速度 / σ。

用法（在 demo/test 里）：
    bridge = VisionBridge(thread, x_max=-0.10)
    ...
    thread.submit(img, now)
    st = bridge.query(now)          # 取最新快照，按延迟外推
    if st: 用 st["xy"] / st["vel"] / st["sigma_mm"]
"""
import time


class VisionBridge:
    def __init__(self, thread, x_max=-0.10, max_age=0.5):
        self.thread = thread
        self.x_max = x_max      # 只考虑这条 x 左侧的目标（可达范围内）
        self.max_age = max_age  # 快照太旧就不外推

    def query(self, t_now=None, dt_extra=0.0):
        """返回 dict(xy, vel, sigma_mm, age) 或 None。"""
        st = self.thread.snapshot()
        if st is None or not st.objects:
            return None
        t_now = time.time() if t_now is None else t_now
        age = max(0.0, t_now - st.t_capture) + dt_extra
        if age > self.max_age:
            return None
        obj = st.front(x_max=self.x_max)
        if obj is None:
            return None
        xy = [obj["xy"][0] + obj["vel"][0] * age,
              obj["xy"][1] + obj["vel"][1] * age]
        # σ 随外推时间增长：每 100 ms 增加 2 mm
        sigma = float(obj["sigma_mm"]) + 2.0 * (age / 0.1)
        return {"xy": xy, "vel": list(obj["vel"]), "sigma_mm": sigma,
                "age": age, "t_capture": st.t_capture, "id": obj["id"]}
