"""虚拟视觉：用真值 + 噪声 + 延迟 + 丢帧 模拟视觉估计。

用途：给策略做"视觉域随机化"微调，让它对 1~3 mm 的估计误差与几十毫秒延迟不敏感，
而不必每次都跑真实的 YOLO 推理（那样训练速度会降一个数量级）。

    sim = VirtualVision(sigma_mm=(0.5, 3.0), delay_steps=(1, 3), dropout=0.02)
    env.set_vision(sim, enable=True)      # 它和 VisionBridge 接口一致：query(t)
"""
from collections import deque

import numpy as np


class VirtualVision:
    def __init__(self, sigma_mm=(0.5, 3.0), vel_sigma=0.03,
                 delay_steps=(1, 3), dropout=0.02, seed=0):
        self.sigma_range = sigma_mm
        self.vel_sigma = vel_sigma
        self.delay_range = delay_steps
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.hist = deque(maxlen=8)
        self.est = None
        self.sigma_mm = float(np.mean(sigma_mm))
        self.miss = 0

    def reset(self):
        self.hist.clear()
        self.est = None
        self.miss = 0
        self.delay = int(self.rng.integers(self.delay_range[0], self.delay_range[1] + 1))
        self.sigma_mm = float(self.rng.uniform(*self.sigma_range))

    def observe(self, truth_pos, truth_vel, dt=0.01):
        """每个物理步调用一次：把真值压入历史，并按延迟+噪声生成估计。"""
        if not self.hist:
            self.reset()
        self.hist.append((np.asarray(truth_pos, float).copy(),
                          np.asarray(truth_vel, float).copy()))
        if self.rng.random() < self.dropout:
            self.miss += 1
            return                      # 丢帧：保持上一次估计
        idx = min(self.delay, len(self.hist) - 1)
        pos, vel = self.hist[-1 - idx]
        noise = self.rng.normal(0.0, self.sigma_mm / 1000.0, 2)
        self.est = {"xy": [float(pos[0] + noise[0]), float(pos[1] + noise[1])],
                    "vel": [float(vel[0] + self.rng.normal(0, self.vel_sigma)),
                            float(vel[1] + self.rng.normal(0, self.vel_sigma))],
                    "sigma_mm": self.sigma_mm}
        self.miss = 0

    def query(self, t_now=None):
        return self.est

    def mark_carried(self, xy, tol=None):
        pass
