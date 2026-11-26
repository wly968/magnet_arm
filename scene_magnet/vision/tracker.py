"""多目标卡尔曼跟踪：恒速模型 + 门限匹配 + 遮挡续命。

单目标状态 x = [px, py, vx, vy]，P 为 4x4。

注意：本机 BLAS 的**矩阵乘法**（A @ B）会硬崩（0xc06d007f），
所以所有矩阵运算一律用 einsum 实现。
"""
from dataclasses import dataclass, field

import numpy as np


def mm(A, B):
    """矩阵乘（einsum 版，规避 BLAS 崩溃）。"""
    return np.einsum("ij,jk->ik", np.asarray(A, float), np.asarray(B, float))


def mv(A, b):
    return np.einsum("ij,j->i", np.asarray(A, float), np.asarray(b, float))


def _F(dt):
    return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def _Q(dt, q):
    q = float(q) ** 2
    return q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                         [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                         [dt ** 3 / 2, 0, dt ** 2, 0],
                         [0, dt ** 3 / 2, 0, dt ** 2]], float)


class KalmanTrack:
    """单目标恒速卡尔曼。"""

    _next_id = 0

    def __init__(self, z, t, sigma_mm=2.0, q_mm=3.0):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.x = np.array([z[0], z[1], 0.0, 0.0], float)
        self.P = np.diag([(sigma_mm / 1000) ** 2, (sigma_mm / 1000) ** 2,
                          0.05 ** 2, 0.05 ** 2])
        self.q = (q_mm / 1000.0)
        self.t = float(t)
        self.n_hits = 1
        self.miss = 0
        self.confirmed = False
        self.carried = False

    def predict(self, t):
        dt = max(float(t) - self.t, 1e-6)
        F = _F(dt)
        self.x = mv(F, self.x)
        self.P = mm(mm(F, self.P), F.T) + _Q(dt, self.q)
        self.t = float(t)

    def update(self, z, sigma_mm):
        R = np.eye(2) * (sigma_mm / 1000.0) ** 2
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = mm(mm(H, self.P), H.T) + R
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = mm(mm(self.P, H.T), Sinv)                 # 4x2
        innov = np.asarray(z, float) - mv(H, self.x)
        self.x = self.x + mv(K, innov)
        self.P = self.P - mm(mm(K, H), self.P)
        self.n_hits += 1
        self.miss = 0
        if self.n_hits >= 2:
            self.confirmed = True
        return float(np.sqrt(mv(Sinv, innov) @ innov))  # NIS 平方根

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:].copy()

    @property
    def sigma_mm(self):
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])) * 1000.0)

    def predicted_pos(self, dt):
        return self.x[:2] + self.x[2:] * dt


@dataclass
class MultiTracker:
    gate_mm: float = 40.0
    max_miss: int = 6
    q_mm: float = 3.0
    tracks: list = field(default_factory=list)

    def step(self, detections, t_now):
        dets = [np.asarray(d.xy, float) for d in detections]
        sig = [float(getattr(d, "sigma_mm", 2.0)) for d in detections]
        for tr in self.tracks:
            tr.predict(t_now)

        gate = self.gate_mm / 1000.0
        cands = []
        for ti, tr in enumerate(self.tracks):
            if tr.carried:
                continue
            for di, z in enumerate(dets):
                d = float(np.linalg.norm(tr.pos - z))
                if d < gate:
                    cands.append((d, ti, di))
        cands.sort()
        used_t, used_d = set(), set()
        for d, ti, di in cands:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(dets[di], sig[di])
            used_t.add(ti); used_d.add(di)

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t and not tr.carried:
                tr.miss += 1
        for di, z in enumerate(dets):
            if di not in used_d:
                self.tracks.append(KalmanTrack(z, t_now, sigma_mm=sig[di], q_mm=self.q_mm))
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return self.tracks

    def mark_carried(self, xy, tol=0.03):
        for tr in self.tracks:
            if np.linalg.norm(tr.pos - np.asarray(xy, float)) < tol:
                tr.carried = True

    def active(self):
        ts = [t for t in self.tracks if not t.carried and t.confirmed]
        return sorted(ts, key=lambda t: -float(t.pos[0]))

    def front(self):
        a = self.active()
        return a[0] if a else None
