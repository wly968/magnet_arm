"""视觉评测：位置误差、数量正确率、光照鲁棒性。

    cd scene_magnet
    python -m vision.eval_vision                # 评测颜色基线
    python -m vision.eval_vision --weights runs/segment/train/weights/best.pt
"""
import argparse

import numpy as np

from .dataset import random_layout
from .scene import VisionScene


def match_errors(dets, truth, tol=0.03):
    """贪心最近邻匹配，返回 (errors_mm, n_matched)。"""
    used = set()
    errs = []
    for tx, ty in truth:
        best, bj = tol, -1
        for j, d in enumerate(dets):
            if j in used:
                continue
            e = float(np.hypot(d.xy[0] - tx, d.xy[1] - ty))
            if e < best:
                best, bj = e, j
        if bj >= 0:
            used.add(bj)
            errs.append(best * 1000.0)
    return errs, len(used)


def bench(detector, scene=None, n=40, levels=(0, 1, 2, 3), seed=0, verbose=True):
    scene = scene or VisionScene()
    rng = np.random.default_rng(seed)
    out = {}
    for lv in levels:
        errs, n_ok, n_truth, n_det = [], 0, 0, 0
        for _ in range(n):
            scene.set_lighting(lv, rng)
            scene.place_cubes(random_layout(scene, rng))
            img = scene.render()
            dets = detector.detect(img)
            e, m = match_errors(dets, scene.cube_positions())
            errs += e
            n_ok += m
            n_truth += len(scene.cube_positions())
            n_det += len(dets)
        rec = {
            "pos_med_mm": float(np.median(errs)) if errs else float("nan"),
            "pos_p95_mm": float(np.percentile(errs, 95)) if errs else float("nan"),
            "recall": n_ok / max(n_truth, 1),
            "count_acc": 1.0 - abs(n_det - n_truth) / max(n_truth, 1),
        }
        out[lv] = rec
        if verbose:
            print(f"  光照档 {lv}: 位置误差 中位 {rec['pos_med_mm']:.2f} mm / "
                  f"95% {rec['pos_p95_mm']:.2f} mm | 召回 {rec['recall']:.0%} | "
                  f"数量准确 {rec['count_acc']:.0%}")
    return out


def bench_tracking(scene=None, detector=None, fps=12.0, steps=60, seed=1, verbose=True):
    """模拟带速 0.12 m/s，中途遮挡 0.3 s，测跟踪误差。"""
    from .tracker import MultiTracker
    scene = scene or VisionScene()
    detector = detector or __import__("vision.detect", fromlist=["ColorDetector"]).ColorDetector()
    rng = np.random.default_rng(seed)
    tr = MultiTracker()
    scene.set_lighting(1, rng)
    scene.place_cubes([(-0.95, 0.65), (-0.70, 0.60)])
    errs, blind = [], 0
    dt = 1.0 / fps
    v = 0.12
    for k in range(steps):
        t = k * dt
        # 前进（被遮挡期间也照常前进，只是看不到）
        pts = [(x + v * dt, y) for x, y in scene.cube_positions()]
        scene.place_cubes(pts)
        occluded = (k * dt > 1.5) and (k * dt < 1.8)      # 0.3 s 遮挡
        dets = [] if occluded else detector.detect(scene.render())
        if occluded:
            blind += 1
        tr.step(dets, t)
        f = tr.front()
        if f is not None:
            tx = max(pts, key=lambda p: p[0])
            errs.append(float(np.hypot(f.pos[0] - tx[0], f.pos[1] - tx[1])) * 1000.0)
    if verbose:
        print(f"  跟踪（含 0.3 s 遮挡 {blind} 帧）：误差 中位 {np.median(errs):.2f} mm / "
              f"最大 {np.max(errs):.2f} mm")
    return {"med_mm": float(np.median(errs)), "max_mm": float(np.max(errs))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="给了就用 YOLO-seg，否则用颜色基线")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    if a.weights:
        from .detect import SegDetector
        det = SegDetector(a.weights)
        print(f"[eval] YOLO-seg: {a.weights}")
    else:
        from .detect import ColorDetector
        det = ColorDetector()
        print("[eval] 颜色基线")
    bench(det, n=a.n)
    bench_tracking(detector=det)
