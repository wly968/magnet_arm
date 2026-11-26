"""YOLO-seg 数据集生成：图像 + 多边形标签 + data.yaml。

标签用方块**顶面四角**的投影多边形（归一化），类别只有 cube。

    cd scene_magnet
    python -m vision.dataset --images 400 --out vision/dataset
"""
import argparse
import os

import cv2
import numpy as np

from .camera import TOP_Z
from .scene import VisionScene

HALF = 0.025          # 方块半边长


def cube_top_polygon(cam, x, y):
    """顶面四角 -> 像素多边形（按顺时针）。"""
    pts = [(x - HALF, y - HALF), (x + HALF, y - HALF),
           (x + HALF, y + HALF), (x - HALF, y + HALF)]
    return [cam.project_point([px, py, TOP_Z]) for px, py in pts]


def random_layout(scene, rng, n_min=1, n_max=6):
    """在 A 带可抓范围内随机摆 1~6 个方块（最小间距 7 cm）。"""
    n = int(rng.integers(n_min, n_max + 1))
    xs = rng.uniform(-1.05, -0.20, n)
    ys = rng.uniform(0.56, 0.74, n)
    pts = []
    for x, y in zip(xs, ys):
        if all((x - px) ** 2 + (y - py) ** 2 > 0.07 ** 2 for px, py in pts):
            pts.append((float(x), float(y)))
    return pts


def write_label(path, polys, w, h):
    lines = []
    for poly in polys:
        if poly is None or any(p is None for p in poly):
            continue
        coords = []
        for u, v in poly:
            coords += [np.clip(u / w, 0, 1), np.clip(v / h, 0, 1)]
        lines.append("0 " + " ".join(f"{c:.6f}" for c in coords))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_dataset(n_images=400, out="dataset", val_ratio=0.2, verbose=True):
    scene = VisionScene()
    rng = np.random.default_rng(0)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    n_val = int(n_images * val_ratio)
    for i in range(n_images):
        split = "val" if i < n_val else "train"
        level = int(rng.integers(0, 4))
        scene.set_lighting(level, rng)
        scene.place_cubes(random_layout(scene, rng))
        img = scene.render()
        polys = [cube_top_polygon(scene.cam, x, y) for x, y in scene.cube_positions()]
        stem = f"{i:05d}"
        cv2.imwrite(os.path.join(out, f"images/{split}/{stem}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        write_label(os.path.join(out, f"labels/{split}/{stem}.txt"), polys,
                    scene.cam.width, scene.cam.height)
        if verbose and i % 50 == 0:
            print(f"  [{i}/{n_images}] level={level} cubes={len(polys)}")
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\n\nnames:\n  0: cube\n")
    if verbose:
        print(f"[dataset] 完成：{n_images} 张（val {n_val}），data.yaml 已写入 {out}")
    return out


def preview(out="dataset", n=6):
    """把前 n 张图的标签画出来，用于肉眼检查。"""
    scene = VisionScene()
    os.makedirs(os.path.join(out, "preview"), exist_ok=True)
    files = sorted(os.listdir(os.path.join(out, "images/train")))[:n]
    for f in files:
        img = cv2.imread(os.path.join(out, "images/train", f))
        h, w = img.shape[:2]
        lbl = os.path.join(out, "labels/train", f.replace(".png", ".txt"))
        for line in open(lbl, encoding="utf-8"):
            vals = [float(v) for v in line.split()[1:]]
            pts = np.array([[vals[i] * w, vals[i + 1] * h] for i in range(0, len(vals), 2)],
                           dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(out, "preview", f), img)
    print(f"[preview] 已写 {len(files)} 张到 {out}/preview")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=400)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    build_dataset(a.images, a.out)
    if a.preview:
        preview(a.out)
