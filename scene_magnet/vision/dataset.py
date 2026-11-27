"""YOLO-seg 数据集生成：图像 + 多边形标签 + data.yaml。

标签用方块**顶面四角**的投影多边形（归一化），类别只有 cube。

    cd scene_magnet
    python -m vision.dataset --images 400 --out vision/dataset
"""
import argparse
import os

import cv2
import numpy as np

from .camera import TOP_Z, project_with_model

HALF = 0.025          # 方块半边长


class ArmScene:
    """用真正的 ArmEnv（带机械臂）当渲染场景：手臂、阴影、遮挡都会出现在训练图里。"""

    def __init__(self, width=640, height=480, arm_motion=True):
        import mujoco

        from envs.arm_env import ArmEnv
        self.env = ArmEnv()
        self.model, self.data = self.env.model, self.env.data
        self.m, self.d = self.model, self.data      # 兼容 VisionScene.set_lighting
        self.cam_id = self.model.camera("birdeye").id
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.arm_motion = arm_motion
        self.home = self.data.qpos[self.env.arm_qpos_adr].copy()
        self._park_all()

    def _park_all(self):
        import mujoco
        for i in range(10):
            try:
                mid = int(self.model.body_mocapid[self.model.body(f"cube_{i}").id])
            except Exception:
                break
            self.data.mocap_pos[mid] = [0.0, 0.0, -5.0]
        mujoco.mj_forward(self.model, self.data)

    def place_cubes(self, positions):
        import mujoco
        self._park_all()
        for i, (x, y) in enumerate(positions[:10]):
            mid = int(self.model.body_mocapid[self.model.body(f"cube_{i}").id])
            self.data.mocap_pos[mid] = [float(x), float(y), 0.375]
        self.placed = [(float(x), float(y)) for x, y in positions[:10]]
        mujoco.mj_forward(self.model, self.data)

    def cube_positions(self):
        return list(self.placed)

    def randomize_arm(self, rng):
        """手臂随机位姿：既可能悬在带上方，也可能正压在方块附近。"""
        import mujoco
        q = self.home + rng.normal(0, 0.45, 4)
        self.data.qpos[self.env.arm_qpos_adr] = q
        mujoco.mj_forward(self.model, self.data)

    def set_lighting(self, level=0, rng=None):
        from .scene import VisionScene
        VisionScene.set_lighting(self, level, rng)      # 复用同一套光照随机化

    def render(self):
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()


def cube_top_polygon(cam, x, y):
    """顶面四角 -> 像素多边形（按顺时针）。cam 可以是 Camera 或 ArmScene。"""
    pts = [(x - HALF, y - HALF), (x + HALF, y - HALF),
           (x + HALF, y + HALF), (x - HALF, y + HALF)]
    if hasattr(cam, "project_point"):
        return [cam.project_point([px, py, TOP_Z]) for px, py in pts]
    return [project_with_model(cam.model, cam.cam_id, cam.width, cam.height, [px, py, TOP_Z])
            for px, py in pts]


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


def augment_image(img, rng, strength=1.0):
    """图像级域随机化：色偏、运动模糊、噪声、亮度/对比度。不改变几何，标签仍有效。"""
    out = img.astype(np.float32)
    # 1) 通道增益（模拟不同色温/白平衡）
    gain = 1.0 + strength * rng.uniform(-0.18, 0.18, 3)
    out *= gain.reshape(1, 1, 3)
    # 2) 亮度/对比度
    out = (out - 128.0) * (1.0 + strength * rng.uniform(-0.25, 0.25)) + 128.0
    out += strength * rng.uniform(-25, 25)
    # 3) 运动模糊（沿 x，方块在带上运动）
    if rng.random() < 0.6:
        k = int(rng.integers(3, 9))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)
    # 4) 高斯噪声
    out += rng.normal(0, strength * rng.uniform(1.0, 5.0), out.shape)
    # 5) 偶发 JPEG 压缩
    if rng.random() < 0.3:
        q = int(rng.integers(55, 95))
        ok, buf = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., :3].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


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


def build_dataset(n_images=400, out="dataset", val_ratio=0.2, verbose=True, with_arm=True):
    if with_arm:
        scene = ArmScene()
    else:
        from .scene import VisionScene
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
        if with_arm and hasattr(scene, "randomize_arm"):
            scene.randomize_arm(rng)
        img = scene.render()
        img = augment_image(img, rng, strength=0.4 + 0.6 * (level / 3.0))
        polys = [cube_top_polygon(getattr(scene, "cam", scene), x, y)
                 for x, y in scene.cube_positions()]
        stem = f"{i:05d}"
        cv2.imwrite(os.path.join(out, f"images/{split}/{stem}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        _w = getattr(getattr(scene, "cam", scene), "width", 640)
        _h = getattr(getattr(scene, "cam", scene), "height", 480)
        write_label(os.path.join(out, f"labels/{split}/{stem}.txt"), polys, _w, _h)
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
