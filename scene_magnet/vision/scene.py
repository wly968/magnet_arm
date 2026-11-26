"""场景封装：vision_scene、方块摆放、光照随机化。

时间节约点：光照只改 MuJoCo 的 light_* 与 headlight 参数，不重新加载模型。
"""
import numpy as np

from .camera import Camera, SCENE, PLANE_Z


class VisionScene:
    """包住 Camera 的 model/data，提供"摆方块 + 随机光照 + 渲染"。"""

    def __init__(self, width=640, height=480, n_cubes=10, scene=SCENE):
        self.cam = Camera(width=width, height=height, scene=scene)
        self.m, self.d = self.cam.model, self.cam.data
        self.mids = []
        for i in range(n_cubes):
            try:
                bid = self.m.body(f"cube_{i}").id
            except Exception:
                break
            self.mids.append(int(self.m.body_mocapid[bid]))
        self.n_cubes = len(self.mids)
        self.placed = []
        self.park_all()

    # ---------- 方块 ----------
    def park_all(self):
        for mid in self.mids:
            self.d.mocap_pos[mid] = [0.0, 0.0, -5.0]
            self.d.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]
        self.placed = []
        self._forward()

    def place_cubes(self, positions):
        """positions: [(x, y), ...]，最多 self.n_cubes 个。"""
        self.park_all()
        self.placed = list(positions)[: self.n_cubes]
        for mid, (x, y) in zip(self.mids, self.placed):
            self.d.mocap_pos[mid] = [float(x), float(y), PLANE_Z]
        self._forward()

    def cube_positions(self):
        return [(float(x), float(y)) for x, y in self.placed]

    # ---------- 光照 ----------
    def set_lighting(self, level=0, rng=None):
        """level 0(无干扰)~3(强干扰)：位置、色温、强度、高光、环境光。"""
        rng = rng or np.random.default_rng()
        m = self.m
        amp = {0: 0.0, 1: 0.35, 2: 0.7, 3: 1.0}.get(int(level), 0.0)
        try:
            base_pos = np.array(m.light_pos[0])
            m.light_pos[0] = base_pos + amp * rng.normal(0, 0.35, 3)
            warm = 1.0 + amp * rng.uniform(-0.35, 0.35)
            m.light_diffuse[0] = np.clip(np.array([1.0, 0.95, 0.9]) * warm
                                         * (1.0 + amp * rng.uniform(-0.3, 0.5)), 0.05, 3.0)
            m.light_specular[0] = np.clip(np.array([0.3, 0.3, 0.3])
                                          * (1.0 + amp * rng.uniform(0, 2.5)), 0.0, 3.0)
        except Exception:
            pass
        try:                     # 环境光（headlight）也跟着变
            m.vis.headlight.ambient[:] = np.clip(0.1 + amp * rng.uniform(-0.05, 0.25), 0, 1)
            m.vis.headlight.diffuse[:] = np.clip(0.4 + amp * rng.uniform(-0.2, 0.4), 0, 1)
        except Exception:
            pass

    def _forward(self):
        import mujoco
        mujoco.mj_forward(self.m, self.d)

    def render(self):
        return self.cam.render()
