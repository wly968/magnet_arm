"""视觉鲁棒性微调环境：在 ArmFineEnv 的基础上注入视觉噪声/延迟/丢帧。

训练出来的 model1 会对"估计位置有 1~3 mm 误差、延迟 1~3 步"的输入更宽容。
"""
import numpy as np

from .arm_fine import ArmFineEnv
from .vision_sim import VirtualVision

# (dist, angle, still, rel)：从较宽松的视觉档位逐步收紧到最终精度
VISION_SCHEDULE = [
    (0.020, 0.980, 4, 0.05),
    (0.017, 0.985, 8, 0.05),
    (0.014, 0.990, 12, 0.05),
    (0.014, 0.992, 15, 0.05),
]


class ArmVisionFineEnv(ArmFineEnv):
    def __init__(self, sigma_mm=(0.5, 3.0), delay_steps=(1, 3), dropout=0.02,
                 seed=0, **kwargs):
        super().__init__(**kwargs)
        self.vision = VirtualVision(sigma_mm=sigma_mm, delay_steps=delay_steps,
                                    dropout=dropout, seed=seed)
        self.use_vision = True

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        if self.vision is not None:
            self.vision.reset()
            self.vision.observe(self.box_pos, self.box_vel)
        return out
