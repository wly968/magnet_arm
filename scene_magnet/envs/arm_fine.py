"""phase 1/2 的**精度微调**环境（dwell 课程 + 线性/指数混合奖励）。

用法：从 `saves/phase1-2.zip` 热启动，在这个环境里继续训练，
阈值由 `FINE_SCHEDULE` 逐步收紧（距离 2 cm → 5 mm，角度 0.96 → 0.993，
静止帧数 1 → 20+），最终要求"贴住并稳住 0.2 秒以上"——既练精度，
也为将来换成真实夹取留出稳定窗口。

    from envs.arm_fine import ArmFineEnv, FINE_SCHEDULE
    env = ArmFineEnv()
"""
import numpy as np

from .arm_env import ArmEnv

# (dist, angle, still_count 阈值)
# still = N 表示需要连续 N+1 帧都满足判据（0.01 s/帧）
FINE_SCHEDULE = [
    (0.020, 0.960, 1),     # 2.0 cm / 16° / 2 帧：从原模型能力起步
    (0.015, 0.975, 3),     # 1.5 cm / 13° / 4 帧
    (0.010, 0.985, 6),     # 1.0 cm /  10° / 7 帧
    (0.007, 0.990, 10),    # 7 mm  /   8° / 11 帧
    (0.005, 0.993, 15),    # 5 mm  /   7° / 16 帧
    (0.004, 0.995, 20),    # 4 mm  /   6° / 21 帧（≈0.21 s 稳定）
]


class ArmFineEnv(ArmEnv):
    """精度微调版：IK 更准、残差更细、高度带更窄、奖励带指数精度漏斗。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ik_iters = 15          # IK 残差：5 次 4.5 mm -> 15 次约 0.1 mm
        self.ik_scale = 0.008       # 残差 0.008 rad ≈ 4 mm/单位动作（原 0.02）
        self.band_tol = 0.012       # 高度带 ±1.2 cm（原 ±3 cm）
        self.phase2_reward_fn = self._fine_reward

    @staticmethod
    def _fine_reward(dist, angle1, action_change, still_count):
        """线性项管远处、指数项管最后几毫米（精度漏斗）。

        指数项：sigma_d = 1 cm，sigma_a = 1-0.98 → 角度误差 2% 尺度。
        两项在零点附近给出很陡的梯度，把一个"大约到位"的策略压到毫米级。
        """
        d_exp = 1.0 - np.exp(-dist / 0.010)
        a = max(0.0, 1.0 - angle1)
        a_exp = 1.0 - np.exp(-a / 0.010)
        reward = (
            -4.0 * dist                     # 距离：线性
            - 12.0 * a                      # 角度：线性（1° ≈ 0.017）
            - 2.0 * d_exp                   # 距离：指数漏斗（1 cm 尺度）
            - 2.0 * a_exp                   # 角度：指数漏斗
            - 0.3 * action_change
            + 0.4 * min(still_count, 8)     # 保持奖励：鼓励"贴住不动"
        )
        return float(reward)
