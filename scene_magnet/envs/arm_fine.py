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
# 规则：从原课程第 5 档出发，之后**每一档只收紧 dist / angle / still 中的一项**，
# 三项轮流推进，避免"三个条件同时变严"导致成功率断崖。
FINE_SCHEDULE = [
    (0.034, 0.965, 2, 0.05),    #  1 起点：原课程第 5 档
    (0.030, 0.965, 2, 0.05),    #  2 dist
    (0.030, 0.972, 2, 0.05),    #  3 angle
    (0.030, 0.972, 4, 0.05),    #  4 still
    (0.026, 0.972, 4, 0.05),    #  5 dist
    (0.026, 0.978, 4, 0.05),    #  6 angle
    (0.026, 0.978, 7, 0.05),    #  7 still
    (0.022, 0.978, 7, 0.05),    #  8 dist
    (0.022, 0.983, 7, 0.05),    #  9 angle
    (0.022, 0.983, 10, 0.05),   # 10 still
    (0.019, 0.983, 10, 0.05),   # 11 dist
    (0.019, 0.987, 10, 0.05),   # 12 angle
    (0.019, 0.987, 13, 0.05),   # 13 still
    (0.016, 0.987, 13, 0.05),   # 14 dist
    (0.016, 0.990, 13, 0.05),   # 15 angle
    (0.016, 0.990, 16, 0.05),   # 16 still
    (0.013, 0.990, 16, 0.05),   # 17 dist
    (0.013, 0.9925, 16, 0.05),  # 18 angle
    (0.013, 0.9925, 19, 0.05),  # 19 still
    (0.010, 0.9925, 19, 0.05),  # 20 dist
    (0.010, 0.994, 19, 0.05),   # 21 angle
    (0.010, 0.994, 21, 0.05),   # 22 still（≈0.21 s 稳定，可接夹取）
]


class ArmFineEnv(ArmEnv):
    """精度微调版：IK 更准、残差更细、高度带更窄、奖励带指数精度漏斗。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_gravity_ff = True  # 关键：抵消 3~4 cm 的伺服静差，静态精度 41 mm -> 0.06 mm
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

    def step(self, action):
        """微调只训 phase 1/2：抓到即终止，不进入 model1 没学过的 phase 3-6。"""
        obs, reward, terminated, truncated, info = super().step(action)
        if self.phase >= 3 and not terminated:
            terminated = True
            info["final_phase"] = self.phase
        return obs, reward, terminated, truncated, info
