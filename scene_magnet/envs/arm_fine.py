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
    # ---- 阶段 A：只收 dist（still 固定 2 帧）----
    (0.034, 0.965, 2, 0.05),
    (0.030, 0.965, 2, 0.05),
    (0.026, 0.965, 2, 0.05),
    (0.022, 0.965, 2, 0.05),
    (0.018, 0.965, 2, 0.05),
    (0.015, 0.965, 2, 0.05),
    (0.012, 0.965, 2, 0.05),
    (0.010, 0.965, 2, 0.05),
    # ---- 阶段 B：只收 angle ----
    (0.010, 0.972, 2, 0.05),
    (0.010, 0.978, 2, 0.05),
    (0.010, 0.983, 2, 0.05),
    (0.010, 0.987, 2, 0.05),
    (0.010, 0.990, 2, 0.05),
    (0.010, 0.9925, 2, 0.05),
    (0.010, 0.994, 2, 0.05),
    # ---- 阶段 C：最后拉 dwell（dist/angle 已固定）----
    (0.010, 0.994, 4, 0.05),
    (0.010, 0.994, 7, 0.05),
    (0.010, 0.994, 10, 0.05),
    (0.010, 0.994, 13, 0.05),
    (0.010, 0.994, 16, 0.05),
    (0.010, 0.994, 19, 0.05),
    (0.010, 0.994, 21, 0.05),
]


class ArmFineEnv(ArmEnv):
    """精度微调版：IK 更准、残差更细、高度带更窄、奖励带指数精度漏斗。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_gravity_ff = True  # 关键：抵消 3~4 cm 的伺服静差，静态精度 41 mm -> 0.06 mm
        self.ik_iters = 15          # IK 残差：5 次 4.5 mm -> 15 次约 0.1 mm
        self.ik_scale = 0.008       # 残差 0.008 rad ≈ 4 mm/单位动作（原 0.02）
        self.band_tol = 0.020       # 高度带 ±2 cm（原来 ±3 cm；再窄会和 dist 判据互相卡死）
        self.use_rel_criterion = False  # 磁吸是运动学吸附，不需要"零相对速度"
        self.still_decay = 1        # dwell 失败只衰减 1，不清零（否则 11 帧连续几乎不可能）
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
