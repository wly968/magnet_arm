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
    # ---- A 段：只收 dist（3.4 cm -> 1.4 cm），still 固定 2 帧 ----
    (0.034, 0.965, 2, 0.05),
    (0.030, 0.965, 2, 0.05),
    (0.027, 0.965, 2, 0.05),
    (0.024, 0.965, 2, 0.05),
    (0.021, 0.965, 2, 0.05),
    (0.019, 0.965, 2, 0.05),
    (0.017, 0.965, 2, 0.05),
    (0.015, 0.965, 2, 0.05),
    (0.014, 0.965, 2, 0.05),
    # ---- B 段：只收 angle（0.965 -> 0.992）----
    (0.014, 0.970, 2, 0.05),
    (0.014, 0.975, 2, 0.05),
    (0.014, 0.980, 2, 0.05),
    (0.014, 0.984, 2, 0.05),
    (0.014, 0.987, 2, 0.05),
    (0.014, 0.990, 2, 0.05),
    (0.014, 0.992, 2, 0.05),
    # ---- C 段：最后拉 dwell（2 -> 15 帧）----
    (0.014, 0.992, 4, 0.05),
    (0.014, 0.992, 6, 0.05),
    (0.014, 0.992, 8, 0.05),
    (0.014, 0.992, 10, 0.05),
    (0.014, 0.992, 12, 0.05),
    (0.014, 0.992, 14, 0.05),
    (0.014, 0.992, 15, 0.05),
]


class ArmFineEnv(ArmEnv):
    """精度微调版：IK 更准、残差更细、高度带更窄、奖励带指数精度漏斗。"""

    # 两套奖励：v1 用于本轮（温和），v2 用于下一轮（陡峭，进一步收敛）
    PROFILES = {
        "v1": dict(w_dist=8.0, w_ang=12.0, eps_d=0.010, w_ed=2.0,
                   eps_a=0.010, w_ea=2.0, hold=0.3, hold_cap=5),
        "v2": dict(w_dist=20.0, w_ang=30.0, eps_d=0.004, w_ed=4.0,
                   eps_a=0.004, w_ea=4.0, hold=0.5, hold_cap=8),
    }

    def __init__(self, reward_profile="v1", **kwargs):
        super().__init__(**kwargs)
        self.use_gravity_ff = True  # 关键：抵消 3~4 cm 的伺服静差，静态精度 41 mm -> 0.06 mm
        self.ik_iters = 15          # IK 残差：5 次 4.5 mm -> 15 次约 0.1 mm
        self.ik_scale = 0.008       # 残差 0.008 rad ≈ 4 mm/单位动作（原 0.02）
        self.band_tol = 0.020       # 高度带 ±2 cm
        self.use_rel_criterion = False  # 磁吸是运动学吸附，不需要"零相对速度"
        self.still_decay = 1        # dwell 失败只衰减 1，不清零
        self.reward_profile = reward_profile
        self.phase2_reward_fn = self._reward

    def _reward(self, dist, angle1, action_change, still_count, all_ok):
        """线性项管远处、指数项管最后几毫米；保持奖励只在**真正满足判据**时发放。

        注意：保持奖励必须挂在 all_ok 上。之前挂在 still_count 上，
        漏积分器会把 still_count 抬高，策略悬在判据外也能拿钱 -> 奖励≈0 的局部最优。
        """
        p = self.PROFILES[self.reward_profile]
        a = max(0.0, 1.0 - angle1)
        r = (-p["w_dist"] * dist
             - p["w_ang"] * a
             - p["w_ed"] * (1.0 - np.exp(-dist / p["eps_d"]))
             - p["w_ea"] * (1.0 - np.exp(-a / p["eps_a"]))
             - 0.3 * action_change)
        if all_ok:
            r += p["hold"] * min(still_count, p["hold_cap"])
        return float(r)

    def step(self, action):
        """微调只训 phase 1/2：抓到即终止，不进入 model1 没学过的 phase 3-6。"""
        obs, reward, terminated, truncated, info = super().step(action)
        if self.phase >= 3 and not terminated:
            terminated = True
            info["final_phase"] = self.phase
        return obs, reward, terminated, truncated, info
