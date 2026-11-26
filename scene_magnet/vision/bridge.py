"""与环境对接：把视觉快照变成观测里的目标位置 / 速度 / σ。

职责
----
- 取最新快照，按 now - t_capture 用速度外推到当前时刻
- 输出 (pos_xy, vel_xy, sigma_xy)，分别填到 obs 的
    obs[11:14] 目标位置（替换 box_pos）
    obs[19:22] σx, σy, σvx, σvy（替换恒零的占位）
- 成功判定仍用 MuJoCo 真值（评估时才知道真值）

计划接口
--------
class VisionBridge:
    attach(env)
    query(t_now) -> (pos_xy, vel_xy, sigma)
    use_vision: bool        # 关闭时退回真值

依赖：vision.vision_thread, vision.tracker
"""

# TODO: 待实现
