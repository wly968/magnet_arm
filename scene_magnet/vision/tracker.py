"""多目标卡尔曼跟踪：预测、关联、遮挡续命。

职责
----
- 单目标恒速卡尔曼：x = [px, py, vx, vy]，P 为 4x4
- 多目标关联：先预测到当前时刻，再用 代价 + 门限 + 匈牙利 匹配
  * 本场景顺序不交叉（单向带、间距 9 cm），可用 x 排序做一致性校验
  * 未匹配检测：连续 2 帧确认后才新建 track（防空检）
  * 未匹配 track：只预测，miss 累加；miss > 5（约 0.5 s）删除
  * 被抓走的方块：标记 carried 并停止预测，避免和下一个方块抢检测

计划接口
--------
class KalmanTrack:
    predict(dt); update(z, R); pos; vel; sigma; miss
class MultiTracker:
    step(detections, t_now) -> [KalmanTrack]
    predicted(dt) -> [[x, y], ...]
    active() -> 按 x 排序的当前目标

依赖：numpy（可选 scipy.optimize.linear_sum_assignment 做匈牙利）
"""

# TODO: 待实现
