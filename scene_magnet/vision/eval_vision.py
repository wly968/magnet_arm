"""视觉模块评测：精度、数量、光照鲁棒性、端到端影响。

职责
----
- 位置误差：与真值比，报中位数与 95 分位（mm）
- 数量正确率：检测数量 == 真值数量的比例
- 光照档位：0~3 各档的位置误差与漏检率
- 遮挡实验：人为遮挡 0.3 s，测恢复后的位置误差
- 端到端：use_vision=True/False 时的接力成功率差

计划接口
--------
bench_detector(detector, n=200) -> dict
bench_lighting(levels=(0,1,2,3)) -> dict
bench_occlusion(seconds=0.3) -> dict
bench_end_to_end() -> dict

验收目标
--------
位置误差中位 < 2 mm、95 分位 < 5 mm；数量正确率 > 98%；
视觉版成功率较真值版下降 < 10 个百分点。
"""

# TODO: 待实现
