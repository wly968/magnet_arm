# tools — 实验与诊断脚本

开发过程中的一次性脚本（评测、诊断、调参、可视化），**不属于运行链路**。
在仓库根目录执行：

```bash
python tools/<脚本名>.py
```

脚本假设工作目录是仓库根目录（内部多用 `scene_magnet/...` 相对路径）。

## 仍然有用的几组

**模型评测**

| 脚本 | 作用 |
|---|---|
| `eval_new_phase2.py` | model1 在六档课程下的成功率（每档 30 局） |
| `eval_model2_now.py` | model2 从 phase 3 到放料完成的成功率 |
| `eval_m3_new.py` | model3 在不同 `home_still` 阈值下的成功率 |
| `full_chain_100.py` | 三模型接力 100 局端到端统计 |
| `det_vs_sto_p35.py` | 同一模型确定性 vs 随机采样的差异（判断"是否靠噪声蒙中"） |

**可达性 / 阈值可行性**

| 脚本 | 作用 |
|---|---|
| `reach_grasp_y.py` / `reach_x.py` | 抓取点在 y / x 方向的可达范围（多起点 IK） |
| `homotopy2.py` | 从真实抓取构型出发，沿 y 连续跟踪，找姿态边界 |
| `reach_new_waypoint.py` | 转移点候选位置的可达性 |
| `angle_effect.py` | 抓取角度阈值对成功率的影响（0.98 / 0.965 / 0.945） |
| `y_sweep_096.py` / `y_sweep_fine.py` | 按 y 分桶扫描成功率，定位"死区" |

**课程 / 训练诊断**

| 脚本 | 作用 |
|---|---|
| `read_eval_p35.py` / `read_eval_p6.py` | 读 EvalCallback 的 `evaluations.npz` 曲线 |
| `analyze_log.py` | 解析 `saves/train_log.txt`，按档位汇总 |
| `ckpt_trend.py` | 同一轮各检查点的成功率趋势（判断是否收敛） |
| `probe_unit.py` | 单独测试课程探测函数（不经训练循环） |
| `stopper_ab.py` | 挡料器开/关、不同投放间隔的 A/B |
| `pool_wait_measure.py` | 方块池模式下每轮的等待步数 |

**动力学 / 动作分辨率**

| 脚本 | 作用 |
|---|---|
| `speed_measure.py` | 关节角速度、末端线速度统计 |
| `twist_metric.py` / `twist_vs_threshold.py` | 转移段"扭动"量化（方向反转次数、yaw 摆动） |
| `scale_requirement.py` | 由 IK 反推所需的 `ctrl_scale` |

## 其他

- `retrain/` 子目录保存训练日志与旧场景变体（如 5-DOF 实验）；
- 其余大量 `smoke_*` / `dbg_*` / `verify_*` 是当时的临时验证脚本，保留仅作记录，可能依赖已经改过的接口。
