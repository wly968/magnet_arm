# scene_magnet — 双传送带取放环境

运行主体。整体设计、结果与训练原理见仓库根目录的 [README.md](../README.md)。

## 快速开始

```bash
cd scene_magnet

python demo.py                              # 连续抓取演示（默认 10 个方块轮转、1.5 s 一个）
python demo.py --headless 30                # 不开窗口，空跑 30 秒自检
python demo.py --interval 2                 # 想要回合间停顿再加这个参数
python demo.py --cubes 1                    # 单方块模式（每轮从带左端重新投一个）

python test.py                              # 先无渲染跑 20 局统计，再开窗口逐局演示
python test.py --no-viewer --episodes 100   # 纯批量评测
```

`test.py` 的模式由拿到的模型决定：

| 拿到 | 模式 | 目标阶段 |
|---|---|---|
| 只有 model1 | `phase 1/2` | 3 |
| model1 + model2 | `phase 1-5` | 6 |
| 三个都有 | 全流程 | 7 |

## 文件

| 文件 | 作用 |
|---|---|
| `envs/arm_env.py` | 主环境：七阶段状态机、IK 基准 + 残差动作、mocap 方块池 |
| `envs/phase34.py` | model2 训练封装：每回合先跑 model1 到 phase 3 |
| `envs/phase6.py` | model3 训练封装：先跑 model1、model2 到 phase 6 |
| `callbacks.py` | 全部回调：`CurriculumCallback`、`PhaseCurriculumCallback`、`PhaseTrackerCallback`、`DiagCallback`、`TrainLogCallback` |
| `train_phase1-2.py` / `train_phase3-5.py` / `train_phase6.py` | 三个阶段各自的训练脚本 |
| `test.py` | 接力评测 + 可视化 |
| `demo.py` | 连续抓取演示 |
| `assets/models/train_scene.xml` | 训练场景（10 个方块槽位） |

## 模型

| 文件 | 负责 | 判定 |
|---|---|---|
| `saves/phase1-2.zip` | phase 0-2 | dist 3.2 cm / angle 0.96 / 2 帧 |
| `saves/phase3-5.zip` | phase 3-5 | dist 3.4 cm / angle 0.965 / 3 帧 |
| `saves/phase6.zip` | phase 6 | 关节偏差 < 0.15 rad / 3 帧 |

`train_phase1-2.py` 读 `saves/phase1-2.zip` 热启动、写 `ppo_arm_reach.zip`；
`train_phase3-5.py` 读 `saves/phase1-2.zip`、写 `ppo_arm_reach.zip`；
`train_phase6.py` 读 `saves/phase1-2.zip` + `saves/phase3-5.zip`、写 `ppo_arm_reach.zip`。
**每个脚本训完要手动把 `ppo_arm_reach.zip` 改名成下一阶段要读的名字。**

## 阈值在哪改

| 内容 | 位置 |
|---|---|
| 课程档位（dist/angle/still/rel） | `callbacks.py` → `CurriculumCallback.SCHEDULE` |
| 评测/演示用的每阶段阈值 | `test.py`、`demo.py` → `TH_PHASE1/2/35/6` |
| phase 1 固定阈值 | `envs/arm_env.py` → `self.phase1_dist / phase1_angle` |
| phase 4 姿态阈值 | `envs/arm_env.py` → `self.phase4_angle` |
| phase 6 关节容差与 dwell | `envs/arm_env.py` → `self.home_tol / home_still_threshold` |
| 刷新区间、带速、投放点 | `envs/arm_env.py` → `spawn_x_range / spawn_y_range / cube_speed / pool_spawn_x` |
| 回合步数上限 | `envs/arm_env.py` → `max_steps`（任务步）/ `max_wait_steps`（phase 0 等待） |

## 训练日志

`TrainLogCallback` 每 2 万步往 `saves/train_log.txt` 写一行：

```
step stage thr(dist/angle/still/rel) probe pos relpos all mrel mdist mang mdz succ tout nan std ev kl
```

判断收敛看 `probe`（确定性成功率）和 `succ`（结束原因），**不要看 `ep_len_mean`**。

## 备注

- 方块是 mocap 运动学体：phase 2 成功后跟随末端下方 6 cm、保持竖直，phase 5 成功后交给 B 带；
- `phase 2` 成功那行的返回值决定回合是否结束（训练 model1 用 `True`，接力/演示用 `False`），
  在 `envs/arm_env.py` 的 phase 2 分支里，只有一行；
- 改阈值/奖励/阶段结构后需要重训对应模型。
