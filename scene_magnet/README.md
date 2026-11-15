# scene_magnet — 抓取放置环境

本目录是运行主体。完整说明见仓库根目录的 [README.md](../README.md)。

## 快速开始

```bash
cd scene_magnet
python test.py     # 三模型接力演示（先跑 100 局统计，再开窗口）
python demo.py     # 键盘遥控
```

## 模型

| 文件 | 负责阶段 |
|---|---|
| `ppo_arm_reach.zip` | phase 1 取料、phase 2 下抓 |
| `saves/phase34_retract.zip` | phase 3 抬起、phase 4 转移、phase 5 放料 |
| `saves/phase6_home.zip` | phase 6 回到初始位置 |
| `saves/phase1.zip` | 训练热启动模型（`train_phase12.py` 使用） |
| `saves/phase2.zip` | 旧版 phase 1/2 模型（`train_phase34.py` 使用） |

## 训练

```bash
python train_phase12.py    # 训练 model1（phase 1/2）
```

model2 / model3 需要用 `envs/phase34.py`、`envs/phase6.py` 的交接封装训练：
先让前序模型跑到目标阶段起点，再只训练这一段策略。

## 说明

- 方块是 mocap 运动学体：phase 2 成功后锁定跟随末端（下方 6 cm、保持竖直），phase 5 成功后解锁；
- `ArmEnv(smooth_alpha=0.5)` 开启动作低通滤波，`phase_entry_steps=6` 在阶段切换处做额外平滑；
- 阶段阈值在 `test.py` 的 `TH_PHASE12 / TH_PHASE35 / TH_PHASE6` 中配置。
