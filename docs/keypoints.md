# 关键点坐标查询指南

配套脚本：[`keypoints.py`](keypoints.py) —— 直接运行会打印下面所有数值，并演示反解。

```bash
python docs/keypoints.py            # show_scene.xml
python docs/keypoints.py train      # train_scene.xml（结构相同）
```

两个 XML 的机械臂关节名、site 名、传送带 geom 名、相机名完全一致；
唯一的结构差异是 **train 的 `cube_0` 是 mocap 体**（强化学习里用来随机放置目标），
show 的 `cube_0` 是普通自由刚体。

---

## 1. 坐标系与关键尺寸

- 世界坐标：**z 向上**，单位米，原点在地面中心。
- 机械臂底座固定：偏航轴在 **(x=0, y=0)**，肩关节高 **z=0.52**。
- 大臂 **0.70 m**、小臂 **0.55 m**（**L1+L2 = 1.25 m 才是水平可达长度**），
  腕连杆 0.10 m、夹爪中心到腕关节 **0.17 m**。
  最后一节腕连杆在抓取时竖直，**不计入水平可达长度**。
- 带面高度 **z=0.35**，方块边长 0.05，静止时方块中心 **z=0.375**。
- 两条传送带关于 **y = x** 对称；带面近边距机械臂轴线 **0.4**，
  底座安装圆盘半径 0.3（与带面留 0.1 间隙）。

```
            +Y
             ↑
  传送带 A ──→（+X）  带面 1.5(x)×0.5(y)，x∈[-1.20,0.30]，y∈[0.40,0.90]
             │       阴影工作区：x∈[-0.50,0.30]（0.8 长）
             │       近边 y=0.40
       底座 (0,0) ────┐
             │        │  传送带 B（-Y）带面 0.5(x)×1.5(y)
             │        │  x∈[0.40,0.90]，y∈[-1.20,0.30]
             │        ↓  阴影工作区：y∈[-0.50,0.30]
```

A 带和 B 带互为关于 `y=x` 的镜像。

---

## 2. 场景里的关键点

| 名称 | 类型 | 含义 |
|---|---|---|
| `carriage` | body | 底座转接座（固定） |
| `turret` | body | 偏航转台 |
| `shoulder` | body | 肩关节中心 |
| `elbow` | body | 肘关节中心 |
| `wrist` | body | 腕关节中心 |
| `gripper` | body | 夹爪坐标原点 |
| `finger_L` / `finger_R` | body | 两个手指体 |
| `ee_site` | site | **夹爪两指中心**（最常用的抓取点） |
| `belt_cam` | camera | **A 带正上方俯视相机**（唯一场景相机） |
| `wrist_cam` | camera | 腕部相机（属于机械臂，跟着工具滚转） |
| `beltA_surface` / `beltB_surface` | geom | 两条带面（物理接触面） |
| `cube_0/1/2` | body | 三个方块（train 的 cube_0 是 mocap） |

### home 姿态下的实测坐标

```
carriage   ( 0.0000,  0.0000,  0.3700)
turret     ( 0.0000,  0.0000,  0.4000)
shoulder   ( 0.0000,  0.0000,  0.5200)
elbow      ( 0.2486, -0.6394,  0.3809)
wrist      ( 0.2229, -0.5734,  0.9263)
gripper    ( 0.2547, -0.6552,  0.9743)
ee_site    ( 0.2770, -0.7124,  1.0078)
```

带面：

```
beltA_surface: 中心 (-0.4500, 0.6500, 0.3440)，半尺寸 (0.75, 0.25, 0.006)，上表面 z=0.3500
beltB_surface: 中心 ( 0.6500, -0.4500, 0.3440)，半尺寸 (0.25, 0.75, 0.006)，上表面 z=0.3500
```

---

## 3. 一行代码取坐标

```python
import mujoco

m = mujoco.MjModel.from_xml_path("scene/show_scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)   # 回到 home
mujoco.mj_forward(m, d)                # ★ 先算正运动学

print(d.body("elbow").xpos)            # 任意 body 的世界坐标 (3,)
print(d.site("ee_site").xpos)          # 任意 site
print(d.geom("beltA_surface").xpos)     # 任意 geom
```

> **重要**：`xpos` / `site_xpos` / `geom_xpos` 都是缓存值。
> 只要改了 `qpos`、执行了 `mj_step`，读之前都要先 `mj_forward(m, d)`
> （只关心位姿时 `mj_kinematics(m, d)` 也行），否则读到的是上一次的旧值。

旋转矩阵 / 朝向：

```python
R = d.site("ee_site").xmat.reshape(3, 3)
print(R[:, 0])     # 夹爪的 +X（指向方向）
print(R[:, 2])     # 夹爪的 +Z
```

---

## 4. 批量取

```python
# 所有 body：d.xpos 形状 (nbody, 3)
for i in range(m.nbody):
    print(m.body(i).name, d.xpos[i])

# 所有 site
for i in range(m.nsite):
    print(m.site(i).name, d.site_xpos[i])

# 关节角 / 速度（注意用 qposadr，不要写死下标）
j = m.joint("joint_elbow").id
print(d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]])
```

---

## 5. 带面与抓取点

带面是盒子，上表面高度 = `geom_xpos[2] + geom_size[2]`：

```python
gid = m.geom("beltA_surface").id
top_z = d.geom_xpos[gid][2] + m.geom_size[gid][2]   # 0.35
```

方块静止时中心比带面高半个边长：**z = 0.375**。

常用抓取点（`ee_site` 应该落在这里）：

```
A 带阴影工作区： (x, 0.65, 0.375)，x ∈ [-0.50, +0.30]
B 带阴影工作区： (0.65, y, 0.375)，y ∈ [-0.50, +0.30]
```

带子全长为 1.5 m，靠机械臂一侧的 0.8 m 是工作区，外侧 0.7 m 是进料/出料段。

---

## 6. 可达范围

按"最后一节腕连杆竖直、夹爪中心 z=0.40"计算（`tools/check_reach.py`）：

| 区域 | 覆盖率 | 最远点 |
|---|---|---|
| A 带阴影工作区（0.8×0.5） | **100%** | 角点 (-0.50, 0.90)，r=1.030 |
| B 带阴影工作区（0.5×0.8） | **100%** | 角点 (0.90, -0.50)，r=1.030 |

L1+L2 = 1.25，最大水平可达 ≈ 1.249，两个最远角都在范围内（约 21% 余量）。

> 改臂长时同步改 `docs/keypoints.py` 顶部的 `L1 / L2` 和 `tools/check_reach.py`。

---

## 7. 反解：给一个目标点，算关节角

假设：夹爪竖直向下（工具滚转 = 0），目标点就是 `ee_site` 的位置。

```python
import math
import numpy as np

BASE_XY = (0.0, 0.0)
SHOULDER_Z = 0.52
L1, L2 = 0.70, 0.55
WRIST_OFF = 0.17          # 腕连杆 0.10 + 夹爪中心 0.07

def ik_down(x, y, z):
    dx, dy = x - BASE_XY[0], y - BASE_XY[1]
    r = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    dz = (z + WRIST_OFF) - SHOULDER_Z          # 腕关节比末端高 0.17
    c2 = (r*r + dz*dz - L1*L1 - L2*L2) / (2*L1*L2)
    c2 = max(-1.0, min(1.0, c2))
    p2 = -math.acos(c2)                        # 负号 = 肘向上
    p1 = math.atan2(dz, r) - math.atan2(L2*math.sin(p2), L1 + L2*math.cos(p2))
    t1, t2 = -p1, -p2
    t3 = math.pi / 2 - (t1 + t2)               # 让夹爪朝下
    return np.array([yaw, t1, t2, t3])         # 填进 ctrl[0:4]
```

示例（脚本里已验证正解误差 < 0.01 mm）：

```
目标 (+0.10, +0.65, 0.40) -> [ +1.418, -0.907, +2.054, +0.424]
目标 (+0.65, +0.10, 0.40) -> [ +0.153, -0.907, +2.054, +0.424]
目标 (-0.40, +0.75, 0.40) -> [ +2.061, -0.758, +1.659, +0.670]
目标 (+0.65, -0.40, 0.40) -> [ -0.552, -0.830, +1.844, +0.558]
```

命令机械臂：

```python
d.ctrl[0:4] = ik_down(0.10, 0.65, 0.40)
d.ctrl[4], d.ctrl[5] = 0.03, -0.03     # 夹爪张开
for _ in range(500):
    mujoco.mj_step(m, d)
```

> 位置伺服在重力下会有 1~2 cm 静差，需要精确到位时做一次
> "走 → 量 `ee_site` → 修正目标 → 再走"的闭环（2~3 次即可）。

---

## 8. qpos / ctrl 索引

两个 XML 的 **机械臂部分完全相同**：

```
qpos:  0 yaw | 1 shoulder | 2 elbow | 3 wrist | 4 tool_roll(锁) | 5 finger_L | 6 finger_R
ctrl:  0 yaw | 1 shoulder | 2 elbow | 3 wrist | 4 finger_L | 5 finger_R
```

方块的 qpos 地址不同：

```
show_scene.xml (nq=28):  cube_0 从 7，cube_1 从 14，cube_2 从 21（三个自由刚体）
train_scene.xml (nq=21): cube_0 是 mocap（不占 qpos），cube_1 从 7，cube_2 从 14
```

home 关键帧：

```
show qpos = [-1.2, 0.20, -1.90, 1.20, 0.0, 0.030, -0.030, ...]
train qpos = [-1.2, 0.20, -1.90, 1.20, 0.0, 0.030, -0.030,
              -0.35, 0.65, 0.40, 1,0,0,0,  -0.25, 0.65, 0.40, 1,0,0,0]
ctrl = [-1.2, 0.20, -1.90, 1.20, 0.030, -0.030]
```

> 不要写死索引：用 `m.jnt_qposadr[m.joint("名字").id]`。
> 工具滚转 `joint_tool_roll` 目前被 `<equality>` 锁在 0，传感器 `tool_roll_q` 可以读它。

---

## 9. 俯视相机 `belt_cam`

```
position: (-0.45, 0.65, 2.55)     # A 带面中心 (-0.45, 0.65) 正上方 2.2 m
xyaxes:   "1 0 0  0 1 0"          # 图像 x=世界 +X，图像 y=世界 +Y，竖直向下看
fovy:     45°
```

- 带面 1.5×0.5 的四角投影到 900×900 图像里是 (80, 327) ~ (820, 573)，整面上表面可见；
- 高度是按"看到半对角线 0.79 m"反推的：45° 视场下最低需要 0.79 / tan(22.5°) = 1.91 m，
  取 **2.2 m**（相机 z=2.55）留约 15% 余量；
- 取图：`renderer.update_scene(d, m.camera("belt_cam").id)`；
- 如果要改高度，保持 `z_cam - 0.35 ≥ 1.91` 就不会裁掉带面角点。
