"""打印场景关键点坐标，并演示"给目标点反解关节角"。

用法：
    python docs/keypoints.py            # 默认 show_scene.xml
    python docs/keypoints.py train      # 用 train_scene.xml

这个脚本就是文档 docs/keypoints.md 的可运行版本。
"""

import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scene", "show_scene.xml" if len(sys.argv) < 2 else
                     ("train_scene.xml" if sys.argv[1] == "train" else sys.argv[1]))

# 机械臂几何常量（与 XML 一致）
BASE_XY = np.array([0.0, 0.0])     # 底座/偏航轴
SHOULDER_Z = 0.52                  # 肩关节高度
L1, L2 = 0.70, 0.55                # 大臂 / 小臂（L1+L2 才是水平可达长度）
WRIST_OFF = 0.10 + 0.07            # 腕连杆 + 夹爪中心到腕关节的距离


def ik_down(x, y, z=0.40):
    """夹爪竖直向下、工具滚转为 0 时的 4 关节反解。

    返回 [yaw, shoulder, elbow, wrist]（弧度），可直接填进 ctrl[0:4]。
    最后一节腕连杆竖直，水平距离只由 L1+L2 决定。
    """
    dx, dy = x - BASE_XY[0], y - BASE_XY[1]
    r = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)

    wrist_z = z + WRIST_OFF
    dz = wrist_z - SHOULDER_Z

    c2 = (r * r + dz * dz - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        raise ValueError(f"目标不可达: ({x:.3f}, {y:.3f}, {z:.3f})")
    p2 = -math.acos(c2)                                  # 负号=肘向上
    p1 = math.atan2(dz, r) - math.atan2(L2 * math.sin(p2),
                                        L1 + L2 * math.cos(p2))
    t1, t2 = -p1, -p2
    t3 = math.pi / 2 - (t1 + t2)
    return np.array([yaw, t1, t2, t3])


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    print(f"模型: {SCENE}")
    print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}\n")

    print("=== 关节 qpos 索引 ===")
    for i in range(m.njnt):
        print(f"  {m.joint(i).name:16s} qpos={m.jnt_qposadr[i]:2d} "
              f"dof={m.jnt_dofadr[i]:2d}")

    print("\n=== 关键 body 的世界坐标（home 姿态）===")
    for name in ("carriage", "turret", "shoulder", "elbow", "wrist", "gripper",
                 "finger_L", "finger_R", "cube_0", "cube_1", "cube_2"):
        p = d.body(name).xpos
        print(f"  {name:10s} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    print("\n=== 关键 site / 相机 ===")
    ee = d.site("ee_site").xpos
    print(f"  ee_site   ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})   <- 夹爪两指中心")
    rot = d.site("ee_site").xmat.reshape(3, 3)
    print(f"  ee 的 +X 轴（夹爪朝向）: {np.round(rot[:, 0], 3)}")

    print("\n=== 传送带带面（关于 y=x 对称）===")
    for n in ("beltA_surface", "beltB_surface"):
        gid = m.geom(n).id
        c = d.geom_xpos[gid]
        half = m.geom_size[gid]
        print(f"  {n}: 中心 ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})  "
              f"半尺寸 {np.round(half, 3)}  上表面 z={c[2] + half[2]:.4f}")

    print("\n=== 静止方块中心（抓取点）===")
    print("  A 带: (x, 0.65, 0.375)   x ∈ [-0.50, +0.30]（阴影工作区）")
    print("  B 带: (0.65, y, 0.375)   y ∈ [-0.50, +0.30]（阴影工作区）")

    print("\n=== 俯视相机 ===")
    cam = m.camera("belt_cam")
    print(f"  belt_cam: pos {np.round(cam.pos, 3)}  fovy {cam.fovy[0]:.1f}°  "
          f"（A 带面中心正上方）")

    print("\n=== 反解示例 ===")
    for x, y, z in ((0.10, 0.65, 0.40), (0.65, 0.10, 0.40),
                    (-0.40, 0.75, 0.40), (0.65, -0.40, 0.40)):
        q = ik_down(x, y, z)
        print(f"  目标 ({x:+.2f}, {y:+.2f}, {z:.2f}) -> ctrl[0:4] = "
              f"[{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f}]")

    # 用正运动学校验一次
    q = ik_down(0.10, 0.65, 0.40)
    d.qpos[0:4] = q
    mujoco.mj_forward(m, d)
    err = np.linalg.norm(d.site("ee_site").xpos - np.array([0.10, 0.65, 0.40]))
    print(f"\n  IK 校验: 目标 (0.10, 0.65, 0.40)，正解末端误差 {err * 1000:.2f} mm")


if __name__ == "__main__":
    main()
