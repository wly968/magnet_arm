"""对比 train_scene.xml 与 show_scene.xml 的物理模型是否一致。

两个场景的机械臂 / 传送带 / 执行器完全一致；
唯一的设计差异：train 的 cube_0 是 mocap 体（RL 里随机放置目标），
show 的 cube_0 是自由的刚体。本脚本按"排除 cube_0"来对比。

放在 scene/ 外面（tools/），不污染交付目录。
用法：python tools/check_models.py
"""

import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "scene")


def load(name):
    m = mujoco.MjModel.from_xml_path(os.path.join(SCENE, name))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def jmap(m):
    out = {}
    for i in range(m.njnt):
        adr, dadr = int(m.jnt_qposadr[i]), int(m.jnt_dofadr[i])
        out[m.joint(i).name] = (
            int(m.jnt_type[i]),
            adr,
            dadr,
            np.array(m.jnt_range[i]),
            float(m.dof_damping[dadr]),
            float(m.dof_armature[dadr]),
        )
    return out


def main():
    mt, dt = load("train_scene.xml")
    ms, ds = load("show_scene.xml")

    ok = True

    def diff(label, a, b, tol=1e-9):
        nonlocal ok
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            print(f"  [FAIL] {label}: shape {a.shape} vs {b.shape}")
            ok = False
            return
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        flag = "OK" if err <= tol else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  [{flag}] {label:24s} max|diff|={err:.3e}")

    print(f"train: nq={mt.nq} nv={mt.nv} nu={mt.nu}  (cube_0 是 mocap，不占 qpos)")
    print(f"show : nq={ms.nq} nv={ms.nv} nu={ms.nu}  (cube_0 是自由刚体)")
    print(f"差异: nq {ms.nq - mt.nq}  nv {ms.nv - mt.nv}  = cube_0 的 freejoint 维数")

    # ---- 运动 body（排除 cube_0）----
    ignore = {"cube_0"}
    moving_t = [mt.body(i).name for i in range(mt.nbody)
                if mt.body_jntnum[i] > 0 and mt.body(i).name not in ignore]
    moving_s = [ms.body(i).name for i in range(ms.nbody)
                if ms.body_jntnum[i] > 0 and ms.body(i).name not in ignore]
    if moving_t != moving_s:
        print("  [FAIL] 运动 body 集合不同:", moving_t, moving_s)
        ok = False
    else:
        print(f"\n运动 body（除 cube_0，{len(moving_t)} 个）: {moving_t}")
    idx_t = [mt.body(n).id for n in moving_t]
    idx_s = [ms.body(n).id for n in moving_s]
    diff("body_mass", mt.body_mass[idx_t], ms.body_mass[idx_s])
    diff("body_inertia", mt.body_inertia[idx_t], ms.body_inertia[idx_s])

    # ---- 关节（按名字，排除 cube_0_joint）----
    jt, js = jmap(mt), jmap(ms)
    common = [n for n in jt if n in js and n != "cube_0_joint"]
    if sorted(common) != sorted(n for n in jt if n != "cube_0_joint"):
        print("  [FAIL] 共同关节缺失")
        ok = False
    labels = {0: "joint_type", 3: "joint_range", 4: "joint_damping", 5: "joint_armature"}
    for key in (0, 3, 4, 5):
        diff(labels[key],
             [jt[n][key] for n in common], [js[n][key] for n in common])

    diff("actuator_gain", mt.actuator_gainprm[:, 0], ms.actuator_gainprm[:, 0])
    diff("actuator_bias", mt.actuator_biasprm[:, 0], ms.actuator_biasprm[:, 0])
    diff("actuator_ctrlrange", mt.actuator_ctrlrange, ms.actuator_ctrlrange)
    diff("actuator_forcerange", mt.actuator_forcerange, ms.actuator_forcerange)

    # ---- home 关键帧：机械臂 7 维 + ctrl 完全一致 ----
    diff("key_qpos[arm 0:7]", dt.qpos[:7], ds.qpos[:7])
    diff("key_ctrl", dt.ctrl, ds.ctrl)
    diff("initial_ee", dt.site_xpos[mt.site("ee_site").id],
         ds.site_xpos[ms.site("ee_site").id])

    # cube_1 / cube_2 的 qpos（train 与 show 相差 cube_0 的 7 维）
    q1t, q1s = jt["cube_1_joint"][1], js["cube_1_joint"][1]
    q2t, q2s = jt["cube_2_joint"][1], js["cube_2_joint"][1]
    diff("key_qpos[cube_1]", dt.qpos[q1t:q1t + 7], ds.qpos[q1s:q1s + 7])
    diff("key_qpos[cube_2]", dt.qpos[q2t:q2t + 7], ds.qpos[q2s:q2s + 7])

    # ---- 碰撞 geom ----
    print("\n碰撞 geom 对比")

    def collide_sig(m):
        sig = []
        for i in range(m.ngeom):
            if m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0:
                continue
            body = m.body(m.geom_bodyid[i]).name
            sig.append((
                body,
                int(m.geom_type[i]),
                tuple(np.round(m.geom_pos[i], 6)),
                tuple(np.round(m.geom_size[i], 6)),
                tuple(np.round(m.geom_friction[i], 6)),
                int(m.geom_condim[i]),
            ))
        return sorted(sig)

    st, ss = collide_sig(mt), collide_sig(ms)
    print(f"  train 碰撞 geom: {len(st)}  show: {len(ss)}")
    if st != ss:
        ok = False
        print("  [FAIL] 碰撞几何不一致")
        for a in st:
            if a not in ss:
                print("    只在 train:", a)
        for b in ss:
            if b not in st:
                print("    只在 show :", b)
    else:
        print("  [OK] 碰撞几何完全一致（位姿/尺寸/摩擦）")

    print("\n结论:", "PASS 机械臂/传送带物理一致（cube_0 按设计不同）" if ok
          else "FAIL 存在差异")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
