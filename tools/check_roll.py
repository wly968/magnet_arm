import mujoco, numpy as np
for name in ("train_scene.xml", "show_scene.xml"):
    m = mujoco.MjModel.from_xml_path("scene/" + name)
    d = mujoco.MjData(m)
    print(f"--- {name}: nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq}")
    for i in range(m.njnt):
        print(f"   {i}: {m.joint(i).name:16s} qposadr={m.jnt_qposadr[i]} range={np.round(m.jnt_range[i],3)}")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    print("   key qpos:", np.round(d.qpos[:9], 3), " ctrl:", np.round(d.ctrl, 2))
    # 给滚转关节一个初始角速度，看看等式约束是否把它拉回 0
    rid = m.joint("joint_tool_roll").id
    adr = m.jnt_qposadr[rid]; dof = m.jnt_dofadr[rid]
    d.qvel[dof] = 20.0
    for _ in range(300):
        mujoco.mj_step(m, d)
    print(f"   roll after kick: qpos={d.qpos[adr]:+.2e} qvel={d.qvel[dof]:+.2e} nefc={d.nefc}")
