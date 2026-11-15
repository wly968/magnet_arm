import time, mujoco
for name in ("train_scene.xml", "show_scene.xml"):
    m = mujoco.MjModel.from_xml_path("scene/" + name)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(100): mujoco.mj_step(m, d)
    t0 = time.perf_counter(); N = 5000
    for _ in range(N): mujoco.mj_step(m, d)
    dt = time.perf_counter() - t0
    print(f"{name:18s} ngeom={m.ngeom:3d}  {N/dt:6.0f} steps/s  ({N*m.opt.timestep/dt:4.1f}x realtime)")
