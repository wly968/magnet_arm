import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(r"scene_magnet/assets/models/train_scene.xml")
for name in ("beltA", "beltB", "carriage"):
    bid = m.body(name).id
    print(name, "pos", np.round(m.body_pos[bid],3))
    for g in range(m.body_geomadr[bid], m.body_geomadr[bid]+m.body_geomnum[bid]):
        print("   geom", m.geom_type[g], "size", np.round(m.geom_size[g],3), "pos", np.round(m.geom_pos[g],3))
