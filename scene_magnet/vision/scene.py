"""场景封装：vision_scene 的加载、方块摆放、光照随机化。

职责
----
- 加载 assets/models/vision_scene.xml（含 birdeye 俯视相机）
- 把 10 个 mocap 方块隐藏 / 摆到指定 (x, y)，拿真值给数据集打标签
- 光照随机化：光源位置、色温、强度、高光、环境光（用于域随机化）

计划接口
--------
VisionScene.reset()                      复位场景，隐藏全部方块
VisionScene.place_cubes(list_of_xy)       摆放方块（y 可随机）
VisionScene.cube_positions()             当前真值，供标签生成
VisionScene.set_lighting(level, rng)     光照随机化，level ∈ {0,1,2,3}
VisionScene.render()                     返回 (H, W, 3) uint8

依赖：vision.camera.Camera
"""

# TODO: 待实现（当前只保留文件占位与职责说明）
