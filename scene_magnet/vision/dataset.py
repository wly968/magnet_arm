"""YOLO-seg 数据集生成：渲染 + 多边形标签 + data.yaml。

职责
----
- 批量渲染带随机光照/噪声的图像
- 标签：方块**顶面四角**投影到像素的四点多边形（归一化），类别只有 cube
- 划分 train / val（8:2），生成 ultralytics 用的 data.yaml

计划接口
--------
project_cube_polygon(cam, cube_xyz) -> [(u, v)] * 4
render_sample(scene, cam, lighting_level) -> (img, polygons)
build_dataset(n_images, out_dir, levels, val_ratio=0.2)
write_data_yaml(out_dir, names=["cube"])

标签格式（YOLO-seg）
--------------------
    <class> x1 y1 x2 y2 x3 y3 x4 y4      # 全部归一化到 [0, 1]

依赖：vision.scene, vision.camera
"""

# TODO: 待实现
