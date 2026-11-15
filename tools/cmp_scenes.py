import re
def summary(p):
    s = open(p, encoding="utf-8").read()
    out = {}
    out["wrist_link"] = re.findall(r'<geom name="wrist_link"[^>]*>', s)
    out["disc"] = re.findall(r'<geom name="magnet_disc"[^>]*>', s)
    out["upper_arm"] = re.findall(r'<geom name="upper_arm"[^>]*>', s)
    out["forearm"] = re.findall(r'<geom name="forearm"[^>]*>', s)
    out["gripper_body"] = re.findall(r'<body name="gripper"[^>]*>', s)
    out["actuators"] = re.findall(r'<position name="[^"]+"[^>]*>', s)
    out["key"] = re.findall(r'<key name="home"[^>]*>', s, re.S)
    out["cubes"] = re.findall(r'<body name="cube_\d+"[^>]*>', s)
    out["fingers"] = re.findall(r'<body name="finger_[LR]"', s)
    return out
for p in ["scene_magnet/assets/models/train_scene.xml", "scene_magnet/assets/models/show_scene.xml"]:
    print("="*20, p)
    for k, v in summary(p).items():
        print(k, "->", v[:6])
