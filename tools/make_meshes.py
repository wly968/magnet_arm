"""生成 show_scene 用的连杆外观网格（过程文件，放在 scene 外面）。

连杆长度与 scene/show_scene.xml 中的关节位置一致：
    upper 0.43, fore 0.39, wrist 0.10
输出覆盖 scene/meshes/*.obj（不新增文件）。
"""

import os

import numpy as np

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scene")
OUT = os.path.join(SCENE, "meshes")


def superellipse(theta, a, b, p=3.2):
    c, s = np.cos(theta), np.sin(theta)
    return a * np.sign(c) * np.abs(c) ** (2.0 / p), b * np.sign(s) * np.abs(s) ** (2.0 / p)


def fix_winding(v, f):
    vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0
    return f[:, ::-1] if vol < 0 else f


def vertex_normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    return n / length


def write_obj(name, v, f):
    f = fix_winding(v, f)
    n = vertex_normals(v, f)
    with open(os.path.join(OUT, name), "w", encoding="ascii") as fh:
        for p in v:
            fh.write("v %.6f %.6f %.6f\n" % tuple(p))
        for p in n:
            fh.write("vn %.6f %.6f %.6f\n" % tuple(p))
        for t in f:
            a, b, c = t + 1
            fh.write("f %d//%d %d//%d %d//%d\n" % (a, a, b, b, c, c))
    print(f"  {name}: {len(v)} verts, {len(f)} faces")


def link_mesh(name, length, hw0, hh0, hw1, hh1, bulge=0.002, nsec=34, nth=36, p=3.2):
    thetas = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    verts = []
    for i in range(nsec + 1):
        t = i / nsec
        x = t * length
        hw = hw0 + (hw1 - hw0) * t + bulge * np.sin(np.pi * t)
        hh = hh0 + (hh1 - hh0) * t + bulge * np.sin(np.pi * t)
        edge = min(x, length - x)
        shrink = 1.0 - 0.05 * np.exp(-((edge / 0.020) ** 2))
        y, z = superellipse(thetas, hw * shrink, hh * shrink, p)
        verts.extend(zip(np.full(nth, x), y, z))

    verts = np.asarray(verts, dtype=float)
    faces = []
    for i in range(nsec):
        for j in range(nth):
            a = i * nth + j
            b = i * nth + (j + 1) % nth
            c = (i + 1) * nth + (j + 1) % nth
            d = (i + 1) * nth + j
            faces.append((a, b, c))
            faces.append((a, c, d))

    c0 = len(verts)
    verts = np.vstack([verts, [0.0, 0.0, 0.0]])
    for j in range(nth):
        faces.append((c0, (j + 1) % nth, j))

    c1 = len(verts)
    verts = np.vstack([verts, [length, 0.0, 0.0]])
    base = nsec * nth
    for j in range(nth):
        faces.append((c1, base + j, base + (j + 1) % nth))

    write_obj(name, verts, np.asarray(faces))


def main():
    os.makedirs(OUT, exist_ok=True)
    link_mesh("upper_link.obj", 0.700,
              hw0=0.040, hh0=0.045, hw1=0.036, hh1=0.041, bulge=0.0025)
    link_mesh("fore_link.obj", 0.550,
              hw0=0.036, hh0=0.041, hw1=0.031, hh1=0.035, bulge=0.0022)
    link_mesh("wrist_link.obj", 0.100,
              hw0=0.031, hh0=0.035, hw1=0.030, hh1=0.034, bulge=0.0012,
              nsec=18, nth=30)


if __name__ == "__main__":
    main()



