"""
OBJ (triSurface) -> 2D categorical raster  (model input 'A')

prepare_all.sh isko automatically chalata hai, geometry/triSurface ke OBJ files se.

Usage:
    python3 rasterize_obj.py <triSurface_dir> <out_dir> [--res 512]

Output (out_dir me):
    geometry_labels.npy   <- RAW class ids (uint8)
    geometry_rgb.png      <- visual check
    geometry_meta.json    <- resolution / extent_xy / class map

Domain extent: ground.obj se; wo na ho (jo is pipeline me nahi banta) to sab OBJ ke union se.
"""
import os
import json
import argparse
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw


def load_obj_triangles_xy(path):
    verts, tris = [], []
    if not os.path.exists(path):
        return np.zeros((0, 2)), []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("//")[0].split("/")[0]) - 1 for p in parts]
                if len(idx) == 3:
                    tris.append(idx)
                elif len(idx) > 3:
                    for k in range(1, len(idx) - 1):
                        tris.append([idx[0], idx[k], idx[k + 1]])
    verts = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 2))
    return verts, tris


CLASSES = {
    "ground":    {"id": 1, "color": (99, 143, 20)},
    "ponds":     {"id": 2, "color": (30, 60, 170)},
    "lawns":     {"id": 3, "color": (0, 170, 170)},
    "pathways":  {"id": 4, "color": (200, 170, 90)},
    "trees":     {"id": 5, "color": (20, 110, 40)},
    "buildings": {"id": 6, "color": (190, 60, 30)},
}
# baad me draw = upar dikhega (buildings sabse upar)
DRAW_ORDER = ["ponds", "lawns", "pathways", "trees", "buildings"]

OBJ_FILENAMES = {
    "ground": "ground.obj",
    "ponds": "ponds.obj",
    "lawns": "lawns.obj",
    "pathways": "pathways.obj",
    "trees": "trees.obj",
    "buildings": "buildings_refined.obj",
}


def rasterize(tri_dir, out_dir, res=512, extent=None):
    if not os.path.isdir(tri_dir):
        raise FileNotFoundError(f"triSurface dir not found: {tri_dir}")

    # ---- domain extent ----
    ground_v, _ = load_obj_triangles_xy(os.path.join(tri_dir, OBJ_FILENAMES["ground"]))
    if ground_v.shape[0] == 0:
        all_v = []
        for fname in OBJ_FILENAMES.values():
            v, _ = load_obj_triangles_xy(os.path.join(tri_dir, fname))
            if v.shape[0]:
                all_v.append(v)
        if not all_v:
            raise RuntimeError(f"No geometry found in any OBJ under {tri_dir}")
        ground_v = np.concatenate(all_v, axis=0)

    x_min, y_min = ground_v.min(axis=0)
    x_max, y_max = ground_v.max(axis=0)
    if extent is not None:   # e.g. wind-model VTK domain -> raster U ke saath same frame me, kuch cut nahi
        x_min, x_max, y_min, y_max = (float(v) for v in extent)
    if x_max == x_min or y_max == y_min:
        raise RuntimeError(f"Degenerate extent X[{x_min},{x_max}] Y[{y_min},{y_max}]")

    W = H = res

    def world_to_px(x, y):
        px = (x - x_min) / (x_max - x_min) * (W - 1)
        py = (y - y_min) / (y_max - y_min) * (H - 1)
        return px, (H - 1) - py  # top = +Y

    rgb_img = Image.new("RGB", (W, H), CLASSES["ground"]["color"])
    draw = ImageDraw.Draw(rgb_img)
    label_img = Image.new("L", (W, H), CLASSES["ground"]["id"])
    label_draw = ImageDraw.Draw(label_img)

    for name in DRAW_ORDER:
        verts, tris = load_obj_triangles_xy(os.path.join(tri_dir, OBJ_FILENAMES[name]))
        if len(tris) == 0:
            print(f"  {name}: empty/missing -> skip")
            continue
        cls = CLASSES[name]
        px = np.array([world_to_px(x, y) for x, y in verts])
        for tri in tris:
            pts = [tuple(px[i]) for i in tri]
            draw.polygon(pts, fill=cls["color"])
            label_draw.polygon(pts, fill=cls["id"])
        print(f"  {name}: {len(tris)} triangles rasterized, id={cls['id']}")

    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "geometry_labels.npy")
    png_path = os.path.join(out_dir, "geometry_rgb.png")
    meta_path = os.path.join(out_dir, "geometry_meta.json")

    np.save(npy_path, np.array(label_img, dtype=np.uint8))
    rgb_img.save(png_path)
    meta = {
        "resolution": [W, H],
        "extent_xy": [float(x_min), float(x_max), float(y_min), float(y_max)],
        "classes": {k: v["id"] for k, v in CLASSES.items()},
        "colors": {k: v["color"] for k, v in CLASSES.items()},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[rasterize_obj] saved -> {npy_path}")
    return {"npy": npy_path, "png": png_path, "meta": meta_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="triSurface OBJ -> 2D geometry raster")
    ap.add_argument("tri_dir", help="geometry/triSurface folder")
    ap.add_argument("out_dir", help="folder for geometry_labels.npy / png / meta")
    ap.add_argument("--res", type=int, default=512, help="raster resolution (default 512)")
    ap.add_argument("--extent", type=float, nargs=4, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                    default=None, help="override raster extent (default: geometry bbox)")
    a = ap.parse_args()
    rasterize(a.tri_dir, a.out_dir, a.res, a.extent)