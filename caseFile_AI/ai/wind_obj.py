#!/usr/bin/env python3
"""
wind_obj.py  --  layout surfaces  ->  the OBJ the wind model (pipeline/wind_api) reads

The wind model voxelises an OBJ onto a 150 x 150 x 50 grid (domain = bounding box + 10 %, height = 2 x the tallest object) and
pre-processes it first (infrenace.preprocess_obj): objects named  site_boundary / vegetation_zone_N / rainwater_pond_N  are removed,
floor-by-floor buildings lose their internal faces, and every object named  tree_*  is forced to a 1 m wide vertical prism.

This file writes exactly the objects that matter for the flow, from the SAME surfaces the thermal part of the case uses
(momentum/constant/triSurface made by json_to_obj.py), so the wind field, the geometry raster and the surface temperatures
describe one and the same layout:

    o building_1     all of buildings_refined.obj (buildings, stacked floors, OSM buildings)
    o tree_<i>       one 1 m wide octagonal prism per tree of the layout JSON (trunk axis = tree position)

Ground, lawns, ponds and pathways are flat and therefore not part of the wind OBJ (the wind model would remove them anyway).

ASSUMPTION (the pipeline zip does not contain its own OBJ writer): a tree is a prism from the ground to
`tree_height` = "total" (tree height, default) or "trunk" (trunk height).  Change it in ai/config.json ("tree_wind_height").
"""
import json
import math
from pathlib import Path


def _read_obj_faces(path):
    """(vertices, faces) of an OBJ (triangles / polygons, 0-based faces)."""
    V, F = [], []
    p = Path(path)
    if not p.is_file():
        return V, F
    for line in p.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            q = line.split()
            V.append((float(q[1]), float(q[2]), float(q[3])))
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            F.append([i - 1 if i > 0 else len(V) + i for i in idx])
    return V, F


def tree_prism(cx, cy, z0, z1, diameter=1.0, sides=8):
    r = 0.5 * diameter
    ring = [(cx + r * math.cos(2 * math.pi * k / sides + math.pi / sides), cy + r * math.sin(2 * math.pi * k / sides + math.pi / sides))
            for k in range(sides)]
    V = [(x, y, z0) for x, y in ring] + [(x, y, z1) for x, y in ring]
    F = []
    for k in range(sides):
        a, b = k, (k + 1) % sides
        F.append([a, b, sides + b])
        F.append([a, sides + b, sides + a])
    for k in range(1, sides - 1):                       # bottom (facing down) and top (facing up) caps
        F.append([0, k + 1, k])
        F.append([sides, sides + k, sides + k + 1])
    return V, F


def write_wind_obj(tri_dir, layout_json, out_path, tree_height="total", trunk_diameter=1.0):
    """Writes out_path; returns {"buildings": n_faces, "trees": n_trees, "zmax": ...}."""
    tri_dir = Path(tri_dir)
    bv, bf = _read_obj_faces(tri_dir / "buildings_refined.obj")
    lines = ["# wind-model OBJ made by ai/wind_obj.py (buildings + tree prisms of the layout)"]
    n_v = 0
    zmax = 0.0
    if bf:
        lines.append("o building_1")
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in bv]
        lines += ["f " + " ".join(str(i + 1) for i in f) for f in bf]
        n_v = len(bv)
        zmax = max(v[2] for v in bv)
    data = json.loads(Path(layout_json).read_text())
    data = data.get("siteOnly_json", data)
    trees = data.get("trees") or []
    n_trees = 0
    for i, t in enumerate(trees):
        try:
            pos, dim = t["position"], t["dimensions"]
            h = float(dim["totalHeight"] if tree_height == "total" else dim["trunkHeight"])
            z0 = float(pos.get("z", 0.0))
            if not (h > 0 and math.isfinite(h)):
                continue
            V, F = tree_prism(float(pos["x"]), float(pos["y"]), z0, z0 + h, trunk_diameter)
        except (KeyError, TypeError, ValueError):
            continue
        n_trees += 1
        lines.append(f"o tree_{i + 1}")
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V]
        lines += ["f " + " ".join(str(n_v + j + 1) for j in f) for f in F]
        n_v += len(V)
        zmax = max(zmax, z0 + h)
    if not bf and not n_trees:
        raise RuntimeError("no buildings and no trees in the layout: nothing for the wind model to read")
    Path(out_path).write_text("\n".join(lines) + "\n")
    return {"building_faces": len(bf), "trees": n_trees, "zmax": zmax}


if __name__ == "__main__":
    import sys
    print(write_wind_obj(sys.argv[1], sys.argv[2], sys.argv[3]))
