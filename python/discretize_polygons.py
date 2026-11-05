#!/usr/bin/env python3
"""
discretize_polygons.py  --  re-discretize the pond / lawn outlines of existing layout json files

Smoothed outlines carry thousands of vertices millimetres apart (and, between smoothed arcs, straight
edges tens of metres long).  This rewrites every pond and grass-area outline so that neighbouring vertices
are at least STEP metres apart (default 1 m): corners are kept, the stretches between them are resampled at
equal spacing.  The area (and a pond's volume) is updated.  Nothing else in the json is touched.

    python3 discretize_polygons.py generation_1.json [generation_2.json ...] [--step 1.0] [--out-dir DIR]
(without --out-dir the files are rewritten in place; a .bak copy is kept)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout_ops as L  # noqa: E402


def seg_stats(p):
    a = np.array([[q["x"], q["y"]] for q in p["coordinates"]])
    s = np.linalg.norm(np.diff(np.vstack([a, a[:1]]), axis=0), axis=1)
    return len(a), s.min(), float(np.median(s)), s.max()


def process(d, step):
    """Ponds first, then lawns; each outline is pulled back from earlier ones / the buildings if its resampled
    version would overlap them more than the original did."""
    from shapely.ops import unary_union
    fp = []
    for b in d.get("buildings", []):
        try:
            g = L._poly(b["geometry"]["footprint"])
            fp.append(g if g.is_valid else g.buffer(0))
        except Exception:
            pass
    keep_out = unary_union(fp) if fp else None
    items = [(key, p) for key, is_pond in (("ponds", True), ("grassAreas", False)) for p in d.get(key, [])]
    polys = []
    for key, p in items:
        poly = L._poly(p["coordinates"])
        polys.append(poly if poly.is_valid else poly.buffer(0))
    stats0 = [seg_stats(p) for _, p in items]
    new = L.discretize_patches(polys, step, keep_out)
    rows = []
    for (key, p), poly, old, (n0, mn0, md0, mx0) in zip(items, new, polys, stats0):
        L.patch_set_polygon(p, poly, key == "ponds", 0)
        n1, mn1, md1, mx1 = seg_stats(p)
        rows.append((key, p.get("id", ""), n0, n1, mn0, mn1, mx1, old.area, p["dimensions"]["area"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    for f in a.files:
        d = json.load(open(f))
        rows = process(d, a.step)
        out = (a.out_dir / f.name) if a.out_dir else f
        if a.out_dir:
            a.out_dir.mkdir(parents=True, exist_ok=True)
        elif not Path(str(f) + ".bak").exists():
            shutil.copy(f, str(f) + ".bak")
        json.dump(d, open(out, "w"), indent=2)
        print(f"{f.name}:")
        for key, i, n0, n1, mn0, mn1, mx1, a0, a1 in rows:
            print(f"  {key:10s} {i:14s} vertices {n0:5d} -> {n1:4d}   shortest edge {mn0:.3f} -> {mn1:.2f} m   "
                  f"longest {mx1:.1f} m   area {a0:.1f} -> {a1:.1f} m2 ({100*(a1-a0)/a0:+.2f} %)")


if __name__ == "__main__":
    main()
