#!/usr/bin/env python3
"""
Merge several triangle-mesh OBJ files into a single OBJ.

Mirrors the layout of the reference "merged.obj":
  * meshes are concatenated in the order given on the command line
  * object/group names, comments ("# trees") and other IDs are dropped
  * duplicate vertices (identical after rounding to --decimals) are welded
  * vertices are renumbered in order of first use by the faces
    (vertices no face refers to are dropped)
  * one flat normal per face is written, faces are  f v//vn v//vn v//vn

Usage:
    python merge_obj.py trees.obj buildings_refined.obj -o merged.obj

Requires: numpy
"""
import argparse
import sys

import numpy as np


def read_obj(path):
    """Return (vertices [N,3] float64, triangles [M,3] int64, 0-based)."""
    verts, tris = [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                # accepts  "1 2 3", "1/1 2/2 3/3", "1//1 ..." and negative indices
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                # fan-triangulate quads / n-gons
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return v, f


def merge(paths, decimals=6):
    all_v, all_f, offset = [], [], 0
    for p in paths:
        v, f = read_obj(p)
        print(f"  {p}: {len(v)} vertices, {len(f)} faces")
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    V = np.vstack(all_v)
    F = np.vstack(all_f)

    # --- weld duplicate vertices -------------------------------------------
    key = np.round(V, decimals) + 0.0            # +0.0 turns -0.0 into 0.0
    _, first_idx, inverse = np.unique(
        key, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    F = inverse[F]                                # faces -> welded vertex ids
    V = V[first_idx]                              # one coordinate per welded vertex

    # --- renumber by first use in the face list, drop unused ------------------
    flat = F.reshape(-1)
    used, first_pos = np.unique(flat, return_index=True)
    order = used[np.argsort(first_pos)]           # old ids, in order of first use
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[order] = np.arange(len(order))
    V = V[order]
    F = remap[F]
    return V, F


def face_normals(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0                             # degenerate face -> zero normal
    return n / ln


def write_obj(path, V, F, N):
    with open(path, "w") as out:
        out.write("\n")                           # same blank first line as reference
        np.savetxt(out, V, fmt="v %.6f %.6f %.6f")
        np.savetxt(out, N + 0.0, fmt="vn %.6f %.6f %.6f")
        fid = F + 1
        nid = np.arange(1, len(F) + 1)
        np.savetxt(
            out,
            np.column_stack([fid[:, 0], nid, fid[:, 1], nid, fid[:, 2], nid]),
            fmt="f %d//%d %d//%d %d//%d",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="OBJ files to merge, in order")
    ap.add_argument("-o", "--output", default="merged.obj")
    ap.add_argument("--decimals", type=int, default=6,
                    help="vertices equal after rounding to this many decimals are welded")
    args = ap.parse_args()

    print("Reading:")
    V, F = merge(args.inputs, args.decimals)
    N = face_normals(V, F)
    write_obj(args.output, V, F, N)
    print(f"Wrote {args.output}: {len(V)} vertices, {len(N)} normals, {len(F)} faces")


if __name__ == "__main__":
    sys.exit(main())