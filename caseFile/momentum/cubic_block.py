#!/usr/bin/env python3
"""
cubic_block.py  --  make the blockMesh background mesh out of CUBES (dx = dy = dz = h)

Run it in the momentum case directory right after objBoundingDim.py (which writes
meshParameters and blockParameters) and before the files are moved into system/:

    python3 cubic_block.py [--cells-across 48] [--h H] [--step 0.25]

What it does
  1. h = longest horizontal side of the box / cells-across, rounded to `step` metres
     (so layouts with slightly different tree / pond extents get the same h).
     --h sets h directly.
  2. The box is GROWN (never shrunk) to a whole number of cells: x and y symmetrically about
     the box centre, z upwards from zMin.  Growth is less than one cell per side.
  3. meshParameters  : new xMin..zMax + xCells yCells zCells zGrading(=1)
     blockParameters : new xMin..yMax (the ground plane of gen_ground_obj.py must cover the
                       blockMesh domain exactly), zGrading 1
  4. system/cubicBlock.json is written for the check after blockMesh.

    python3 cubic_block.py --verify

reads constant/polyMesh/points (run it right after blockMesh) and stops with exit code 1 if
the cells are not cubes of side h or the cell count differs from Nx*Ny*Nz.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

KEYS = ("xMin", "xMax", "yMin", "yMax", "zMin", "zMax")


def read_params(path):
    out, lines = {}, []
    for line in Path(path).read_text().splitlines():
        lines.append(line)
        m = re.match(r"\s*(\w+)\s+([-+0-9.eE]+)\s*;", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out, lines


def write_params(path, lines, new):
    """Replace the value of every key in `new` (append missing keys), keep other lines."""
    done, out = set(), []
    for line in lines:
        m = re.match(r"\s*(\w+)\s+([-+0-9.eE]+)\s*;", line)
        if m and m.group(1) in new:
            out.append(f"{m.group(1)}\t{new[m.group(1)]};")
            done.add(m.group(1))
        else:
            out.append(line)
    for k, v in new.items():
        if k not in done:
            out.append(f"{k}\t{v};")
    Path(path).write_text("\n".join(out) + "\n")


def fmt(v):
    return f"{v:.10g}"


def plan(P, cells_across, h_fixed, step):
    Lx, Ly, Lz = P["xMax"] - P["xMin"], P["yMax"] - P["yMin"], P["zMax"] - P["zMin"]
    if min(Lx, Ly, Lz) <= 0:
        sys.exit(f"cubic_block: non-positive box size {Lx} x {Ly} x {Lz}")
    if h_fixed:
        h = float(h_fixed)
    else:
        h = round(max(Lx, Ly) / cells_across / step) * step
        h = max(h, step)
    # the +1e-6 relative slack keeps an exact multiple from gaining a whole extra cell
    N = [int(math.ceil(L / h * (1.0 - 1e-9))) for L in (Lx, Ly, Lz)]
    cx, cy = 0.5 * (P["xMin"] + P["xMax"]), 0.5 * (P["yMin"] + P["yMax"])
    new = dict(
        xMin=cx - 0.5 * N[0] * h, xMax=cx + 0.5 * N[0] * h,
        yMin=cy - 0.5 * N[1] * h, yMax=cy + 0.5 * N[1] * h,
        zMin=P["zMin"], zMax=P["zMin"] + N[2] * h)
    return h, N, new, (Lx, Ly, Lz)


def do_plan(a):
    mp, mlines = read_params("meshParameters")
    bp, blines = read_params("blockParameters")
    for k in KEYS:
        if k not in mp:
            sys.exit(f"cubic_block: {k} missing in meshParameters")
    h, N, new, L = plan(mp, a.cells_across, a.h, a.step)
    # meshParameters: extents + cells
    mnew = {k: fmt(v) for k, v in new.items()}
    mnew.update(xCells=str(N[0]), yCells=str(N[1]), zCells=str(N[2]), zGrading="1")
    write_params("meshParameters", mlines, mnew)
    # blockParameters: same x / y extents (ground plane), z keeps its own meaning
    bnew = {k: fmt(new[k]) for k in ("xMin", "xMax", "yMin", "yMax")}
    bnew["zGrading"] = "1"
    write_params("blockParameters", blines, bnew)
    Path("system").mkdir(exist_ok=True)
    Path("system/cubicBlock.json").write_text(json.dumps(
        dict(h=h, N=N, xMin=new["xMin"], yMin=new["yMin"], zMin=new["zMin"]), indent=1))
    print(f"[cubic_block] cube size h = {h:g} m")
    print(f"[cubic_block] box {L[0]:.2f} x {L[1]:.2f} x {L[2]:.2f} m -> "
          f"{N[0] * h:.2f} x {N[1] * h:.2f} x {N[2] * h:.2f} m  =  {N[0]} x {N[1]} x {N[2]} cubes "
          f"({N[0] * N[1] * N[2]} background cells)")
    print(f"[cubic_block] growth per side: x {0.5 * (N[0] * h - L[0]):.2f} m, y {0.5 * (N[1] * h - L[1]):.2f} m, "
          f"top {N[2] * h - L[2]:.2f} m")


def do_verify(a):
    spec = json.loads(Path("system/cubicBlock.json").read_text())
    h, N = spec["h"], spec["N"]
    pts = Path("constant/polyMesh/points")
    if not pts.exists():
        sys.exit("cubic_block --verify: constant/polyMesh/points not found (run blockMesh first)")
    txt = pts.read_text(errors="ignore")
    if "format      binary" in txt[:600] or "format binary" in txt[:600]:
        sys.exit("cubic_block --verify: points file is binary (set writeFormat ascii in controlDict)")
    vals = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", txt)
    n = (N[0] + 1) * (N[1] + 1) * (N[2] + 1)
    if len(vals) != n:
        print(f"cubic_block --verify: {len(vals)} points, expected {n} for {N[0]}x{N[1]}x{N[2]} cells", file=sys.stderr)
        sys.exit(1)
    bad = 0
    for ax, name in enumerate("xyz"):
        u = sorted({round(float(v[ax]), 6) for v in vals})
        d = [b - c for b, c in zip(u[1:], u[:-1])]
        dev = max(abs(x - h) for x in d)
        if len(u) != N[ax] + 1 or dev > 1e-4 * h + 1e-5:
            print(f"cubic_block --verify: {name} spacing differs from h={h:g} by up to {dev:.3g} m "
                  f"({len(u) - 1} cells, expected {N[ax]})", file=sys.stderr)
            bad += 1
    if bad:
        sys.exit(1)
    print(f"[cubic_block] blockMesh check OK: {N[0]} x {N[1]} x {N[2]} cubes of {h:g} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-across", type=float, default=48, help="cells along the longer horizontal side (default 48)")
    ap.add_argument("--h", type=float, default=None, help="cube size in metres (overrides --cells-across)")
    ap.add_argument("--step", type=float, default=0.25, help="h is rounded to a multiple of this (m, default 0.25)")
    ap.add_argument("--verify", action="store_true", help="check the blockMesh result instead of planning")
    a = ap.parse_args()
    do_verify(a) if a.verify else do_plan(a)


if __name__ == "__main__":
    main()
