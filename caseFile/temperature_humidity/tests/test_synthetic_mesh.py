#!/usr/bin/env python3
"""
test_synthetic_mesh.py  --  check the python half of the case WITHOUT OpenFOAM

Builds a small polyMesh (points / faces / boundary only) whose `buildings` patch is made of the very
triangles of the layout (buildings + tree crowns), whose `ground` patch is a grid of 20 m quads, plus
dummy inlet / outlet / top faces.  Then it runs  epw_to_weather -> run_thermal -> map_surface_temperature
and compares the temperature written to every `buildings` face with the temperature of the triangle it
came from.  A correct mapping reproduces it to ~0 K.

    python3 tests/test_synthetic_mesh.py --json layout.json --epw weather.epw --month 5 --day 20 --hour-end 15
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CASE / "tools"))
import foam_io  # noqa: E402
from run_thermal import read_obj_vertices  # noqa: E402


def read_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V).reshape(-1, 3), F


def write_mesh(case, pts, patches):
    pm = case / "constant/polyMesh"
    pm.mkdir(parents=True, exist_ok=True)
    allf, bnd, start = [], [], 0
    for name, ptype, faces in patches:
        bnd.append((name, ptype, len(faces), start))
        allf.extend(faces)
        start += len(faces)
    (pm / "points").write_text(foam_io.foam_header("vectorField", "points", "constant/polyMesh") +
                               f"{len(pts)}\n(\n" + "\n".join(f"({p[0]:.6f} {p[1]:.6f} {p[2]:.6f})" for p in pts) + "\n)\n")
    (pm / "faces").write_text(foam_io.foam_header("faceList", "faces", "constant/polyMesh") +
                              f"{len(allf)}\n(\n" + "\n".join(f"{len(f)}(" + " ".join(map(str, f)) + ")" for f in allf) + "\n)\n")
    b = foam_io.foam_header("polyBoundaryMesh", "boundary", "constant/polyMesh") + f"{len(bnd)}\n(\n"
    for n, t, nf, sf in bnd:
        extra = "        inGroups        List<word> 1(wall);\n" if t == "wall" else ""
        b += f"    {n}\n    {{\n        type            {t};\n{extra}        nFaces          {nf};\n        startFace       {sf};\n    }}\n"
    (pm / "boundary").write_text(b + ")\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--epw", required=True, type=Path)
    ap.add_argument("--month", type=int, default=5)
    ap.add_argument("--day", type=int, default=20)
    ap.add_argument("--hour-end", type=int, default=15)
    ap.add_argument("--work", type=Path, default=Path("/tmp/uhi_synthetic_test"))
    ap.add_argument("--max-tree-faces", type=int, default=60000)
    ap.add_argument("--layout", choices=("legacy", "split"), default="legacy",
                    help="legacy: one `buildings` patch (buildings + trees);  split: obj_buildings_refined, obj_trees, "
                         "obj_lawns, obj_ponds + ground (the patch names of the current momentum pipeline)")
    ap.add_argument("--casefile", type=Path, default=CASE.parent,
                    help="folder that holds momentum/ and surface_temperature/ (default: the parent of this case)")
    a = ap.parse_args()

    work = a.work
    if work.exists():
        shutil.rmtree(work)
    case = work / "temperature_humidity"
    mom = work / "momentum"
    (case / "config").mkdir(parents=True)
    mom.mkdir(parents=True)
    shutil.copy(CASE / "config/surface_types.json", case / "config/surface_types.json")
    shutil.copytree(a.casefile / "momentum", mom, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("constant", "0", "processor*", "[1-9]*"))
    shutil.copytree(a.casefile / "surface_temperature", work / "surface_temperature", dirs_exist_ok=True)

    # same obj + shift as run_thermal (no buildingsOrig.obj -> bounding-box rule)
    raw = work / "raw"
    raw.mkdir()
    import contextlib
    import io
    from run_thermal import make_geometry_obj
    with contextlib.redirect_stdout(io.StringIO()):
        make_geometry_obj(mom / "json_to_obj.py", a.json, raw)
    bv = read_obj_vertices(raw / "buildings_refined.obj")
    tv = read_obj_vertices(raw / "trees.obj")
    allv = np.vstack([v for v in (bv, tv) if len(v)])
    lo, hi = allv.min(0), allv.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0.0])

    pts, faces = [], []

    def add_surface(path, drop_floor):
        V, F = read_obj(path)
        if not len(V):
            return []
        V = V + shift
        base = len(pts)
        pts.extend(V.tolist())
        out = []
        for f in F:
            p = V[f]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if drop_floor and p[:, 2].max() < 1e-6:
                continue
            out.append([base + i for i in f][::-1])      # reversed: OpenFOAM boundary normals point out of the fluid, into the solid
        return out

    bfaces = add_surface(raw / "buildings_refined.obj", True)
    tfaces = add_surface(raw / "trees.obj", False)
    if len(tfaces) > a.max_tree_faces:
        rng = np.random.default_rng(1)
        tfaces = [tfaces[i] for i in sorted(rng.choice(len(tfaces), a.max_tree_faces, replace=False))]
    bpatch = bfaces + tfaces
    half = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1]) + 60.0
    n = int(np.ceil(2 * half / 20.0))
    g0 = len(pts)
    xs = np.linspace(-half, half, n + 1)
    for y in xs:
        for x in xs:
            pts.append([x, y, 0.0])
    gfaces = []
    for j in range(n):
        for i in range(n):
            p0 = g0 + j * (n + 1) + i
            gfaces.append([p0, p0 + (n + 1), p0 + (n + 1) + 1, p0 + 1])     # clockwise seen from +z (normal down)
    lawn_f, pond_f = [], []
    if a.layout == "split":
        from run_thermal import json_polygons, point_in_polygon
        polys = json_polygons(a.json)
        keep = []
        for fq in gfaces:
            c = np.mean([pts[i] for i in fq], axis=0)
            x, y = np.array([c[0] - shift[0]]), np.array([c[1] - shift[1]])
            if any(point_in_polygon(x, y, pl)[0] for pl in polys["pond"]):
                pond_f.append(fq)
            elif any(point_in_polygon(x, y, pl)[0] for pl in polys["lawn"]):
                lawn_f.append(fq)
            else:
                keep.append(fq)
        gfaces = keep
    d0 = len(pts)
    for z in (0, 1):
        pts.extend([[-half, -half, 50 * z + 1], [half, -half, 50 * z + 1], [half, half, 50 * z + 1], [-half, half, 50 * z + 1]])
    P = np.array(pts)
    write_mesh(case, P, [("inlet", "patch", [[d0, d0 + 1, d0 + 2, d0 + 3]]),
                         ("outlet", "patch", [[d0 + 4, d0 + 5, d0 + 6, d0 + 7]]),
                         ("top", "symmetry", [[d0 + 3, d0 + 2, d0 + 1, d0]]),
                         ("ground", "wall", gfaces)] +
                        ([("buildings", "wall", bpatch)] if a.layout == "legacy" else
                         [("obj_buildings_refined", "wall", bfaces), ("obj_lawns", "wall", lawn_f),
                          ("obj_ponds", "wall", pond_f), ("obj_trees", "wall", tfaces)]))
    print(f"synthetic mesh ({a.layout}): {len(bfaces)} building faces, {len(tfaces)} tree faces, ground {len(gfaces)} faces"
          f" (+{len(lawn_f)} lawn, {len(pond_f)} pond), shift {shift.round(3)}")

    py = sys.executable
    def run(*args):
        r = subprocess.run([py, *map(str, args)], capture_output=True, text=True)
        sys.stdout.write(r.stdout[-1800:])
        if r.returncode:
            sys.stderr.write(r.stderr[-3000:])
            raise SystemExit(f"FAILED: {args[0]}")
        if r.stderr.strip():
            print("stderr:", r.stderr.strip()[-600:])
    run(CASE / "tools/epw_to_weather.py", "--epw", a.epw, "--month", a.month, "--day", a.day,
        "--hour-end", a.hour_end, "--out", case / "weatherData")
    run(CASE / "tools/run_thermal.py", "--case", case, "--momentum", mom, "--json", a.json,
        "--config", case / "config/surface_types.json", "--shift", *[f"{x:.6f}" for x in shift])
    run(CASE / "tools/map_surface_temperature.py", "--case", case)

    # ---- compare with the source triangles ---------------------------------------------------------
    S = np.load(case / "surfaces/thermal_surfaces.npz")
    mesh = foam_io.read_mesh(case)
    bname = "buildings" if a.layout == "legacy" else "obj_buildings_refined"
    ids, fc = foam_io.patch_faces(mesh, bname)
    C, Sv, A = foam_io.face_geometry(mesh["points"], fc)
    txt = (case / "0/T").read_text()
    def patch_values(text, name):
        mm = re.search(r"\b" + name + r"\s*\{[^}]*?value\s+(?:nonuniform List<scalar>\s*(\d+)\s*\(([^)]*)\)|uniform\s+(\S+);)", text, re.S)
        if mm.group(2) is not None:
            return np.array(mm.group(2).split(), dtype=float)
        return np.array([float(mm.group(3))])
    bT = patch_values(txt, bname) - 273.15
    assert len(bT) == len(fc), (len(bT), len(fc))
    cen_b = S["b_tri"].mean(axis=1)
    from scipy.spatial import cKDTree
    kd = cKDTree(cen_b) if len(cen_b) else None
    err_b = []
    tree_T = float(S["ambient_C"])
    ntree_ok = 0
    nb_faces = len(bfaces)
    if kd is not None:
        # coincident triangles of touching blocks exist in this synthetic mesh (a real snappyHexMesh mesh has no
        # buried faces); compare each face with the source cell of the SAME centroid AND the same outward normal.
        Nf = -Sv[:nb_faces] / A[:nb_faces, None]      # outward from the building (patch normal reversed)
        import map_surface_temperature as mst
        sel_idx, exp_T = [], []
        for fi in range(nb_faces):
            near = kd.query_ball_point(C[fi], 3.0)
            near = [c for c in near if float(np.dot(S["b_normal"][c], Nf[fi])) > 0.99]
            if not near:
                continue
            P3 = mesh["points"][fc[fi]]
            mids = [0.5 * (P3[i] + P3[(i + 1) % 3]) for i in range(3)]
            probes = ([C[fi]] + [w * C[fi] + (1 - w) * v for v in P3[:3] for w in (0.1, 0.5)]
                      + [0.1 * C[fi] + 0.9 * mdp for mdp in mids])
            hits_all = []
            for pr in probes:
                d = mst.point_tri_distance(np.repeat(pr[None], len(near), 0), S["b_tri"][near])
                hits_all.append([near[i] for i in np.flatnonzero(d < 1e-3)])
            own = set(hits_all[0])
            # skip faces that overlap another coplanar source cell (stacked / touching blocks in the layout;
            # a real snappyHexMesh mesh has no such buried faces)
            if len(own) == 1 and all(set(h) == own for h in hits_all):
                sel_idx.append(fi)
                exp_T.append(S["b_T"][next(iter(own))])
        sel_idx, exp_T = np.array(sel_idx), np.array(exp_T)
        err_b = np.abs(bT[sel_idx] - exp_T)
        print(f"building faces compared: {len(sel_idx)} / {nb_faces} (rest: buried or downward faces that the thermal "
              f"model does not treat as outdoor surfaces)   max |dT| = {err_b.max():.4f} K, mean {err_b.mean():.4f} K, "
              f">0.1 K: {(err_b > 0.1).sum()}")
        # every face with a noticeable difference must be explained by another coplanar source cell overlapping
        # part of it (stacked / touching blocks); otherwise it is a mapping error
        unexplained = 0
        for k_ in np.flatnonzero(err_b > 0.1):
            fi = sel_idx[k_]
            sp_, _ = mst.face_samples(mesh["points"], [fc[fi]], A[fi:fi + 1], [200])
            near = [c for c in kd.query_ball_point(C[fi], 3.0) if float(np.dot(S["b_normal"][c], Nf[fi])) > 0.99]
            cnt = np.zeros(len(sp_), int)
            for c in near:
                d = mst.point_tri_distance(sp_, np.repeat(S["b_tri"][c][None], len(sp_), 0))
                cnt += d < 1e-6
            if cnt.max() < 2:
                unexplained += 1
        print(f"faces with |dT| > 0.1 K: {(err_b > 0.1).sum()}, not explained by overlapping cells: {unexplained}")
        assert unexplained == 0, "building mapping error"
        assert err_b.mean() < 0.2
    if a.layout == "split":
        qtxt = (case / "0/qFlux").read_text()
        for pn, nm in (("obj_trees", len(tfaces)), ("obj_lawns", len(lawn_f)), ("obj_ponds", len(pond_f))):
            if nm == 0:
                continue
            Tp, Fp = patch_values(txt, pn) - 273.15, patch_values(qtxt, pn)
            assert len(Tp) == nm and len(Fp) == nm, (pn, len(Tp), nm)
            print(f"{pn:12s} {nm:6d} faces: T {Tp.min():.1f}..{Tp.max():.1f} C   vapour flux {Fp.min()*3600:.3f}..{Fp.max()*3600:.3f} mm/h")
            if pn == "obj_trees":
                assert np.abs(Tp - tree_T).max() < 1e-3
            else:
                assert Fp.min() > 0, pn + ": a wet patch without evaporation"
        # the building flux must be zero
        assert patch_values(qtxt, bname).max() == 0.0
    elif len(tfaces):
        e = np.abs(bT[nb_faces:] - tree_T)
        print(f"tree faces: {len(tfaces)}   max |T - ambient| = {e.max():.3f} K, 95th pct {np.percentile(e,95):.3f} K "
              "(faces on crown/trunk junctions or touching buildings can mix)")
    rep = json.loads((case / "surfaces/mapping_report.json").read_text())
    rb = rep['buildings'][bname]
    print("mapping report (buildings):", json.dumps({k: rb[k] for k in
          ('samples_without_normal_match', 'samples_farther_than_tolerance', 'T_min', 'T_max')}))
    print("ground:", json.dumps({k: rep['ground'][k] for k in ('faces_borrowed_from_neighbour', 'T_min', 'T_max',
          'faces_with_pond', 'faces_with_flux')}), json.dumps(rep['ground']['area_by_class_m2']))
    print("OK")


if __name__ == "__main__":
    main()
