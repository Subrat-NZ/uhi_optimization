#!/usr/bin/env python3
"""
run_thermal.py  --  layout JSON  ->  surface temperatures in the MESH frame

1. runs the momentum case's json_to_obj.py on the layout JSON (same geometry the mesh was built from)
2. moves the buildings / trees into the mesh frame (same shift the momentum pipeline applied)
3. builds ground.obj from the mesh's own `ground` patch faces (1 face -> 1..n triangles), and the
   fraction of every ground face that is pond / lawn / pathway / bare ground (from the JSON polygons)
4. runs surface_temperature/solar_thermal_model.py over the spin-up window (weatherData/MMDD) for
   the buildings, and once per ground material class that is present
5. writes  <work>/thermal_surfaces.npz  and  <work>/thermal_report.json

Two stages (--stage), so that the slow thermal model can run WHILE the momentum case is meshed and solved:
    raw   needs NO mesh: starts as soon as gen_ground_obj.py of the momentum case has finished.  Geometry and ground come from
          the surfaces that case has in constant/triSurface (buildings_refined.obj, trees.obj, ground.obj, lawns.obj, ponds.obj,
          pathways.obj; mesh frame).  Writes surfaces/thermal_raw.npz (+ thermal_report_raw.json).
    bind  needs the finished mesh: reads thermal_raw.npz and hands the ground temperatures to the faces of the mesh's ground /
          lawn / pond / pathway patches -> surfaces/thermal_surfaces.npz + thermal_report.json (same files as before).
    all   (default) the original single pass on the mesh ground patches.

Nothing in the thermal model is modified; it is imported and its own functions are called.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foam_io  # noqa: E402

CLASSES = ["default", "pathway", "lawn", "pond"]


# --------------------------------------------------------------------------- helpers
def load_module(path):
    spec = importlib.util.spec_from_file_location("solar_thermal_model", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solar_thermal_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def module_problems(tm):
    """What this script needs from solar_thermal_model.py that the given module does not have."""
    import inspect
    need = ["load_weather_folder", "WallSettings", "GroundSettings", "build_faces_from_obj", "TriangleRayTracer",
            "load_obj_mesh", "_simulate", "build_ground_faces_from_obj"]
    miss = [n for n in need if not hasattr(tm, n)]
    if hasattr(tm, "_simulate"):
        params = inspect.signature(tm._simulate).parameters
        miss += [f"_simulate(..., {q}=)" for q in ("surface_group", "wall_settings", "ray_tracer", "shadow_samples",
                                                  "t_initial_c") if q not in params]
    if hasattr(tm, "GroundSettings") and not hasattr(tm.GroundSettings, "slab_settings"):
        miss.append("GroundSettings.slab_settings")
    return miss


def api_kind(tm):
    """'layers'    : newer solar_thermal_model.py (build_layer_faces / load_obj_checked / prepare_obj_bundle)
       'ground_fn' : older one with build_ground_faces_from_obj (and the bundled copy)
       None        : neither, cannot be used"""
    layer_api = ["build_layer_faces", "load_obj_checked", "BuildingMaterials", "load_weather_folder", "WallSettings",
                 "GroundSettings", "build_faces_from_obj", "TriangleRayTracer", "_simulate"]
    if all(hasattr(tm, n) for n in layer_api):
        import inspect
        params = inspect.signature(tm._simulate).parameters
        if all(q in params for q in ("surface_group", "wall_settings", "ray_tracer", "shadow_samples", "t_initial_c")) \
                and hasattr(tm.GroundSettings, "slab_settings"):
            return "layers"
    return "ground_fn" if not module_problems(tm) else None


def read_obj_vertices(path):
    v = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                v.append((float(p[1]), float(p[2]), float(p[3])))
    return np.array(v, dtype=float).reshape(-1, 3)


def make_geometry_obj(j2o_path, json_path, out_dir, building_mesh_size=None):
    """buildings_refined.obj + trees.obj exactly as the momentum case's json_to_obj.py writes them.

    Only the buildings and trees are produced (ponds / lawns / pathways are flat and are taken from the json
    polygons here, and tessellating them is by far the slowest part of json_to_obj.py).  The functions and
    arguments are the ones main() of json_to_obj.py uses, including TREE_CROWN_SHAPE from the environment.
    """
    spec = importlib.util.spec_from_file_location("json_to_obj_mod", str(j2o_path))
    mod = importlib.util.module_from_spec(spec)
    old_cwd = os.getcwd()
    try:
        os.chdir(Path(j2o_path).parent)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    site = mod._extract_site(data)
    buildings = list(site.get("buildings", [])) + list(mod.collect_osm_buildings(site))
    trees = list(site.get("trees", []))
    crown = os.environ.get("TREE_CROWN_SHAPE", "sphere").strip().lower()
    crown = {"spherical": "sphere", "conical": "cone", "spheroidal": "spheroid",
             "ellipsoidal": "ellipsoid"}.get(crown, crown)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if building_mesh_size:
        size = building_mesh_size
    elif hasattr(mod, "compute_mesh_size"):                 # older json_to_obj.py
        size = mod.compute_mesh_size(buildings)
    else:                                                   # newer: sizes come from the module's MeshPolicy (--coarsen 3)
        size = mod.POLICY.building
    mod.buildings_to_obj(buildings, str(out_dir / "buildings_refined.obj"), size)
    mod.trees_to_obj(trees, str(out_dir / "trees.obj"), crown_shape=crown)
    return dict(buildings=len(buildings), trees=len(trees), building_mesh_size=size, crown_shape=crown)


def shift_obj(src, dst, shift):
    """Copy an OBJ, adding `shift` to every vertex (faces untouched)."""
    sx, sy, sz = shift
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            if line.startswith("v "):
                p = line.split()
                fo.write(f"v {float(p[1]) + sx:.6f} {float(p[2]) + sy:.6f} {float(p[3]) + sz:.6f}\n")
            else:
                fo.write(line)


def obj_counts(path):
    nv = nf = 0
    if Path(path).is_file():
        with open(path) as f:
            for line in f:
                nv += line.startswith("v ")
                nf += line.startswith("f ")
    return nv, nf


def find_shift(momentum, regenerated_vertices, override=None, mesh=None, roles=None):
    """Shift that maps the layout JSON frame to the mesh frame (new = old + shift)."""
    if override is not None:
        return np.array(override, dtype=float), "command line"
    tri = Path(momentum) / "constant/triSurface"
    orig, new = tri / "buildingsOrig.obj", tri / "buildings.obj"
    if orig.is_file() and new.is_file():
        a, b = read_obj_vertices(orig), read_obj_vertices(new)
        if len(a) == len(b) and len(a):
            d = b - a
            s = np.median(d, axis=0)
            if np.abs(d - s).max() < 1e-2:
                return s, "momentum buildingsOrig.obj -> buildings.obj"
            print("[run_thermal] WARNING: buildingsOrig.obj and buildings.obj differ by more than a "
                  "constant shift; falling back to the bounding-box rule", file=sys.stderr)
    # the mesh itself: fit the regenerated building / tree vertices onto the building / tree patch faces
    if mesh is not None and roles is not None and len(regenerated_vertices):
        names = roles["by_role"].get("buildings", []) + roles["by_role"].get("trees", [])
        if names:
            from scipy.spatial import cKDTree
            cen = np.vstack([foam_io.face_geometry(mesh["points"], foam_io.patch_faces(mesh, n)[1])[0] for n in names])
            kd = cKDTree(cen)
            v = regenerated_vertices[:: max(1, len(regenerated_vertices) // 20000)]
            lo, hi = regenerated_vertices.min(0), regenerated_vertices.max(0)
            s0 = 0.5 * (cen.min(0) + cen.max(0)) - 0.5 * (lo + hi)
            s0[2] = 0.0
            sh = s0.copy()
            for _ in range(30):
                d, idx = kd.query(v + sh)
                step = np.median(cen[idx] - (v + sh), axis=0)
                step[2] = 0.0
                sh += step
                if np.abs(step).max() < 1e-3:
                    break
            d, _ = kd.query(v + sh)
            return sh, (f"fitted to the mesh {names} (median distance regenerated vertex -> nearest face centre "
                        f"{np.median(d):.2f} m; accuracy about one cell size)")
    # fallback: translate.sh rule  -(bbox centre) in x and y, z unchanged
    if len(regenerated_vertices):
        lo, hi = regenerated_vertices.min(0), regenerated_vertices.max(0)
        c = 0.5 * (lo + hi)
        return np.array([-c[0], -c[1], 0.0]), "bounding-box centre of regenerated geometry (translate.sh rule)"
    return np.zeros(3), "none (no geometry)"


def point_in_polygon(px, py, poly):
    """Vectorised even-odd test. poly (n,2)."""
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= cond & (px < xint)
        j = i
    return inside


def json_polygons(json_path):
    data = json.loads(Path(json_path).read_text())
    site = data.get("siteOnly_json", data)

    def polys(items):
        out = []
        for it in items or []:
            c = it.get("coordinates") or []
            if len(c) >= 3:
                pts = np.array([[p["x"], p["y"]] for p in c], dtype=float)
                if np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                out.append(pts)
        return out

    return dict(pond=polys(site.get("ponds")), lawn=polys(site.get("grassAreas")),
                pathway=polys(site.get("roads")) + polys(site.get("pathways")))


def sample_points_in_face(pts, face, n):
    """n deterministic points spread over a polygon face (fan from the centre)."""
    P = pts[face]
    c = P.mean(axis=0)
    k = len(P)
    # low-discrepancy points in a triangle (c, P[i], P[i+1]) cycling through the fan
    out = np.empty((n, 3))
    g = 0.7548776662466927
    for s in range(n):
        u = (0.5 + s * g) % 1.0
        v = (0.5 + s * g * g) % 1.0
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        i = s % k
        a, b = P[i], P[(i + 1) % k]
        out[s] = c + u * (a - c) + v * (b - c)
    return out


def read_obj_tris(path):
    """(vertices (n,3), triangles (m,3)) of an OBJ; polygons are fan-triangulated."""
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                q = line.split()
                V.append((float(q[1]), float(q[2]), float(q[3])))
            elif line.startswith("f "):
                ix = [int(t.split("/")[0]) for t in line.split()[1:]]
                ix = [i - 1 if i > 0 else len(V) + i for i in ix]
                for k in range(1, len(ix) - 1):
                    F.append((ix[0], ix[k], ix[k + 1]))
    return np.array(V, dtype=float).reshape(-1, 3), np.array(F, dtype=np.int64).reshape(-1, 3)


GROUND_FILES = (("default", "ground.obj"), ("lawn", "lawns.obj"), ("pond", "ponds.obj"), ("pathway", "pathways.obj"))


def bind_ground(raw, gtri_xyz, class_names, max_dist_warn=8.0):
    """Ground results of the raw stage for the triangles of the MESH ground patches.
    For every class that has results: each mesh triangle is sampled at its centre and at three points 70 % of the way to its corners;
    every sample takes the value of the nearest raw triangle (centroid) of that class that has a value (triangles under buildings have
    none) and the four values are averaged.  Returns (dict of per-class arrays T, H, Rni, worst nearest distance)."""
    from scipy.spatial import cKDTree
    rtri = np.asarray(raw["gr_tri"], dtype=float)
    rc = rtri.mean(axis=1)
    C = gtri_xyz.mean(axis=1)
    samples = [C] + [C + 0.7 * (gtri_xyz[:, i] - C) for i in range(3)]
    out, worst = {}, 0.0
    for cname in class_names:
        key = f"g_T_{cname}"
        if key not in raw.files:
            continue
        have = np.isfinite(raw[key])
        kd = cKDTree(rc[have][:, :2])
        acc = {k: np.zeros(len(C)) for k in ("T", "H", "Rni")}
        src = {"T": raw[key][have], "H": raw[f"g_H_{cname}"][have], "Rni": raw[f"g_Rni_{cname}"][have]}
        for smp in samples:
            d, idx = kd.query(smp[:, :2])
            worst = max(worst, float(d.max()))
            for k in acc:
                acc[k] += src[k][idx] / len(samples)
        out[cname] = acc
    return out, worst


def auto(value, fallback):
    return fallback if (isinstance(value, str) and value.lower() == "auto") else float(value)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=Path("."), help="this case (needs constant/polyMesh)")
    ap.add_argument("--momentum", type=Path, default=None, help="momentum case (default ../momentum)")
    ap.add_argument("--json", type=Path, default=None, help="layout json (default: the only *.json in the momentum case)")
    ap.add_argument("--target", type=Path, default=None, help="weatherData/target.json")
    ap.add_argument("--config", type=Path, default=None, help="surface_types.json")
    ap.add_argument("--thermal-model", type=Path, default=None)
    ap.add_argument("--geometry", choices=("auto", "triSurface", "json"), default="auto",
                    help="auto: use constant/triSurface/buildings_refined.obj + trees.obj of the momentum case when they "
                         "exist (the surfaces snappyHexMesh used), else regenerate them from the layout json")
    ap.add_argument("--bundled-thermal-model", action="store_true",
                    help="use tools/bundled/solar_thermal_model.py even if ../surface_temperature has one")
    ap.add_argument("--json-to-obj", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None, help="output folder (default <case>/surfaces)")
    ap.add_argument("--shift", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"))
    ap.add_argument("--shadow-samples", type=int, choices=(1, 3), default=3)
    ap.add_argument("--ground-samples", type=int, default=9, help="points per ground face for the class fractions")
    ap.add_argument("--building-mesh-size", type=float, default=None, help="passed to json_to_obj.py")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--stage", choices=("all", "raw", "bind"), default="all",
                    help="raw: thermal model on the momentum case's surfaces, no mesh needed; bind: map thermal_raw.npz onto the "
                         "mesh; all: both in one pass on the mesh ground (default)")
    ap.add_argument("--ground-dir", type=Path, default=None,
                    help="raw stage: folder with ground.obj, lawns.obj, ponds.obj, pathways.obj (default <momentum>/constant/triSurface)")
    ap.add_argument("--buildings-obj", type=Path, default=None,
                    help="TEST/FALLBACK: use this OBJ (already in the mesh frame, e.g. the momentum case's "
                         "constant/triSurface/buildings.obj) as the only geometry instead of running json_to_obj. "
                         "Trees inside it are then treated as buildings.")
    a = ap.parse_args(argv)

    case = a.case.resolve()
    momentum = (a.momentum or case.parent / "momentum").resolve()
    work = (a.work or case / "surfaces").resolve()
    work.mkdir(parents=True, exist_ok=True)
    target_path = a.target or case / "weatherData/target.json"
    cfg = json.loads((a.config or case / "config/surface_types.json").read_text())
    tm_path = a.thermal_model or case.parent / "surface_temperature/solar_thermal_model.py"
    j2o = a.json_to_obj or momentum / "json_to_obj.py"
    tgt = json.loads(Path(target_path).read_text())
    wdir = Path(tgt["weather_folder"])
    if not wdir.is_absolute():
        wdir = (Path(target_path).parent.parent / wdir) if not wdir.exists() else wdir.resolve()
    jpath = a.json
    if jpath is None and not a.buildings_obj:
        cands = sorted(momentum.glob("*.json"))
        if len(cands) != 1:
            raise SystemExit(f"need --json (found {len(cands)} *.json in {momentum})")
        jpath = cands[0]
    jpath = Path(jpath).resolve() if jpath else None
    t0 = time.time()
    report = {"json": str(jpath) if jpath else None, "momentum": str(momentum), "weather_folder": str(wdir),
              "target": tgt.get("target")}

    # ---- 1/2. json -> obj -> mesh frame ------------------------------------------------
    raw = work / "raw"
    raw.mkdir(exist_ok=True)
    stage = a.stage
    mesh = roles = None
    if stage != "raw":
        mesh = foam_io.read_mesh(case)
        roles = foam_io.classify_patches(mesh, cfg.get("patches"))
        report["patches"] = roles["by_role"]
        print("[run_thermal] patches by role: " + "; ".join(f"{r}: {', '.join(n)}" for r, n in roles["by_role"].items()))
    tri_dir = momentum / "constant/triSurface"
    if stage != "bind":
        tri_b, tri_t = tri_dir / "buildings_refined.obj", tri_dir / "trees.obj"
        nb_tri = obj_counts(tri_b)[1] if tri_b.is_file() else 0
        nt_tri = obj_counts(tri_t)[1] if tri_t.is_file() else 0
        use_tri = (not a.buildings_obj) and a.geometry in ("auto", "triSurface") and (nb_tri > 0 or nt_tri > 0)
        if stage == "raw" and not use_tri:
            raise SystemExit(f"--stage raw needs buildings_refined.obj / trees.obj with faces in {tri_dir} "
                             "(json_to_obj.py of the momentum case has to have run)")
        if a.geometry == "triSurface" and not use_tri:
            raise SystemExit(f"--geometry triSurface: no buildings_refined.obj / trees.obj with faces in {tri_dir}")
        if use_tri:
            # the surfaces snappyHexMesh itself used: exactly the mesh geometry, in the mesh frame (no shift, no regeneration)
            shift = np.zeros(3)
            have_b, have_t = nb_tri > 0, nt_tri > 0
            if have_b:
                shutil.copy(tri_b, work / "buildings.obj")
            if have_t:
                shutil.copy(tri_t, work / "trees.obj")
            allv = np.vstack([read_obj_vertices(x) for x, h in ((tri_b, have_b), (tri_t, have_t)) if h])
            if mesh is not None:
                fit, how_fit = find_shift(Path("/nonexistent"), allv, None, mesh, roles)
            else:
                fit, how_fit = np.zeros(3), "not checked here (raw stage: no mesh yet; the bind stage checks it)"
            report["shift"] = dict(value=[0, 0, 0], source=f"constant/triSurface used by snappyHexMesh (mesh frame); "
                                                          f"check: {how_fit}; residual {np.round(fit, 3).tolist()}")
            report["geometry_check"] = dict(source=f"{tri_b.name if have_b else ''} {tri_t.name if have_t else ''} from {tri_dir}",
                                            mesh_fit_residual_shift=fit.tolist())
            print(f"[run_thermal] geometry: constant/triSurface/{tri_b.name if have_b else '-'} ({nb_tri} faces) + "
                  f"{tri_t.name if have_t else '-'} ({nt_tri} faces), mesh frame; mesh-fit residual shift "
                  f"{np.round(fit, 3).tolist()} m")
            if np.abs(fit[:2]).max() > 1.0:
                print("[run_thermal] WARNING: the triSurface geometry does not sit on the building / tree patches of the "
                      "mesh (residual shift > 1 m): was the mesh made from other surfaces?", file=sys.stderr)
        elif a.buildings_obj:
            shutil.copy(a.buildings_obj, work / "buildings.obj")
            shift = np.zeros(3)
            report["shift"] = dict(value=[0, 0, 0], source="--buildings-obj (already in the mesh frame)")
            have_b = obj_counts(work / "buildings.obj")[1] > 0
            have_t = False
            print(f"[run_thermal] using {a.buildings_obj} as geometry (mesh frame, no trees)")
        else:
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                info = make_geometry_obj(j2o, jpath, raw, a.building_mesh_size)
            (work / "json_to_obj.log").write_text(buf.getvalue())
            report["json_to_obj"] = info
            bv = read_obj_vertices(raw / "buildings_refined.obj") if (raw / "buildings_refined.obj").is_file() else np.zeros((0, 3))
            tv = read_obj_vertices(raw / "trees.obj") if (raw / "trees.obj").is_file() else np.zeros((0, 3))
            allv = np.vstack([bv, tv])
            shift, how = find_shift(momentum, allv, a.shift, mesh, roles)
            report["shift"] = dict(value=shift.tolist(), source=how)
            print(f"[run_thermal] json -> obj done: {len(bv)} building vertices, {len(tv)} tree vertices")
            print(f"[run_thermal] frame shift (json -> mesh) = {shift.round(4).tolist()}  [{how}]")
            have_b = len(bv) > 0 and obj_counts(raw / "buildings_refined.obj")[1] > 0
            have_t = len(tv) > 0 and obj_counts(raw / "trees.obj")[1] > 0
            if have_b:
                shift_obj(raw / "buildings_refined.obj", work / "buildings.obj", shift)
            if have_t:
                shift_obj(raw / "trees.obj", work / "trees.obj", shift)
            mom_obj = momentum / "constant/triSurface/buildings.obj"
            report["geometry_check"] = None
            if mom_obj.is_file() and len(allv):
                from scipy.spatial import cKDTree
                mv = read_obj_vertices(mom_obj)
                mine = allv + shift
                d, _ = cKDTree(mv).query(mine)
                frac1 = float(np.mean(d < 5e-3))
                d2, _ = cKDTree(mine).query(mv)
                frac2 = float(np.mean(d2 < 5e-3))
                report["geometry_check"] = dict(regenerated_vertices_found_in_mesh_geometry=frac1,
                                                mesh_geometry_vertices_found_in_regenerated=frac2)
                print(f"[run_thermal] geometry check vs momentum buildings.obj: {100*frac1:.1f} % of regenerated "
                      f"vertices and {100*frac2:.1f} % of the mesh geometry vertices coincide")
                if min(frac1, frac2) < 0.98:
                    print("[run_thermal] WARNING: the layout json does not reproduce the geometry the mesh was made "
                          "from (different json, tree-crown shape or mesh size?). Surface temperatures will be "
                          "mapped to the nearest surface, which may be wrong.", file=sys.stderr)

    RAWZ = None
    if stage == "bind":
        rawp = work / "thermal_raw.npz"
        if not rawp.is_file():
            raise SystemExit(f"--stage bind: {rawp} not found (run --stage raw first)")
        RAWZ = np.load(rawp)
        shift = np.array(RAWZ["shift"], dtype=float)
        rrp = work / "thermal_report_raw.json"
        if rrp.is_file():
            rr = json.loads(rrp.read_text())
            if rr.get("target") and tgt.get("target") and rr["target"] != tgt["target"]:
                raise SystemExit(f"--stage bind: thermal_raw.npz was made for the target hour {rr['target']}, this case is set to "
                                 f"{tgt['target']} - run --stage raw again")
            report.update({k: v for k, v in json.loads(rrp.read_text()).items() if k not in ("json", "momentum", "weather_folder")})
        # the building / tree surfaces must sit on the mesh's building / tree patches (they were copied from constant/triSurface)
        _bo, _to = work / "buildings.obj", work / "trees.obj"
        _allv = [read_obj_vertices(x) for x in (_bo, _to) if x.is_file() and obj_counts(x)[1] > 0]
        if _allv:
            _fit, _how = find_shift(Path("/nonexistent"), np.vstack(_allv), None, mesh, roles)
            report["geometry_check"] = dict(source="constant/triSurface (raw stage)", mesh_fit_residual_shift=_fit.tolist(), how=_how)
            print(f"[run_thermal] geometry vs mesh patches: residual shift {np.round(_fit, 3).tolist()} m")
            if np.abs(_fit[:2]).max() > 1.0:
                print("[run_thermal] WARNING: the triSurface geometry does not sit on the building / tree patches of the "
                      "mesh (residual shift > 1 m): was the mesh made from other surfaces?", file=sys.stderr)
    if stage != "raw":
        # ---- 3. ground from the mesh -----------------------------------------------------
        # ground-like patches: `ground` (class fractions from the json polygons) + flat lawn / pond / pathway patches (one class each)
        gfaces, gpatch, gforce = [], [], []
        gnames = [n for role in ("ground", "lawn", "pond", "pathway") for n in roles["by_role"].get(role, [])]
        if not gnames:
            raise SystemExit(f"no ground / lawn / pond patch in the mesh (patches: {[x['name'] for x in mesh['patches']]})")
        for pi_, n in enumerate(gnames):
            fcs = foam_io.patch_faces(mesh, n)[1]
            role = roles["role"][n]
            gfaces += fcs
            gpatch += [pi_] * len(fcs)
            gforce += [-1 if role == "ground" else CLASSES.index(role)] * len(fcs)
        gpatch, gforce = np.array(gpatch), np.array(gforce)
        pts = mesh["points"]
        tris, tri_face = [], []
        for fi, f in enumerate(gfaces):
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
                tri_face.append(fi)
        tris = np.array(tris)
        tri_face = np.array(tri_face)
        gtri_xyz = pts[tris]
        used = np.unique(tris)
        remap = -np.ones(len(pts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        with open(work / "ground.obj", "w") as f:
            f.write("# ground triangles taken from the polyMesh `ground` patch (mesh frame)\n")
            for p in pts[used]:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for t in tris:
                f.write(f"f {remap[t[0]] + 1} {remap[t[1]] + 1} {remap[t[2]] + 1}\n")
        zspan = float(np.ptp(pts[used][:, 2]))
        report["ground"] = dict(faces=len(gfaces), triangles=len(tris), z_span_m=zspan, patches=gnames,
                                faces_per_patch=[int((gpatch == i).sum()) for i in range(len(gnames))])
        if zspan > 1e-3:
            print(f"[run_thermal] WARNING: ground patch is not flat (z span {zspan:.3f} m); the thermal "
                  "model treats every ground triangle with its own normal.", file=sys.stderr)

        # class fractions per ground face from the JSON polygons (mesh frame = json + shift)
        polys = json_polygons(jpath) if jpath else dict(pond=[], lawn=[], pathway=[])
        ns = max(1, a.ground_samples)
        frac = np.zeros((len(gfaces), len(CLASSES)))
        # all sample points of all ground faces at once; a polygon is only tested for the points inside its bounding box
        free = np.flatnonzero(gforce < 0)
        if len(free):
            SP = np.vstack([sample_points_in_face(pts, gfaces[fi], ns) for fi in free])
            sx, sy = SP[:, 0] - shift[0], SP[:, 1] - shift[1]
            taken = np.zeros(len(sx), dtype=bool)
            for name in ("pond", "pathway", "lawn"):          # priority: pond > pathway > lawn
                m = np.zeros(len(sx), dtype=bool)
                for poly in polys[name]:
                    cand = np.flatnonzero((sx >= poly[:, 0].min()) & (sx <= poly[:, 0].max()) &
                                          (sy >= poly[:, 1].min()) & (sy <= poly[:, 1].max()))
                    if len(cand):
                        m[cand] |= point_in_polygon(sx[cand], sy[cand], poly)
                m &= ~taken
                frac[free, CLASSES.index(name)] = m.reshape(len(free), ns).mean(axis=1)
                taken |= m
            frac[free, 0] = 1.0 - taken.reshape(len(free), ns).mean(axis=1)
        for fi in np.flatnonzero(gforce >= 0):              # a whole lawn / pond patch face is that class
            frac[fi] = 0.0
            frac[fi, gforce[fi]] = 1.0
        C, S, A = foam_io.face_geometry(pts, gfaces)
        cls_area = (frac * A[:, None]).sum(axis=0)
        json_area = {}
        for name in ("pond", "pathway", "lawn"):
            json_area[name] = float(sum(abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
                                        for p in polys[name]))
        report["ground_classes"] = {c: dict(mapped_area_m2=float(cls_area[i]), json_area_m2=json_area.get(c))
                                    for i, c in enumerate(CLASSES)}
        print("[run_thermal] ground classes (mapped on mesh faces vs. json polygon area, m2): " +
              ", ".join(f"{c} {cls_area[i]:.0f}/{json_area.get(c, float('nan')):.0f}" for i, c in enumerate(CLASSES) if c != "default"))

    # ---- 3'. RAW stage: ground from the surfaces of the momentum case (no mesh) --------------------
    # ground.obj = bare ground with holes where lawns / ponds / pathways are, lawns.obj / ponds.obj / pathways.obj fill them (all in the
    # mesh frame, flat at z = 0).  Every file is one material class.
    if stage == "raw":
        gdir = (a.ground_dir or tri_dir)
        Vs, Fs, gcls, off = [], [], [], 0
        for cname, fn in GROUND_FILES:
            fp = Path(gdir) / fn
            if not fp.is_file():
                continue
            V_, F_ = read_obj_tris(fp)
            if not len(F_):
                continue
            Vs.append(V_)
            Fs.append(F_ + off)
            gcls += [CLASSES.index(cname)] * len(F_)
            off += len(V_)
            print(f"[run_thermal] ground surface {fn}: {len(F_)} triangles -> class '{cname}'")
        if not Fs:
            raise SystemExit(f"--stage raw: no ground triangles (ground.obj, lawns.obj, ...) in {gdir}")
        gV, gF, gcls = np.vstack(Vs), np.vstack(Fs), np.array(gcls)
        e1, e2 = gV[gF[:, 1]] - gV[gF[:, 0]], gV[gF[:, 2]] - gV[gF[:, 0]]
        nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        gF[nz < 0] = gF[nz < 0][:, [0, 2, 1]]                 # normals up
        gtri_xyz = gV[gF]
        area = 0.5 * np.abs(nz)
        cls_area = np.array([area[gcls == i].sum() for i in range(len(CLASSES))])
        with open(work / "ground.obj", "w") as f:
            f.write("# ground triangles of the momentum case surfaces (ground, lawns, ponds, pathways; mesh frame)\n")
            for q in gV:
                f.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
            for t in gF:
                f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        polys = json_polygons(jpath) if jpath else dict(pond=[], lawn=[], pathway=[])
        json_area = {n: float(sum(abs(0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])) for q in polys[n]))
                     for n in ("pond", "pathway", "lawn")}
        report["ground"] = dict(source=str(gdir), triangles=int(len(gF)), z_span_m=float(np.ptp(gtri_xyz[:, :, 2])))
        report["ground_classes"] = {c: dict(surface_area_m2=float(cls_area[i]), json_area_m2=json_area.get(c))
                                    for i, c in enumerate(CLASSES)}
        print("[run_thermal] ground classes (surface files vs. json polygon area, m2): " +
              ", ".join(f"{c} {cls_area[i]:.0f}/{json_area.get(c, float('nan')):.0f}" for i, c in enumerate(CLASSES) if c != "default"))

    # ---- BIND stage: raw results -> faces of the mesh ground / lawn / pond / pathway patches --------------
    if stage == "bind":
        t1 = time.time()
        mapped, worst = bind_ground(RAWZ, gtri_xyz, CLASSES)
        out = {k: RAWZ[k] for k in ("b_tri", "b_T", "b_normal", "t_tri", "t_normal", "t_T", "ambient_C")}
        out.update(g_tri_face=tri_face, g_frac=frac, g_class_names=np.array(CLASSES), g_patch_names=np.array(gnames),
                   g_patch_of_face=gpatch, shift=shift)
        for cname, m in mapped.items():
            out[f"g_T_{cname}"], out[f"g_H_{cname}"], out[f"g_Rni_{cname}"] = m["T"], m["H"], m["Rni"]
            for k in ("Tdeep", "k", "depth", "eps"):
                out[f"g_{k}_{cname}"] = float(RAWZ[f"g_{k}_{cname}"])
            print(f"[run_thermal] bind: class '{cname}': T_surface {m['T'].min():.1f} .. {m['T'].max():.1f} C on {len(m['T'])} mesh triangles")
        np.savez_compressed(work / "thermal_surfaces.npz", **out)
        report["bind"] = dict(mesh_triangles=int(len(gtri_xyz)), worst_nearest_distance_m=worst, seconds=time.time() - t1)
        report["counts"] = dict(building_cells=int(len(out["b_tri"])), tree_cells=int(len(out["t_tri"])),
                                ground_triangles=int(len(gtri_xyz)))
        (work / "thermal_report.json").write_text(json.dumps(report, indent=2))
        print(f"[run_thermal] bind done in {time.time()-t1:.0f} s (worst nearest raw-triangle distance {worst:.1f} m) -> {work/'thermal_surfaces.npz'}")
        if worst > 8.0:
            print("[run_thermal] WARNING: some mesh ground faces are > 8 m from a raw ground triangle of that class: "
                  "is the mesh domain the one gen_ground_obj.py covered?", file=sys.stderr)
        return 0

    # ---- 4. thermal model --------------------------------------------------------------
    tm = load_module(tm_path)
    kind = api_kind(tm)
    miss = [] if kind else module_problems(tm)
    report["thermal_model"] = dict(path=str(tm_path), api=kind, missing=miss)
    if kind is None or a.bundled_thermal_model:
        bundled = Path(__file__).resolve().parent / "bundled/solar_thermal_model.py"
        if kind is None:
            print(f"[run_thermal] WARNING: {tm_path} lacks {miss} (a version this case does not know).\n"
                  f"[run_thermal] Using the copy bundled with this case: {bundled}", file=sys.stderr)
        tm = load_module(bundled)
        kind = api_kind(tm)
        report["thermal_model"].update(used=str(bundled), api=kind)
        if kind is None:
            raise SystemExit(f"bundled thermal model is incomplete: {module_problems(tm)}")
    print(f"[run_thermal] thermal model: {report['thermal_model'].get('used', tm_path)} (api: {kind})")
    scene = None
    if kind == "layers" and jpath is not None:
        try:
            scene = tm.load_scenes([str(jpath)])[0]
        except Exception as exc:                                  # materials then fall back to the model's defaults
            print(f"[run_thermal] note: could not read building materials from {jpath} ({exc})", file=sys.stderr)
    wx = tm.load_weather_folder(str(wdir))
    records = [dict(hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
                    sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
                    i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i]) for i in range(len(wx["hour"]))]
    last_hour = float(wx["hour"][-1])
    annual_mean = tgt.get("annual_mean_air_temperature_C")
    if annual_mean is None:
        annual_mean = float(np.mean(wx["T_amb_C"]))
    w = cfg.get("walls", {})
    ws = tm.WallSettings(thickness_m=float(w.get("thickness_m", 0.23)), nodes=int(w.get("nodes", 5)),
                         indoor_temperature_c=float(w.get("indoor_temperature_C", 24.0)),
                         indoor_h_wm2k=float(w.get("indoor_h_Wm2K", 8.0)))

    btris_xyz, btri_T, btri_normal = [], [], []
    b_loaded = False
    tracer_tris = []
    building_faces = []
    building_tracer = None
    if have_b:
        if kind == "layers":
            bmesh = tm.load_obj_checked(str(work / "buildings.obj"), "buildings")
            if bmesh is not None:
                faces_b, caps, _, _ = tm.build_faces_from_obj(mesh=bmesh, materials=tm.BuildingMaterials(scene))
                building_faces = faces_b
                btri_arr = bmesh[0][bmesh[1]]
                tracer_tris.extend(btri_arr.tolist())
                building_tracer = tm.TriangleRayTracer(btri_arr)
        else:
            faces_b, caps, _, occ = tm.build_faces_from_obj(str(work / "buildings.obj"))
            building_faces = faces_b
            btri = [t for o in occ.values() if o is not None for t in o.triangles]
            tracer_tris.extend(btri)
            if btri:
                building_tracer = tm.TriangleRayTracer(btri)
    trees_v = trees_f = None
    if have_t:
        if kind == "layers":
            tmesh = tm.load_obj_checked(str(work / "trees.obj"), "trees")
            trees_v, trees_f = tmesh if tmesh is not None else (np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
        else:
            trees_v, trees_f = tm.load_obj_mesh(str(work / "trees.obj"))
        if len(trees_f):
            tracer_tris.extend(trees_v[np.asarray(trees_f, dtype=int)].tolist())
        else:
            have_t = False
    tracer = None
    if tracer_tris and not a.no_shadows:
        print(f"[run_thermal] ray tracer: {len(tracer_tris)} occluding triangles", flush=True)
        tracer = tm.TriangleRayTracer(tracer_tris)

    # buildings
    if building_faces:
        t1 = time.time()
        fdf, _ = tm._simulate(building_faces, caps, records, t_initial_c=None, ray_tracer=tracer,
                              shadow_samples=a.shadow_samples, wall_settings=ws)
        fin = fdf[np.isclose(fdf["hour"], last_hour)].set_index("face_id")
        keep = [f for f in building_faces if f.face_id in fin.index]
        btris_xyz = np.array([f.vertices[:3] for f in keep])
        btri_T = fin.loc[[f.face_id for f in keep], "T_surface_C"].to_numpy(float)
        btri_normal = np.array([f.normal for f in keep])
        print(f"[run_thermal] buildings: {len(keep)} cells, T_surface "
              f"{btri_T.min():.1f} .. {btri_T.max():.1f} C   ({time.time()-t1:.0f} s)", flush=True)
    else:
        btris_xyz = np.zeros((0, 3, 3))
        btri_T = np.zeros(0)
        btri_normal = np.zeros((0, 3))

    # trees (geometry only, ambient temperature)
    ambient_last = float(wx["T_amb_C"][-1])
    if have_t:
        ttri = trees_v[np.asarray(trees_f, dtype=int)]
        tn = np.cross(ttri[:, 1] - ttri[:, 0], ttri[:, 2] - ttri[:, 0])
        ta = np.linalg.norm(tn, axis=1)
        ok = ta > 1e-12
        ttri, tn = ttri[ok], tn[ok] / ta[ok, None]
    else:
        ttri = np.zeros((0, 3, 3))
        tn = np.zeros((0, 3))

    # ground, one run per material class that occurs
    ng = len(gtri_xyz)
    gT = {}
    gH = {}                                   # convective coefficient used by the thermal model
    gR = {}                                   # isothermal net radiation of every ground triangle
    removed = np.zeros(ng, dtype=bool)
    for ci, cname in enumerate(CLASSES):
        if cls_area[ci] <= 0:
            continue
        th = cfg["ground"][cname]["thermal"]
        gkw = dict(
            depth_m=float(th.get("depth_m", 0.23)), nodes=int(th.get("nodes", 5)),
            deep_temperature_c=auto(th.get("deep_temperature_C", "auto"), annual_mean),
            thermal_conductivity=float(th["conductivity"]), density=float(th["density"]),
            specific_heat=float(th["specific_heat"]), albedo=float(th["albedo"]),
            emissivity=float(th["emissivity"]))
        sim_extra = {}
        if "wetness" in getattr(tm.GroundSettings, "__dataclass_fields__", {}):
            # no evaporation inside the thermal model: lawn / pond evaporation is the analytic wet_surface balance
            # of map_surface_temperature.py (the same Rni is used there), soil and asphalt stay dry.
            gkw["wetness"] = 0.0
            sim_extra["wetness"] = 0.0
        gs = tm.GroundSettings(**gkw)
        t1 = time.time()
        if kind == "layers":
            gV, gF = tm.load_obj_checked(str(work / "ground.obj"), "ground")
            gfaces_m = tm.build_layer_faces("ground", gV, gF, gs, building_tracer, None, "up")
            from scipy.spatial import cKDTree
            clean2orig = cKDTree(gtri_xyz.mean(axis=1)).query(gV[gF].mean(axis=1))[1]
        else:
            gfaces_m = tm.build_ground_faces_from_obj(str(work / "ground.obj"), building_tracer, gs)
            clean2orig = None
        cap = gs.density * gs.specific_heat * gs.depth_m * sum(f.area for f in gfaces_m)
        gdf, _ = tm._simulate(gfaces_m, {"ground": cap}, records, t_initial_c=None, ray_tracer=tracer,
                              shadow_samples=a.shadow_samples, wall_settings=gs.slab_settings(),
                              surface_group="ground", **sim_extra)
        fin = gdf[np.isclose(gdf["hour"], last_hour)].set_index("face_id")
        T = np.full(ng, np.nan)
        H = np.full(ng, np.nan)
        R = np.full(ng, np.nan)
        for f in gfaces_m:
            idx = int(f.face_id.rsplit("_", 1)[1])
            if clean2orig is not None:
                idx = int(clean2orig[idx])
            ts_, ta_ = fin.at[f.face_id, "T_surface_C"], fin.at[f.face_id, "T_amb_C"]
            T[idx] = ts_
            H[idx] = fin.at[f.face_id, "h_conv_Wm2K"]
            # isothermal net radiation (what the surface would absorb at T_surface = T_air): absorbed shortwave
            # (albedo and shadows included) minus net longwave loss evaluated at the air temperature
            h_rad = 4.0 * gs.emissivity * 5.670374419e-8 * (ta_ + 273.15) ** 3
            q_lw_amb = fin.at[f.face_id, "q_longwave_out_Wm2"] - h_rad * (ts_ - ta_)
            R[idx] = fin.at[f.face_id, "q_solar_absorbed_Wm2"] - q_lw_amb
        gT[cname], gH[cname] = T, H
        gR[cname] = R
        removed |= np.isnan(T)
        print(f"[run_thermal] ground class '{cname}': T_surface {np.nanmin(T):.1f} .. {np.nanmax(T):.1f} C "
              f"({time.time()-t1:.0f} s)", flush=True)

    out = dict(
        b_tri=btris_xyz, b_T=btri_T, b_normal=btri_normal,
        t_tri=ttri, t_normal=tn, t_T=np.full(len(ttri), ambient_last),
        g_class_names=np.array(CLASSES), ambient_C=ambient_last, shift=shift,
    )
    if stage == "raw":
        out.update(gr_tri=gtri_xyz, gr_cls=gcls)            # ground triangles of the momentum surfaces (no mesh indices yet)
    else:
        out.update(g_tri_face=tri_face, g_frac=frac, g_patch_names=np.array(gnames), g_patch_of_face=gpatch)
    for k, v in gT.items():
        out[f"g_T_{k}"] = v
        out[f"g_H_{k}"] = gH[k]
        out[f"g_Rni_{k}"] = gR[k]
        out[f"g_Tdeep_{k}"] = float(auto(cfg["ground"][k]["thermal"].get("deep_temperature_C", "auto"), annual_mean))
        out[f"g_k_{k}"] = float(cfg["ground"][k]["thermal"]["conductivity"])
        out[f"g_depth_{k}"] = float(cfg["ground"][k]["thermal"].get("depth_m", 0.23))
        out[f"g_eps_{k}"] = float(cfg["ground"][k]["thermal"]["emissivity"])
    nm_npz = "thermal_raw.npz" if stage == "raw" else "thermal_surfaces.npz"
    nm_rep = "thermal_report_raw.json" if stage == "raw" else "thermal_report.json"
    np.savez_compressed(work / nm_npz, **out)
    report["counts"] = dict(building_cells=int(len(btris_xyz)), tree_cells=int(len(ttri)),
                            ground_triangles=int(ng), ground_triangles_removed_under_buildings=int(removed.sum()))
    report["ambient_C"] = ambient_last
    report["stage"] = stage
    report["seconds"] = time.time() - t0
    (work / nm_rep).write_text(json.dumps(report, indent=2))
    print(f"[run_thermal] done in {time.time()-t0:.0f} s -> {work/nm_npz}" +
          ("   (next: --stage bind, once the momentum mesh exists)" if stage == "raw" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
