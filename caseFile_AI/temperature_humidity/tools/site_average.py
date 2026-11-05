#!/usr/bin/env python3
"""
site_average.py  --  site-averaged air temperature / specific humidity / wind speed at pedestrian height

Reads the LATEST time of a (reconstructed) airTqFoam run: T, q, U, and the mesh, and averages them over
the cells whose vertical extent contains the chosen height (default 2 m) and whose centre lies inside the
site boundary polygon. Weights are the horizontal cell areas (V / dz).

Writes  <case>/postProcessing/site_average.json  (T_C, q, RH_pct, wind speed, ... and the inlet reference).
The site polygon is taken from constant/triSurface/site_boundary_shifted.obj of the momentum case (mesh frame);
without it the bounding box of the `buildings` patch is used.
UTCI also needs Tmrt, which is not part of this case.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foam_io  # noqa: E402
from run_thermal import point_in_polygon, read_obj_vertices  # noqa: E402


def e_sat_pa(t_c):
    return 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def cell_geometry(mesh):
    pts, faces = mesh["points"], mesh["faces"]
    own, nei = mesh["owner"], mesh["neighbour"]
    ncell = int(max(own.max(), nei.max() if len(nei) else 0)) + 1
    Cf, Sf, _ = foam_io.face_geometry(pts, faces)
    nint = len(nei)
    c0 = np.zeros((ncell, 3))
    cnt = np.zeros(ncell)
    np.add.at(c0, own, Cf)
    np.add.at(cnt, own, 1)
    np.add.at(c0, nei, Cf[:nint])
    np.add.at(cnt, nei, 1)
    c0 /= cnt[:, None]
    vol = np.zeros(ncell)
    cen = np.zeros(ncell, dtype=float).reshape(-1, 1).repeat(3, axis=1)
    pyr_o = np.einsum("fj,fj->f", Sf, Cf - c0[own]) / 3.0
    ce_o = 0.75 * Cf + 0.25 * c0[own]
    np.add.at(vol, own, pyr_o)
    np.add.at(cen, own, pyr_o[:, None] * ce_o)
    pyr_n = -np.einsum("fj,fj->f", Sf[:nint], Cf[:nint] - c0[nei]) / 3.0
    ce_n = 0.75 * Cf[:nint] + 0.25 * c0[nei]
    np.add.at(vol, nei, pyr_n)
    np.add.at(cen, nei, pyr_n[:, None] * ce_n)
    cen /= vol[:, None]
    fz0 = np.array([pts[f][:, 2].min() for f in faces])
    fz1 = np.array([pts[f][:, 2].max() for f in faces])
    z0 = np.full(ncell, np.inf)
    z1 = np.full(ncell, -np.inf)
    np.minimum.at(z0, own, fz0)
    np.maximum.at(z1, own, fz1)
    np.minimum.at(z0, nei, fz0[:nint])
    np.maximum.at(z1, nei, fz1[:nint])
    return cen, vol, z0, z1, Cf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=Path("."))
    ap.add_argument("--time", default=None, help="time directory (default: latest)")
    ap.add_argument("--site-obj", type=Path, default=None)
    ap.add_argument("--target", type=Path, default=None)
    ap.add_argument("--heights", type=float, nargs="+", default=[2.0])
    a = ap.parse_args(argv)
    case = a.case.resolve()
    tname = a.time or foam_io.latest_time(case)
    if tname is None:
        raise SystemExit("no time directory found")
    tdir = case / tname
    mesh = foam_io.read_mesh(case, with_cells=True)
    cen, vol, z0, z1, Cf = cell_geometry(mesh)
    n = len(vol)
    T = foam_io.read_internal_field(tdir / "T", n) - 273.15
    q = foam_io.read_internal_field(tdir / "q", n)
    # airTqFoam does not write U (it is only read), so take it from the result time if present, else the frozen
    # momentum field copied into 0/
    upath = next((p for p in (tdir / "U", case / "0" / "U") if p.is_file()), None)
    if upath is None:
        raise SystemExit(f"no U found in {tdir} or {case / '0'}")
    U = foam_io.read_internal_field(upath, n, rank="vector")
    spd = np.linalg.norm(U, axis=1)
    tgt = json.loads(Path(a.target or case / "weatherData/target.json").read_text())
    p = tgt["boundary"]["pressure_Pa"]

    site = a.site_obj
    if site is None:
        for c in (case.parent / "momentum/constant/triSurface/site_boundary_shifted.obj",
                  case / "constant/triSurface/site_boundary_shifted.obj"):
            if c.is_file():
                site = c
                break
    if site and Path(site).is_file():
        poly = read_obj_vertices(site)[:, :2]
        # order the loop by angle about its centre (the file is one convex-ish quad/polygon)
        c = poly.mean(axis=0)
        poly = poly[np.argsort(np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0]))]
        inside = point_in_polygon(cen[:, 0], cen[:, 1], poly)
        site_desc = str(site)
    else:
        roles = foam_io.classify_patches(mesh)
        names = roles["by_role"].get("buildings", []) + roles["by_role"].get("trees", [])
        b_ids = np.concatenate([foam_io.patch_faces(mesh, n)[0] for n in names])
        bc = Cf[b_ids]
        lo, hi = bc.min(axis=0), bc.max(axis=0)
        inside = (cen[:, 0] >= lo[0]) & (cen[:, 0] <= hi[0]) & (cen[:, 1] >= lo[1]) & (cen[:, 1] <= hi[1])
        site_desc = "bounding box of the building / tree patches"

    out = {"time": tname, "site_region": site_desc, "heights": {}}
    for h in a.heights:
        sel = inside & (z0 <= h) & (z1 > h)   # one cell per vertical column
        if not sel.any():
            out["heights"][str(h)] = None
            continue
        w = vol[sel] / np.maximum(z1[sel] - z0[sel], 1e-9)
        Tm, qm, sm = (np.sum(w * x[sel]) / w.sum() for x in (T, q, spd))
        e = qm * p / (0.622 + 0.378 * qm)
        out["heights"][str(h)] = dict(cells=int(sel.sum()), area_m2=float(w.sum()),
                                      T_C=float(Tm), q_kg_per_kg=float(qm),
                                      RH_pct=float(100 * e / e_sat_pa(Tm)), wind_speed_ms=float(sm),
                                      T_C_min=float(T[sel].min()), T_C_max=float(T[sel].max()))
    b = tgt["boundary"]
    out["inlet_reference"] = dict(T_C=b["air_temperature_C"], q_kg_per_kg=b["specific_humidity_kg_per_kg"],
                                  RH_pct=b["relative_humidity_pct"])
    (case / "postProcessing").mkdir(exist_ok=True)
    (case / "postProcessing/site_average.json").write_text(json.dumps(out, indent=2))
    for h, v in out["heights"].items():
        if v:
            print(f"[site_average] z={h} m, {v['cells']} cells: T {v['T_C']:.2f} C  q {v['q_kg_per_kg']*1000:.2f} g/kg  "
                  f"RH {v['RH_pct']:.0f} %  |U| {v['wind_speed_ms']:.2f} m/s   (inlet: {b['air_temperature_C']:.1f} C, "
                  f"{b['specific_humidity_kg_per_kg']*1000:.2f} g/kg)")
        else:
            print(f"[site_average] z={h} m: no cells inside the site region")
    return 0


if __name__ == "__main__":
    sys.exit(main())
