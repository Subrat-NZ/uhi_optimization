#!/usr/bin/env python3
"""
compute_utci.py  --  site area-averaged UTCI at pedestrian height from an airTqFoam run

Inputs (all inside the case, produced by the earlier steps):
    latest time        T, q, U                 air temperature, specific humidity, wind (frozen momentum field)
    0/T                wall temperatures       ground / lawn / pond patch faces  -> long-wave from the ground
    surfaces/          buildings.obj trees.obj the occluders the thermal model used (mesh frame)
                       thermal_surfaces.npz    building surface temperatures
    weatherData/       target.json + the weather folder (sun vector, DNI, DHI, sky infrared)
    layout JSON        site polygon  (site.coordinates, mesh frame)

Method (the SolarCal formula of select_epw_cases.py, evaluated per point with the local surroundings):
    sample points      cells of the CFD mesh that contain z = --height inside the site, area-weighted (stratified sampling);
                       T, q, |U| at --height from a quadratic fit through the cell-centre values of that column (see below)
    sun                direct beam through the building + tree BVH (thermal model's TriangleRayTracer);
                       a tree crown transmits --tree-transmissivity (default 0.15) of the beam and of the diffuse sky
    sky                cosine-weighted hemisphere rays (--sky-rays) -> open / tree-blocked / building-blocked fractions
    short wave         S = a_k * ( 0.5*DHI_eff + 0.5*albedo*GHI_local + fp*DNI*sunlit )
    long wave          L = eps_p * ( 0.5*(f_open*IR_sky + f_tree*sigma*Ta^4 + f_bld*sigma*Tb^4) + 0.5*eps_g*sigma*Tg^4 )
                       Tg = the mapped ground / lawn / pond surface temperature below the point, Tb = mean building surface T
    Tmrt               ((S + L) / (eps_p sigma))^(1/4)
    UTCI               pythermalcomfort.models.utci (same call as select_epw_cases.py) with Ta, RH, |U| of the CFD cell
With nothing around (f_open = 1, Tg = Ta) this is exactly the rural Tmrt of select_epw_cases.py, so the layout
objective measures only what the layout changes: shade, sky view, surface temperature, wind, air T and humidity.

Height of the sample points (--height, e.g. 0.75): the CFD gives one value per cell (its centre), so T, q and |U| at --height are
reconstructed from the vertical column of cells at each sample point: a quadratic through the centre values of the three cells
whose centres are nearest to --height (Lagrange form, also for unequal cell heights; extrapolation when --height lies below the
first centre).  The overshoot is limited (--fit-overshoot), |U| and q are kept >= 0.  Falls back to linear (two cells) or to the cell
value (one cell, e.g. under a roof).  `--interp none` takes the cell value as before, `--interp linear` uses two cells.
How many points were fitted and how far the fit moved the values is printed and written to utci.json ("interpolation").

Wind: |U| of the CFD cell is used as it is (--wind-factor 1).  The momentum case has a uniform inlet equal to the
10 m wind of the EPW, so the open-field wind at 2 m already is about that value.

Writes  <out> (default <case>/../UTCI_details.txt, the file run_candidates.py parses):
        === Processing target hour ... ===
        Site area-averaged: UTCI=xx.xx C (category), ...
and <case>/postProcessing/utci.json, <case>/postProcessing/utci_points.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foam_io  # noqa: E402
import run_thermal as RT  # noqa: E402
from site_average import cell_geometry, e_sat_pa  # noqa: E402

SIGMA = 5.670374419e-8

CATEGORIES = [(46.0, "extreme heat stress"), (38.0, "very strong heat stress"), (32.0, "strong heat stress"),
              (26.0, "moderate heat stress"), (9.0, "no thermal stress"), (0.0, "slight cold stress"),
              (-13.0, "moderate cold stress"), (-27.0, "strong cold stress"), (-40.0, "very strong cold stress")]


def category(v):
    for lo, name in CATEGORIES:
        if v > lo:
            return name
    return "extreme cold stress"


def patch_values(text, name):
    import re
    m = re.search(r"\b" + re.escape(name) + r"\s*\{[^}]*?value\s+(?:nonuniform\s+List<scalar>\s*(\d+)\s*\(([^)]*)\)|uniform\s+(\S+);)",
                  text, re.S)
    if m is None:
        raise ValueError(f"patch {name} not found in 0/T")
    if m.group(2) is not None:
        return np.array(m.group(2).split(), dtype=float)
    return np.array([float(m.group(3))])


def hemisphere_rays(n):
    """n directions on the upper hemisphere, cosine-weighted (equal weights = cosine-weighted sampling):
    Fibonacci spiral in (sin^2 theta, phi)."""
    k = (np.arange(n) + 0.5) / n
    sin_t = np.sqrt(k)                       # sin^2(theta) uniform -> cosine-weighted sampling
    cos_t = np.sqrt(1.0 - k)
    phi = np.arange(n) * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def stratified_pick(w, n):
    """Indices chosen with probability proportional to w by systematic sampling (deterministic)."""
    c = np.cumsum(w)
    c /= c[-1]
    u = (np.arange(n) + 0.5) / n
    return np.searchsorted(c, u)


def horizontal_links(mesh, cen):
    """vertical neighbours of the cells: for every internal face that is (nearly) horizontal, the lower and the upper cell.
    Returns CSR tables  up_ptr, up_idx  (cells above each cell) and  dn_ptr, dn_idx  (cells below each cell).
    (a refinement step gives a cell up to 4 neighbours above or below)"""
    pts, faces = mesh["points"], mesh["faces"]
    own, nei = mesh["owner"], mesh["neighbour"]
    nint = len(nei)
    ncell = len(cen)
    nv = np.fromiter((len(f) for f in faces[:nint]), dtype=int, count=nint)
    horiz = np.zeros(nint, dtype=bool)
    for L in np.unique(nv):                       # faces with the same number of vertices together
        ids = np.flatnonzero(nv == L)
        idx = np.array([faces[i] for i in ids]).reshape(len(ids), L)
        px, py, pz = pts[idx, 0], pts[idx, 1], pts[idx, 2]
        ext = np.maximum(np.ptp(px, axis=1), np.ptp(py, axis=1))
        horiz[ids] = np.ptp(pz, axis=1) < 0.05 * ext
    f = np.flatnonzero(horiz)
    a, b = own[:nint][f], nei[f]
    a_is_lower = cen[a, 2] < cen[b, 2]
    lower = np.where(a_is_lower, a, b)
    upper = np.where(a_is_lower, b, a)

    def csr(src, dst):
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        ptr = np.zeros(ncell + 1, dtype=np.int64)
        np.add.at(ptr, src + 1, 1)
        return np.cumsum(ptr), dst
    return csr(lower, upper) + csr(upper, lower)


def fit_column_values(h, pick, cen, links, fields, order=2, overshoot=0.5, nonneg=("q", "U")):
    """values of `fields` (dict name -> per-cell array) at height h above the cell centres of the sample columns.
    For every cell in `pick`: walk down (up to 2) and up (up to 2) through the vertical neighbours (the neighbour whose centre is
    nearest in x, y), take the order+1 cells whose centres are nearest to h and evaluate the Lagrange polynomial through their
    centre values at h.  The result is limited to [min - overshoot*spread, max + overshoot*spread] of the cell values used.
    Returns (dict of arrays, stats dict)."""
    up_ptr, up_idx, dn_ptr, dn_idx = links
    npnt = len(pick)
    out = {k: np.empty(npnt) for k in fields}
    used = np.zeros(npnt, dtype=int)
    limited = np.zeros(npnt, dtype=bool)
    z_first = np.full(npnt, np.nan)
    z_used = np.full(npnt, np.nan)

    def nxt(cur, ptr, idx, xy):
        c = idx[ptr[cur]:ptr[cur + 1]]
        if len(c) == 0:
            return None
        if len(c) == 1:
            return int(c[0])
        return int(c[np.argmin(np.hypot(cen[c, 0] - xy[0], cen[c, 1] - xy[1]))])

    for i, c0 in enumerate(pick):
        xy = cen[c0, :2]
        below, cur = [], int(c0)
        for _ in range(2):
            cur = nxt(cur, dn_ptr, dn_idx, xy)
            if cur is None:
                break
            below.append(cur)
        above, cur = [], int(c0)
        for _ in range(2):
            cur = nxt(cur, up_ptr, up_idx, xy)
            if cur is None:
                break
            above.append(cur)
        col = np.array(below[::-1] + [int(c0)] + above)
        zc = cen[col, 2]
        z_first[i] = zc[0] if not below else cen[below[-1], 2]
        m = min(order + 1, len(col))
        sel = np.sort(np.argsort(np.abs(zc - h), kind="stable")[:m])
        cs, zs = col[sel], zc[sel]
        if m >= 2 and np.min(np.diff(zs)) < 1e-9:                  # two cells with the same centre height: not a column
            m = 1
        used[i] = m - 1
        z_used[i] = float(np.mean(zs))
        for k, arr in fields.items():
            v = arr[cs]
            if m == 1:
                out[k][i] = arr[c0]
                continue
            r = 0.0
            for j in range(m):                                      # Lagrange form
                w = 1.0
                for l in range(m):
                    if l != j:
                        w *= (h - zs[l]) / (zs[j] - zs[l])
                r += w * v[j]
            lo, hi = float(v.min()), float(v.max())
            sp = hi - lo
            rl = min(max(r, lo - overshoot * sp), hi + overshoot * sp)
            if abs(rl - r) > 1e-9 * (abs(r) + 1.0):
                limited[i] = True
            if k in nonneg:
                rl = max(rl, 0.0)
            out[k][i] = rl
    stats = dict(points=int(npnt), quadratic=int((used == 2).sum()), linear=int((used == 1).sum()), cell_value=int((used == 0).sum()),
                 limited=int(limited.sum()), mean_distance_to_used_centres_m=float(np.nanmean(np.abs(z_used - h))),
                 mean_first_centre_z_m=float(np.nanmean(z_first)))
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=Path("."))
    ap.add_argument("--momentum", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None, help="layout json (site polygon); default: the only *.json in the momentum case")
    ap.add_argument("--time", default=None)
    ap.add_argument("--height", type=float, default=2.0, help="sample height above the ground [m] (default 2)")
    ap.add_argument("--interp", choices=["quadratic", "linear", "none"], default="quadratic",
                    help="T, q, |U| at --height from the cell-centre values of the vertical column: quadratic (3 cells, default), linear (2 cells) "
                         "or none (value of the cell that contains --height)")
    ap.add_argument("--fit-overshoot", type=float, default=0.5,
                    help="the fitted value may leave [min, max] of the cell values used by at most this share of their spread (default 0.5)")
    ap.add_argument("--points", type=int, default=1500, help="number of sample points (area-weighted)")
    ap.add_argument("--sky-rays", type=int, default=48)
    ap.add_argument("--tree-transmissivity", type=float, default=0.15)
    ap.add_argument("--albedo", type=float, default=0.20, help="ground albedo for reflected short wave (as select_epw_cases.py)")
    ap.add_argument("--abs-k", type=float, default=0.70)
    ap.add_argument("--emis-p", type=float, default=0.97)
    ap.add_argument("--emis-g", type=float, default=0.95)
    ap.add_argument("--wind-factor", type=float, default=1.0, help="multiplies the CFD wind before UTCI")
    ap.add_argument("--thermal-model", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default <case>/../UTCI_details.txt")
    a = ap.parse_args(argv)

    t0 = time.time()
    case = a.case.resolve()
    momentum = (a.momentum or case.parent / "momentum").resolve()
    tname = a.time or foam_io.latest_time(case)
    if tname is None or float(tname) <= 0.0:
        print("[utci] no result time directory (only 0/): airTqFoam wrote nothing, it crashed or diverged", file=sys.stderr)
        return 3
    tdir = case / tname
    tgt = json.loads((case / "weatherData/target.json").read_text())
    b = tgt["boundary"]
    pres = b["pressure_Pa"]

    # ---- mesh, fields, sample points
    mesh = foam_io.read_mesh(case, with_cells=True)
    cen, vol, z0, z1, Cf = cell_geometry(mesh)
    n = len(vol)
    T = foam_io.read_internal_field(tdir / "T", n) - 273.15
    q = foam_io.read_internal_field(tdir / "q", n)
    upath = next((p for p in (tdir / "U", case / "0" / "U") if p.is_file()), None)
    if upath is None:
        raise SystemExit("no U found")
    U = foam_io.read_internal_field(upath, n, rank="vector")
    spd = np.linalg.norm(U, axis=1)

    jpath = a.json
    if jpath is None:
        c = sorted(momentum.glob("*.json"))
        jpath = c[0] if len(c) == 1 else None
    poly = None
    if jpath is not None:
        d = json.loads(Path(jpath).read_text())
        co = (d.get("siteOnly_json", d).get("site") or {}).get("coordinates")
        if co and len(co) >= 3:
            poly = np.array([[p["x"], p["y"]] for p in co], dtype=float)
            if np.allclose(poly[0], poly[-1]):
                poly = poly[:-1]
    if poly is not None:
        inside = RT.point_in_polygon(cen[:, 0], cen[:, 1], poly)
        site_desc = f"site polygon of {Path(jpath).name}"
    else:
        roles0 = foam_io.classify_patches(mesh)
        names = roles0["by_role"].get("buildings", []) + roles0["by_role"].get("trees", [])
        ids = np.concatenate([foam_io.patch_faces(mesh, nm)[0] for nm in names])
        lo, hi = Cf[ids].min(0), Cf[ids].max(0)
        inside = (cen[:, 0] >= lo[0]) & (cen[:, 0] <= hi[0]) & (cen[:, 1] >= lo[1]) & (cen[:, 1] <= hi[1])
        site_desc = "bounding box of the building / tree patches"
    # one cell per vertical column: z0 <= h < z1 (a cell that ends exactly at h is not counted twice with the cell above it)
    sel = np.flatnonzero(inside & (z0 <= a.height) & (z1 > a.height))
    if not len(sel):
        raise SystemExit("no cells at the chosen height inside the site")
    w_cell = vol[sel] / np.maximum(z1[sel] - z0[sel], 1e-9)
    pick = sel[stratified_pick(w_cell, min(a.points, 20 * len(sel)))]
    P = np.column_stack([cen[pick, 0], cen[pick, 1], np.full(len(pick), a.height)])
    Ta, qa, ws = T[pick], q[pick], spd[pick]
    fit_stats = dict(method="cell value")
    if a.interp != "none":
        t_fit = time.time()
        links = horizontal_links(mesh, cen)
        fit, fit_stats = fit_column_values(a.height, pick, cen, links, {"T": T, "q": q, "U": spd}, order=2 if a.interp == "quadratic" else 1,
                                           overshoot=a.fit_overshoot)
        fit_stats["method"] = a.interp
        fit_stats["mean_change"] = dict(T_C=float(np.mean(fit["T"] - Ta)), q=float(np.mean(fit["q"] - qa)), U_ms=float(np.mean(fit["U"] - ws)))
        fit_stats["mean_abs_change"] = dict(T_C=float(np.mean(np.abs(fit["T"] - Ta))), q=float(np.mean(np.abs(fit["q"] - qa))),
                                            U_ms=float(np.mean(np.abs(fit["U"] - ws))))
        Ta, qa, ws = fit["T"], fit["q"], fit["U"]
        print(f"[utci] values at z = {a.height:g} m: {a.interp} fit through the cell centres of each column: "
              f"{fit_stats.get('quadratic', 0)} quadratic, {fit_stats.get('linear', 0)} linear, {fit_stats.get('cell_value', 0)} cell value, "
              f"{fit_stats['limited']} limited; mean |change| T {fit_stats['mean_abs_change']['T_C']:.2f} C, "
              f"|U| {fit_stats['mean_abs_change']['U_ms']:.2f} m/s; first centre at z = {fit_stats['mean_first_centre_z_m']:.2f} m "
              f"[{time.time() - t_fit:.0f} s]", file=sys.stderr)
    e = qa * pres / (0.622 + 0.378 * qa)
    RH = np.clip(100.0 * e / e_sat_pa(Ta), 1.0, 100.0)

    # ---- did airTqFoam diverge?  T must stay between the wall / inlet temperatures, q must stay physical.
    # (exit code 3 lets the caller retry with a more robust scheme instead of putting a garbage UTCI into the optimisation)
    roles = foam_io.classify_patches(mesh)
    Ttxt = (case / "0/T").read_text()
    wall_vals = [np.array([b["air_temperature_C"]])]
    for role in ("buildings", "trees", "ground", "lawn", "pond", "pathway"):
        for nm in roles["by_role"].get(role, []):
            try:
                wall_vals.append(patch_values(Ttxt, nm) - 273.15)
            except ValueError:
                pass
    wv = np.concatenate(wall_vals)
    margin = 3.0
    # T is bounded by the wall / inlet temperatures: anything outside means the solution blew up.  q is NOT held to that test over
    # the whole domain: next to an evaporating lawn / pond / tree face the imposed vapour flux can leave a few cells with q far from
    # physical (several % negative or above saturation) while the converged field at pedestrian height is fine.  What matters for
    # the UTCI is q at the sample points, so q is judged there (below), and only a blown-up field (non-finite, |q| > 1) stops here.
    if (not np.isfinite(T).all() or not np.isfinite(q).all() or T.min() < wv.min() - margin or T.max() > wv.max() + margin
            or np.abs(q).max() > 1.0):
        print(f"[utci] DIVERGED: T {np.nanmin(T):.1f} .. {np.nanmax(T):.1f} C (walls / inlet {wv.min():.1f} .. {wv.max():.1f} C), "
              f"q {np.nanmin(q):.4f} .. {np.nanmax(q):.4f} kg/kg", file=sys.stderr)
        return 3
    q_out = float(np.mean((q < -1e-3) | (q > 0.1)))
    if q_out > 0:
        print(f"[utci] note: q outside 0 .. 0.1 kg/kg in {100 * q_out:.2f} % of all cells (q {q.min():.4f} .. {q.max():.4f}); "
              "checking it at the sample points", file=sys.stderr)
    # q at the sample points: saturation limit of the local air temperature
    q_sat = 0.622 * e_sat_pa(Ta) / (pres - 0.378 * e_sat_pa(Ta))
    bad_pts = (qa < -1e-3) | (qa > 1.05 * q_sat)
    if bad_pts.mean() > 0.05:
        print(f"[utci] DIVERGED: q is unphysical at {100 * bad_pts.mean():.1f} % of the {len(qa)} sample points "
              f"(q {qa.min():.4f} .. {qa.max():.4f} kg/kg)", file=sys.stderr)
        return 3
    if bad_pts.any():
        print(f"[utci] note: q clipped to 0 .. saturation at {int(bad_pts.sum())} of {len(qa)} sample points", file=sys.stderr)
    qa = np.clip(qa, 0.0, q_sat)
    e = qa * pres / (0.622 + 0.378 * qa)
    RH = np.clip(100.0 * e / e_sat_pa(Ta), 1.0, 100.0)

    # ---- surface temperatures: ground below each point, mean building surface
    gx, gT = [], []
    for role in ("ground", "lawn", "pond", "pathway"):
        for nm in roles["by_role"].get(role, []):
            ids, fcs = foam_io.patch_faces(mesh, nm)
            C, _, _ = foam_io.face_geometry(mesh["points"], fcs)
            v = patch_values(Ttxt, nm) - 273.15 if True else None
            v = np.full(len(fcs), v[0]) if len(v) == 1 else v
            gx.append(C[:, :2])
            gT.append(v)
    from scipy.spatial import cKDTree
    gx, gT = np.vstack(gx), np.concatenate(gT)
    Tg = gT[cKDTree(gx).query(P[:, :2])[1]]
    S = np.load(case / "surfaces/thermal_surfaces.npz")
    if len(S["b_T"]):
        bt = S["b_tri"]
        barea = 0.5 * np.linalg.norm(np.cross(bt[:, 1] - bt[:, 0], bt[:, 2] - bt[:, 0]), axis=1)
        Tb = float(np.sum(S["b_T"] * barea) / barea.sum())
    else:
        Tb = float(b["air_temperature_C"])

    # ---- weather of the target hour
    tm_path = a.thermal_model or case.parent / "surface_temperature/solar_thermal_model.py"
    tm = RT.load_module(tm_path)
    if RT.api_kind(tm) is None:
        tm = RT.load_module(HERE / "bundled/solar_thermal_model.py")
    wdir = Path(tgt["weather_folder"])
    if not wdir.is_absolute():
        wdir = case / wdir
    wx = tm.load_weather_folder(str(wdir))
    sun = np.asarray(wx["sun_vec"][-1], float)
    sun /= np.linalg.norm(sun)
    alt = float(np.degrees(np.arcsin(np.clip(sun[2], -1, 1))))
    dni, dhi = float(wx["IDN_Wm2"][-1]), float(wx["Idif_Wm2"][-1])
    hir = float(b.get("horizontal_ir_Wm2") or 0.0)
    if hir <= 0.0:                                   # no sky infrared in the EPW: clear-sky estimate
        ta_amb = b["air_temperature_C"] + 273.15
        hir = 0.8 * SIGMA * ta_amb ** 4
    up = alt > 0.0
    dni, dhi = (dni, dhi) if up else (0.0, 0.0)

    # ---- occluders and rays
    def load_tris(name):
        f = case / "surfaces" / name
        if not f.is_file() or RT.obj_counts(f)[1] == 0:
            return None
        if hasattr(tm, "load_obj_checked"):
            m = tm.load_obj_checked(str(f), name)
            return None if m is None else m[0][m[1]]
        v, fcs = tm.load_obj_mesh(str(f))
        return np.asarray(v)[np.asarray(fcs, dtype=int)]

    btri, ttri = load_tris("buildings.obj"), load_tris("trees.obj")
    trB = tm.TriangleRayTracer(btri) if btri is not None else None
    trT = tm.TriangleRayTracer(ttri) if ttri is not None else None
    off = np.array([0.0, 0.0, 1e-3])

    def blocked(tracer, origins, d):
        return tracer.blocked(origins, d) if tracer is not None else np.zeros(len(origins), bool)

    # points inside a building (a vertical ray hits a roof) are dropped
    under_roof = blocked(trB, P + off, np.array([0.0, 0.0, 1.0]))
    keep = ~under_roof
    if keep.sum() < 10:
        raise SystemExit("fewer than 10 sample points outside buildings")
    P, Ta, qa, ws, RH, Tg = P[keep], Ta[keep], qa[keep], ws[keep], RH[keep], Tg[keep]

    tau = a.tree_transmissivity
    sunlit = np.zeros(len(P))
    if up:
        bb, bt_ = blocked(trB, P + off, sun), blocked(trT, P + off, sun)
        sunlit = np.where(bb, 0.0, np.where(bt_, tau, 1.0))
    dirs = hemisphere_rays(a.sky_rays)
    f_open = np.zeros(len(P))
    f_tree = np.zeros(len(P))
    f_bld = np.zeros(len(P))
    for d in dirs:
        bb, bt_ = blocked(trB, P + off, d), blocked(trT, P + off, d)
        f_bld += bb
        f_tree += (~bb) & bt_
        f_open += (~bb) & (~bt_)
    f_open, f_tree, f_bld = f_open / len(dirs), f_tree / len(dirs), f_bld / len(dirs)

    # ---- Tmrt (SolarCal form of select_epw_cases.py, local surroundings)
    ta_k = Ta + 273.15
    dhi_eff = dhi * (f_open + tau * f_tree)
    ghi_loc = dni * sunlit * max(sun[2], 0.0) + dhi_eff
    fp = 0.308 * np.cos(np.radians(alt * (0.998 - alt ** 2 / 50000.0))) if up else 0.0
    s_short = a.abs_k * (0.5 * dhi_eff + 0.5 * a.albedo * ghi_loc + fp * dni * sunlit)
    l_up = f_open * hir + f_tree * SIGMA * ta_k ** 4 + f_bld * 0.95 * SIGMA * (Tb + 273.15) ** 4
    l_dn = a.emis_g * SIGMA * (Tg + 273.15) ** 4
    s_long = a.emis_p * (0.5 * l_up + 0.5 * l_dn)
    tmrt = ((s_short + s_long) / (a.emis_p * SIGMA)) ** 0.25 - 273.15

    # ---- UTCI
    from pythermalcomfort.models import utci as _utci
    v = np.clip(ws * a.wind_factor, 0.5, 17.0)
    tr = np.clip(tmrt, Ta - 30.0, Ta + 70.0)
    res = _utci(tdb=Ta, tr=tr, v=v, rh=RH, units="SI", limit_inputs=False)
    u = np.asarray(getattr(res, "utci", res), dtype=float)
    ok = np.isfinite(u)
    if ok.sum() < 0.5 * len(u):
        raise SystemExit("UTCI could not be evaluated for most points")
    um = float(u[ok].mean())

    # rural reference (same formula, open field, air at the inlet condition)
    ta0 = b["air_temperature_C"]
    s0 = a.abs_k * (0.5 * dhi + 0.5 * a.albedo * (dni * max(sun[2], 0) + dhi) + fp * dni)
    l0 = a.emis_p * (0.5 * hir + 0.5 * a.emis_g * SIGMA * (ta0 + 273.15) ** 4)
    tm0 = ((s0 + l0) / (a.emis_p * SIGMA)) ** 0.25 - 273.15
    r0 = _utci(tdb=ta0, tr=tm0, v=float(np.clip(b["wind_speed_10m_ms"], 0.5, 17.0)), rh=b["relative_humidity_pct"],
               units="SI", limit_inputs=False)
    u0 = float(getattr(r0, "utci", r0))

    dt = tgt["target"]
    summary = dict(
        time=tname, site=site_desc, points=int(ok.sum()), height_m=a.height,
        utci_C=um, utci_min_C=float(u[ok].min()), utci_max_C=float(u[ok].max()), category=category(um),
        rural_utci_C=u0, rural_tmrt_C=float(tm0),
        Tair_C=float(Ta.mean()), RH_pct=float(RH.mean()), wind_ms=float(ws.mean()), Tmrt_C=float(tmrt[ok].mean()),
        sunlit_fraction=float(sunlit.mean()), sky_open=float(f_open.mean()), sky_tree=float(f_tree.mean()),
        sky_building=float(f_bld.mean()), ground_T_C=float(Tg.mean()), building_T_C=Tb,
        sun_altitude_deg=alt, dni=dni, dhi=dhi, target=dt, interpolation=fit_stats, seconds=round(time.time() - t0, 1))
    (case / "postProcessing").mkdir(exist_ok=True)
    (case / "postProcessing/utci.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(case / "postProcessing/utci_points.csv", np.column_stack([P[ok], Ta[ok], RH[ok], ws[ok], tmrt[ok], u[ok], sunlit[ok], f_open[ok]]),
               delimiter=",", header="x,y,z,Tair_C,RH_pct,wind_ms,Tmrt_C,UTCI_C,sunlit,sky_open", comments="")
    out = a.out or case.parent / "UTCI_details.txt"
    out.write_text(
        f"=== Processing target hour {dt['year']}-{dt['month']:02d}-{dt['day']:02d} hour_end {dt['hour_end']:02d} "
        f"(CFD time {tname}) ===\n"
        f"Sample points: {int(ok.sum())} at z={a.height} m inside {site_desc}\n"
        f"Mean: Tair={summary['Tair_C']:.2f} C  RH={summary['RH_pct']:.1f} %  wind={summary['wind_ms']:.2f} m/s  "
        f"Tmrt={summary['Tmrt_C']:.2f} C  (sunlit {100*summary['sunlit_fraction']:.0f} %, sky open {100*summary['sky_open']:.0f} %, "
        f"tree {100*summary['sky_tree']:.0f} %, building {100*summary['sky_building']:.0f} %)\n"
        f"Values at z={a.height:g} m: " + (f"{a.interp} fit in z through the cell centres of the column ({fit_stats.get('quadratic', 0)} quadratic, "
                                          f"{fit_stats.get('linear', 0)} linear, {fit_stats.get('cell_value', 0)} cell value)\n" if a.interp != "none" else "cell value\n") +
        f"Rural reference: UTCI={u0:.2f} C (Tmrt {tm0:.2f} C)\n"
        f"Site area-averaged: UTCI={um:.2f} C ({category(um)}), Delta UTCI vs rural {um - u0:+.2f} C, "
        f"range {summary['utci_min_C']:.1f} .. {summary['utci_max_C']:.1f} C\n")
    print(f"[utci] {int(ok.sum())} points: Ta {Ta.mean():.1f} C  RH {RH.mean():.0f} %  |U| {ws.mean():.2f} m/s  Tmrt {tmrt[ok].mean():.1f} C "
          f"-> UTCI {um:.2f} C ({category(um)}), rural {u0:.2f} C   [{time.time()-t0:.0f} s] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# #!/usr/bin/env python3
# """
# compute_utci.py  --  site area-averaged UTCI at pedestrian height from an airTqFoam run

# Inputs (all inside the case, produced by the earlier steps):
#     latest time        T, q, U                 air temperature, specific humidity, wind (frozen momentum field)
#     0/T                wall temperatures       ground / lawn / pond patch faces  -> long-wave from the ground
#     surfaces/          buildings.obj trees.obj the occluders the thermal model used (mesh frame)
#                        thermal_surfaces.npz    building surface temperatures
#     weatherData/       target.json + the weather folder (sun vector, DNI, DHI, sky infrared)
#     layout JSON        site polygon  (site.coordinates, mesh frame)

# Method (the SolarCal formula of select_epw_cases.py, evaluated per point with the local surroundings):
#     sample points      cells of the CFD mesh that contain z = 2 m inside the site, area-weighted (stratified sampling)
#     sun                direct beam through the building + tree BVH (thermal model's TriangleRayTracer);
#                        a tree crown transmits --tree-transmissivity (default 0.15) of the beam and of the diffuse sky
#     sky                cosine-weighted hemisphere rays (--sky-rays) -> open / tree-blocked / building-blocked fractions
#     short wave         S = a_k * ( 0.5*DHI_eff + 0.5*albedo*GHI_local + fp*DNI*sunlit )
#     long wave          L = eps_p * ( 0.5*(f_open*IR_sky + f_tree*sigma*Ta^4 + f_bld*sigma*Tb^4) + 0.5*eps_g*sigma*Tg^4 )
#                        Tg = the mapped ground / lawn / pond surface temperature below the point, Tb = mean building surface T
#     Tmrt               ((S + L) / (eps_p sigma))^(1/4)
#     UTCI               pythermalcomfort.models.utci (same call as select_epw_cases.py) with Ta, RH, |U| of the CFD cell
# With nothing around (f_open = 1, Tg = Ta) this is exactly the rural Tmrt of select_epw_cases.py, so the layout
# objective measures only what the layout changes: shade, sky view, surface temperature, wind, air T and humidity.

# Wind: |U| of the CFD cell is used as it is (--wind-factor 1).  The momentum case has a uniform inlet equal to the
# 10 m wind of the EPW, so the open-field wind at 2 m already is about that value.

# Writes  <out> (default <case>/../UTCI_details.txt, the file run_candidates.py parses):
#         === Processing target hour ... ===
#         Site area-averaged: UTCI=xx.xx C (category), ...
# and <case>/postProcessing/utci.json, <case>/postProcessing/utci_points.csv
# """
# import argparse
# import json
# import sys
# import time
# from pathlib import Path

# import numpy as np

# HERE = Path(__file__).resolve().parent
# sys.path.insert(0, str(HERE))
# import foam_io  # noqa: E402
# import run_thermal as RT  # noqa: E402
# from site_average import cell_geometry, e_sat_pa  # noqa: E402

# SIGMA = 5.670374419e-8

# CATEGORIES = [(46.0, "extreme heat stress"), (38.0, "very strong heat stress"), (32.0, "strong heat stress"),
#               (26.0, "moderate heat stress"), (9.0, "no thermal stress"), (0.0, "slight cold stress"),
#               (-13.0, "moderate cold stress"), (-27.0, "strong cold stress"), (-40.0, "very strong cold stress")]


# def category(v):
#     for lo, name in CATEGORIES:
#         if v > lo:
#             return name
#     return "extreme cold stress"


# def patch_values(text, name):
#     import re
#     m = re.search(r"\b" + re.escape(name) + r"\s*\{[^}]*?value\s+(?:nonuniform\s+List<scalar>\s*(\d+)\s*\(([^)]*)\)|uniform\s+(\S+);)",
#                   text, re.S)
#     if m is None:
#         raise ValueError(f"patch {name} not found in 0/T")
#     if m.group(2) is not None:
#         return np.array(m.group(2).split(), dtype=float)
#     return np.array([float(m.group(3))])


# def hemisphere_rays(n):
#     """n directions on the upper hemisphere, cosine-weighted (equal weights = cosine-weighted sampling):
#     Fibonacci spiral in (sin^2 theta, phi)."""
#     k = (np.arange(n) + 0.5) / n
#     sin_t = np.sqrt(k)                       # sin^2(theta) uniform -> cosine-weighted sampling
#     cos_t = np.sqrt(1.0 - k)
#     phi = np.arange(n) * np.pi * (3.0 - np.sqrt(5.0))
#     return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


# def stratified_pick(w, n):
#     """Indices chosen with probability proportional to w by systematic sampling (deterministic)."""
#     c = np.cumsum(w)
#     c /= c[-1]
#     u = (np.arange(n) + 0.5) / n
#     return np.searchsorted(c, u)


# def main(argv=None):
#     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--case", type=Path, default=Path("."))
#     ap.add_argument("--momentum", type=Path, default=None)
#     ap.add_argument("--json", type=Path, default=None, help="layout json (site polygon); default: the only *.json in the momentum case")
#     ap.add_argument("--time", default=None)
#     ap.add_argument("--height", type=float, default=2.0, help="sample height above the ground [m] (default 2)")
#     ap.add_argument("--points", type=int, default=1500, help="number of sample points (area-weighted)")
#     ap.add_argument("--sky-rays", type=int, default=48)
#     ap.add_argument("--tree-transmissivity", type=float, default=0.15)
#     ap.add_argument("--albedo", type=float, default=0.20, help="ground albedo for reflected short wave (as select_epw_cases.py)")
#     ap.add_argument("--abs-k", type=float, default=0.70)
#     ap.add_argument("--emis-p", type=float, default=0.97)
#     ap.add_argument("--emis-g", type=float, default=0.95)
#     ap.add_argument("--wind-factor", type=float, default=1.0, help="multiplies the CFD wind before UTCI")
#     ap.add_argument("--thermal-model", type=Path, default=None)
#     ap.add_argument("--out", type=Path, default=None, help="default <case>/../UTCI_details.txt")
#     a = ap.parse_args(argv)

#     t0 = time.time()
#     case = a.case.resolve()
#     momentum = (a.momentum or case.parent / "momentum").resolve()
#     tname = a.time or foam_io.latest_time(case)
#     if tname is None or float(tname) <= 0.0:
#         print("[utci] no result time directory (only 0/): airTqFoam wrote nothing, it crashed or diverged", file=sys.stderr)
#         return 3
#     tdir = case / tname
#     tgt = json.loads((case / "weatherData/target.json").read_text())
#     b = tgt["boundary"]
#     pres = b["pressure_Pa"]

#     # ---- mesh, fields, sample points
#     mesh = foam_io.read_mesh(case, with_cells=True)
#     cen, vol, z0, z1, Cf = cell_geometry(mesh)
#     n = len(vol)
#     T = foam_io.read_internal_field(tdir / "T", n) - 273.15
#     q = foam_io.read_internal_field(tdir / "q", n)
#     upath = next((p for p in (tdir / "U", case / "0" / "U") if p.is_file()), None)
#     if upath is None:
#         raise SystemExit("no U found")
#     U = foam_io.read_internal_field(upath, n, rank="vector")
#     spd = np.linalg.norm(U, axis=1)

#     jpath = a.json
#     if jpath is None:
#         c = sorted(momentum.glob("*.json"))
#         jpath = c[0] if len(c) == 1 else None
#     poly = None
#     if jpath is not None:
#         d = json.loads(Path(jpath).read_text())
#         co = (d.get("siteOnly_json", d).get("site") or {}).get("coordinates")
#         if co and len(co) >= 3:
#             poly = np.array([[p["x"], p["y"]] for p in co], dtype=float)
#             if np.allclose(poly[0], poly[-1]):
#                 poly = poly[:-1]
#     if poly is not None:
#         inside = RT.point_in_polygon(cen[:, 0], cen[:, 1], poly)
#         site_desc = f"site polygon of {Path(jpath).name}"
#     else:
#         roles0 = foam_io.classify_patches(mesh)
#         names = roles0["by_role"].get("buildings", []) + roles0["by_role"].get("trees", [])
#         ids = np.concatenate([foam_io.patch_faces(mesh, nm)[0] for nm in names])
#         lo, hi = Cf[ids].min(0), Cf[ids].max(0)
#         inside = (cen[:, 0] >= lo[0]) & (cen[:, 0] <= hi[0]) & (cen[:, 1] >= lo[1]) & (cen[:, 1] <= hi[1])
#         site_desc = "bounding box of the building / tree patches"
#     # one cell per vertical column: z0 <= h < z1 (a cell that ends exactly at h is not counted twice with the cell above it)
#     sel = np.flatnonzero(inside & (z0 <= a.height) & (z1 > a.height))
#     if not len(sel):
#         raise SystemExit("no cells at the chosen height inside the site")
#     w_cell = vol[sel] / np.maximum(z1[sel] - z0[sel], 1e-9)
#     pick = sel[stratified_pick(w_cell, min(a.points, 20 * len(sel)))]
#     P = np.column_stack([cen[pick, 0], cen[pick, 1], np.full(len(pick), a.height)])
#     Ta, qa, ws = T[pick], q[pick], spd[pick]
#     e = qa * pres / (0.622 + 0.378 * qa)
#     RH = np.clip(100.0 * e / e_sat_pa(Ta), 1.0, 100.0)

#     # ---- did airTqFoam diverge?  T must stay between the wall / inlet temperatures, q must stay physical.
#     # (exit code 3 lets the caller retry with a more robust scheme instead of putting a garbage UTCI into the optimisation)
#     roles = foam_io.classify_patches(mesh)
#     Ttxt = (case / "0/T").read_text()
#     wall_vals = [np.array([b["air_temperature_C"]])]
#     for role in ("buildings", "trees", "ground", "lawn", "pond", "pathway"):
#         for nm in roles["by_role"].get(role, []):
#             try:
#                 wall_vals.append(patch_values(Ttxt, nm) - 273.15)
#             except ValueError:
#                 pass
#     wv = np.concatenate(wall_vals)
#     margin = 3.0
#     # T is bounded by the wall / inlet temperatures: anything outside means the solution blew up.  q is NOT held to that test over
#     # the whole domain: next to an evaporating lawn / pond / tree face the imposed vapour flux can leave a few cells with q far from
#     # physical (several % negative or above saturation) while the converged field at pedestrian height is fine.  What matters for
#     # the UTCI is q at the sample points, so q is judged there (below), and only a blown-up field (non-finite, |q| > 1) stops here.
#     if (not np.isfinite(T).all() or not np.isfinite(q).all() or T.min() < wv.min() - margin or T.max() > wv.max() + margin
#             or np.abs(q).max() > 1.0):
#         print(f"[utci] DIVERGED: T {np.nanmin(T):.1f} .. {np.nanmax(T):.1f} C (walls / inlet {wv.min():.1f} .. {wv.max():.1f} C), "
#               f"q {np.nanmin(q):.4f} .. {np.nanmax(q):.4f} kg/kg", file=sys.stderr)
#         return 3
#     q_out = float(np.mean((q < -1e-3) | (q > 0.1)))
#     if q_out > 0:
#         print(f"[utci] note: q outside 0 .. 0.1 kg/kg in {100 * q_out:.2f} % of all cells (q {q.min():.4f} .. {q.max():.4f}); "
#               "checking it at the sample points", file=sys.stderr)
#     # q at the sample points: saturation limit of the local air temperature
#     q_sat = 0.622 * e_sat_pa(Ta) / (pres - 0.378 * e_sat_pa(Ta))
#     bad_pts = (qa < -1e-3) | (qa > 1.05 * q_sat)
#     if bad_pts.mean() > 0.05:
#         print(f"[utci] DIVERGED: q is unphysical at {100 * bad_pts.mean():.1f} % of the {len(qa)} sample points "
#               f"(q {qa.min():.4f} .. {qa.max():.4f} kg/kg)", file=sys.stderr)
#         return 3
#     if bad_pts.any():
#         print(f"[utci] note: q clipped to 0 .. saturation at {int(bad_pts.sum())} of {len(qa)} sample points", file=sys.stderr)
#     qa = np.clip(qa, 0.0, q_sat)
#     e = qa * pres / (0.622 + 0.378 * qa)
#     RH = np.clip(100.0 * e / e_sat_pa(Ta), 1.0, 100.0)

#     # ---- surface temperatures: ground below each point, mean building surface
#     gx, gT = [], []
#     for role in ("ground", "lawn", "pond", "pathway"):
#         for nm in roles["by_role"].get(role, []):
#             ids, fcs = foam_io.patch_faces(mesh, nm)
#             C, _, _ = foam_io.face_geometry(mesh["points"], fcs)
#             v = patch_values(Ttxt, nm) - 273.15 if True else None
#             v = np.full(len(fcs), v[0]) if len(v) == 1 else v
#             gx.append(C[:, :2])
#             gT.append(v)
#     from scipy.spatial import cKDTree
#     gx, gT = np.vstack(gx), np.concatenate(gT)
#     Tg = gT[cKDTree(gx).query(P[:, :2])[1]]
#     S = np.load(case / "surfaces/thermal_surfaces.npz")
#     if len(S["b_T"]):
#         bt = S["b_tri"]
#         barea = 0.5 * np.linalg.norm(np.cross(bt[:, 1] - bt[:, 0], bt[:, 2] - bt[:, 0]), axis=1)
#         Tb = float(np.sum(S["b_T"] * barea) / barea.sum())
#     else:
#         Tb = float(b["air_temperature_C"])

#     # ---- weather of the target hour
#     tm_path = a.thermal_model or case.parent / "surface_temperature/solar_thermal_model.py"
#     tm = RT.load_module(tm_path)
#     if RT.api_kind(tm) is None:
#         tm = RT.load_module(HERE / "bundled/solar_thermal_model.py")
#     wdir = Path(tgt["weather_folder"])
#     if not wdir.is_absolute():
#         wdir = case / wdir
#     wx = tm.load_weather_folder(str(wdir))
#     sun = np.asarray(wx["sun_vec"][-1], float)
#     sun /= np.linalg.norm(sun)
#     alt = float(np.degrees(np.arcsin(np.clip(sun[2], -1, 1))))
#     dni, dhi = float(wx["IDN_Wm2"][-1]), float(wx["Idif_Wm2"][-1])
#     hir = float(b.get("horizontal_ir_Wm2") or 0.0)
#     if hir <= 0.0:                                   # no sky infrared in the EPW: clear-sky estimate
#         ta_amb = b["air_temperature_C"] + 273.15
#         hir = 0.8 * SIGMA * ta_amb ** 4
#     up = alt > 0.0
#     dni, dhi = (dni, dhi) if up else (0.0, 0.0)

#     # ---- occluders and rays
#     def load_tris(name):
#         f = case / "surfaces" / name
#         if not f.is_file() or RT.obj_counts(f)[1] == 0:
#             return None
#         if hasattr(tm, "load_obj_checked"):
#             m = tm.load_obj_checked(str(f), name)
#             return None if m is None else m[0][m[1]]
#         v, fcs = tm.load_obj_mesh(str(f))
#         return np.asarray(v)[np.asarray(fcs, dtype=int)]

#     btri, ttri = load_tris("buildings.obj"), load_tris("trees.obj")
#     trB = tm.TriangleRayTracer(btri) if btri is not None else None
#     trT = tm.TriangleRayTracer(ttri) if ttri is not None else None
#     off = np.array([0.0, 0.0, 1e-3])

#     def blocked(tracer, origins, d):
#         return tracer.blocked(origins, d) if tracer is not None else np.zeros(len(origins), bool)

#     # points inside a building (a vertical ray hits a roof) are dropped
#     under_roof = blocked(trB, P + off, np.array([0.0, 0.0, 1.0]))
#     keep = ~under_roof
#     if keep.sum() < 10:
#         raise SystemExit("fewer than 10 sample points outside buildings")
#     P, Ta, qa, ws, RH, Tg = P[keep], Ta[keep], qa[keep], ws[keep], RH[keep], Tg[keep]

#     tau = a.tree_transmissivity
#     sunlit = np.zeros(len(P))
#     if up:
#         bb, bt_ = blocked(trB, P + off, sun), blocked(trT, P + off, sun)
#         sunlit = np.where(bb, 0.0, np.where(bt_, tau, 1.0))
#     dirs = hemisphere_rays(a.sky_rays)
#     f_open = np.zeros(len(P))
#     f_tree = np.zeros(len(P))
#     f_bld = np.zeros(len(P))
#     for d in dirs:
#         bb, bt_ = blocked(trB, P + off, d), blocked(trT, P + off, d)
#         f_bld += bb
#         f_tree += (~bb) & bt_
#         f_open += (~bb) & (~bt_)
#     f_open, f_tree, f_bld = f_open / len(dirs), f_tree / len(dirs), f_bld / len(dirs)

#     # ---- Tmrt (SolarCal form of select_epw_cases.py, local surroundings)
#     ta_k = Ta + 273.15
#     dhi_eff = dhi * (f_open + tau * f_tree)
#     ghi_loc = dni * sunlit * max(sun[2], 0.0) + dhi_eff
#     fp = 0.308 * np.cos(np.radians(alt * (0.998 - alt ** 2 / 50000.0))) if up else 0.0
#     s_short = a.abs_k * (0.5 * dhi_eff + 0.5 * a.albedo * ghi_loc + fp * dni * sunlit)
#     l_up = f_open * hir + f_tree * SIGMA * ta_k ** 4 + f_bld * 0.95 * SIGMA * (Tb + 273.15) ** 4
#     l_dn = a.emis_g * SIGMA * (Tg + 273.15) ** 4
#     s_long = a.emis_p * (0.5 * l_up + 0.5 * l_dn)
#     tmrt = ((s_short + s_long) / (a.emis_p * SIGMA)) ** 0.25 - 273.15

#     # ---- UTCI
#     from pythermalcomfort.models import utci as _utci
#     v = np.clip(ws * a.wind_factor, 0.5, 17.0)
#     tr = np.clip(tmrt, Ta - 30.0, Ta + 70.0)
#     res = _utci(tdb=Ta, tr=tr, v=v, rh=RH, units="SI", limit_inputs=False)
#     u = np.asarray(getattr(res, "utci", res), dtype=float)
#     ok = np.isfinite(u)
#     if ok.sum() < 0.5 * len(u):
#         raise SystemExit("UTCI could not be evaluated for most points")
#     um = float(u[ok].mean())

#     # rural reference (same formula, open field, air at the inlet condition)
#     ta0 = b["air_temperature_C"]
#     s0 = a.abs_k * (0.5 * dhi + 0.5 * a.albedo * (dni * max(sun[2], 0) + dhi) + fp * dni)
#     l0 = a.emis_p * (0.5 * hir + 0.5 * a.emis_g * SIGMA * (ta0 + 273.15) ** 4)
#     tm0 = ((s0 + l0) / (a.emis_p * SIGMA)) ** 0.25 - 273.15
#     r0 = _utci(tdb=ta0, tr=tm0, v=float(np.clip(b["wind_speed_10m_ms"], 0.5, 17.0)), rh=b["relative_humidity_pct"],
#                units="SI", limit_inputs=False)
#     u0 = float(getattr(r0, "utci", r0))

#     dt = tgt["target"]
#     summary = dict(
#         time=tname, site=site_desc, points=int(ok.sum()), height_m=a.height,
#         utci_C=um, utci_min_C=float(u[ok].min()), utci_max_C=float(u[ok].max()), category=category(um),
#         rural_utci_C=u0, rural_tmrt_C=float(tm0),
#         Tair_C=float(Ta.mean()), RH_pct=float(RH.mean()), wind_ms=float(ws.mean()), Tmrt_C=float(tmrt[ok].mean()),
#         sunlit_fraction=float(sunlit.mean()), sky_open=float(f_open.mean()), sky_tree=float(f_tree.mean()),
#         sky_building=float(f_bld.mean()), ground_T_C=float(Tg.mean()), building_T_C=Tb,
#         sun_altitude_deg=alt, dni=dni, dhi=dhi, target=dt, seconds=round(time.time() - t0, 1))
#     (case / "postProcessing").mkdir(exist_ok=True)
#     (case / "postProcessing/utci.json").write_text(json.dumps(summary, indent=2))
#     np.savetxt(case / "postProcessing/utci_points.csv", np.column_stack([P[ok], Ta[ok], RH[ok], ws[ok], tmrt[ok], u[ok], sunlit[ok], f_open[ok]]),
#                delimiter=",", header="x,y,z,Tair_C,RH_pct,wind_ms,Tmrt_C,UTCI_C,sunlit,sky_open", comments="")
#     out = a.out or case.parent / "UTCI_details.txt"
#     out.write_text(
#         f"=== Processing target hour {dt['year']}-{dt['month']:02d}-{dt['day']:02d} hour_end {dt['hour_end']:02d} "
#         f"(CFD time {tname}) ===\n"
#         f"Sample points: {int(ok.sum())} at z={a.height} m inside {site_desc}\n"
#         f"Mean: Tair={summary['Tair_C']:.2f} C  RH={summary['RH_pct']:.1f} %  wind={summary['wind_ms']:.2f} m/s  "
#         f"Tmrt={summary['Tmrt_C']:.2f} C  (sunlit {100*summary['sunlit_fraction']:.0f} %, sky open {100*summary['sky_open']:.0f} %, "
#         f"tree {100*summary['sky_tree']:.0f} %, building {100*summary['sky_building']:.0f} %)\n"
#         f"Rural reference: UTCI={u0:.2f} C (Tmrt {tm0:.2f} C)\n"
#         f"Site area-averaged: UTCI={um:.2f} C ({category(um)}), Delta UTCI vs rural {um - u0:+.2f} C, "
#         f"range {summary['utci_min_C']:.1f} .. {summary['utci_max_C']:.1f} C\n")
#     print(f"[utci] {int(ok.sum())} points: Ta {Ta.mean():.1f} C  RH {RH.mean():.0f} %  |U| {ws.mean():.2f} m/s  Tmrt {tmrt[ok].mean():.1f} C "
#           f"-> UTCI {um:.2f} C ({category(um)}), rural {u0:.2f} C   [{time.time()-t0:.0f} s] -> {out}")
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())
