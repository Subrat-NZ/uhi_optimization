#!/usr/bin/env python3
"""
ai_utci.py  --  site area-averaged UTCI from the AI fields  (the AI twin of temperature_humidity/tools/compute_utci.py)

Same method, same radiation / UTCI code path as compute_utci.py (SolarCal Tmrt from sun, sky view, surface temperatures; pythermalcomfort UTCI), the
same outputs in the same places:  UTCI_details.txt  (line "Site area-averaged: UTCI=xx.xx C ..."),
temperature_humidity/postProcessing/{utci.json, utci_points.csv}  and the images.   Only where Ta, RH and |U| come from differs:

    OpenFOAM case   cells of the CFD mesh at z = 2 m  (quadratic fit through the column of cells)
    AI case         pixels of the AI rasters inside the site polygon: T from the T model, RH from the RH model, |U| from the wind model
                    (the rasters are the models' pedestrian-level layer, z-index 1 of the wind grid)

Everything else is read from the thermal raw stage of the OpenFOAM case (tools/run_thermal.py --stage raw, no mesh needed):
    surfaces/thermal_raw.npz        building surface T, ground triangles + their material-class T (ground below each point)
    surfaces/buildings.obj trees.obj  the occluders for sun and sky rays
Because the AI rasters are what the models predict, the objective is a surrogate: confirm the best layouts with the OpenFOAM case
(nsga2.py --verify-case-file).

Exit code 3 = the AI fields are not usable (mostly NaN / unphysical).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "temperature_humidity" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(HERE))
import compute_utci as CU  # noqa: E402  (category, hemisphere_rays, stratified_pick)
import run_thermal as RT  # noqa: E402
from site_average import e_sat_pa  # noqa: E402

SIGMA = 5.670374419e-8
CONFIG = json.loads((HERE / "config.json").read_text()) if (HERE / "config.json").is_file() else {}


def site_polygon(jpath):
    d = json.loads(Path(jpath).read_text())
    co = (d.get("siteOnly_json", d).get("site") or {}).get("coordinates")
    if not co or len(co) < 3:
        return None
    poly = np.array([[p["x"], p["y"]] for p in co], dtype=float)
    return poly[:-1] if np.allclose(poly[0], poly[-1]) else poly


def pixel_xy(shape, ext):
    """x, y of every pixel (rasterize_obj.py: col 0 = x_min, last col = x_max; row 0 = y_max, last row = y_min)."""
    H, W = shape
    x = ext[0] + np.arange(W) / (W - 1) * (ext[1] - ext[0])
    y = ext[3] - np.arange(H) / (H - 1) * (ext[3] - ext[2])
    return np.meshgrid(x, y)


def fill_nan(a, max_px=3):
    """NaN pixels next to valid ones take the nearest valid value (within max_px); the rest stay NaN."""
    from scipy import ndimage
    bad = ~np.isfinite(a)
    if not bad.any() or bad.all():
        return a
    d, idx = ndimage.distance_transform_edt(bad, return_indices=True)
    out = a.copy()
    near = bad & (d <= max_px)
    out[near] = a[tuple(i[near] for i in idx)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=Path("."), help="the job folder (run/NAME/<season>)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--points", type=int, default=int(CONFIG.get("points", 1500)))
    ap.add_argument("--sky-rays", type=int, default=int(CONFIG.get("sky_rays", 48)))
    ap.add_argument("--tree-transmissivity", type=float, default=0.15)
    ap.add_argument("--albedo", type=float, default=0.20)
    ap.add_argument("--abs-k", type=float, default=0.70)
    ap.add_argument("--emis-p", type=float, default=0.97)
    ap.add_argument("--emis-g", type=float, default=0.95)
    ap.add_argument("--wind-factor", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None, help="default <case>/UTCI_details.txt")
    a = ap.parse_args(argv)
    t0 = time.time()
    case = a.case.resolve()
    th = case / "temperature_humidity"
    jpath = a.json or (sorted(case.glob("*.json")) or [None])[0]
    tgt = json.loads((th / "weatherData" / "target.json").read_text())
    b, dt = tgt["boundary"], tgt["target"]

    F = np.load(case / "ai_work" / "fields.npz")
    T_K, RH_pct, U = F["T_K"], F["RH_pct"], F["U_ms"]
    ext = [float(v) for v in F["extent"]]
    poly = site_polygon(jpath) if jpath else None
    X, Y = pixel_xy(T_K.shape, ext)
    if poly is None:
        raise SystemExit("no site polygon in the layout JSON")
    inside = RT.point_in_polygon(X.ravel(), Y.ravel(), poly).reshape(X.shape)
    if not inside.any():
        raise SystemExit("the site polygon covers no pixel of the AI rasters (is the site outside the wind model domain?)")
    # how much of the site lies inside the wind model's domain (bounding box of the buildings + 10 %)? the rest is not evaluated
    pix_area = (ext[1] - ext[0]) / (T_K.shape[1] - 1) * (ext[3] - ext[2]) / (T_K.shape[0] - 1)
    poly_area = 0.5 * abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])))
    coverage = float(min(1.0, inside.sum() * pix_area / max(poly_area, 1e-9)))
    if coverage < 0.9:
        print(f"[utci] WARNING: only {100 * coverage:.0f} % of the site lies inside the wind model domain (the rest is not evaluated)", file=sys.stderr)
    T_K, RH_pct, U = fill_nan(T_K), fill_nan(RH_pct), fill_nan(U)
    valid = inside & np.isfinite(T_K) & np.isfinite(RH_pct) & np.isfinite(U)
    if valid.sum() < 0.3 * inside.sum():
        print(f"[utci] AI fields are valid at only {100 * valid.sum() / inside.sum():.0f} % of the site pixels", file=sys.stderr)
        return 3
    idx = np.flatnonzero(valid.ravel())
    pick = idx[CU.stratified_pick(np.ones(len(idx)), min(a.points, len(idx)))] if len(idx) > a.points else idx
    P = np.column_stack([X.ravel()[pick], Y.ravel()[pick], np.full(len(pick), 2.0)])
    Ta = T_K.ravel()[pick] - 273.15
    RH = np.clip(RH_pct.ravel()[pick], 1.0, 100.0)
    ws = U.ravel()[pick]
    if not (np.isfinite(Ta).all() and Ta.min() > -60 and Ta.max() < 70):
        print(f"[utci] AI temperature is unphysical: {np.nanmin(Ta):.1f} .. {np.nanmax(Ta):.1f} C", file=sys.stderr)
        return 3

    # ---- surface temperatures: ground below each point, mean building surface (thermal raw stage)
    from scipy.spatial import cKDTree
    Z = np.load(th / "surfaces" / "thermal_raw.npz")
    classes = [str(c) for c in Z["g_class_names"]]
    gtri, gcls = Z["gr_tri"], Z["gr_cls"]
    gT = np.full(len(gtri), np.nan)
    for ci, cn in enumerate(classes):
        if f"g_T_{cn}" in Z.files:
            m = gcls == ci
            gT[m] = Z[f"g_T_{cn}"][m]
    cen = gtri.mean(axis=1)[:, :2]
    ok_g = np.isfinite(gT)
    if not ok_g.any():
        raise SystemExit("thermal_raw.npz has no ground temperature")
    Tg = gT[ok_g][cKDTree(cen[ok_g]).query(P[:, :2])[1]]
    if len(Z["b_T"]):
        bt = Z["b_tri"]
        barea = 0.5 * np.linalg.norm(np.cross(bt[:, 1] - bt[:, 0], bt[:, 2] - bt[:, 0]), axis=1)
        Tb = float(np.sum(Z["b_T"] * barea) / barea.sum())
    else:
        Tb = float(b["air_temperature_C"])

    # ---- weather of the target hour (sun, DNI, DHI, sky infrared) -- identical to compute_utci.py
    tm_path = th.parent / "surface_temperature" / "solar_thermal_model.py"
    tm = RT.load_module(tm_path)
    if RT.api_kind(tm) is None:
        tm = RT.load_module(TOOLS / "bundled" / "solar_thermal_model.py")
    wdir = Path(tgt["weather_folder"])
    if not wdir.is_absolute():
        wdir = th / wdir
    if not wdir.is_dir():
        wdir = th / "weatherData" / wdir.name
    wx = tm.load_weather_folder(str(wdir))
    sun = np.asarray(wx["sun_vec"][-1], float)
    sun /= np.linalg.norm(sun)
    alt = float(np.degrees(np.arcsin(np.clip(sun[2], -1, 1))))
    dni, dhi = float(wx["IDN_Wm2"][-1]), float(wx["Idif_Wm2"][-1])
    hir = float(b.get("horizontal_ir_Wm2") or 0.0)
    if hir <= 0.0:
        hir = 0.8 * SIGMA * (b["air_temperature_C"] + 273.15) ** 4
    up = alt > 0.0
    dni, dhi = (dni, dhi) if up else (0.0, 0.0)

    def load_tris(name):
        f = th / "surfaces" / name
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

    keep = ~blocked(trB, P + off, np.array([0.0, 0.0, 1.0]))          # points under a roof are dropped
    if keep.sum() < 10:
        raise SystemExit("fewer than 10 sample points outside buildings")
    P, Ta, ws, RH, Tg = P[keep], Ta[keep], ws[keep], RH[keep], Tg[keep]
    tau = a.tree_transmissivity
    sunlit = np.zeros(len(P))
    if up:
        bb, bt_ = blocked(trB, P + off, sun), blocked(trT, P + off, sun)
        sunlit = np.where(bb, 0.0, np.where(bt_, tau, 1.0))
    dirs = CU.hemisphere_rays(a.sky_rays)
    f_open, f_tree, f_bld = np.zeros(len(P)), np.zeros(len(P)), np.zeros(len(P))
    for d in dirs:
        bb, bt_ = blocked(trB, P + off, d), blocked(trT, P + off, d)
        f_bld += bb
        f_tree += (~bb) & bt_
        f_open += (~bb) & (~bt_)
    f_open, f_tree, f_bld = f_open / len(dirs), f_tree / len(dirs), f_bld / len(dirs)

    # ---- Tmrt (SolarCal, local surroundings) and UTCI -- identical to compute_utci.py
    ta_k = Ta + 273.15
    dhi_eff = dhi * (f_open + tau * f_tree)
    ghi_loc = dni * sunlit * max(sun[2], 0.0) + dhi_eff
    fp = 0.308 * np.cos(np.radians(alt * (0.998 - alt ** 2 / 50000.0))) if up else 0.0
    s_short = a.abs_k * (0.5 * dhi_eff + 0.5 * a.albedo * ghi_loc + fp * dni * sunlit)
    l_up = f_open * hir + f_tree * SIGMA * ta_k ** 4 + f_bld * 0.95 * SIGMA * (Tb + 273.15) ** 4
    l_dn = a.emis_g * SIGMA * (Tg + 273.15) ** 4
    s_long = a.emis_p * (0.5 * l_up + 0.5 * l_dn)
    tmrt = ((s_short + s_long) / (a.emis_p * SIGMA)) ** 0.25 - 273.15
    from pythermalcomfort.models import utci as _utci
    v = np.clip(ws * a.wind_factor, 0.5, 17.0)
    tr = np.clip(tmrt, Ta - 30.0, Ta + 70.0)
    res = _utci(tdb=Ta, tr=tr, v=v, rh=RH, units="SI", limit_inputs=False)
    u = np.asarray(getattr(res, "utci", res), dtype=float)
    ok = np.isfinite(u)
    if ok.sum() < 0.5 * len(u):
        raise SystemExit("UTCI could not be evaluated for most points")
    um = float(u[ok].mean())
    ta0 = b["air_temperature_C"]
    s0 = a.abs_k * (0.5 * dhi + 0.5 * a.albedo * (dni * max(sun[2], 0) + dhi) + fp * dni)
    l0 = a.emis_p * (0.5 * hir + 0.5 * a.emis_g * SIGMA * (ta0 + 273.15) ** 4)
    tm0 = ((s0 + l0) / (a.emis_p * SIGMA)) ** 0.25 - 273.15
    r0 = _utci(tdb=ta0, tr=tm0, v=float(np.clip(b["wind_speed_10m_ms"], 0.5, 17.0)), rh=b["relative_humidity_pct"], units="SI", limit_inputs=False)
    u0 = float(getattr(r0, "utci", r0))

    site_desc = f"site polygon of {Path(jpath).name}, AI rasters"
    rep = json.loads((case / "ai_work" / "ai_report.json").read_text()) if (case / "ai_work" / "ai_report.json").is_file() else {}
    summary = dict(
        time="AI", site=site_desc, points=int(ok.sum()), height_m=2.0,
        utci_C=um, utci_min_C=float(u[ok].min()), utci_max_C=float(u[ok].max()), category=CU.category(um),
        rural_utci_C=u0, rural_tmrt_C=float(tm0),
        Tair_C=float(Ta.mean()), RH_pct=float(RH.mean()), wind_ms=float(ws.mean()), Tmrt_C=float(tmrt[ok].mean()),
        sunlit_fraction=float(sunlit.mean()), sky_open=float(f_open.mean()), sky_tree=float(f_tree.mean()),
        sky_building=float(f_bld.mean()), ground_T_C=float(Tg.mean()), building_T_C=Tb,
        sun_altitude_deg=alt, dni=dni, dhi=dhi, target=dt, interpolation=dict(method="AI 2D raster (pedestrian layer)"),
        source="ai", site_coverage=coverage, ai=rep, seconds=round(time.time() - t0, 1))
    pp = th / "postProcessing"
    pp.mkdir(exist_ok=True)
    (pp / "utci.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(pp / "utci_points.csv", np.column_stack([P[ok], Ta[ok], RH[ok], ws[ok], tmrt[ok], u[ok], sunlit[ok], f_open[ok]]),
               delimiter=",", header="x,y,z,Tair_C,RH_pct,wind_ms,Tmrt_C,UTCI_C,sunlit,sky_open", comments="")
    out = a.out or case / "UTCI_details.txt"
    out.write_text(
        f"=== Processing target hour {dt['year']}-{dt['month']:02d}-{dt['day']:02d} hour_end {dt['hour_end']:02d} (AI fields) ===\n"
        f"Sample points: {int(ok.sum())} at the AI raster layer inside {site_desc}\n"
        f"Mean: Tair={summary['Tair_C']:.2f} C  RH={summary['RH_pct']:.1f} %  wind={summary['wind_ms']:.2f} m/s  "
        f"Tmrt={summary['Tmrt_C']:.2f} C  (sunlit {100 * summary['sunlit_fraction']:.0f} %, sky open {100 * summary['sky_open']:.0f} %, "
        f"tree {100 * summary['sky_tree']:.0f} %, building {100 * summary['sky_building']:.0f} %)\n"
        f"Values: T, RH and |U| from the AI models (wind -> T -> RH), back end {rep.get('backend', '?')}\n"
        f"Rural reference: UTCI={u0:.2f} C (Tmrt {tm0:.2f} C)\n"
        f"Site area-averaged: UTCI={um:.2f} C ({CU.category(um)}), Delta UTCI vs rural {um - u0:+.2f} C, "
        f"range {summary['utci_min_C']:.1f} .. {summary['utci_max_C']:.1f} C\n")
    print(f"[utci] {int(ok.sum())} points: Ta {Ta.mean():.1f} C  RH {RH.mean():.0f} %  |U| {ws.mean():.2f} m/s  Tmrt {tmrt[ok].mean():.1f} C "
          f"-> UTCI {um:.2f} C ({CU.category(um)}), rural {u0:.2f} C   [{time.time() - t0:.0f} s] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
