#!/usr/bin/env python3
"""
run_ai_case.py  --  the AI replacement of "momentum + airTqFoam":  layout -> wind (U) -> T -> RH  as 2D fields at pedestrian level

Called by ../Allrun (inside run/NAME/<season>/) after the geometry (momentum/constant/triSurface, made by the SAME json_to_obj.py /
gen_ground_obj.py as the OpenFOAM case) and the weather of the target hour (temperature_humidity/weatherData/target.json, made by the
SAME epw_to_weather.py) exist.

    1  wind OBJ        ai/wind_obj.py: buildings_refined.obj + one prism per tree          -> ai_work/wind.obj
    2  wind model      inlet vector of the target hour (EPW speed + direction, model frame) -> ai_work/wind_2d/U_2d.npy (|U|, 512 x 512)
    3  raster          pipeline/scripts/rasterize_obj.py on the wind model's own XY domain  -> ai_work/raster2d/geometry_labels.npy
    4  inlets          T, u, w on the wind model's inlet edge(s), values of the target hour -> ai_work/raster2d/<MMDD>_<t>_{T,u,w}_inlet.npy
    5  T model, RH model (pipeline/worker/handler.py, same call as the pipeline)           -> ai_work/T, ai_work/RH
    6  fields.npz      T [K], RH [%], |U| [m/s] on one grid + its extent                     -> ai_work/fields.npz   (read by ai_utci.py)

The inlet values come from target.json (the boundary of the OpenFOAM case), not from the pipeline's extractWeatherData.py, so the AI case sees
EXACTLY the weather the OpenFOAM case would see for the same hour (including --north-deg).
The layout JSON is NOT flipped again: the JSONs of the NSGA are already in the frame json_to_obj.py reads (run_all_codes flips the base once).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backend as B  # noqa: E402
import wind_obj  # noqa: E402

CONFIG = B.CONFIG
_SAFE = {"identity": (lambda a: a, (1, 1)), "flip_up_down": (np.flipud, (1, -1)), "flip_left_right": (np.fliplr, (-1, 1)),
         "rotate_180": (lambda a: np.rot90(a, 2), (-1, -1))}


def say(msg):
    print(f"[ai] {msg}", flush=True)


def write_scalar(path, t, v, label):
    Path(path).write_text(f"// {label}\n(\n( {t:.0f} {v:.6f} )\n)\n")


def write_vector(path, t, v, label):
    Path(path).write_text(f"// {label}\n(\n( {t:.0f} ( {v[0]:.10f} {v[1]:.10f} {v[2]:.10f} ) )\n)\n")


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum()) / float(u)


def align_wind(raster_dir, flow, min_iou):
    """Buildings of the geometry raster vs. solid cells of U_2d.  Identity must match; if only a mirror matches (the wind model's OBJ frame is
    mirrored against ours) U_2d AND the flow vector are mirrored together, as the pipeline does.  Otherwise: stop (never use misplaced fields)."""
    geo = np.load(raster_dir / "geometry_labels.npy")
    meta = json.loads((raster_dir / "geometry_meta.json").read_text())
    bclass = int(meta.get("classes", {}).get("buildings", 6))
    u = np.load(raster_dir / "U_2d.npy")
    if u.shape != geo.shape:
        raise B.AIError(f"wind field {u.shape} and geometry raster {geo.shape} differ in size")
    gb = geo == bclass
    if not gb.any():
        return flow, {"transform": "identity", "iou": None, "note": "no building in the layout"}
    solid = (~np.isfinite(u)) | (u == 0)
    sc = {n: iou(gb, f(solid)) for n, (f, _s) in _SAFE.items()}
    best = max(sc, key=sc.get)
    if sc["identity"] >= min_iou:
        return flow, {"transform": "identity", "iou": sc["identity"]}
    if sc[best] >= min_iou and sc[best] - sc["identity"] >= 0.15:
        f, (sx, sy) = _SAFE[best]
        np.save(raster_dir / "U_2d.npy", np.ascontiguousarray(f(u)).astype(np.float32))
        flow = dict(flow)
        flow["ux"], flow["uy"] = float(flow["ux"]) * sx, float(flow["uy"]) * sy
        flow["direction_deg"] = math.degrees(math.atan2(flow["uy"], flow["ux"])) % 360.0
        swap = {"flip_up_down": (("y_min", "y_max"),), "flip_left_right": (("x_min", "x_max"),),
                "rotate_180": (("x_min", "x_max"), ("y_min", "y_max"))}[best]
        m = {}
        for a, b in swap:
            m[a], m[b] = b, a
        flow["inlet_edges_vtk_original"] = list(flow.get("inlet_edges_vtk_original", flow.get("inlet_edges_vtk", [])))
        flow["inlet_edges_vtk"] = [m.get(e, e) for e in flow.get("inlet_edges_vtk", [])]
        (raster_dir / "U_flow_direction.json").write_text(json.dumps(flow, indent=2))
        return flow, {"transform": best, "iou_before": sc["identity"], "iou": sc[best]}
    raise B.AIError(f"the wind field does not match the geometry (building overlap {sc['identity']:.2f}, best {sc[best]:.2f} with '{best}', "
                    f"need {min_iou:.2f}): the wind model's OBJ and the geometry raster are not in the same frame")


def newest(folder, pattern):
    c = sorted(Path(folder).glob(pattern))
    if not c:
        raise B.AIError(f"no {pattern} in {folder}")
    return c[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=Path("."), help="the job folder (run/NAME/<season>)")
    ap.add_argument("--json", type=Path, default=None, help="layout json (default: the only *.json in --case)")
    ap.add_argument("--backend", default=None, help="runpod | local | mock | auto (default: $UHI_AI_BACKEND or ai/config.json)")
    a = ap.parse_args(argv)
    t0 = time.time()
    case = a.case.resolve()
    js = a.json or (sorted(case.glob("*.json")) or [None])[0]
    if js is None:
        raise SystemExit("no layout *.json in the case folder")
    th = case / "temperature_humidity"
    tgt = json.loads((th / "weatherData" / "target.json").read_text())
    b, dt = tgt["boundary"], tgt["target"]
    tri = case / "momentum" / "constant" / "triSurface"
    work = case / "ai_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    log = work / "ai.log"

    be = B.Backend(a.backend, case)
    say(f"back end: {be.name}   pipeline: {be.pipe}")

    # 1 wind OBJ
    info = wind_obj.write_wind_obj(tri, js, work / "wind.obj", CONFIG.get("tree_wind_height", "total"))
    say(f"wind OBJ: {info['building_faces']} building faces, {info['trees']} trees, height {info['zmax']:.1f} m")

    # 2 wind model, inlet = the target hour (speed and direction from target.json, model frame)
    speed = max(float(b["wind_speed_10m_ms"]), float(CONFIG.get("min_calm_speed_ms", 0.5)))
    frm = math.radians(float(b["wind_from_deg_model"]))
    ux, uy = -speed * math.sin(frm), -speed * math.cos(frm)
    ux, uy = (0.0 if abs(ux) < 1e-10 else round(ux, 6)), (0.0 if abs(uy) < 1e-10 else round(uy, 6))
    t1 = time.time()
    be.wind(work / "wind.obj", ux, uy, 0.0, work / "wind_2d", log, label=f"wind {speed:.1f} m/s from {b['wind_from_deg_model']:.0f}")
    say(f"wind model: {speed:.2f} m/s from {b['wind_from_deg_model']:.0f} deg -> ({ux:.3f}, {uy:.3f})   [{time.time() - t1:.0f} s]")

    # 3 geometry raster on the wind model's XY domain
    pipe_scripts = be.pipe / "scripts"
    sys.path.insert(0, str(pipe_scripts))
    import rasterize_obj as RO
    ras = work / "raster2d"
    ras.mkdir()
    ext = [float(v) for v in (work / "wind_2d" / "wind_extent.txt").read_text().split()]
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        RO.rasterize(str(tri), str(ras), res=int(CONFIG.get("raster_res", 512)), extent=ext)
    shutil.copy2(work / "wind_2d" / "U_2d.npy", ras / "U_2d.npy")
    shutil.copy2(work / "wind_2d" / "U_flow_direction.json", ras / "U_flow_direction.json")
    flow = json.loads((ras / "U_flow_direction.json").read_text())
    flow, al = align_wind(ras, flow, float(CONFIG.get("min_align_iou", 0.8)))
    say(f"wind field vs geometry: {al['transform']}" + (f" (building overlap {al['iou']:.2f})" if al.get("iou") is not None else ""))

    # 4 inlets: values of the target hour on the wind model's inlet edge(s)
    mmdd = f"{dt['month']:02d}{dt['day']:02d}"
    wroot = work / "weather" / "weatherData"
    (wroot / mmdd).mkdir(parents=True)
    (work / "weather" / "date_and_time.xyz").write_text(f"#MM,DD HH\n{dt['month']:02d},{dt['day']:02d}  {dt['hour_end']:02d}\n")
    tt = 3600.0                       # one time step, so the models cannot match anything else
    write_vector(wroot / mmdd / "windVelocity", tt, (ux, uy, 0.0), "wind velocity of the target hour")
    write_scalar(wroot / mmdd / "windSpeed", tt, speed, "wind speed")
    write_scalar(wroot / mmdd / "Tambient", tt, float(b["air_temperature_C"]) + 273.15, "T [K]")
    write_scalar(wroot / mmdd / "wambient", tt, float(b["specific_humidity_kg_per_kg"]), "specific humidity [kg/kg]")
    write_scalar(wroot / mmdd / "atmosphericPressure", tt, float(b["pressure_Pa"]), "pressure [Pa]")
    sys.path.insert(0, str(pipe_scripts))
    import wind_inlets
    with contextlib.redirect_stdout(buf):
        wind_inlets.regenerate_inlets(str(ras), str(wroot), str(ras / "U_2d.npy"), flow)
    if not list(ras.glob("*_T_inlet.npy")):
        raise B.AIError("no inlet edge: the wind model reports no inflow edge, the T / RH models have no inlet to start from")

    # 5 T model, RH model
    t1 = time.time()
    be.t_rh(ras, wroot, work / "T", work / "RH", log, label=f"{mmdd} {dt['hour_end']:02d}h")
    say(f"T and RH models   [{time.time() - t1:.0f} s]")
    T = np.load(newest(work / "T", "*_T_predicted.npy")).astype(np.float64)
    RH = np.load(newest(work / "RH", "*_RH_predicted.npy")).astype(np.float64)
    U = np.load(ras / "U_2d.npy").astype(np.float64)
    if not (T.shape == RH.shape == U.shape):
        raise B.AIError(f"field shapes differ: T {T.shape} RH {RH.shape} U {U.shape}")
    np.savez_compressed(work / "fields.npz", T_K=T, RH_pct=RH, U_ms=U, extent=np.array(ext))
    ok = np.isfinite(T) & np.isfinite(RH) & np.isfinite(U)
    rep = dict(backend=be.name, wind_ms=speed, wind_from_deg_model=float(b["wind_from_deg_model"]), alignment=al, extent_xy=ext,
               T_C=[float(np.nanmin(T) - 273.15), float(np.nanmax(T) - 273.15)], RH_pct=[float(np.nanmin(RH)), float(np.nanmax(RH))],
               U_ms=[float(np.nanmin(U)), float(np.nanmax(U))], valid_fraction=float(ok.mean()), seconds=round(time.time() - t0, 1))
    (work / "ai_report.json").write_text(json.dumps(rep, indent=2))
    say(f"fields: T {rep['T_C'][0]:.1f} .. {rep['T_C'][1]:.1f} C   RH {rep['RH_pct'][0]:.0f} .. {rep['RH_pct'][1]:.0f} %   "
        f"|U| {rep['U_ms'][0]:.2f} .. {rep['U_ms'][1]:.2f} m/s   [{rep['seconds']:.0f} s]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except B.AIError as exc:
        print(f"[ai] ERROR: {exc}", file=sys.stderr)
        sys.exit(4)
