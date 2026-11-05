"""
UHI Instant pipeline: scene JSON + EPW + wind OBJ -> T and RH maps for the hottest day of the EPW year.

    1. hottest hour of the EPW year       -> job-local date_and_time.xyz            (this host)
    2. wind + pressure + facade Cp        -> wind_2d/Summer/{U_2d.npy, flow, extent, facade_cp_summary.txt} (RunPod)
    3. scripts/prepare_all.sh             -> geometry, weather, rasters, inlets      (this host, CPU)
    4. wind field <-> geometry alignment  (mirror check), inlets on the wind model's inlet edge(s)
    5. for every clock time in date_and_time.xyz (09, 13/14, 17): T model then RH model    (RunPod)

The wind model, T model and RH model all run on the same RunPod endpoint (worker/handler.py). Nothing goes
through S3; the wind VTK never leaves the GPU worker.

Layout written under <run_dir>/output:
    wind_2d/Summer/                    U_2d.npy + U_flow_direction.json + wind_extent.txt + facade_cp_summary.txt
    raster2d/seasons/<Season>/         U_2d.npy + U_flow_direction.json aligned to the geometry raster
    rht_work/<HHMM>/raster2d|T|RH      per-time working folders
    rht_results/<HHMM>_T.png, <HHMM>_RH.png, summary.json      <- what the API serves
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app import runpod_client, wind_ai

logger = logging.getLogger("rht_api.pipeline")

IMAGE_NAME_RE = re.compile(r"^[0-9]{4}_(T|RH)\.png$")
SEASON = "Summer"  # UHI Instant uses only the Summer wind

# angle (deg) between the season's wind-model direction and the EPW wind at that hour
GOOD_ANGLE_DEG = 22.5
FAIR_ANGLE_DEG = 45.0


# ─── helpers ────────────────────────────────────────────────────────────────────
def _run(cmd: List[str], log_path: Path, *, cwd: Optional[Path] = None, env: Optional[dict] = None,
         timeout: int = 1800, append: bool = False) -> int:
    with open(log_path, "a" if append else "w") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None,
            env=env or os.environ.copy(), timeout=timeout,
        )
    return proc.returncode


def _tail(path: Path, n: int = 1500) -> str:
    try:
        return path.read_text(errors="replace")[-n:]
    except Exception:  # noqa: BLE001
        return ""


def _hour_label(hour: Optional[int]) -> str:
    if hour is None:
        return "--"
    h12 = hour % 12 or 12
    return f"{h12} {'AM' if hour < 12 else 'PM'}"


def _quality(angle: float) -> str:
    if angle <= GOOD_ANGLE_DEG:
        return "good"
    if angle <= FAIR_ANGLE_DEG:
        return "fair"
    return "poor"


def _load_script_module(scripts_dir: Path, name: str):
    """Import scripts/<name>.py (numpy-only helpers; no torch)."""
    p = str(scripts_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module(name)


# ─── step 1: hottest-day date file (per job) ────────────────────────────────────
_PEAK_RE = re.compile(r"Peak temperature:\s*([-\d.]+)\s*C on (\d{2})/(\d{2}) at hour (\d{2})")


def make_job_date_file(epw_path: Path, out_dir: Path, scripts_dir: Path, log_path: Path,
                       env: Optional[dict] = None) -> Tuple[Path, Dict[str, Any]]:
    """
    Copy scripts/date_and_time.xyz into the job folder and run scripts/update_date_time.py on the COPY, so the
    run uses the hottest hour of the whole EPW year (shared template is never modified -> parallel jobs are safe).
    Returns (path of the job's date_and_time.xyz, {"temp_c", "month", "day", "hour"}).
    """
    template = scripts_dir / "date_and_time.xyz"
    job_file = out_dir / "date_and_time.xyz"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, job_file)
    proc = subprocess.run(
        [sys.executable, str(scripts_dir / "update_date_time.py"), str(epw_path), str(job_file)],
        capture_output=True, text=True, env=env or os.environ.copy(), timeout=300,
    )
    with open(log_path, "a") as log:
        log.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"update_date_time.py failed (rc={proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}")
    m = _PEAK_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(f"update_date_time.py gave unexpected output: {proc.stdout[-300:]}")
    peak = {"temp_c": float(m.group(1)), "month": int(m.group(2)), "day": int(m.group(3)), "hour": int(m.group(4))}
    logger.info("hottest hour of the EPW year: %s", peak)
    return job_file, peak


# ─── step 4: put the wind field into the geometry raster's frame ────────────────
# The wind model builds its own OBJ from the scene, and that OBJ can be mirrored relative to the geometry raster
# built here (flip_json / json_to_obj). Then U_2d shows the buildings in one place and geometry_labels.npy in
# another, and the T / RH models paint "ghost" buildings. We measure it (building mask of the geometry vs. solid
# mask of U_2d) and, when a pure mirror fixes it, mirror U_2d AND the flow vector together (a Y-mirror also flips
# uy, otherwise the wind <-> EPW direction matching would be wrong).
# Only orientation-independent transforms are applied automatically (flip up-down / left-right / rotate 180).
ALIGN_MIN_IOU = float(os.getenv("ALIGN_MIN_IOU", "0.80"))   # accept a layout only if the masks overlap this much
ALIGN_MIN_GAIN = 0.15                                       # and a transform must beat identity by this much

# name -> (array transform, (sign_ux, sign_uy), edge-name pairs to swap in the informational flow fields)
_SAFE_TRANSFORMS = {
    "identity": (lambda a: a, (1, 1), ()),
    "flip_up_down": (np.flipud, (1, -1), (("y_min", "y_max"),)),
    "flip_left_right": (np.fliplr, (-1, 1), (("x_min", "x_max"),)),
    "rotate_180": (lambda a: np.rot90(a, 2), (-1, -1), (("x_min", "x_max"), ("y_min", "y_max"))),
}


def _solid_mask(u: np.ndarray) -> np.ndarray:
    return (~np.isfinite(u)) | (u == 0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum()) / float(union)


def _transform_flow(flow: dict, signs: Tuple[int, int], swaps: tuple) -> dict:
    sx, sy = signs
    out = dict(flow)
    out["ux"] = float(flow["ux"]) * sx
    out["uy"] = float(flow["uy"]) * sy
    out["direction_deg"] = math.degrees(math.atan2(out["uy"], out["ux"])) % 360.0
    if flow.get("domain_mean_direction_deg") is not None:
        d = math.radians(float(flow["domain_mean_direction_deg"]))
        out["domain_mean_direction_deg"] = math.degrees(math.atan2(sy * math.sin(d), sx * math.cos(d))) % 360.0
    # edges exactly as the wind model reported them (before any frame alignment) -> used for the T/u/w inlets
    out["inlet_edges_vtk_original"] = list(flow.get("inlet_edges_vtk_original", flow.get("inlet_edges_vtk", [])))
    swap = {}
    for a, b in swaps:
        swap[a], swap[b] = b, a
    if isinstance(flow.get("inlet_edges_vtk"), list):
        out["inlet_edges_vtk"] = [swap.get(e, e) for e in flow["inlet_edges_vtk"]]
    if isinstance(flow.get("edge_inflow_flux"), dict):
        out["edge_inflow_flux"] = {swap.get(k, k): v for k, v in flow["edge_inflow_flux"].items()}
    return out


def align_wind_fields(
    raster_dir: Path, season_flows: Dict[str, dict], min_iou: Optional[float] = None
) -> Tuple[Dict[str, dict], List[dict], Dict[str, dict]]:
    """
    Returns (aligned_flows, skipped_seasons, report). Rewrites raster2d/seasons/<S>/U_2d.npy and
    U_flow_direction.json in place when a mirror is needed. A season whose wind field cannot be matched to the
    geometry is skipped (never shown) instead of producing misplaced results.
    """
    min_iou = ALIGN_MIN_IOU if min_iou is None else min_iou
    geo = np.load(raster_dir / "geometry_labels.npy")
    building_class = 6
    meta_path = raster_dir / "geometry_meta.json"
    if meta_path.is_file():
        try:
            building_class = int(json.loads(meta_path.read_text()).get("classes", {}).get("buildings", 6))
        except Exception:  # noqa: BLE001
            pass
    geo_b = geo == building_class

    aligned: Dict[str, dict] = {}
    skipped: List[dict] = []
    report: Dict[str, dict] = {}
    for season, flow in season_flows.items():
        sdir = raster_dir / "seasons" / season
        u = np.load(sdir / "U_2d.npy")
        if u.shape != geo.shape:
            skipped.append({"season": season, "reason": f"wind field {u.shape} and geometry {geo.shape} differ in size"})
            continue
        solid = _solid_mask(u)
        scores = {name: _iou(geo_b, fn(solid)) for name, (fn, _s, _w) in _SAFE_TRANSFORMS.items()}
        ident = scores["identity"]
        best = max(scores, key=scores.get)
        if ident >= min_iou:
            chosen = "identity"
        elif scores[best] >= min_iou and scores[best] - ident >= ALIGN_MIN_GAIN:
            chosen = best
        else:
            skipped.append({"season": season, "reason": (
                f"wind field does not match the geometry (building overlap {ident:.2f}, best {scores[best]:.2f} "
                f"with '{best}', need {min_iou:.2f}) - skipped instead of showing misplaced results")})
            continue
        fn, signs, swaps = _SAFE_TRANSFORMS[chosen]
        new_flow = flow
        if chosen != "identity":
            np.save(sdir / "U_2d.npy", np.ascontiguousarray(fn(u)).astype(np.float32))
            new_flow = _transform_flow(flow, signs, swaps)
            new_flow["frame_alignment"] = {"transform": chosen, "iou_before": ident, "iou_after": scores[chosen]}
            (sdir / "U_flow_direction.json").write_text(json.dumps(new_flow, indent=2))
            logger.info("season %s: wind field aligned to geometry with %s (IoU %.2f -> %.2f)",
                        season, chosen, ident, scores[chosen])
        aligned[season] = new_flow
        report[season] = {"transform": chosen, "iou_before": round(ident, 3), "iou_after": round(scores[chosen], 3)}
    return aligned, skipped, report


# ─── step 5a: pick the wind field per target time ───────────────────────────────
def select_matches(rpt, raster_dir: Path, weather_dir: Path,
                   season_flows: Dict[str, dict]) -> Tuple[List[dict], List[dict]]:
    """
    For every clock time listed in date_and_time.xyz choose the season whose wind-model flow
    direction is closest to the EPW wind direction at that time (speed ignored).
    Returns (matches, skipped_times).
    """
    cases = rpt.find_cases(str(raster_dir))
    hours_by_date = rpt.load_hours(os.path.join(os.path.dirname(os.path.normpath(str(weather_dir))),
                                                "date_and_time.xyz"))

    # angle diff of every (date, timestep) against every season
    table: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for season, flow in season_flows.items():
        try:
            _best, rows = rpt.match_timestep(cases, str(weather_dir), flow)
        except RuntimeError:
            continue  # no non-calm timestep at all
        for r in rows:
            key = (r["date"], r["timestep"])
            slot = table.setdefault(key, {"row": r, "angles": {}})
            slot["angles"][season] = float(r["angle_diff_deg"])

    matches: List[dict] = []
    skipped: List[dict] = []
    max_angle = float(os.getenv("MAX_MATCH_ANGLE_DEG", "180"))

    for date, ts, _t_inlet, _u_inlet in cases:
        hour_list = hours_by_date.get(date, [])
        slot = table.get((date, ts))
        if slot is None:
            # calm wind (direction undefined) or no data for this timestep. Only report real target times.
            continue
        row = slot["row"]
        hour = row.get("hour")
        if hour is None or hour not in hour_list:
            continue  # t=0 initial-condition step etc. is not a user-facing time
        season, diff = min(slot["angles"].items(), key=lambda kv: kv[1])
        if diff > max_angle:
            skipped.append({"clock": row.get("clock"), "reason": f"best wind direction match is {diff:.0f} deg "
                                                                  f"(limit {max_angle:.0f})"})
            continue
        flow = season_flows[season]
        matches.append({
            "date": date, "timestep": ts, "hour": hour, "clock": row.get("clock"),
            "name_time": row.get("name_time"), "label": _hour_label(hour),
            "season": season,
            "season_wind_direction_deg": float(flow["direction_deg"]),
            "epw_wind_direction_deg": float(row["wind_direction_deg"]),
            "angle_diff_deg": float(diff),
            "match_quality": _quality(float(diff)),
            "all_angles_deg": {k: round(v, 2) for k, v in sorted(slot["angles"].items())},
        })
    matches.sort(key=lambda m: (m["date"], m["hour"]))
    return matches, skipped


# ─── step 5b: per-time T then RH on RunPod ──────────────────────────────────────
def _build_view_raster(raster_dir: Path, season_dir: Path, view_dir: Path, date: str, ts: str) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raster_dir / "geometry_labels.npy", view_dir / "geometry_labels.npy")
    shutil.copy2(season_dir / "U_2d.npy", view_dir / "U_2d.npy")
    shutil.copy2(season_dir / "U_flow_direction.json", view_dir / "U_flow_direction.json")
    for kind in ("T", "u", "w"):
        src = raster_dir / f"{date}_{ts}_{kind}_inlet.npy"
        if src.is_file():
            shutil.copy2(src, view_dir / src.name)


def _local_copy(path_in_summary: str, folder: Path) -> Path:
    p = Path(path_in_summary)
    return p if p.is_file() else folder / p.name


def _predict_one(match: dict, *, raster_dir: Path, weather_dir: Path, work_dir: Path, results_dir: Path,
                 log_path: Path) -> dict:
    hhmm = match["name_time"]
    base = work_dir / hhmm
    if base.exists():
        shutil.rmtree(base)
    view = base / "raster2d"
    t_out, rh_out = base / "T", base / "RH"
    _build_view_raster(raster_dir, raster_dir / "seasons" / match["season"], view, match["date"], match["timestep"])

    # T then RH in ONE RunPod call (the T field stays on the worker); writes the same files the scripts write
    runpod_client.predict_t_rh(view, weather_dir, t_out, rh_out, log_path, label=match["label"])

    t_summary, rh_summary = t_out / "predictions_summary.json", rh_out / "predictions_RH_summary.json"
    if not t_summary.is_file() or not rh_summary.is_file():
        raise RuntimeError(f"T/RH result incomplete for {match['label']}:\n{_tail(log_path)}")
    t_info = json.loads(t_summary.read_text())["matched"]
    rh_info = json.loads(rh_summary.read_text())
    t_png, rh_png = _local_copy(t_info["png"], t_out), _local_copy(rh_info["rh_png"], rh_out)
    for png in (t_png, rh_png):
        if not png.is_file():
            raise RuntimeError(f"{png.name} missing for {match['label']}")
    shutil.copy2(t_png, results_dir / f"{hhmm}_T.png")
    shutil.copy2(rh_png, results_dir / f"{hhmm}_RH.png")

    entry: Dict[str, Any] = dict(match)
    entry["T"] = {
        "png": f"{hhmm}_T.png",
        "min_k": float(t_info["T_min"]), "max_k": float(t_info["T_max"]),
        "min_c": float(t_info["T_min"]) - 273.15, "max_c": float(t_info["T_max"]) - 273.15,
    }
    entry["RH"] = {
        "png": f"{hhmm}_RH.png",
        "min_percent": float(rh_info["RH_min_percent"]),
        "max_percent": float(rh_info["RH_max_percent"]),
        "pressure_pa": float(rh_info.get("pressure_pa", 0.0)),
    }
    return entry


# ─── public entry point ─────────────────────────────────────────────────────────
def run(
    *,
    run_dir: Path,
    scene_json_path: Path,
    epw_path: Path,
    wind_obj_path: Path,
    wind_direction: float,
    wind_speed: float,
    app_root: Path,
    prepare_timeout: int,
) -> Dict[str, Any]:
    scripts_dir = app_root / "scripts"
    out_dir = run_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raster_dir = out_dir / "raster2d"
    weather_dir = out_dir / "weather" / "weatherData"
    work_dir = out_dir / "rht_work"
    results_dir = out_dir / "rht_results"
    log_path = out_dir / "pipeline.log"
    log_path.write_text("")
    for d in (work_dir, results_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable

    # 1. hottest hour of the EPW year -> job-local date_and_time.xyz (picked up by prepare_all.sh)
    date_file, peak_day = make_job_date_file(epw_path, out_dir, scripts_dir, log_path, env)

    # 2. wind model on RunPod -> 2D wind field on the wind model's own domain (no VTK here)
    wind_2d_dir = wind_ai.run_summer_wind(wind_obj_path, wind_direction, wind_speed,
                                          out_dir=out_dir / "wind_2d" / SEASON, log_path=log_path)

    # 3. geometry, weather, rasters and inlets on the wind domain
    env_prepare = env.copy()
    env_prepare["WIND_2D_DIR"] = str(wind_2d_dir)
    env_prepare["DATE_AND_TIME_FILE"] = str(date_file)
    prep_log = out_dir / "prepare_all.log"
    rc = _run(["bash", str(scripts_dir / "prepare_all.sh"), str(scene_json_path), str(epw_path), str(out_dir)],
              prep_log, env=env_prepare, timeout=prepare_timeout)
    if rc != 0:
        raise RuntimeError(f"prepare_all.sh failed (rc={rc}):\n{_tail(prep_log, 3000)}")

    # 4. wind field <-> geometry alignment, then inlets on the wind model's inlet edge(s)
    sdir = raster_dir / "seasons" / SEASON
    sdir.mkdir(parents=True, exist_ok=True)
    for fname in ("U_2d.npy", "U_flow_direction.json"):
        shutil.copy2(raster_dir / fname, sdir / fname)
    season_flows = {SEASON: json.loads((sdir / "U_flow_direction.json").read_text())}
    season_flows, skipped_seasons, alignment = align_wind_fields(raster_dir, season_flows)
    if not season_flows:
        raise RuntimeError(f"The wind field does not match the geometry: {skipped_seasons}")
    wi = _load_script_module(scripts_dir, "wind_inlets")
    wi.regenerate_inlets(str(raster_dir), str(weather_dir), str(sdir / "U_2d.npy"), season_flows[SEASON])

    # 5. match every user-facing time to the wind field, then T + RH per time on RunPod
    rpt = _load_script_module(scripts_dir, "run_predict_T")
    matches, skipped_times = select_matches(rpt, raster_dir, weather_dir, season_flows)
    if not matches:
        raise RuntimeError(f"No time could be matched to a wind-model direction: {skipped_times}")

    cp_summary_path = wind_2d_dir / "facade_cp_summary.txt"
    summary: Dict[str, Any] = {
        "version": 1,
        "peak_day": peak_day,
        "facade_cp": {
            "summary_file": str(cp_summary_path.relative_to(run_dir)),
            "summary": cp_summary_path.read_text(encoding="utf-8"),
        },
        "seasons_used": {s: {"wind_model_direction_deg": float(f["direction_deg"])} for s, f in season_flows.items()},
        "skipped_seasons": skipped_seasons,
        "alignment": alignment,
        "skipped_times": skipped_times,
        "times": [
            _predict_one(m, raster_dir=raster_dir, weather_dir=weather_dir, work_dir=work_dir,
                         results_dir=results_dir, log_path=log_path)
            for m in matches
        ],
        "predictions_enabled": True,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
