#!/usr/bin/env python3
"""
Generate a T prediction for the ONE matching timestep using raster2d/
(U_2d + inlets + geometry).

The wind model's U field (from VTK) corresponds to a single wind direction.
The script extracts that direction from
raster2d/U_flow_direction.json (derived from the VTK U vector), compares it
with the windVelocity direction at every timestep in weatherData, and selects
the timestep with the smallest angular difference, ignoring wind speed.

T_model/predict_T.py is executed only for the selected timestep, using its
corresponding inlet conditions.

Inputs for the selected (<MMDD>, <t>) timestep:

    geometry : <raster_dir>/geometry_labels.npy
    U (aux)  : <raster_dir>/U_2d.npy
               (wind field obtained from the wind model's VTK output)
    T inlet  : <raster_dir>/<MMDD>_<t>_T_inlet.npy
               (EPW data)
    U inlet  : <raster_dir>/<MMDD>_<t>_u_inlet.npy
               (EPW data)

Output (--out_dir):
    <MMDD>_<HHMM>_T_predicted.npy
    <MMDD>_<HHMM>_T_predicted.png
    <MMDD>_<HHMM>_T_predicted_meta.json

    predictions_summary.json
        Contains the matched timestep and the angular differences for all
        candidate timesteps.

    HHMM represents the clock time of the matched timestep.

Usage:
    python scripts/run_predict_T.py --raster_dir <run>/output/raster2d ^
        --weather_dir <run>/output/weather/weatherData ^
        --out_dir <run>/output/prediction_T --model_dir T_model
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_to_2d_aligned import find_inlet_edges, read_vector_table  # noqa: E402

T_INLET_RE = re.compile(r"^(?P<date>\d{4})_(?P<ts>[^_]+)_T_inlet\.npy$")


def find_cases(raster_dir):
    cases = []
    for name in os.listdir(raster_dir):
        m = T_INLET_RE.match(name)
        if not m:
            continue
        date, ts = m.group("date"), m.group("ts")
        u_inlet = os.path.join(raster_dir, f"{date}_{ts}_u_inlet.npy")
        if not os.path.isfile(u_inlet):
            raise FileNotFoundError(f"u_inlet missing for {name}: {u_inlet}")
        cases.append((date, ts, os.path.join(raster_dir, name), u_inlet))
    cases.sort(key=lambda c: (c[0], float(c[1])))
    if not cases:
        raise FileNotFoundError(f"No *_T_inlet.npy found in {raster_dir}")
    return cases


def _ts_str(t):
    return f"{int(t)}" if float(t).is_integer() else f"{t}"


def load_hours(xyz_path):
    """date_and_time.xyz -> {"0320": [9, 13, 17]}"""
    out = {}
    if not os.path.isfile(xyz_path):
        return out
    with open(xyz_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            mm, dd = (int(x) for x in parts[0].split(","))
            out.setdefault(f"{mm:02d}{dd:02d}", []).extend(int(h) for h in parts[1:])
    return {k: sorted(v) for k, v in out.items()}


def clock_hour(date, ts, table_times, hours_by_date):
    """t=0 IC hour (first hour - 1), phir listed hours (extractWeatherData.py jaisa). Na mile to None."""
    hrs = hours_by_date.get(date)
    if not hrs:
        return None
    times = sorted(table_times)
    labels = [_ts_str(t) for t in times]
    if ts not in labels:
        return None
    if len(times) == len(hrs) + 1:
        seq = [(hrs[0] - 1) % 24] + hrs
    elif len(times) == len(hrs):
        seq = hrs
    else:
        return None
    return seq[labels.index(ts)]


def angle_diff_deg(v1, v2):
    """Smallest angle between two horizontal vectors (deg, 0..180). Speed ignore."""
    a1 = np.degrees(np.arctan2(v1[1], v1[0]))
    a2 = np.degrees(np.arctan2(v2[1], v2[0]))
    return float(abs((a1 - a2 + 180.0) % 360.0 - 180.0))


def match_timestep(cases, weather_dir, flow):
    """(date, ts) jiska windVelocity direction VTK flow direction se sabse close ho."""
    u_vec = np.array([flow["ux"], flow["uy"], 0.0])
    hours_by_date = load_hours(os.path.join(os.path.dirname(os.path.normpath(weather_dir)),
                                            "date_and_time.xyz"))
    rows = []
    for date, ts, t_inlet, u_inlet in cases:
        wfile = os.path.join(weather_dir, date, "windVelocity")
        table = read_vector_table(wfile)
        key = next((t for t in table if _ts_str(t) == ts), None)
        if key is None:
            raise KeyError(f"t={ts} not in {wfile}")
        w = table[key]
        if float(np.hypot(w[0], w[1])) < 1e-9:      # calm wind -> direction undefined
            continue
        hour = clock_hour(date, ts, table.keys(), hours_by_date)
        rows.append({
            "date": date, "timestep": ts,
            "hour": hour,
            "clock": f"{hour:02d}:00" if hour is not None else None,
            "name_time": f"{hour:02d}00" if hour is not None else f"t{ts}",
            "wind_direction_deg": float(np.degrees(np.arctan2(w[1], w[0])) % 360.0),
            "angle_diff_deg": angle_diff_deg(u_vec, w),
            "inlet_edges": find_inlet_edges(w),
            "t_inlet": t_inlet, "u_inlet": u_inlet,
        })
    if not rows:
        raise RuntimeError("no timestep with non-zero wind to match against")
    rows.sort(key=lambda r: r["angle_diff_deg"])
    return rows[0], rows


def warn_if_out_of_range(label, path, lo, hi):
    """normalize() values ko chup-chaap [lo, hi] me clip kar deta hai -> unit mismatch pakadne ke liye."""
    a = np.load(path)
    a = a[np.isfinite(a)]
    if a.size and (a.min() < lo or a.max() > hi):
        print(f"[warn] {label} {os.path.basename(path)} range {a.min():.4f}..{a.max():.4f} "
              f"is outside norm_stats [{lo}, {hi}] -> will be clipped (unit mismatch?)",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Run T_model/predict_T.py for all timesteps in raster2d")
    ap.add_argument("--raster_dir", required=True)
    ap.add_argument("--weather_dir", required=True, help="output/weather/weatherData (<MMDD>/windVelocity)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True, help="folder with predict_T.py, latest_net_G.pth, norm_stats.json")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--no_smooth", action="store_true")
    a = ap.parse_args()

    model_dir = os.path.abspath(a.model_dir)
    checkpoint = os.path.join(model_dir, "latest_net_G.pth")
    norm_stats = os.path.join(model_dir, "norm_stats.json")
    geometry = os.path.join(a.raster_dir, "geometry_labels.npy")
    u_2d = os.path.join(a.raster_dir, "U_2d.npy")
    for p in (os.path.join(model_dir, "predict_T.py"), checkpoint, norm_stats, geometry, u_2d):
        if not os.path.isfile(p):
            print(f"[predict_T] ERROR: missing file: {p}", file=sys.stderr)
            return 1

    sys.path.insert(0, model_dir)
    import predict_T as ptm  # noqa: E402  (heavy: torch/cv2/matplotlib)

    with open(norm_stats) as f:
        stats = json.load(f)

    dir_json = os.path.join(a.raster_dir, "U_flow_direction.json")
    if not os.path.isfile(dir_json):
        print(f"[predict_T] ERROR: {dir_json} missing (vtk_to_2d.py did not find a 'U' vector in the VTK)",
              file=sys.stderr)
        return 1
    with open(dir_json) as f:
        flow = json.load(f)

    cases = find_cases(a.raster_dir)
    best, rows = match_timestep(cases, a.weather_dir, flow)

    print(f"[predict_T] wind model U direction = {flow['direction_deg']:.1f} deg "
          f"(ux={flow['ux']:.4f}, uy={flow['uy']:.4f}), inlet edges = {find_inlet_edges(np.array([flow['ux'], flow['uy'], 0.0]))}")
    for r in rows:
        mark = "  <== MATCH" if r is best else ""
        print(f"    {r['date']} {r['clock'] or '--:--'} (t={r['timestep']:>6}): wind dir {r['wind_direction_deg']:6.1f} deg  "
              f"diff {r['angle_diff_deg']:6.1f} deg  edges {r['inlet_edges']}{mark}")
    if best["angle_diff_deg"] > 45.0:
        print(f"[warn] best match still {best['angle_diff_deg']:.1f} deg off -- no timestep really "
              f"matches the wind model direction", file=sys.stderr)

    os.makedirs(a.out_dir, exist_ok=True)
    warn_if_out_of_range("T inlet", best["t_inlet"], stats["T"]["min"], stats["T"]["max"])
    warn_if_out_of_range("U inlet", best["u_inlet"], stats["U"]["min"], stats["U"]["max"])
    print(f"[predict_T] running model for {best['date']} {best['clock'] or ''} (t={best['timestep']})")
    res = ptm.predict_T(
        geometry_path=geometry,
        aux_path=u_2d,
        target_inlet_path=best["t_inlet"],
        aux_inlet_path=best["u_inlet"],
        checkpoint_path=checkpoint,
        norm_stats_path=norm_stats,
        out_dir=a.out_dir,
        case_name=best["date"],
        timestep=best["name_time"],      # file name = <MMDD>_<HHMM>_T_predicted.*
        device=a.device,
        smooth=not a.no_smooth,
    )
    arr = res["array"]
    valid = np.isfinite(arr)
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))

    # PNG: colour scale = min..max of THIS prediction, colour bar ke saath, image ke upar koi text nahi.
    # Colour bar degrees Celsius me dikhta hai (K - 273.15). Sirf image ka display badalta hai:
    # saved .npy aur predictions_summary.json (T_min/T_max) Kelvin me hi rehte hain -- RH model K padhta hai.
    K0 = 273.15
    try:
        ptm.render_prediction_image(
            arr - K0, valid, vmin - K0, vmax - K0, res["png"],
            field_name="Temperature", units="\u00b0C", cmap="jet",
            smooth=not a.no_smooth, title=None,
        )
    except Exception as exc:  # noqa: BLE001  (predict_T already wrote a PNG)
        print(f"[warn] custom PNG render failed, keeping predict_T's PNG ({exc})", file=sys.stderr)

    matched = {
        "date": best["date"],
        "clock": best["clock"],
        "timestep": best["timestep"],
        "angle_diff_deg": best["angle_diff_deg"],
        "npy": res["npy"], "png": res["png"], "meta": res["meta"],
        "T_min": vmin, "T_max": vmax,
    }
    candidates = [{k: r[k] for k in ("date", "clock", "timestep", "wind_direction_deg", "angle_diff_deg", "inlet_edges")}
                  for r in rows]
    with open(os.path.join(a.out_dir, "predictions_summary.json"), "w") as f:
        json.dump({"wind_model_direction_deg": flow["direction_deg"],
                   "matched": matched, "candidates": candidates}, f, indent=2)
    print(f"[predict_T] done -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())