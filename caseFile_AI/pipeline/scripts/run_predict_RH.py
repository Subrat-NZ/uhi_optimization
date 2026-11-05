#!/usr/bin/env python3
"""
Run the RH model AFTER the T model for the same matched timestep.

Workflow:
    run_predict_T.py  ->  <out>/prediction_T/predictions_summary.json
                          (matched date / timestep / T .npy file)

    run_predict_RH.py ->  Reads the same (date, timestep) and runs
                          RH_model/predict_RH.py with the following inputs:

        geometry     : <raster_dir>/geometry_labels.npy
        aux (U)      : <raster_dir>/U_2d.npy
                       (wind field obtained from the wind model's VTK output)
        aux2 (T)     : Matched <MMDD>_<HHMM>_T_predicted.npy
                       (output from the T model)
        target inlet : <raster_dir>/<MMDD>_<t>_w_inlet.npy
                       (EPW data for the matched timestep)
        pressure     : weatherData/<MMDD>/atmosphericPressure at the matched
                       timestep (EPW data, in Pa)

Output (--out_dir):
    <MMDD>_<HHMM>_w_predicted.npy / .png
    <MMDD>_<HHMM>_RH_predicted.npy / .png
    *_meta.json
    predictions_RH_summary.json

Usage:
    python scripts/run_predict_RH.py --raster_dir <run>/output/raster2d ^
        --pred_T_dir <run>/output/prediction_T ^
        --weather_dir <run>/output/weather/weatherData ^
        --out_dir <run>/output/prediction_RH --model_dir RH_model
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_to_2d_aligned import read_scalar_table  # noqa: E402


def find_checkpoint(model_dir, explicit=None):
    """--checkpoint > latest_net_G.pth > sabse bada <epoch>_net_G.pth (jaise 495_net_G.pth)."""
    if explicit:
        return explicit
    latest = os.path.join(model_dir, "latest_net_G.pth")
    if os.path.isfile(latest):
        return latest
    best, best_ep = None, -1
    for p in glob.glob(os.path.join(model_dir, "*_net_G.pth")):
        m = re.match(r"^(\d+)_net_G\.pth$", os.path.basename(p))
        if m and int(m.group(1)) > best_ep:
            best, best_ep = p, int(m.group(1))
    return best


def _ts_key(table, ts):
    return next((t for t in table if (f"{int(t)}" if float(t).is_integer() else f"{t}") == ts), None)


def range_check(label, arr, lo, hi):
    a = arr[np.isfinite(arr)]
    if a.size and (a.min() < lo or a.max() > hi):
        print(f"[warn] {label} range {a.min():.5g}..{a.max():.5g} is outside norm_stats [{lo:.5g}, {hi:.5g}] "
              f"-> model input clip hoga (unit mismatch?)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="RH model for the timestep matched by run_predict_T.py")
    ap.add_argument("--raster_dir", required=True)
    ap.add_argument("--pred_T_dir", required=True, help="prediction_T folder (predictions_summary.json + T npy)")
    ap.add_argument("--weather_dir", required=True, help="output/weather/weatherData")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True, help="folder with predict_RH.py, *_net_G.pth, norm_stats.json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    model_dir = os.path.abspath(a.model_dir)
    checkpoint = find_checkpoint(model_dir, a.checkpoint)
    norm_stats = os.path.join(model_dir, "norm_stats.json")
    summary_path = os.path.join(a.pred_T_dir, "predictions_summary.json")
    geometry = os.path.join(a.raster_dir, "geometry_labels.npy")
    u_2d = os.path.join(a.raster_dir, "U_2d.npy")

    need = [os.path.join(model_dir, "predict_RH.py"), norm_stats, geometry, u_2d, summary_path]
    if not checkpoint:
        print(f"[predict_RH] ERROR: no *_net_G.pth in {model_dir}", file=sys.stderr)
        return 1
    for p in need + [checkpoint]:
        if not os.path.isfile(p):
            print(f"[predict_RH] ERROR: missing file: {p}", file=sys.stderr)
            return 1

    with open(summary_path) as f:
        matched = json.load(f)["matched"]
    date, ts = matched["date"], str(matched["timestep"])
    t_pred = matched["npy"]
    if not os.path.isfile(t_pred):                       # summary me purana absolute path ho to
        t_pred = os.path.join(a.pred_T_dir, os.path.basename(t_pred))
    w_inlet = os.path.join(a.raster_dir, f"{date}_{ts}_w_inlet.npy")
    for p in (t_pred, w_inlet):
        if not os.path.isfile(p):
            print(f"[predict_RH] ERROR: missing file: {p}", file=sys.stderr)
            return 1

    # HHMM label -- T model wala hi (file names ek jaise rahen): <MMDD>_<HHMM>_T_predicted.npy
    m = re.match(rf"^{date}_(?P<lbl>[^_]+)_T_predicted\.npy$", os.path.basename(t_pred))
    label = m.group("lbl") if m else f"t{ts}"

    with open(norm_stats) as f:
        stats = json.load(f)
    for k in ("U", "T", "w"):
        if k not in stats:
            print(f"[predict_RH] ERROR: {norm_stats} me '{k}' min/max nahi hai (RH model ko U, T, w chahiye)",
                  file=sys.stderr)
            return 1

    # w units: norm_stats["w"] kg/kg (max ~0.03) hai ya g/kg (max ~30)? field_to_2d_aligned kg/kg deta hai.
    w_units = "g/kg" if stats["w"]["max"] > 0.5 else "kg/kg"
    os.makedirs(a.out_dir, exist_ok=True)
    w_in_path = w_inlet
    if w_units == "g/kg":
        w_arr = np.load(w_inlet).astype(np.float32) * 1000.0
        w_in_path = os.path.join(a.out_dir, f"{date}_{ts}_w_inlet_gkg.npy")
        np.save(w_in_path, w_arr)
        print("[predict_RH] norm_stats w g/kg me hai -> w inlet ko x1000 kiya")

    # pressure: EPW ka atmosphericPressure, matched timestep pe (Bengaluru ~91 kPa; 101325 galat RH deta)
    pressure = 101325.0
    pfile = os.path.join(a.weather_dir, date, "atmosphericPressure")
    if os.path.isfile(pfile):
        tab = read_scalar_table(pfile)
        k = _ts_key(tab, ts)
        if k is not None and tab[k] > 10000:
            pressure = float(tab[k])
    else:
        print("[warn] atmosphericPressure table nahi mila -> 101325 Pa use ho raha hai", file=sys.stderr)

    range_check("U", np.load(u_2d), stats["U"]["min"], stats["U"]["max"])
    range_check("T (T model output)", np.load(t_pred), stats["T"]["min"], stats["T"]["max"])
    range_check("w inlet", np.load(w_in_path), stats["w"]["min"], stats["w"]["max"])

    print(f"[predict_RH] matched by T model: {date} {matched.get('clock') or ''} (t={ts}, "
          f"angle diff {matched.get('angle_diff_deg', float('nan')):.1f} deg, match_ok={matched.get('match_ok')})")
    print(f"[predict_RH] T raster = {os.path.basename(t_pred)} | w inlet = {os.path.basename(w_inlet)} | "
          f"pressure = {pressure:.0f} Pa | w_units = {w_units} | checkpoint = {os.path.basename(checkpoint)}")

    sys.path.insert(0, model_dir)
    import predict_RH as prh  # noqa: E402  (heavy: torch/cv2/matplotlib)

    res = prh.predict_T(          # predict_RH.py me function ka naam predict_T hi hai (target = w)
        geometry_path=geometry,
        aux_path=u_2d,
        aux2_path=t_pred,
        target_inlet_path=w_in_path,
        checkpoint_path=checkpoint,
        norm_stats_path=norm_stats,
        out_dir=a.out_dir,
        case_name=date,
        timestep=label,
        target_field="w",
        use_inlet=True,
        device=a.device,
        pressure_pa=pressure,
        w_units=w_units,
    )

    rh = res["rh_array"]
    out = {
        "date": date, "timestep": ts, "clock": matched.get("clock"),
        "T_input": t_pred, "w_inlet": w_inlet, "pressure_pa": pressure, "w_units": w_units,
        "checkpoint": checkpoint,
        "RH_min_percent": float(np.nanmin(rh)), "RH_max_percent": float(np.nanmax(rh)),
        "rh_npy": res["rh_npy"], "rh_png": res["rh_png"], "w_png": res["w_png"],
        "w_npy": os.path.join(a.out_dir, f"{date}_{label}_w_predicted.npy"),
        "meta": res["meta"],
    }
    with open(os.path.join(a.out_dir, "predictions_RH_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[predict_RH] done -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())