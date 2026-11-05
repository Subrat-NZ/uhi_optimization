#!/usr/bin/env python3
"""
select_epw_cases.py

Automatic weather-case selection for the UHI layout optimizer.
Derived from analyze_epw_utci_extremes.py (same rural-UTCI method), changed so that:

  * it needs no summary.json: every *.epw in --epw-dir (or a single --epw) is processed;
  * for each EPW it writes <out-dir>/<epw-stem>_cases.json, the file the pipeline reads;
  * the two simulation cases are single HOURS (peak heat, peak cold), not days;
  * each case carries every boundary value the solvers need (temperature, RH, specific
    humidity, pressure, wind speed and direction, solar radiation, sky temperature,
    sun position and sun vector in the model frame);
  * it also selects 5 hot + 5 cold "check hours" (95th-99th / 1st-5th UTCI percentile,
    different days) used after the search to test the annual claim;
  * a human-readable report is still written (as before).

UTCI method is unchanged: rural UTCI from EPW data, open-field Tmrt (SolarCal-style).
    pip install pythermalcomfort numpy

Project folders (relative to the folder that contains python/):
    epw/      input  : the *.epw weather files
    jsons/    output : <epw-stem>_cases.json for each EPW, plus cases_index.json
    output/   output : utci_extremes_report.txt

Usage (run from anywhere; no arguments needed):
    python python/select_epw_cases.py
    python python/select_epw_cases.py --percentile 99 --occupied 7 20 --north-deg 0
    python python/select_epw_cases.py --epw epw/Delhi.epw
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from pythermalcomfort.models import utci as _utci

ROOT = Path(__file__).resolve().parent.parent      # folder that contains python/
EPW_HEADER_LINES = 8

# 0-indexed columns of an EPW data row
COL_YEAR, COL_MONTH, COL_DAY, COL_HOUR = 0, 1, 2, 3
COL_DRYBULB, COL_DEWPT, COL_RH, COL_PRESSURE = 6, 7, 8, 9
COL_HIR, COL_GHI, COL_DNI, COL_DHI = 12, 13, 14, 15
COL_WDIR, COL_WIND = 20, 21
MIN_COLS = COL_WIND + 1

SIGMA = 5.670374419e-8
MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

UTCI_CATEGORIES = [
    (46.0, "extreme heat stress"), (38.0, "very strong heat stress"),
    (32.0, "strong heat stress"), (26.0, "moderate heat stress"),
    (9.0, "no thermal stress"), (0.0, "slight cold stress"),
    (-13.0, "moderate cold stress"), (-27.0, "strong cold stress"),
    (-40.0, "very strong cold stress"),
]
EXTREME_COLD_LABEL = "extreme cold stress"


def utci_category(value):
    for lower, label in UTCI_CATEGORIES:
        if value > lower:
            return label
    return EXTREME_COLD_LABEL


# ---------------------------------------------------------------------------
# EPW reading
# ---------------------------------------------------------------------------
def parse_epw(epw_path):
    """Return (header, data). header: city, lat, lon, tz, elev. data: dict of arrays."""
    rows, header = [], {}
    with epw_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                header = {"city": row[1], "country": row[3],
                          "lat": float(row[6]), "lon": float(row[7]),
                          "tz": float(row[8]),
                          "elev": float(row[9]) if len(row) > 9 else 0.0}
            if i < EPW_HEADER_LINES or len(row) < MIN_COLS:
                continue
            try:
                rows.append([float(row[c]) for c in (
                    COL_YEAR, COL_MONTH, COL_DAY, COL_HOUR, COL_DRYBULB, COL_RH,
                    COL_HIR, COL_GHI, COL_DNI, COL_DHI, COL_WIND,
                    COL_WDIR, COL_PRESSURE, COL_DEWPT)])
            except ValueError:
                continue
    if not rows:
        return header, None
    a = np.array(rows, dtype=float)
    data = {"year": a[:, 0].astype(int), "month": a[:, 1].astype(int),
            "day": a[:, 2].astype(int), "hour": a[:, 3].astype(int),
            "ta": a[:, 4], "rh": a[:, 5], "hir": a[:, 6], "ghi": a[:, 7],
            "dni": a[:, 8], "dhi": a[:, 9], "wind": a[:, 10], "wdir": a[:, 11],
            "pres": a[:, 12], "dewpt": a[:, 13]}
    data["ta"][data["ta"] >= 99.0] = np.nan
    data["rh"][data["rh"] >= 999.0] = np.nan
    data["hir"][data["hir"] >= 9999.0] = np.nan
    for k in ("ghi", "dni", "dhi"):
        data[k][data[k] >= 9999.0] = np.nan
    data["wind"][data["wind"] >= 999.0] = np.nan
    data["wdir"][data["wdir"] >= 999.0] = np.nan
    data["pres"][data["pres"] >= 999999.0] = np.nan
    data["dewpt"][data["dewpt"] >= 99.0] = np.nan
    return header, data


# ---------------------------------------------------------------------------
# Solar position and mean radiant temperature
# ---------------------------------------------------------------------------
def solar_position(lat, lon, tz, year, month, day, hour_end):
    """Solar elevation and azimuth (deg, azimuth clockwise from true north), mid-hour."""
    doy = np.array([date(int(y), int(m), int(d)).timetuple().tm_yday
                    for y, m, d in zip(year, month, day)], dtype=float)
    clock_hour = hour_end - 0.5
    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (clock_hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    true_solar_min = clock_hour * 60.0 + eqtime + 4.0 * lon - 60.0 * tz
    ha = np.radians(true_solar_min / 4.0 - 180.0)
    lat_r = np.radians(lat)
    cos_zen = np.clip(np.sin(lat_r) * np.sin(decl)
                      + np.cos(lat_r) * np.cos(decl) * np.cos(ha), -1.0, 1.0)
    elev = np.radians(90.0) - np.arccos(cos_zen)
    cos_az = (np.sin(decl) - np.sin(elev) * np.sin(lat_r)) / \
             np.maximum(np.cos(elev) * np.cos(lat_r), 1e-9)
    phi = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    az = np.where(ha > 0.0, 360.0 - phi, phi)       # afternoon -> west of south
    return np.degrees(elev), az


def mean_radiant_temperature(data, header, albedo=0.20, abs_k=0.70, emis_p=0.97,
                             emis_ground=0.95):
    elev, az = solar_position(header["lat"], header["lon"], header["tz"],
                              data["year"], data["month"], data["day"], data["hour"])
    sun_up = elev > 0.0
    elev_pos = np.where(sun_up, elev, 0.0)
    fp = 0.308 * np.cos(np.radians(elev_pos * (0.998 - elev_pos ** 2 / 50000.0)))
    fp = np.where(sun_up, fp, 0.0)
    ghi = np.where(sun_up, np.nan_to_num(data["ghi"], nan=0.0), 0.0)
    dni = np.where(sun_up, np.nan_to_num(data["dni"], nan=0.0), 0.0)
    dhi = np.where(sun_up, np.nan_to_num(data["dhi"], nan=0.0), 0.0)
    ta_k = data["ta"] + 273.15
    l_ground = emis_ground * SIGMA * ta_k ** 4
    s_short = abs_k * (0.5 * dhi + 0.5 * albedo * ghi + fp * dni)
    s_long = emis_p * (0.5 * data["hir"] + 0.5 * l_ground)
    tmrt = ((s_short + s_long) / (emis_p * SIGMA)) ** 0.25 - 273.15
    return tmrt, elev, az


def compute_utci(data, header):
    tmrt, elev, az = mean_radiant_temperature(data, header)
    wind = np.clip(data["wind"], 0.5, 17.0)
    res = _utci(tdb=data["ta"], tr=tmrt, v=wind, rh=data["rh"],
                units="SI", limit_inputs=True)
    values = np.asarray(getattr(res, "utci", res), dtype=float)
    return values, tmrt, elev, az


# ---------------------------------------------------------------------------
# Humidity helpers
# ---------------------------------------------------------------------------
def e_sat_pa(t_c):
    """Saturation vapour pressure over water (Buck 1981), Pa."""
    return 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def specific_humidity(ta_c, rh_pct, p_pa):
    e = rh_pct / 100.0 * e_sat_pa(ta_c)
    return 0.622 * e / (p_pa - 0.378 * e)


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------
def window_mask(data, h0, h1):
    """EPW hour N (1..24) is inside the window if h0 < N <= h1 (local standard time)."""
    return (data["hour"] > h0) & (data["hour"] <= h1)


def pick_index(utci, mask, mode, percentile):
    """Index of the case hour. mode 'max' (heat) or 'min' (cold).
    percentile=100 -> absolute peak; <100 -> hour whose UTCI is closest to the
    P-th (heat) or (100-P)-th (cold) percentile of the window hours."""
    u = np.where(mask, utci, np.nan)
    if np.all(np.isnan(u)):
        return None
    if percentile >= 100.0:
        return int(np.nanargmax(u) if mode == "max" else np.nanargmin(u))
    q = percentile if mode == "max" else 100.0 - percentile
    target = np.nanpercentile(u, q)
    return int(np.nanargmin(np.abs(u - target)))


def pick_check_hours(data, utci, mask, mode, n, heat_thr, cold_thr, band=(95.0, 99.0)):
    """n hours on different days from the percentile band (hot) or its mirror (cold)."""
    u = np.where(mask, utci, np.nan)
    lo_q, hi_q = band if mode == "max" else (100.0 - band[1], 100.0 - band[0])
    lo, hi = np.nanpercentile(u, lo_q), np.nanpercentile(u, hi_q)
    cand = np.where((u >= lo) & (u <= hi))[0]
    if mode == "max":
        cand = cand[u[cand] > heat_thr]
    else:
        cand = cand[u[cand] < cold_thr]
    best_per_day = {}
    for i in cand:
        k = (data["month"][i], data["day"][i])
        if k not in best_per_day or (u[i] > u[best_per_day[k]] if mode == "max"
                                     else u[i] < u[best_per_day[k]]):
            best_per_day[k] = i
    idx = sorted(best_per_day.values(), key=lambda i: u[i])
    if len(idx) > n:
        idx = [idx[j] for j in np.linspace(0, len(idx) - 1, n).round().astype(int)]
    return [int(i) for i in idx]


def rotate_to_model(bearing_true_deg, north_deg):
    """True-north bearing -> bearing in the model frame (model +y is rotated
    north_deg clockwise from true north)."""
    return (bearing_true_deg - north_deg) % 360.0


def case_record(name, i, data, header, utci, tmrt, elev, az, north_deg, notes_extra=None):
    ta, rh = float(data["ta"][i]), float(data["rh"][i])
    p = data["pres"][i]
    if not np.isfinite(p):
        p = 101325.0 * (1.0 - 2.25577e-5 * header["elev"]) ** 5.25588
    p = float(p)
    q = float(specific_humidity(ta, rh, p))
    ws = float(data["wind"][i]) if np.isfinite(data["wind"][i]) else float("nan")
    wd = float(data["wdir"][i]) if np.isfinite(data["wdir"][i]) else float("nan")
    wd_model = rotate_to_model(wd, north_deg) if np.isfinite(wd) else float("nan")
    # flow vector (where the air goes), model axes x = east-ish, y = north-ish
    flow = ([-np.sin(np.radians(wd_model)), -np.cos(np.radians(wd_model))]
            if np.isfinite(wd_model) else [float("nan")] * 2)
    az_m = rotate_to_model(float(az[i]), north_deg)
    el = float(elev[i])
    sun_vec = [float(np.sin(np.radians(az_m)) * np.cos(np.radians(el))),
               float(np.cos(np.radians(az_m)) * np.cos(np.radians(el))),
               float(np.sin(np.radians(el)))]
    hir = float(data["hir"][i])
    t_sky = float((hir / SIGMA) ** 0.25 - 273.15) if np.isfinite(hir) else float("nan")
    notes = []
    if np.isfinite(ws) and ws < 0.5:
        notes.append("calm: EPW wind below 0.5 m/s; use a floor of 0.5 m/s at the inlet "
                     "(UTCI clamps at 0.5 too) and expect weak, buoyancy-influenced flow")
    if el <= 0:
        notes.append("sun below horizon: no shortwave")
    if int(data["hour"][i]) - 1 == 0:
        notes.append("hour 00 in date_and_time.xyz: extractWeatherData.py takes the initial-condition hour "
                     "from hour 23 of the SAME date; prefer --occupied 1 24 (or similar) to avoid this")
    if notes_extra:
        notes += notes_extra
    return {
        "name": name,
        "timestamp": {"year": int(data["year"][i]), "month": int(data["month"][i]),
                      "day": int(data["day"][i]), "hour_end": int(data["hour"][i]),
                      "hour_start": int(data["hour"][i]) - 1,   # hour used in date_and_time.xyz
                      "label": f"{MONTH_NAMES[int(data['month'][i])]} {int(data['day'][i]):02d}, "
                               f"hour {int(data['hour'][i]):02d} (local standard time)"},
        "epw_row_index": i,
        "rural_utci_C": float(utci[i]), "rural_tmrt_C": float(tmrt[i]),
        "category": utci_category(float(utci[i])),
        "boundary": {
            "air_temperature_C": ta, "relative_humidity_pct": rh,
            "dew_point_C": float(data["dewpt"][i]) if np.isfinite(data["dewpt"][i]) else None,
            "pressure_Pa": p, "specific_humidity_kg_per_kg": q,
            "wind_speed_10m_ms": ws, "wind_speed_floor_applied_ms": max(ws, 0.5) if np.isfinite(ws) else None,
            "wind_from_deg_true": wd, "wind_from_deg_model": wd_model,
            "wind_flow_unit_vector_xy_model": flow,
            "wind_reference_height_m": 10.0,
            "ghi_Wm2": float(np.nan_to_num(data["ghi"][i])),
            "dni_Wm2": float(np.nan_to_num(data["dni"][i])),
            "dhi_Wm2": float(np.nan_to_num(data["dhi"][i])),
            "horizontal_infrared_Wm2": hir, "sky_temperature_C": t_sky,
            "sun_elevation_deg": el, "sun_azimuth_deg_true": float(az[i]),
            "sun_azimuth_deg_model": float(az_m),
            "sun_unit_vector_to_sun_xyz_model": sun_vec,
        },
        "notes": notes,
    }


def stress_status(peak, thr, mode, margin):
    """'present' if the rural peak crosses the threshold; 'marginal' if within `margin`
    K of it (urban surfaces can still push the street across); else 'absent'."""
    gap = peak - thr if mode == "max" else thr - peak
    if gap > 0:
        return "present"
    if gap > -margin:
        return "marginal"
    return "absent"


def build_cases(epw_path, args):
    header, data = parse_epw(epw_path)
    if data is None:
        raise ValueError("no usable data rows")
    utci, tmrt, elev, az = compute_utci(data, header)
    valid = ~np.isnan(utci)
    if not valid.any():
        raise ValueError("UTCI could not be computed (inputs missing / out of range)")
    mask = window_mask(data, args.occupied[0], args.occupied[1]) & valid

    ann = {"utci_max_C": float(np.nanmax(utci)), "utci_min_C": float(np.nanmin(utci)),
           "utci_mean_C": float(np.nanmean(utci)),
           "heat_hours": int(np.sum(utci > args.heat_threshold)),
           "cold_hours": int(np.sum(utci < args.cold_threshold)),
           "valid_hours": int(valid.sum()), "total_hours": int(len(utci)),
           "window_hours": int(mask.sum())}

    out = {"epw_file": epw_path.name,
           "location": {k: header[k] for k in ("city", "country", "lat", "lon", "tz", "elev")},
           "settings": {"heat_threshold_C": args.heat_threshold,
                        "cold_threshold_C": args.cold_threshold,
                        "occupied_window_hour_end": list(args.occupied),
                        "percentile": args.percentile, "north_deg": args.north_deg,
                        "stress_margin_K": args.margin},
           "annual_rural": ann}

    for name, mode, thr in (("heat", "max", args.heat_threshold),
                            ("cold", "min", args.cold_threshold)):
        i = pick_index(utci, mask, mode, args.percentile)
        if i is None:
            out[f"{name}_case"] = None
            out[f"{name}_status"] = "absent"
            continue
        peak_win = float(utci[i])
        status = stress_status(peak_win, thr, mode, args.margin)
        out[f"{name}_status"] = status
        extra = []
        # outlier diagnostic: how isolated is the chosen peak?
        lo, hi = max(i - 3, 0), min(i + 4, len(utci))
        nb = np.delete(utci[lo:hi], i - lo)
        nb = nb[~np.isnan(nb)]
        if nb.size:
            gap = abs(peak_win - float(np.median(nb)))
            if gap > 5.0:
                extra.append(f"isolated peak: {gap:.1f} K away from the median of the "
                             "3 hours either side; consider --percentile 99")
        out[f"{name}_case"] = case_record(name, i, data, header, utci, tmrt, elev, az,
                                          args.north_deg, extra) if status != "absent" else None

    checks = {}
    for name, mode in (("hot", "max"), ("cold", "min")):
        ids = pick_check_hours(data, utci, mask, mode, args.n_check,
                               args.heat_threshold, args.cold_threshold)
        checks[name] = [case_record(f"check_{name}_{k + 1}", i, data, header, utci, tmrt,
                                    elev, az, args.north_deg) for k, i in enumerate(ids)]
    out["check_hours"] = checks
    return out, ann


def format_report_block(epw_name, out):
    ann, lines = out["annual_rural"], []
    lines.append(f"{epw_name}  ({out['location']['city']}, {out['location']['country']})")
    lines.append(f"  Annual rural UTCI: max {ann['utci_max_C']:.1f}C, min {ann['utci_min_C']:.1f}C, "
                 f"mean {ann['utci_mean_C']:.1f}C ({ann['valid_hours']}/{ann['total_hours']} valid hours)")
    lines.append(f"  Heat-stress hours: {ann['heat_hours']}   Cold-stress hours: {ann['cold_hours']}")
    for n in ("heat", "cold"):
        c = out[f"{n}_case"]
        if c is None:
            lines.append(f"  {n.capitalize()} case: none (status {out[f'{n}_status']})")
        else:
            b = c["boundary"]
            lines.append(f"  {n.capitalize()} case [{out[f'{n}_status']}]: {c['timestamp']['label']}  "
                         f"UTCI {c['rural_utci_C']:.1f}C ({c['category']}); Ta {b['air_temperature_C']:.1f}C, "
                         f"RH {b['relative_humidity_pct']:.0f}%, wind {b['wind_speed_10m_ms']:.1f} m/s "
                         f"from {b['wind_from_deg_true']:.0f} deg, GHI {b['ghi_Wm2']:.0f} W/m2")
            for note in c["notes"]:
                lines.append(f"      note: {note}")
    for n in ("hot", "cold"):
        lines.append(f"  Check hours ({n}): " + ", ".join(
            f"{c['timestamp']['label'].split(' (')[0]} UTCI {c['rural_utci_C']:.1f}C"
            for c in out["check_hours"][n]) if out["check_hours"][n] else f"  Check hours ({n}): none")
    return lines


def main():
    ap = argparse.ArgumentParser(description="Select peak heat / cold UTCI hours and write cases.json per EPW.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--epw-dir", type=Path, default=ROOT / "epw",
                     help="Directory; every *.epw in it is processed (default: epw/).")
    src.add_argument("--epw", type=Path, help="A single EPW file instead of the whole folder.")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "jsons",
                    help="Where <stem>_cases.json files go (default: jsons/).")
    ap.add_argument("--report", type=Path, default=ROOT / "output" / "utci_extremes_report.txt",
                    help="Text report path (default: output/utci_extremes_report.txt).")
    ap.add_argument("--heat-threshold", type=float, default=26.0)
    ap.add_argument("--cold-threshold", type=float, default=9.0)
    ap.add_argument("--margin", type=float, default=3.0,
                    help="K below/above the threshold within which stress is still 'marginal' (case kept).")
    ap.add_argument("--percentile", type=float, default=100.0,
                    help="100 = absolute peak hour (default). 99 = hour near the 99th (heat) / 1st (cold) UTCI percentile.")
    ap.add_argument("--occupied", type=float, nargs=2, default=[0, 24], metavar=("START", "END"),
                    help="Only hours with START < hour_end <= END are eligible (local standard time). "
                         "Default 0 24 = all hours; e.g. 7 20 for pedestrian hours.")
    ap.add_argument("--north-deg", type=float, default=0.0,
                    help="Clockwise angle of the model +y axis from true north (0 = +y is north).")
    ap.add_argument("--n-check", type=int, default=5, help="Check hours per season (default 5).")
    args = ap.parse_args()

    files = [args.epw] if args.epw else sorted(args.epw_dir.glob("*.epw"))
    if not files:
        raise SystemExit(f"No .epw files found in {args.epw_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["UHI case selection report", "=" * 78,
             f"Heat stress: rural UTCI > {args.heat_threshold:g} C. Cold stress: < {args.cold_threshold:g} C.",
             f"Case = single hour. Percentile {args.percentile:g}; eligible hours {args.occupied[0]:g}-{args.occupied[1]:g}.",
             "Rural UTCI = open-field UTCI from EPW data; street-level UTCI will differ.", "=" * 78, ""]
    index = []
    for f in files:
        try:
            out, _ = build_cases(f, args)
        except (ValueError, IndexError) as e:
            lines += [f"{f.name}\n  ERROR: {e}; skipped.", ""]
            index.append({"epw_file": f.name, "status": "error", "message": str(e)})
            continue
        cj = args.out_dir / f"{f.stem}_cases.json"
        cj.write_text(json.dumps(out, indent=2, allow_nan=True), encoding="utf-8")
        lines += format_report_block(f.name, out) + [f"  -> {cj}", ""]
        index.append({"epw_file": f.name, "status": "ok", "cases_json": str(cj),
                      "heat_status": out["heat_status"], "cold_status": out["cold_status"]})
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (args.out_dir / "cases_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"{sum(1 for e in index if e['status'] == 'ok')}/{len(files)} EPW files processed; "
          f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
