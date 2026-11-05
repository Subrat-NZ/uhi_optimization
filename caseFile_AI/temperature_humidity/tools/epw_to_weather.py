#!/usr/bin/env python3
"""
epw_to_weather.py  --  EPW + target hour  ->  weatherData/ time tables + target.json

Produces
  <out>/<MMDD>/{Tambient,windSpeed,IDN,Idif,sunPosVector,relativeHumidity,
                atmosphericPressure,specificHumidity,windDirection}
        OpenFOAM-style time tables ( t value ) in SECONDS, hourly, in the format that
        surface_temperature/solar_thermal_model.py reads (load_weather_folder).
        The window is   <spin-up hours>  +  the target hour.  Record 0 is the initial
        state of the slab, each later record is the constant boundary over the hour that
        ENDS at that record (EPW convention: row "hour 14" = 13:00-14:00), so the LAST
        record is the target hour and the thermal state at the last record is the state
        at the end of the target hour with the target hour's weather.
  <out>/target.json
        boundary values for the target hour: Ta, RH, p, q (specific humidity),
        wind, plus spin-up means (used for the pond water temperature).

The target hour comes from ONE of
  --date-file date_and_time.xyz   (the same file the momentum case reads; its hour is the hour
                                   START, so the EPW row is hour_end = HH + 1, exactly as in
                                   run_candidates.py)
  --cases <epw>_cases.json --season heat|cold    (select_epw_cases.py output)
  --month M --day D --hour-end H

Axes: model +x = East, +y = North, rotated by --north-deg exactly like select_epw_cases.py
(model +y is rotated north_deg clockwise from true north; default 0).

No dependency on the rest of the project: standard library + numpy.
"""
import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

EPW_HEADER_LINES = 8
SIGMA = 5.670374419e-8


# --------------------------------------------------------------------------- EPW
def parse_epw(path):
    rows, header = [], {}
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                header = dict(city=row[1], country=row[3], lat=float(row[6]), lon=float(row[7]),
                              tz=float(row[8]), elev=float(row[9]) if len(row) > 9 else 0.0)
            if i < EPW_HEADER_LINES or len(row) < 22:
                continue
            try:
                rows.append([float(row[c]) for c in (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 20, 21)])
            except ValueError:
                continue
    if not rows:
        raise SystemExit(f"{path}: no usable EPW data rows")
    a = np.array(rows)
    d = dict(year=a[:, 0].astype(int), month=a[:, 1].astype(int), day=a[:, 2].astype(int),
             hour=a[:, 3].astype(int), ta=a[:, 4], dew=a[:, 5], rh=a[:, 6], pres=a[:, 7],
             hir=a[:, 8], ghi=a[:, 9], dni=a[:, 10], dhi=a[:, 11], wdir=a[:, 12], wind=a[:, 13])
    d["ta"][d["ta"] >= 99] = np.nan
    d["dew"][d["dew"] >= 99] = np.nan
    d["rh"][d["rh"] >= 999] = np.nan
    d["pres"][d["pres"] >= 999999] = np.nan
    d["hir"][d["hir"] >= 9999] = np.nan
    for k in ("ghi", "dni", "dhi"):
        d[k][d[k] >= 9999] = 0.0
    d["wind"][d["wind"] >= 999] = np.nan
    d["wdir"][d["wdir"] >= 999] = np.nan
    return header, d


def solar_position(lat, lon, tz, year, month, day, hour_end):
    """Elevation and azimuth (deg, clockwise from true north), at mid-hour of the EPW hour."""
    doy = np.array([date(int(y), int(m), int(dd)).timetuple().tm_yday
                    for y, m, dd in zip(year, month, day)], dtype=float)
    clock = hour_end - 0.5
    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (clock - 12.0) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = clock * 60.0 + eqt + 4.0 * lon - 60.0 * tz
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cz = np.clip(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha), -1, 1)
    el = np.radians(90.0) - np.arccos(cz)
    caz = (np.sin(dec) - np.sin(el) * np.sin(la)) / np.maximum(np.cos(el) * np.cos(la), 1e-9)
    phi = np.degrees(np.arccos(np.clip(caz, -1, 1)))
    az = np.where(ha > 0, 360.0 - phi, phi)
    return np.degrees(el), az


def e_sat_pa(t_c):
    """Saturation vapour pressure over water (Buck 1981), Pa."""
    return 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def q_from_rh(t_c, rh_pct, p_pa):
    e = rh_pct / 100.0 * e_sat_pa(t_c)
    return 0.622 * e / (p_pa - 0.378 * e)


def wet_bulb_thermo(t_c, q, p_pa):
    """Thermodynamic wet-bulb temperature, deg C: q_sat(Tw) - q = cp/L (T - Tw)  (adiabatic saturation,
    Lewis number 1).  Solved by bisection; Tw <= T."""
    L = 2.501e6 - 2370.0 * t_c
    cp = 1005.0 * (1.0 - q) + 1860.0 * q
    lo, hi = t_c - 40.0, t_c
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e = e_sat_pa(mid)
        f = 0.622 * e / (p_pa - 0.378 * e) - q - cp / L * (t_c - mid)
        if f > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def wet_bulb_stull(t_c, rh_pct):
    """Stull (2011) wet-bulb approximation, deg C (valid ~5-99 % RH, -20..50 C)."""
    return (t_c * np.arctan(0.151977 * np.sqrt(rh_pct + 8.313659)) + np.arctan(t_c + rh_pct)
            - np.arctan(rh_pct - 1.676331) + 0.00391838 * rh_pct ** 1.5 * np.arctan(0.023101 * rh_pct)
            - 4.686035)


# ------------------------------------------------------------------ target selection
def target_from_date_file(path):
    txt = Path(path).read_text()
    lines = [l.strip() for l in txt.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise SystemExit(f"{path}: no date line (expected e.g. '07,15  14')")
    m = re.match(r"(\d+)\s*,\s*(\d+)\s+(\d+)", lines[-1])
    if not m:
        raise SystemExit(f"{path}: cannot parse '{lines[-1]}' (expected 'MM,DD  HH')")
    mo, dd, hh = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return mo, dd, hh + 1                                   # hour START in file -> EPW hour_end


def target_from_cases(path, season):
    c = json.loads(Path(path).read_text())
    rec = c.get(f"{season}_case")
    if not rec:
        raise SystemExit(f"{path}: no '{season}_case'")
    ts = rec["timestamp"]
    return int(ts["month"]), int(ts["day"]), int(ts["hour_end"]), int(rec["epw_row_index"]), float(
        c.get("settings", {}).get("north_deg", 0.0))


# ---------------------------------------------------------------- foam file writers
def _w_scalar(path, name, unit, times, values):
    with open(path, "w") as f:
        f.write(f"// {name} [{unit}]  hourly time table written by epw_to_weather.py\n(\n")
        for t, v in zip(times, values):
            f.write(f"( {t:.0f} {v:.8g} )\n")
        f.write(")\n")


def _w_vector(path, name, unit, times, vec):
    with open(path, "w") as f:
        f.write(f"// {name} [{unit}]  hourly time table written by epw_to_weather.py\n(\n")
        for t, v in zip(times, vec):
            f.write(f"( {t:.0f} ( {v[0]:.8g} {v[1]:.8g} {v[2]:.8g} ) )\n")
        f.write(")\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epw", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date-file", type=Path)
    g.add_argument("--cases", type=Path)
    g.add_argument("--month", type=int)
    ap.add_argument("--season", choices=("heat", "cold"), help="with --cases")
    ap.add_argument("--day", type=int)
    ap.add_argument("--hour-end", type=int, help="EPW hour 1..24 (with --month/--day)")
    ap.add_argument("--spinup-hours", type=int, default=24,
                    help="hours simulated before the target hour so the wall/soil slabs forget the "
                         "initial state (default 24; the EPW is wrapped around the year end)")
    ap.add_argument("--north-deg", type=float, default=None,
                    help="model +y is this many degrees clockwise from true north "
                         "(default: value in the cases file, else 0)")
    ap.add_argument("--wind-floor", type=float, default=0.5, help="minimum wind speed, m/s (default 0.5)")
    ap.add_argument("--out", type=Path, default=Path("weatherData"))
    a = ap.parse_args(argv)

    header, d = parse_epw(a.epw)
    n = len(d["ta"])
    north = a.north_deg
    row = None
    if a.date_file:
        mo, dd, he = target_from_date_file(a.date_file)
    elif a.cases:
        if not a.season:
            ap.error("--cases needs --season")
        mo, dd, he, row, n0 = target_from_cases(a.cases, a.season)
        north = n0 if north is None else north
    else:
        if a.day is None or a.hour_end is None:
            ap.error("--month needs --day and --hour-end")
        mo, dd, he = a.month, a.day, a.hour_end
    north = 0.0 if north is None else north

    if row is None:
        hit = np.flatnonzero((d["month"] == mo) & (d["day"] == dd) & (d["hour"] == he))
        if not len(hit):
            raise SystemExit(f"EPW has no row for month {mo} day {dd} hour_end {he}")
        row = int(hit[0])

    N = int(a.spinup_hours)
    rows = (np.arange(row - N, row + 1)) % n
    nrec = len(rows)
    times = np.arange(nrec) * 3600.0

    elev, az = solar_position(header["lat"], header["lon"], header["tz"],
                              d["year"], d["month"], d["day"], d["hour"])
    az_m = np.radians((az - north) % 360.0)
    el_r = np.radians(elev)
    sunvec = np.stack([np.sin(az_m) * np.cos(el_r), np.cos(az_m) * np.cos(el_r), np.sin(el_r)], axis=1)

    ta = d["ta"].copy()
    bad = ~np.isfinite(ta[rows])
    if bad.any():
        raise SystemExit("EPW dry-bulb temperature is missing inside the spin-up window")
    p = d["pres"].copy()
    pfill = 101325.0 * (1.0 - 2.25577e-5 * header["elev"]) ** 5.25588
    p = np.where(np.isfinite(p), p, pfill)
    rh = np.where(np.isfinite(d["rh"]), d["rh"], 50.0)
    wind = np.where(np.isfinite(d["wind"]), d["wind"], a.wind_floor)
    wind_f = np.maximum(wind, a.wind_floor)
    wdir = np.where(np.isfinite(d["wdir"]), d["wdir"], 0.0)
    q = q_from_rh(ta, rh, p)

    mmdd = f"{d['month'][row]:02d}{d['day'][row]:02d}"
    out = a.out / mmdd
    out.mkdir(parents=True, exist_ok=True)
    r = rows
    _w_scalar(out / "Tambient", "Tambient", "K", times, ta[r] + 273.15)
    _w_scalar(out / "windSpeed", "windSpeed", "m/s", times, wind_f[r])
    _w_scalar(out / "windDirection", "windDirection (from, true north)", "deg", times, wdir[r])
    _w_scalar(out / "IDN", "direct normal irradiance", "W/m2", times, d["dni"][r])
    _w_scalar(out / "Idif", "diffuse horizontal irradiance", "W/m2", times, d["dhi"][r])
    _w_vector(out / "sunPosVector", "unit vector TO the sun, model axes (x E, y N, z up)", "-", times, sunvec[r])
    _w_scalar(out / "relativeHumidity", "relativeHumidity", "%", times, rh[r])
    _w_scalar(out / "atmosphericPressure", "atmosphericPressure", "Pa", times, p[r])
    _w_scalar(out / "specificHumidity", "specificHumidity", "kg/kg", times, q[r])

    k24 = max(1, min(24, nrec - 1))
    last = rows[-k24:]
    ts = {"year": int(d["year"][row]), "month": int(d["month"][row]), "day": int(d["day"][row]),
          "hour_end": int(d["hour"][row]), "hour_start": int(d["hour"][row]) - 1, "epw_row_index": int(row)}
    wd_model = (wdir[row] - north) % 360.0
    tgt = {
        "epw_file": Path(a.epw).name,
        "location": header,
        "target": ts,
        "weather_folder": str(out),
        "spinup_hours": N,
        "north_deg": north,
        "boundary": {
            "air_temperature_C": float(ta[row]),
            "air_temperature_K": float(ta[row] + 273.15),
            "relative_humidity_pct": float(rh[row]),
            "pressure_Pa": float(p[row]),
            "specific_humidity_kg_per_kg": float(q[row]),
            "wind_speed_10m_ms": float(wind_f[row]),
            "wind_speed_epw_ms": float(wind[row]),
            "wind_from_deg_true": float(wdir[row]),
            "wind_from_deg_model": float(wd_model),
            "dni_Wm2": float(d["dni"][row]), "dhi_Wm2": float(d["dhi"][row]),
            "sun_elevation_deg": float(elev[row]),
            "air_density_kg_m3": float(p[row] / (287.05 * (ta[row] + 273.15))),
            "wet_bulb_C": float(wet_bulb_thermo(ta[row], q[row], p[row])),
            "wet_bulb_stull_C": float(wet_bulb_stull(ta[row], rh[row])),
            "ghi_Wm2": float(d["ghi"][row]),
            "horizontal_ir_Wm2": float(d["hir"][row]) if np.isfinite(d["hir"][row]) else None,
        },
        "annual_mean_air_temperature_C": float(np.nanmean(d["ta"])),
        "spinup_means": {
            "hours_used": int(k24),
            "air_temperature_C_mean": float(np.mean(ta[last])),
            "wet_bulb_C_mean": float(np.mean(wet_bulb_stull(ta[last], rh[last]))),
            "specific_humidity_mean": float(np.mean(q[last])),
        },
    }
    (a.out / "target.json").write_text(json.dumps(tgt, indent=2))
    b = tgt["boundary"]
    print(f"[epw_to_weather] target: {ts['month']:02d}/{ts['day']:02d} hour_end {ts['hour_end']:02d}  "
          f"(EPW row {row}), window {nrec} records -> {out}")
    print(f"[epw_to_weather] Ta {b['air_temperature_C']:.1f} C  RH {b['relative_humidity_pct']:.0f} %  "
          f"q {b['specific_humidity_kg_per_kg']*1000:.2f} g/kg  p {b['pressure_Pa']:.0f} Pa  "
          f"wind {b['wind_speed_10m_ms']:.2f} m/s from {b['wind_from_deg_true']:.0f} deg  "
          f"sun elev {b['sun_elevation_deg']:.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
