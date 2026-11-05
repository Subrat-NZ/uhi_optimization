import glob
import argparse
import pandas as pd
import pvlib
import numpy as np
import os
import re


# -------------------------------------------------------------------------
# Read EPW
# -------------------------------------------------------------------------
def find_epw(pattern):
    """Resolve a regex (optionally with a directory prefix) to a single EPW file."""
    directory, name_pattern = os.path.split(pattern)
    directory = directory or "."

    if not os.path.isdir(directory):
        raise SystemExit(f"Directory not found: {directory}")

    regex = re.compile(name_pattern)
    matches = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(".epw") and regex.search(f)
    )

    if not matches:
        raise SystemExit(f"No .epw file in '{directory}' matches regex: {name_pattern}")
    if len(matches) > 1:
        raise SystemExit(
            f"Regex '{name_pattern}' matched multiple files:\n  " + "\n  ".join(matches)
        )
    return os.path.join(directory, matches[0])


parser = argparse.ArgumentParser(description="Process an EPW weather file")
parser.add_argument(
    "--epw", required=True,
    help=r"EPW path or regex on the filename, e.g. epw/file_name.epw or 'epw/.*Bengaluru.*\.epw'",
)
args = parser.parse_args()

epw_file = find_epw(args.epw)
data, meta = pvlib.iotools.read_epw(epw_file)

print(f"EPW file     : {epw_file}")
print(f"EPW location : {meta.get('Name', epw_file)}")
print(f"  lat={meta['latitude']}  lon={meta['longitude']}  UTC{meta['TZ']:+.4g}")
print(f"  Times in date_and_time.xyz are treated as local standard time (UTC{meta['TZ']:+.4g})")

# -------------------------------------------------------------------------
# Solar position — computed with the timezone-aware EPW index.
# date_and_time.xyz hours are in local standard time of the EPW location.
# pvlib uses the tz offset embedded in data.index to convert local → UTC
# internally, so solar geometry is correct for any EPW regardless of timezone.
# -------------------------------------------------------------------------
solar_position = pvlib.solarposition.get_solarposition(
    time=data.index,          # tz-aware local time from EPW → pvlib → UTC
    latitude=meta['latitude'],
    longitude=meta['longitude']
)
data["elevation"] = solar_position["apparent_elevation"].values
data["azimuth"]   = solar_position["azimuth"].values

# Strip timezone (keep local wall-clock values for hour-based matching).
# pvlib read_epw returns timestamps at the START of each hourly interval:
# index 09:00 = EPW data for the 9AM–10AM period → matches hour==9 mask.
# No ±30-minute shift needed.
times = data.index.tz_localize(None)
data.index = times

# -------------------------------------------------------------------------
# Monthly, same-hour wind fallback
# -------------------------------------------------------------------------
# Calm/near-calm EPW records are unsuitable for the inlet wind field. For
# each (month, hour), form a representative wind using only days whose wind
# speed is at least this threshold. Direction is a circular quantity, so it
# must not be averaged with an ordinary arithmetic mean (e.g. 359 and 1
# degrees should average to 0 degrees, not 180 degrees).
MIN_WIND_SPEED = 0.75


def circular_mean_degrees(values):
    """Return the circular mean of finite angles in degrees, in [0, 360)."""
    angles = np.asarray(values, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan

    radians = np.radians(angles)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    if np.hypot(mean_sin, mean_cos) < 1.0e-12:
        # The observations have no uniquely defined mean direction.
        return np.nan
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)


monthly_hourly_wind = {}
for (month_number, hour_number), group in data.groupby(
    [data.index.month, data.index.hour]
):
    qualifying = group[group["wind_speed"] >= MIN_WIND_SPEED]
    if qualifying.empty:
        continue

    mean_direction = circular_mean_degrees(qualifying["wind_direction"])
    if np.isfinite(mean_direction):
        monthly_hourly_wind[(int(month_number), int(hour_number))] = (
            float(qualifying["wind_speed"].mean()),
            mean_direction,
            len(qualifying),
        )

# -------------------------------------------------------------------------
# Wet-bulb temperature — Stull (2011) empirical formula
#   T  in °C, RH in % (0–100)  →  Tw in °C
# -------------------------------------------------------------------------
def wet_bulb_stull(T, RH):
    Tw = (T  * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
          + np.arctan(T + RH)
          - np.arctan(RH - 1.676331)
          + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
          - 4.686035)
    return Tw

data["wet_bulb"] = wet_bulb_stull(data["temp_air"], data["relative_humidity"])

# -------------------------------------------------------------------------
# Specific humidity [kg/kg]  —  derived from T, RH, atmospheric pressure
#   Tetens saturation vapour pressure → actual vapour pressure → q
# -------------------------------------------------------------------------
def specific_humidity(T_C, RH_pct, P_Pa):
    es = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))   # saturation e [Pa]
    e  = (RH_pct / 100.0) * es                           # actual e [Pa]
    return 0.622 * e / (P_Pa - 0.378 * e)                # q [kg/kg]

data["specific_humidity"] = specific_humidity(
    data["temp_air"], data["relative_humidity"], data["atmospheric_pressure"]
)

def saturation_specific_humidity(T_C, P_Pa):
    es = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    return 0.622 * es / (P_Pa - 0.378 * es)

data["pond_saturation_specific_humidity"] = saturation_specific_humidity(
    data["wet_bulb"],
    data["atmospheric_pressure"],
)

# Cloud cover fraction [0–1]  (EPW stores total_sky_cover in tenths 0–10)
data["cloud_cover_frac"] = data["total_sky_cover"] / 10.0

# -------------------------------------------------------------------------
# Parse date_and_time.xyz
#   Format:  #MM,DD  HH  HH  HH ...
# -------------------------------------------------------------------------
timestamps = []   # list of (month, day, hour) tuples

with open("date_and_time.xyz") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        md    = parts[0].split(',')
        month = int(md[0])
        day   = int(md[1])
        hours = [int(h) for h in parts[1:]]
        for h in hours:
            timestamps.append((month, day, h))

# Sort chronologically so output time increases monotonically
timestamps.sort()

# -------------------------------------------------------------------------
# Time-axis helper  (year-independent day-of-year seconds)
# -------------------------------------------------------------------------
from datetime import date as _date
from collections import defaultdict

def _doy_seconds(month, day, hour):
    """Seconds from Jan 1 00:00 for a given month/day/hour (year-independent)."""
    doy = _date(2000, month, day).timetuple().tm_yday - 1   # 0-indexed
    return doy * 86400 + hour * 3600

# -------------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------------
# All extracted weather data is written to weatherData/<MMDD>/ subdirectories.
# Nothing is written into the OpenFOAM case directories.
os.makedirs("weatherData", exist_ok=True)

def write_scalar_table(filepath, time_list, values, label=""):
    with open(filepath, "w") as f:
        f.write(f"// {label}\n")
        f.write("(\n")
        for t, v in zip(time_list, values):
            f.write(f"( {t:.0f} {v:.6f} )\n")
        f.write(")\n")

def write_vector_table(filepath, time_list, vx, vy, vz, label=""):
    with open(filepath, "w") as f:
        f.write(f"// {label}\n")
        f.write("(\n")
        for t, x, y, z in zip(time_list, vx, vy, vz):
            f.write(f"( {t:.0f} ( {x:.10f} {y:.10f} {z:.10f} ) )\n")
        f.write(")\n")

# -------------------------------------------------------------------------
# Group timestamps by date and write per-date subdirectories
# -------------------------------------------------------------------------
dates_dict = defaultdict(list)
for (month, day, hour) in timestamps:
    dates_dict[(month, day)].append(hour)

for (month, day), hours in sorted(dates_dict.items()):
    date_str  = f"{month:02d}{day:02d}"
    out_dir   = f"weatherData/{date_str}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Extract EPW rows for this date's hours (sorted) ---
    hours_sorted = sorted(hours)
    date_rows = []
    for hour in hours_sorted:
        mask = (
            (data.index.month == month) &
            (data.index.day   == day)   &
            (data.index.hour  == hour)
        )
        rows = data[mask]
        if rows.empty:
            print(f"Warning: No EPW data for {month:02d}/{day:02d} hour {hour:02d}")
            continue
        date_rows.append(rows.iloc[0])

    if not date_rows:
        print(f"Warning: No data found for {month:02d}/{day:02d}, skipping.")
        continue

    # --- Prepend the IC hour (one hour before the first listed time) ---
    ic_hour = hours_sorted[0] - 1
    if ic_hour < 0:
        ic_hour = 23   # midnight wrap
    ic_mask = (
        (data.index.month == month) &
        (data.index.day   == day)   &
        (data.index.hour  == ic_hour)
    )
    ic_rows = data[ic_mask]
    if ic_rows.empty:
        print(f"Warning: No EPW data for IC hour {ic_hour:02d}:00 on {month:02d}/{day:02d}, skipping IC row.")
    else:
        date_rows.insert(0, ic_rows.iloc[0])

    sel = pd.DataFrame(date_rows)

    # --- Uniform time axis: 0=IC hour, 3600=1st sim hour, 7200=2nd, … ---
    # t=0 carries IC-hour data; t=3600, 7200, 10800 carry the listed sim hours.
    time_sec = [i * 3600 for i in range(len(date_rows))]

    # --- Sun position vector (FROM scene TOWARD sun, unit vector) ---
    # pvlib azimuth: 0=N, 90=E, 180=S, 270=W (clockwise from North)
    # Case axes: x=East, y=North, z=Up
    # When sun is above the horizon (elev > 0), z > 0.
    elev  = np.radians(sel["elevation"])
    azim  = np.radians(sel["azimuth"])
    sun_x =  np.cos(elev) * np.sin(azim)
    sun_y =  np.cos(elev) * np.cos(azim)
    sun_z =  np.sin(elev)

    # --- Wind velocity vector (x=East, y=North, z=Up) ---
    # If this date/hour is below 1.0 m/s, replace its wind for windVelocity
    # with the mean speed and circular-mean direction for the same hour over
    # all qualifying days (speed >= 1.0 m/s) in that month.
    velocity_speed = sel["wind_speed"].astype(float).copy()
    velocity_direction = sel["wind_direction"].astype(float).copy()
    wind_replacements = []

    for row_number, timestamp in enumerate(sel.index):
        original_speed = float(velocity_speed.iloc[row_number])
        if original_speed >= MIN_WIND_SPEED:
            continue

        key = (int(timestamp.month), int(timestamp.hour))
        fallback = monthly_hourly_wind.get(key)
        if fallback is None:
            print(
                f"Warning: No wind speed >= {MIN_WIND_SPEED:.1f} m/s for "
                f"month {timestamp.month:02d}, hour {timestamp.hour:02d}; "
                "using the original wind for windVelocity."
            )
            continue

        mean_speed, mean_direction, sample_count = fallback
        velocity_speed.iloc[row_number] = mean_speed
        velocity_direction.iloc[row_number] = mean_direction
        wind_replacements.append(
            (timestamp, original_speed, mean_speed, mean_direction, sample_count)
        )

    # EPW wind_direction = bearing wind comes FROM; vector points toward.
    wd_rad = np.radians(velocity_direction)
    wind_x = (-velocity_speed * np.sin(wd_rad)).values
    wind_y = (-velocity_speed * np.cos(wd_rad)).values
    wind_z = np.zeros(len(sel))

    # --- Write all table files ---
    write_vector_table(f"{out_dir}/sunPosVector", time_sec,
        sun_x.values, sun_y.values, sun_z.values,
        "Sun position vector (from scene toward sun, unit vector)")
    write_scalar_table(f"{out_dir}/solarElevation", time_sec,
        sel["elevation"].values,
        "Solar elevation angle [degrees above horizon]")
    write_scalar_table(f"{out_dir}/solarAzimuth", time_sec,
        sel["azimuth"].values,
        "Solar azimuth angle [degrees clockwise from North]")
    write_scalar_table(f"{out_dir}/IDN", time_sec,
        sel["dni"].values,
        "Direct Normal Irradiance [W/m2]")
    write_scalar_table(f"{out_dir}/Idif", time_sec,
        sel["dhi"].values,
        "Diffuse Horizontal Irradiance [W/m2]")
    write_scalar_table(f"{out_dir}/Tambient", time_sec,
        (sel["temp_air"] + 273.15).values,
        "Dry Bulb Temperature [K]")
    write_scalar_table(f"{out_dir}/wetBulbTemperature", time_sec,
        (sel["wet_bulb"] + 273.15).values,
        "Wet Bulb Temperature [K]")
    write_scalar_table(f"{out_dir}/wambient", time_sec,
        sel["specific_humidity"].values,
        "Specific Humidity [kg/kg]")
    write_scalar_table(f"{out_dir}/wPonds", time_sec,
        sel["pond_saturation_specific_humidity"].values,
        "Saturated specific humidity at pond wet-bulb surface [kg/kg]")
    write_scalar_table(f"{out_dir}/relativeHumidity", time_sec,
        (sel["relative_humidity"] / 100.0).values,
        "Relative Humidity [-]")
    write_scalar_table(f"{out_dir}/atmosphericPressure", time_sec,
        sel["atmospheric_pressure"].values,
        "Atmospheric Pressure [Pa]")
    write_scalar_table(f"{out_dir}/cloudCover", time_sec,
        sel["cloud_cover_frac"].values,
        "Cloud cover fraction [0-1]  (0=clear, 1=overcast)")
    write_vector_table(f"{out_dir}/windVelocity", time_sec,
        wind_x, wind_y, wind_z,
        "Wind velocity vector [m/s]  (x=East, y=North, z=Up)")
    write_scalar_table(f"{out_dir}/windSpeed", time_sec,
        sel["wind_speed"].values,
        "Wind Speed [m/s]")

    # --- Print summary for this date ---
    print(f"\n{month:02d}/{day:02d}  →  {out_dir}/")
    for i, (t, ts) in enumerate(zip(time_sec, sel.index)):
        row = sel.iloc[i]
        print(f"  t={t:6.0f}s  {ts.strftime('%H:%M')}  "
              f"T={row['temp_air']:.1f}°C  Tw={row['wet_bulb']:.1f}°C  "
              f"q={row['specific_humidity']:.5f}kg/kg  "
              f"elev={row['elevation']:.1f}°  az={row['azimuth']:.1f}°  "
              f"cloud={row['cloud_cover_frac']:.1f}  wind={row['wind_speed']:.1f}m/s")
    for ts, old_speed, new_speed, new_direction, sample_count in wind_replacements:
        print(
            f"    windVelocity fallback {ts.strftime('%m/%d %H:%M')}: "
            f"{old_speed:.2f} m/s -> {new_speed:.2f} m/s from "
            f"{new_direction:.1f} degrees ({sample_count} qualifying days)"
        )

print(
    "\nDone. weatherData/ subdirectories created: "
    + "  ".join(f"{m:02d}{d:02d}" for m, d in sorted(dates_dict.keys()))
    + "\n\nEnd\n"
)

