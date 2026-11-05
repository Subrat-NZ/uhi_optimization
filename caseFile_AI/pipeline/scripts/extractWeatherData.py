import glob

import pandas as pd
import pvlib
import numpy as np
import os

# -------------------------------------------------------------------------
# Read EPW
# -------------------------------------------------------------------------
epw_file = glob.glob("*.epw")[0]
data, meta = pvlib.iotools.read_epw(epw_file)

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

def _wind_vector(row):
    """Return the horizontal velocity vector represented by one EPW row."""
    wd_rad = np.radians(float(row["wind_direction"]))
    speed = float(row["wind_speed"])
    return np.array([
        -speed * np.sin(wd_rad),
        -speed * np.cos(wd_rad),
        0.0,
    ])

def wind_vectors_with_zero_fallback(sel, all_data):
    """Build wind vectors, replacing calm records with nearby/day data.

    A zero vector is replaced by the average of the preceding and following
    hourly vectors. If the following hour is also calm (or unavailable), the
    vector at the maximum wind speed on that record's calendar day is used.
    Returned speeds are always the magnitudes of the final vectors.
    """
    vectors = []
    speeds = []

    for timestamp, row in sel.iterrows():
        vector = _wind_vector(row)

        if np.linalg.norm(vector) == 0.0:
            position = all_data.index.get_loc(timestamp)
            previous_row = all_data.iloc[position - 1] if position > 0 else None
            next_row = all_data.iloc[position + 1] if position + 1 < len(all_data) else None
            next_vector = _wind_vector(next_row) if next_row is not None else np.zeros(3)

            if next_row is not None and np.linalg.norm(next_vector) != 0.0 and previous_row is not None:
                vector = (_wind_vector(previous_row) + next_vector) / 2.0
                source = "average of previous and next hour"
            else:
                same_day = all_data[
                    (all_data.index.month == timestamp.month) &
                    (all_data.index.day == timestamp.day)
                ]
                max_row = same_day.loc[same_day["wind_speed"].idxmax()]
                vector = _wind_vector(max_row)
                source = "daily maximum wind velocity"

            print(
                f"Wind fallback: {timestamp.strftime('%m/%d %H:%M')} was calm; "
                f"using {source} ({np.linalg.norm(vector):.3f} m/s)."
            )

        vectors.append(vector)
        speeds.append(np.linalg.norm(vector))

    return np.asarray(vectors), np.asarray(speeds)

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
    # EPW wind_direction = bearing wind comes FROM; vector points toward
    wind_vectors, wind_speeds = wind_vectors_with_zero_fallback(sel, data)
    wind_x = wind_vectors[:, 0]
    wind_y = wind_vectors[:, 1]
    wind_z = wind_vectors[:, 2]

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
        wind_speeds,
        "Wind Speed [m/s]")

    # --- Print summary for this date ---
    print(f"\n{month:02d}/{day:02d}  →  {out_dir}/")
    for i, (t, ts) in enumerate(zip(time_sec, sel.index)):
        row = sel.iloc[i]
        print(f"  t={t:6.0f}s  {ts.strftime('%H:%M')}  "
              f"T={row['temp_air']:.1f}°C  Tw={row['wet_bulb']:.1f}°C  "
              f"q={row['specific_humidity']:.5f}kg/kg  "
              f"elev={row['elevation']:.1f}°  az={row['azimuth']:.1f}°  "
              f"cloud={row['cloud_cover_frac']:.1f}  wind={wind_speeds[i]:.1f}m/s")

print(
    "\nDone. weatherData/ subdirectories created: "
    + "  ".join(f"{m:02d}{d:02d}" for m, d in sorted(dates_dict.keys()))
    + "\n\nEnd\n"
)
