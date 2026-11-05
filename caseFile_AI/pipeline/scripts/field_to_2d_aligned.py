"""
Weather data -> 2D inlet rasters  (sirf T_inlet, u_inlet, w_inlet ke .npy)

prepare_all.sh isko automatically chalata hai, weatherData/<MMDD>/ aur
geometry_meta.json (rasterize_obj.py se) ke saath -> grid A se aligned.

Usage:
    python3 field_to_2d_aligned.py <weatherData_dir> <geometry_meta.json> <out_dir>
        [--timesteps 3600,7200]

Har date folder (<MMDD>) aur har timestep ke liye 3 npy:
    <MMDD>_<t>_T_inlet.npy   <- Tambient   [K]
    <MMDD>_<t>_u_inlet.npy   <- windSpeed  [m/s]
    <MMDD>_<t>_w_inlet.npy   <- wambient   [kg/kg]

Grid (H, W) float32: sirf inlet edge ke pixels me value, baaki sab NaN.
Inlet edge = jis domain edge ka outward normal . windVelocity < 0.
Row 0 = y_max (geometry raster jaisa).
"""
import os
import re
import json
import argparse
import numpy as np

# (naam in output, weatherData file)
INLET_FIELDS = [
    ("T_inlet", "Tambient"),
    ("u_inlet", "windSpeed"),
    ("w_inlet", "wambient"),
]

DOMAIN_EDGE_NORMALS = {
    "x_min": np.array([-1.0, 0.0, 0.0]),
    "x_max": np.array([1.0, 0.0, 0.0]),
    "y_min": np.array([0.0, -1.0, 0.0]),
    "y_max": np.array([0.0, 1.0, 0.0]),
}

_NUM = r"[-+0-9.eE]+"


def read_scalar_table(path):
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    entries = [(float(m.group(1)), float(m.group(2)))
               for m in re.finditer(rf"\(\s*({_NUM})\s+({_NUM})\s*\)", content)]
    if not entries:
        raise ValueError(f"No scalar entries parsed from {path}")
    entries.sort(key=lambda e: e[0])
    return {t: v for t, v in entries}


def read_vector_table(path):
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    pat = rf"\(\s*({_NUM})\s+\(\s*({_NUM})\s+({_NUM})\s+({_NUM})\s*\)\s*\)"
    entries = [(float(m.group(1)), [float(m.group(2)), float(m.group(3)), float(m.group(4))])
               for m in re.finditer(pat, content)]
    if not entries:
        raise ValueError(f"No vector entries parsed from {path}")
    entries.sort(key=lambda e: e[0])
    return {t: np.asarray(v) for t, v in entries}


def find_inlet_edges(wind_velocity, tol=1e-12):
    return [name for name, n in DOMAIN_EDGE_NORMALS.items()
            if float(np.dot(n, wind_velocity)) < -tol]


def build_inlet_grid(W, H, inlet_edges, value):
    grid = np.full((H, W), np.nan, dtype=np.float32)
    edge_slices = {
        "x_min": (slice(None), 0),
        "x_max": (slice(None), -1),
        "y_min": (-1, slice(None)),
        "y_max": (0, slice(None)),
    }
    for edge in inlet_edges:
        grid[edge_slices[edge]] = value
    return grid


def generate_inlets(weather_root, geometry_meta_path, out_dir, timesteps=None):
    with open(geometry_meta_path) as f:
        W, H = json.load(f)["resolution"]

    if not os.path.isdir(weather_root):
        raise FileNotFoundError(f"weatherData dir not found: {weather_root}")
    date_dirs = sorted(d for d in os.listdir(weather_root)
                       if os.path.isdir(os.path.join(weather_root, d)))
    if not date_dirs:
        raise RuntimeError(f"No <MMDD> folders in {weather_root}")

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for date in date_dirs:
        ddir = os.path.join(weather_root, date)
        wind = read_vector_table(os.path.join(ddir, "windVelocity"))
        tables = {name: read_scalar_table(os.path.join(ddir, fname))
                  for name, fname in INLET_FIELDS}

        times = sorted(wind.keys())
        if timesteps is not None:
            times = [t for t in times if t in timesteps]

        for t in times:
            edges = find_inlet_edges(wind[t])
            ts = f"{int(t)}" if float(t).is_integer() else f"{t}"
            for name, _ in INLET_FIELDS:
                grid = build_inlet_grid(W, H, edges, tables[name][t])
                path = os.path.join(out_dir, f"{date}_{ts}_{name}.npy")
                np.save(path, grid)
                saved.append(path)
            print(f"[inlet] {date} t={ts}: inlet edges = {edges or 'none (calm wind)'}")

    print(f"[inlet] {len(saved)} npy files saved -> {out_dir}")
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="weatherData -> T/u/w inlet 2D npy")
    ap.add_argument("weather_root", help="folder jisme <MMDD>/ date folders hain (weatherData)")
    ap.add_argument("geometry_meta", help="geometry_meta.json (rasterize_obj.py se)")
    ap.add_argument("out_dir", help="npy yahan save honge")
    ap.add_argument("--timesteps", default=None,
                    help="comma-separated seconds, e.g. 3600,7200 (default: table ke sab times)")
    a = ap.parse_args()
    ts = {float(x) for x in a.timesteps.split(",")} if a.timesteps else None
    generate_inlets(a.weather_root, a.geometry_meta, a.out_dir, ts)
