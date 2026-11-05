#!/usr/bin/env python3
"""
Wind-model-direction inlets  (T_inlet / u_inlet / w_inlet)

Direction : the inlet edge(s) of the WIND MODEL  (raster2d/U_flow_direction.json -> "inlet_edges_vtk"),
            used for T, u and w alike -> all three inlets always sit on the same edge(s).
Values    : from the EPW, separately for every time step (so the 3 times differ):
              u_inlet = windSpeed   [m/s]
              T_inlet = Tambient    [K]
              w_inlet = wambient    [kg/kg]
            (constant value on the inlet edge(s), same as before -- only the edge now follows the wind model)

Overwrites raster2d/<MMDD>_<t>_{T,u,w}_inlet.npy that field_to_2d_aligned.py wrote from the EPW wind direction.

Usage:
    python3 wind_inlets.py <raster2d_dir> <weatherData_dir> [--u2d U_2d.npy] [--flow U_flow_direction.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_to_2d_aligned import read_scalar_table, read_vector_table, build_inlet_grid  # noqa: E402

EDGE_SLICES = {
    "x_min": (slice(None), 0),
    "x_max": (slice(None), -1),
    "y_min": (-1, slice(None)),
    "y_max": (0, slice(None)),
}


def regenerate_inlets(raster_dir, weather_root, u2d_path=None, flow=None):
    """Rewrite all T/u/w inlet npy in raster_dir: wind model's inlet edges, EPW values per time.
    (u2d_path is only used to get the grid size; falls back to geometry_meta.json.)"""
    if flow is None:
        with open(os.path.join(raster_dir, "U_flow_direction.json")) as f:
            flow = json.load(f)
    # edges exactly as the wind model reports them (original names, before the geometry-frame alignment flip)
    edges = [e for e in (flow.get("inlet_edges_vtk_original") or flow.get("inlet_edges_vtk", [])) if e in EDGE_SLICES]
    if not edges:
        print("[wind_inlets] WARNING: wind model has no clear inlet edge -> EPW-direction inlets kept unchanged")
        return []

    ref = u2d_path or os.path.join(raster_dir, "U_2d.npy")
    H, W = np.load(ref, mmap_mode="r").shape

    written = []
    for date in sorted(d for d in os.listdir(weather_root) if os.path.isdir(os.path.join(weather_root, d))):
        ddir = os.path.join(weather_root, date)
        times = sorted(read_vector_table(os.path.join(ddir, "windVelocity")).keys())
        t_tab = read_scalar_table(os.path.join(ddir, "Tambient"))
        u_tab = read_scalar_table(os.path.join(ddir, "windSpeed"))
        w_tab = read_scalar_table(os.path.join(ddir, "wambient"))
        for t in times:
            ts = f"{int(t)}" if float(t).is_integer() else f"{t}"
            for kind, grid in (("T", build_inlet_grid(W, H, edges, t_tab[t])),
                               ("u", build_inlet_grid(W, H, edges, u_tab[t])),
                               ("w", build_inlet_grid(W, H, edges, w_tab[t]))):
                p = os.path.join(raster_dir, f"{date}_{ts}_{kind}_inlet.npy")
                np.save(p, grid)
                written.append(p)
            print(f"[wind_inlets] {date} t={ts}: edges={edges}  T={t_tab[t]:.2f} K  "
                  f"u={u_tab[t]:.2f} m/s  w={w_tab[t]:.5f}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T/u/w inlets: wind model direction, EPW values")
    ap.add_argument("raster_dir")
    ap.add_argument("weather_root")
    ap.add_argument("--u2d", default=None)
    ap.add_argument("--flow", default=None)
    a = ap.parse_args()
    flow = json.load(open(a.flow)) if a.flow else None
    regenerate_inlets(a.raster_dir, a.weather_root, a.u2d, flow)