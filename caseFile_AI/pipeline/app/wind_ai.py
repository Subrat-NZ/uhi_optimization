"""
Summer wind for UHI Instant -- wind model on the UHI RunPod worker (same endpoint and image as T + RH).

The worker (worker/handler.py, task "wind") runs the wind model, reconstructs pressure and facade Cp on its CPU,
keeps the VTK on its own disk and returns the wind raster plus a compact Cp summary:

    this host                                              UHI RunPod worker
    wind OBJ (gzip, inline) + ux, uy, uz       ->          wind model -> result.vtk
    <out_dir>/U_2d.npy, flow, extent, Cp       <-          pressure reconstruction -> facade Cp -> vtk_to_2d.py

scripts/prepare_all.sh then uses that folder (WIND_2D_DIR): the raster extent is the VTK extent, so the result is
bit-identical to converting the VTK on this host.

.env:
    RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID    the UHI endpoint (app/runpod_client.py)
    WIND_RUNPOD_TIMEOUT_SEC               one wind run incl. queue wait + cold start   (default 1800)
"""
from __future__ import annotations

import base64
import gzip
import math
import os
from pathlib import Path

from app import runpod_client

WIND_FILES = ("U_2d.npy", "U_flow_direction.json", "wind_extent.txt", "facade_cp_summary.txt")


class WindAiError(RuntimeError):
    pass


def wind_vector(speed: float, direction_deg: float) -> tuple[float, float, float]:
    """Identical to WindAI's WindPipeline._compute_wind_vector (the model expects this convention)."""
    theta = math.radians(direction_deg % 360.0)
    ux = speed * math.cos(theta)
    uy = speed * math.sin(theta)
    if abs(ux) < 1e-10:
        ux = 0.0
    if abs(uy) < 1e-10:
        uy = 0.0
    return round(ux, 6), round(uy, 6), 0.0


def run_summer_wind(obj_path: Path, wind_direction: float, wind_speed: float, out_dir: Path,
                    log_path: Path) -> Path:
    """Run wind inference on RunPod; return the U raster, flow metadata, extent, and facade Cp summary."""
    ux, uy, uz = wind_vector(wind_speed, wind_direction)
    payload = {
        "task": "wind",
        "obj_gz_b64": base64.b64encode(gzip.compress(obj_path.read_bytes())).decode("ascii"),
        "ux": ux, "uy": uy, "uz": uz,
        "res": int(os.getenv("RASTER_RES", "512")),
        "z_index": int(os.getenv("VTK_Z_INDEX", "1")),
        "label": "Summer wind",
    }
    with open(log_path, "a") as f:
        f.write(f"[wind_ai] dir={wind_direction} speed={wind_speed} -> ux={ux} uy={uy}\n")
    try:
        output = runpod_client.call(payload, log_path, float(os.getenv("WIND_RUNPOD_TIMEOUT_SEC", "1800")),
                                    "wind model")
    except runpod_client.RunpodError as exc:
        raise WindAiError(f"wind model failed: {exc}") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    runpod_client.safe_extract(base64.b64decode(output["result_b64"]), out_dir)
    missing = [n for n in WIND_FILES if not (out_dir / n).is_file()]
    if missing:
        raise WindAiError(f"wind model result is missing {', '.join(missing)}")
    return out_dir
