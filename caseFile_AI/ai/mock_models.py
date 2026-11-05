#!/usr/bin/env python3
"""
mock_models.py  --  a numpy-only stand-in for pipeline/worker/handler.py  (NO neural networks, NO GPU)

handler(event) takes the same {"input": {...}} the RunPod worker takes ("wind" task and "rht" task) and answers with the same
{"ok": True, "result_b64": <tar.gz>, ...} files, so the whole case (wind OBJ -> rasters -> inlets -> T / RH -> UTCI -> NSGA) can be
tested on a laptop.  The fields are NOT physical predictions, they only have the right shape and trends:
    |U|  speed * (0.4 .. 1) growing with the distance from the buildings, NaN inside buildings
    T    inlet T + warm near buildings and on paths, cool over lawns / ponds / trees, a little cooler where it is windy
    RH   from the inlet humidity ratio and that T
Select with  UHI_AI_BACKEND=mock .  Never use its numbers for a decision.
"""
import base64
import gzip
import io
import json
import math
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage

EDGES = {"x_min": (-1, 0), "x_max": (1, 0), "y_min": (0, -1), "y_max": (0, 1)}


def _pack(root, include_npy=True):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(Path(root).rglob("*")):
            if p.is_file() and (include_npy or p.suffix != ".npy"):
                tar.add(p, arcname=str(p.relative_to(root)))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _inlet_edges(ux, uy):
    return [n for n, (nx, ny) in EDGES.items() if nx * ux + ny * uy < -1e-12]


def _read_obj(text):
    V, F = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            q = line.split()
            V.append((float(q[1]), float(q[2]), float(q[3])))
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V), F


def _png(path, a, label, cmap="jet"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(a, cmap=cmap)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
    except Exception:                                       # a missing PNG never fails a mock run
        Path(path).write_bytes(b"")


def wind_job(inp, work):
    from PIL import Image, ImageDraw
    res = int(inp.get("res", 512))
    V, F = _read_obj(gzip.decompress(base64.b64decode(inp["obj_gz_b64"])).decode())
    x0, x1, y0, y1 = V[:, 0].min(), V[:, 0].max(), V[:, 1].min(), V[:, 1].max()
    dx, dy = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    ext = [float(x0 - dx), float(x1 + dx), float(y0 - dy), float(y1 + dy)]       # = the wind model domain: bbox + 10 %
    img = Image.new("L", (res, res), 0)
    dr = ImageDraw.Draw(img)
    for f in F:
        pts = [((V[i, 0] - ext[0]) / (ext[1] - ext[0]) * (res - 1), (res - 1) - (V[i, 1] - ext[2]) / (ext[3] - ext[2]) * (res - 1)) for i in f]
        if len(pts) >= 3:
            dr.polygon(pts, fill=1)
    solid = np.array(img, dtype=bool)
    solid = ndimage.binary_closing(solid, iterations=1)
    pix = max((ext[1] - ext[0]) / res, 1e-6)
    d = ndimage.distance_transform_edt(~solid) * pix
    ux, uy = float(inp["ux"]), float(inp["uy"])
    speed = math.hypot(ux, uy)
    U = (speed * (0.4 + 0.6 * (1.0 - np.exp(-d / 6.0)))).astype(np.float32)
    U[solid] = np.nan
    out = work / "out"
    out.mkdir(parents=True)
    np.save(out / "U_2d.npy", U)
    edges = _inlet_edges(ux, uy)
    flow = {"ux": ux, "uy": uy, "direction_deg": math.degrees(math.atan2(uy, ux)) % 360.0, "method": "mock",
            "inlet_edges_vtk": edges, "inlet_edges_vtk_original": edges, "domain_mean_direction_deg": math.degrees(math.atan2(uy, ux)) % 360.0}
    (out / "U_flow_direction.json").write_text(json.dumps(flow, indent=2))
    (out / "wind_extent.txt").write_text(" ".join(repr(v) for v in ext) + "\n")
    (out / "facade_cp_summary.txt").write_text("mock wind model: no facade Cp\n")
    return {"result_b64": _pack(out), "extent": ext, "inference_seconds": 0.0}


def _scalar(path, t_key=None):
    vals = [float(m) for m in __import__("re").findall(r"\(\s*[-+0-9.eE]+\s+([-+0-9.eE]+)\s*\)", Path(path).read_text())]
    return vals[-1]


def rht_job(inp, work):
    from app.runpod_client import safe_extract
    safe_extract(base64.b64decode(inp["bundle_b64"]), work / "in")
    ras = work / "in" / "raster2d"
    wd = work / "in" / "weather" / "weatherData"
    geo = np.load(ras / "geometry_labels.npy")
    U = np.load(ras / "U_2d.npy")
    tin = sorted(ras.glob("*_T_inlet.npy"))[0]
    date, ts = tin.name.split("_")[0], tin.name.split("_")[1]
    T0 = float(np.nanmax(np.load(tin)))
    w0 = float(np.nanmax(np.load(ras / f"{date}_{ts}_w_inlet.npy")))
    p = _scalar(wd / date / "atmosphericPressure") if (wd / date / "atmosphericPressure").is_file() else 101325.0
    hh = "0000"
    xyz = work / "in" / "weather" / "date_and_time.xyz"
    if xyz.is_file():
        for line in xyz.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                hh = f"{int(line.split()[1]):02d}00"
    bld = geo == 6
    dist = ndimage.distance_transform_edt(~bld)
    warm = 3.0 * np.exp(-dist / 12.0) + 2.0 * (geo == 4) - 2.0 * np.isin(geo, (2, 3, 5)) - 0.5 * np.nan_to_num(U, nan=0.0) / max(float(np.nanmax(U)), 1e-6)
    T = T0 + ndimage.gaussian_filter(warm, 2.0)
    T = T.astype(np.float32)
    T[bld] = np.nan
    wf = w0 + 0.0015 * ndimage.gaussian_filter(np.isin(geo, (2, 3, 5)).astype(float), 3.0)
    es = 610.94 * np.exp(17.625 * (T - 273.15) / ((T - 273.15) + 243.04))
    e = wf * p / (0.622 + 0.378 * wf)
    RH = np.clip(100.0 * e / es, 1.0, 100.0).astype(np.float32)
    t_out, rh_out = work / "out" / "T", work / "out" / "RH"
    t_out.mkdir(parents=True)
    rh_out.mkdir(parents=True)
    tn = t_out / f"{date}_{hh}_T_predicted.npy"
    np.save(tn, T)
    _png(t_out / f"{date}_{hh}_T_predicted.png", T - 273.15, "T [C]")
    (t_out / "predictions_summary.json").write_text(json.dumps({"matched": {
        "date": date, "clock": f"{hh[:2]}:00", "timestep": ts, "angle_diff_deg": 0.0, "npy": str(tn),
        "png": str(t_out / f"{date}_{hh}_T_predicted.png"), "T_min": float(np.nanmin(T)), "T_max": float(np.nanmax(T))}}))
    if inp.get("run_rh", True):
        rn = rh_out / f"{date}_{hh}_RH_predicted.npy"
        np.save(rn, RH)
        _png(rh_out / f"{date}_{hh}_RH_predicted.png", RH, "RH [%]", "viridis")
        (rh_out / "predictions_RH_summary.json").write_text(json.dumps({
            "date": date, "timestep": ts, "pressure_pa": p, "RH_min_percent": float(np.nanmin(RH)), "RH_max_percent": float(np.nanmax(RH)),
            "rh_npy": str(rn), "rh_png": str(rh_out / f"{date}_{hh}_RH_predicted.png")}))
    return {"result_b64": _pack(work / "out", True), "worker_dirs": {"T": str(t_out), "RH": str(rh_out)}}


def handler(event):
    inp = event.get("input") or {}
    task = inp.get("task") or ("wind" if "obj_gz_b64" in inp else "rht")
    with tempfile.TemporaryDirectory(prefix=f"mock_{task}_") as tmp:
        try:
            res = wind_job(inp, Path(tmp)) if task == "wind" else rht_job(inp, Path(tmp))
            return {"ok": True, "task": task, **res, "log": f"[mock] {task} done", "seconds": 0.0}
        except Exception as exc:  # noqa: BLE001
            import traceback
            return {"error": f"{exc}\n{traceback.format_exc()[-2000:]}"}
