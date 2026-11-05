"""
RunPod Serverless worker for UHI: wind model + T model + RH model in ONE container, all loaded once.

Two kinds of job (event["input"]["task"]):

  "wind"  wind OBJ + (ux, uy, uz) -> wind_api/model/infrenace.py -> result.vtk on THIS worker's disk
          -> pressure_calculator/run_pressure.py -> updated p field in result.vtk
          -> pressure_calculator/facade_cp.py -> per-building Cp summary
          -> scripts/vtk_extent.py + scripts/vtk_to_2d.py -> U_2d.npy (512x512) + U_flow_direction.json
          The 27 MB VTK never leaves the worker; the host gets ~1 MB and needs no S3.
          (Bit-identical to what prepare_all.sh makes from the VTK, because the raster extent IS the VTK extent.)
  "rht"   (default) per-time rasters -> scripts/run_predict_T.py -> scripts/run_predict_RH.py, the T field is
          read by the RH model straight from this worker's disk.

Input "wind":
    obj_gz_b64   gzip + base64 of the wind OBJ
    ux, uy, uz   wind vector (app/wind_ai.py wind_vector(), same convention as WindAI)
    res          raster resolution (default 512), z_index: VTK layer for the 2D slice (default 1)
Output "wind":
    {"ok": true, "result_b64": tar.gz of U_2d.npy + U_flow_direction.json + wind_extent.txt +
     facade_cp_summary.txt, "extent": [...], ...}

Input "rht":
    bundle_b64   tar.gz: raster2d/{geometry_labels,U_2d,*_inlet}.npy + U_flow_direction.json,
                         weather/date_and_time.xyz, weather/weatherData/<MMDD>/{windVelocity,atmosphericPressure}
    run_rh       bool (default true), return_npy: bool (default false)
Output "rht":
    {"ok": true, "result_b64": tar.gz of T/ + RH/, "worker_dirs": {"T": ..., "RH": ...}, "log": ..., "seconds": ...}

Any failure returns {"error": "..."} -> RunPod marks the job FAILED.

Local test without RunPod (CPU is fine):
    APP_ROOT=. python worker/handler.py --test <run>/output/rht_work/0900/raster2d <run>/output/weather/weatherData
    APP_ROOT=. python worker/handler.py --test-wind <wind.obj> <speed m/s> <direction deg>
"""
from __future__ import annotations

import base64
import contextlib
import functools
import gzip
import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

APP_ROOT = Path(os.getenv("APP_ROOT", "/app")).resolve()
SCRIPTS_DIR = APP_ROOT / "scripts"
T_MODEL_DIR = APP_ROOT / "T_model"
RH_MODEL_DIR = APP_ROOT / "RH_model"
DEVICE = os.getenv("PREDICT_DEVICE", "").strip() or None
WIND_ENABLED = os.getenv("WIND_ENABLED", "1") != "0"  # 0 = T + RH only (e.g. a test worker without the wind model)

for _p in (APP_ROOT, SCRIPTS_DIR, RH_MODEL_DIR, T_MODEL_DIR):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402
import predict_RH as prh  # noqa: E402
import predict_T as ptm  # noqa: E402
import run_predict_RH  # noqa: E402
import run_predict_T  # noqa: E402

from app.runpod_client import safe_extract  # noqa: E402

# predict_T() / predict_RH's predict_T() call load_generator() on every prediction. Cache it so each 268 MB
# checkpoint is read once per container instead of once per job (eval() + no_grad: safe to reuse).
ptm.load_generator = functools.lru_cache(maxsize=None)(ptm.load_generator)
prh.load_generator = functools.lru_cache(maxsize=None)(prh.load_generator)

RH_INPUT_NC = 11  # geometry one-hot + U + T + w inlet (predict_RH.py with use_inlet=True)
_LOG_TAIL = 20000
_wind = None  # wind_api.model.infrenace, imported once (it loads best.ckpt at import time)
_DEVICE_INFO = "not loaded"  # e.g. "cuda:NVIDIA H100 80GB HBM3, 4 cpus" -- printed in every job log


def _warmup() -> None:
    """Load all models once. T/RH with the same (path, device) the scripts use, so the cache keys match."""
    global _wind, _DEVICE_INFO
    started = time.monotonic()
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    ptm.load_generator(str(T_MODEL_DIR / "latest_net_G.pth"), device)
    prh.load_generator(run_predict_RH.find_checkpoint(str(RH_MODEL_DIR)), device, RH_INPUT_NC)
    if WIND_ENABLED:
        _wind = importlib.import_module("wind_api.model.infrenace")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    _DEVICE_INFO = f"{device}:{gpu}, {len(os.sched_getaffinity(0))} cpus, torch {torch.__version__}"
    print(f"[worker] models loaded on {_DEVICE_INFO} in {time.monotonic() - started:.1f}s "
          f"(T, RH{', wind' if _wind else ''})", flush=True)


def _run_script(module, argv: list, log) -> int:
    """module.main() with argv, stdout/stderr into log; sys.argv / sys.path restored afterwards."""
    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = [module.__file__] + argv
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return int(module.main() or 0)
    finally:
        sys.argv, sys.path[:] = old_argv, old_path


def _pack(out_dir: Path, include_npy: bool = True) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and (include_npy or p.suffix != ".npy"):
                tar.add(p, arcname=str(p.relative_to(out_dir)))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _script(args: list, log, timeout: int = 900) -> str:
    """Run scripts/<x>.py as a subprocess (pyvista / VTK work); returns stdout, raises on failure."""
    proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout)
    log.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(args[0]).name} failed (rc={proc.returncode}): {proc.stderr.strip()[-500:]}")
    return proc.stdout


def _wind_job(inp: dict, work: Path, log) -> dict:
    if _wind is None:
        raise RuntimeError("wind model is disabled on this worker (WIND_ENABLED=0)")
    res, z_index = int(inp.get("res", 512)), int(inp.get("z_index", 1))
    job = work / "wind_job"  # run_inference names its temp files after this folder
    obj, vtk, out = job / "wind.obj", job / "vtk" / "result.vtk", work / "out"
    obj.parent.mkdir(parents=True)
    out.mkdir()
    obj.write_bytes(gzip.decompress(base64.b64decode(inp["obj_gz_b64"])))

    t0 = time.monotonic()
    print(f"[worker] {inp.get('label') or ''} wind model on {_DEVICE_INFO} ux={inp['ux']} uy={inp['uy']} uz={inp.get('uz', 0)}",
          file=log)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        _wind.run_inference(obj, float(inp["ux"]), float(inp["uy"]), float(inp.get("uz", 0.0)), vtk)
    infer_sec = time.monotonic() - t0

    pressure_vtk = vtk.with_name("result_pressure.vtk")
    _script([
        str(APP_ROOT / "pressure_calculator" / "run_pressure.py"),
        str(vtk), "--output", str(pressure_vtk), "--p-inlet", "101325",
    ], log, timeout=int(os.getenv("PRESSURE_CALCULATOR_TIMEOUT_SEC", "1800")))
    if not pressure_vtk.is_file() or pressure_vtk.stat().st_size == 0:
        raise RuntimeError("Pressure calculator did not produce a valid VTK")
    pressure_vtk.replace(vtk)

    inlet_speed = (float(inp["ux"]) ** 2 + float(inp["uy"]) ** 2 + float(inp.get("uz", 0.0)) ** 2) ** 0.5
    cp_summary = out / "facade_cp_summary.txt"
    _script([
        str(APP_ROOT / "pressure_calculator" / "facade_cp.py"),
        str(vtk), "--obj", str(obj), "--pressure-field", "p_calc",
        "--p-ref", "101325", "--uref", str(inlet_speed),
        "--output", str(job / "facade_cp.vtk"), "--csv", str(job / "facade_cp.csv"),
        "--summary", str(cp_summary),
    ], log, timeout=int(os.getenv("FACADE_CP_TIMEOUT_SEC", "900")))
    if not cp_summary.is_file() or cp_summary.stat().st_size == 0:
        raise RuntimeError("Facade Cp calculator did not produce a summary")

    # the host rasterizes on exactly this extent (prepare_all.sh, WIND_2D_DIR) -> pixel-aligned with the geometry
    extent = [float(v) for v in _script([str(SCRIPTS_DIR / "vtk_extent.py"), str(vtk)], log).split()]
    meta = job / "meta.json"
    meta.write_text(json.dumps({"extent_xy": extent, "resolution": [res, res]}))
    _script([str(SCRIPTS_DIR / "vtk_to_2d.py"), str(vtk), str(out / "U_2d.npy"), "--meta", str(meta),
             "--res", str(res), "--z-index", str(z_index)], log)
    if not (out / "U_flow_direction.json").is_file():
        raise RuntimeError("VTK has no 'U' vector -> U_flow_direction.json not written")
    (out / "wind_extent.txt").write_text(" ".join(repr(v) for v in extent) + "\n")
    print(f"[worker] wind done: inference {infer_sec:.1f}s, VTK {vtk.stat().st_size / 1e6:.1f} MB kept on worker",
          file=log)
    return {"result_b64": _pack(out), "extent": extent, "inference_seconds": round(infer_sec, 2)}


def _rht_job(inp: dict, work: Path, log) -> dict:
    safe_extract(base64.b64decode(inp["bundle_b64"]), work / "in")
    raster = work / "in" / "raster2d"
    weather = work / "in" / "weather" / "weatherData"
    t_out, rh_out = work / "out" / "T", work / "out" / "RH"
    device = ["--device", DEVICE] if DEVICE else []
    print(f"[worker] {inp.get('label') or ''} T model on {_DEVICE_INFO}", file=log)

    t0 = time.monotonic()
    rc = _run_script(run_predict_T, [
        "--raster_dir", str(raster), "--weather_dir", str(weather),
        "--out_dir", str(t_out), "--model_dir", str(T_MODEL_DIR), *device,
    ], log)
    if rc != 0:
        raise RuntimeError(f"predict_T failed (rc={rc})")
    t_sec = time.monotonic() - t0

    t0 = time.monotonic()
    if inp.get("run_rh", True):
        rc = _run_script(run_predict_RH, [
            "--raster_dir", str(raster), "--pred_T_dir", str(t_out), "--weather_dir", str(weather),
            "--out_dir", str(rh_out), "--model_dir", str(RH_MODEL_DIR), *device,
        ], log)
        if rc != 0:
            raise RuntimeError(f"predict_RH failed (rc={rc})")
    print(f"[worker] T {t_sec:.2f}s, RH {time.monotonic() - t0:.2f}s (model + PNG rendering)", file=log)
    return {
        "result_b64": _pack(work / "out", bool(inp.get("return_npy"))),
        "worker_dirs": {"T": str(t_out), "RH": str(rh_out)},
    }


def handler(event: dict) -> dict:
    inp = event.get("input") or {}
    task = inp.get("task") or ("wind" if "obj_gz_b64" in inp else "rht")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"{task}_") as tmp:
        log = io.StringIO()
        try:
            if task == "wind":
                result = _wind_job(inp, Path(tmp), log)
            elif task == "rht":
                result = _rht_job(inp, Path(tmp), log)
            else:
                raise ValueError(f"unknown task {task!r} (expected 'wind' or 'rht')")
            seconds = round(time.monotonic() - started, 2)
            print(f"[worker] {task} done in {seconds}s", file=log)
            return {"ok": True, "task": task, **result, "log": log.getvalue()[-_LOG_TAIL:], "seconds": seconds}
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc(), file=log)
            text = log.getvalue()
            print(text[-_LOG_TAIL:], flush=True)  # RunPod console Logs tab
            return {"error": f"{exc}\n{text[-3000:]}"}


def _local_test(raster_dir: str, weather_dir: str) -> None:
    from app.runpod_client import _bundle

    out = handler({"input": {
        "bundle_b64": base64.b64encode(_bundle(Path(raster_dir), Path(weather_dir))).decode("ascii"),
        "run_rh": True, "label": "local-test",
    }})
    _print_result(out)


def _local_test_wind(obj: str, speed: float, direction: float) -> None:
    from app.wind_ai import wind_vector

    ux, uy, uz = wind_vector(speed, direction)
    out = handler({"input": {
        "task": "wind", "obj_gz_b64": base64.b64encode(gzip.compress(Path(obj).read_bytes())).decode("ascii"),
        "ux": ux, "uy": uy, "uz": uz, "label": "local-test",
    }})
    _print_result(out)


def _print_result(out: dict) -> None:
    if "error" in out:
        sys.exit(out["error"])
    print(out["log"])
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(out["result_b64"])), mode="r:gz") as tar:
        print(json.dumps({"seconds": out["seconds"], "files": tar.getnames(), "extent": out.get("extent")},
                         indent=2))


if __name__ == "__main__":
    _warmup()
    if len(sys.argv) == 4 and sys.argv[1] == "--test":
        _local_test(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "--test-wind":
        _local_test_wind(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    else:
        import runpod

        runpod.serverless.start({"handler": handler})
