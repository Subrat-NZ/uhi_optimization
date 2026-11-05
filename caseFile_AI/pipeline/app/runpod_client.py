"""
RunPod Serverless client for the UHI worker (worker/handler.py: wind model + T model + RH model, one image).

Two job types go to the same endpoint, both as inline JSON (no S3):
    task "wind"  app/wind_ai.py -> run_job()        wind OBJ in, 2D wind field out (~1 MB)
    task "rht"   predict_t_rh() below, one timestep: T model, then RH model on that T field inside the same
                 worker (the T prediction never leaves it)

    this host                                         RunPod worker (models loaded once per container)
    raster2d/{geometry,U_2d,inlets} + 2 weather  -->  run_predict_T.py -> run_predict_RH.py
    tables, gzipped inline in the JSON (~1 MB)   <--  T/ + RH/ (png, json; ~0.4 MB) written into t_out / rh_out

.env:
    RUNPOD_API_KEY=...           RunPod API key (Bearer)
    RUNPOD_ENDPOINT_ID=...       the UHI endpoint (image jagathmcd/uhi-rht-worker)
    RUNPOD_TIMEOUT_SEC=900       one T+RH call incl. queue wait + cold start
    RUNPOD_RETURN_NPY=0          1 = also bring back the .npy fields (~2 MB more per call)

Cancelling: a submitted RunPod job never outlives the API job that started it.
    * exception / timeout / poll failure while waiting  -> call() cancels it
    * API job marked failed or cancelled meanwhile      -> should_stop hook stops the wait, call() cancels it
    * graceful shutdown (PM2 stop/restart, SIGTERM)     -> shutdown() cancels every job this process waits on
    * hard crash / SIGKILL / reboot                     -> the job id is in the job record (on_submit hook);
                                                           main.py cancels it on the next startup
    * this host never comes back                        -> RunPod drops the job itself: every job is submitted
                                                           with policy executionTimeout + ttl
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("rht_api.runpod")

API_BASE = os.getenv("RUNPOD_API_BASE", "https://api.runpod.ai/v2").rstrip("/")
TIMEOUT_SEC = float(os.getenv("RUNPOD_TIMEOUT_SEC", "900"))
RETURN_NPY = os.getenv("RUNPOD_RETURN_NPY", "0") == "1"

# RunPod /run accepts at most 10 MB of JSON; stay below it.
MAX_PAYLOAD_BYTES = 9 * 1024 * 1024
# 429/503 = request refused, safe to resend; anything else after submit is never resent (no duplicate GPU runs)
_RETRY_STATUSES = {429, 503}
_MAX_RETRIES = 3
_MAX_POLL_ERRORS = 5  # consecutive /status failures before giving up (and cancelling)
_TERMINAL_FAILED = {"FAILED", "CANCELLED", "TIMED_OUT"}
# raster2d files the T / RH scripts read; the rest (png previews, seasons/) stays here
_RASTER_FILES = ("geometry_labels.npy", "U_2d.npy", "U_flow_direction.json")
_WEATHER_TABLES = ("windVelocity", "atmosphericPressure")
# RunPod keeps a job this long past our own timeout (queue + run), then deletes it whatever its state
_TTL_MARGIN_SEC = 600


class RunpodError(RuntimeError):
    pass


class RunpodCancelled(RunpodError):
    """The wait was stopped on purpose (API job failed/cancelled, server shutting down)."""


class _JobEnded(RunpodError):
    """RunPod reports the job FAILED / CANCELLED / TIMED_OUT: nothing left to cancel."""


# ─── logging ────────────────────────────────────────────────────────────────────
def _emit(level: int, msg: str, log_path: Optional[Path] = None) -> None:
    """Server log (with the API job context) and, when given, the job's pipeline.log."""
    logger.log(level, msg)
    if log_path is None:
        return
    try:
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {logging.getLevelName(level)} [runpod] {msg}\n")
    except OSError as exc:
        logger.warning("Could not write to %s: %s", log_path, exc)


# ─── job tracking (so every way an API job ends also cancels its RunPod job) ────
_shutdown = threading.Event()
_active_lock = threading.Lock()
_active: Dict[str, str] = {}  # RunPod job id -> endpoint, for jobs this process is waiting on
_hooks = threading.local()


@contextlib.contextmanager
def job_hooks(on_submit: Callable[[str, str], None], on_settled: Callable[[str], None],
              should_stop: Callable[[], Optional[str]]):
    """
    For calls made by this thread inside the block:
        on_submit(runpod_id, endpoint)  right after submit (persist it, so a crash can still cancel it)
        on_settled(runpod_id)           once the RunPod job is finished or cancelled
        should_stop() -> reason | None  checked before submit and on every status poll
    """
    _hooks.current = (on_submit, on_settled, should_stop)
    try:
        yield
    finally:
        _hooks.current = None


def _check_stop() -> None:
    if _shutdown.is_set():
        raise RunpodCancelled("server is shutting down")
    hooks = getattr(_hooks, "current", None)
    reason = hooks[2]() if hooks else None
    if reason:
        raise RunpodCancelled(reason)


def cancel(job_id: str, endpoint: Optional[str] = None, reason: str = "unspecified", *,
           attempts: int = 3, timeout: float = 15, log_path: Optional[Path] = None) -> bool:
    """POST /cancel/{job_id}. True once RunPod accepted it or no longer knows the job (nothing left to stop)."""
    _emit(logging.WARNING, f"RunPod job {job_id}: cancel requested (reason: {reason})", log_path)
    err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            st = _request("POST", f"cancel/{job_id}", timeout=timeout, endpoint=endpoint)
            _emit(logging.WARNING, f"RunPod job {job_id}: cancelled (RunPod status: {st.get('status', 'unknown')})",
                  log_path)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                _emit(logging.INFO, f"RunPod job {job_id}: not found on RunPod, nothing to cancel", log_path)
                return True
            err = exc
        except (urllib.error.URLError, TimeoutError, OSError, RunpodError) as exc:
            err = exc
        if attempt < attempts:
            _emit(logging.WARNING, f"RunPod job {job_id}: cancel attempt {attempt}/{attempts} failed ({err}), "
                                   f"retrying in {2 * attempt}s", log_path)
            time.sleep(2 * attempt)
    _emit(logging.ERROR, f"RunPod job {job_id}: cancel failed after {attempts} attempt(s): {err}. "
                         "It stays on the job record and is retried at the next startup; RunPod stops it "
                         "at its execution timeout regardless", log_path)
    return False


def shutdown() -> int:
    """Stop all waits in this process and cancel their RunPod jobs (PM2 kills us soon). Returns count."""
    with _active_lock:
        jobs = list(_active)
    _shutdown.set()  # wakes every waiting thread at once; each cancels its own job in call()
    if not jobs:
        return 0
    logger.warning("Server shutdown: cancelling %d in-flight RunPod job(s): %s", len(jobs), ", ".join(jobs))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with _active_lock:
            if not _active:
                return len(jobs)
        time.sleep(0.1)
    # threads stuck in an HTTP call: cancel what is left from here, in parallel
    with _active_lock:
        left = list(_active.items())
    threads = [threading.Thread(target=cancel, args=(jid, ep, "server shutdown"),
                                kwargs={"attempts": 1, "timeout": 5}, daemon=True)
               for jid, ep in left]
    for t in threads:
        t.start()
    for t in threads:
        t.join(6)
    return len(jobs)


def config_problems() -> list:
    """Missing settings, for /health."""
    missing = [k for k in ("RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID") if not os.getenv(k, "").strip()]
    return [f"{', '.join(missing)} not set in .env"] if missing else []


# ─── bundle in / out ────────────────────────────────────────────────────────────
def _bundle(raster_dir: Path, weather_dir: Path) -> bytes:
    """tar.gz of what run_predict_T.py + run_predict_RH.py read, laid out as raster2d/ + weather/."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in _RASTER_FILES:
            if (raster_dir / name).is_file():
                tar.add(raster_dir / name, arcname=f"raster2d/{name}")
        for p in sorted(raster_dir.glob("*_inlet.npy")):
            tar.add(p, arcname=f"raster2d/{p.name}")
        # run_predict_T reads <weather_dir>/../date_and_time.xyz for the clock hours
        xyz = weather_dir.parent / "date_and_time.xyz"
        if xyz.is_file():
            tar.add(xyz, arcname="weather/date_and_time.xyz")
        for day in sorted(d for d in weather_dir.iterdir() if d.is_dir()):
            for table in _WEATHER_TABLES:
                if (day / table).is_file():
                    tar.add(day / table, arcname=f"weather/weatherData/{day.name}/{table}")
    return buf.getvalue()


def safe_extract(data: bytes, dest: Path) -> None:
    """Extract a tar.gz, refusing absolute paths / '..' (also used by worker/handler.py)."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise RunpodError(f"unsafe path in result archive: {m.name}")
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, **kw)


def _place_results(output: Dict[str, Any], t_out: Path, rh_out: Optional[Path]) -> None:
    """Unpack the worker's T/ and RH/ into t_out / rh_out; point paths in the summaries at the local copies."""
    with tempfile.TemporaryDirectory(prefix="runpod_result_") as tmp:
        safe_extract(base64.b64decode(output["result_b64"]), Path(tmp))
        remote = output.get("worker_dirs") or {}
        for sub, local in (("T", t_out), ("RH", rh_out)):
            src = Path(tmp) / sub
            if local is None or not src.is_dir():
                continue
            if local.exists():
                shutil.rmtree(local)
            shutil.copytree(src, local)
            for js in local.glob("*.json"):
                text = js.read_text()
                for rsub, rdir in remote.items():
                    target = {"T": t_out, "RH": rh_out}.get(rsub)
                    if rdir and target is not None:
                        text = text.replace(rdir, str(target))
                js.write_text(text)


# ─── RunPod HTTP ────────────────────────────────────────────────────────────────
def _request(method: str, path: str, body: Optional[dict] = None, timeout: float = 60,
             endpoint: Optional[str] = None) -> Dict[str, Any]:
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    endpoint = (endpoint or os.getenv("RUNPOD_ENDPOINT_ID", "")).strip()
    if not api_key or not endpoint:
        raise RunpodError("RunPod is not configured in .env: set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
    req = urllib.request.Request(
        f"{API_BASE}/{endpoint}/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _start_job(payload: dict, label: str, timeout_sec: float, endpoint: str,
               log_path: Optional[Path]) -> str:
    # RunPod-side limits: the job dies there even if this host is gone and never cancels it
    policy = {"executionTimeout": max(5000, int(timeout_sec * 1000)),
              "ttl": int((timeout_sec + _TTL_MARGIN_SEC) * 1000)}
    for attempt in range(1, _MAX_RETRIES + 1):
        _check_stop()
        try:
            job = _request("POST", "run", {"input": payload, "policy": policy}, endpoint=endpoint)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                wait = 10 * attempt
                _emit(logging.WARNING, f"{label}: submit rejected with HTTP {exc.code} "
                                       f"(attempt {attempt}/{_MAX_RETRIES}), retrying in {wait}s", log_path)
                time.sleep(wait)
                continue
            msg = f"{label}: RunPod submit failed with HTTP {exc.code}: {text}"
            _emit(logging.ERROR, msg, log_path)
            raise RunpodError(msg) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            msg = f"{label}: RunPod submit failed: {exc}"
            _emit(logging.ERROR, msg, log_path)
            raise RunpodError(msg) from exc
        if not job.get("id"):
            msg = f"{label}: RunPod accepted the request but returned no job id: {str(job)[:300]}"
            _emit(logging.ERROR, msg, log_path)
            raise RunpodError(msg)
        return str(job["id"])
    raise RunpodError(f"{label}: RunPod submit failed after {_MAX_RETRIES} attempts")


def _wait_for_job(job_id: str, timeout_sec: float, endpoint: str, log_path: Optional[Path]) -> Dict[str, Any]:
    """Poll /status every 1 s, backing off to 4 s, until done; raises on failure, stop or timeout_sec."""
    deadline = time.monotonic() + timeout_sec
    delay, errors, last = 1.0, 0, None
    while time.monotonic() < deadline:
        _shutdown.wait(delay)
        _check_stop()
        delay = min(delay * 1.5, 4.0)
        try:
            st = _request("GET", f"status/{job_id}", timeout=30, endpoint=endpoint)
            errors = 0
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # HTTPError is a URLError
            errors += 1
            if errors >= _MAX_POLL_ERRORS:
                raise RunpodError(f"RunPod job {job_id}: status check failed {errors} times in a row: {exc}") from exc
            _emit(logging.WARNING, f"RunPod job {job_id}: status check failed "
                                   f"({errors}/{_MAX_POLL_ERRORS}): {exc}", log_path)
            continue
        status = st.get("status")
        if status != last:
            _emit(logging.INFO, f"RunPod job {job_id}: status {last or 'SUBMITTED'} -> {status}", log_path)
            last = status
        if status == "COMPLETED":
            return st
        if status in _TERMINAL_FAILED:
            detail = str(st.get("error") or st.get("output") or "no details")[-1500:]
            raise _JobEnded(f"RunPod job {job_id}: ended with status {status}: {detail}")
    raise RunpodError(f"RunPod job {job_id}: no result within the {int(timeout_sec)}s timeout")


def call(payload: dict, log_path: Path, timeout_sec: float, label: str,
         endpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    One worker job: submit, wait, check the worker's answer, append its log to log_path.
    Returns the worker output ({"ok": true, "result_b64": ..., ...}); raises RunpodError otherwise.
    Whatever ends the wait early (error, timeout, stop, shutdown) cancels the RunPod job first.
    """
    size = len(json.dumps(payload))
    if size > MAX_PAYLOAD_BYTES:
        msg = (f"{label}: RunPod input is {size / 1e6:.1f} MB, over the "
               f"{MAX_PAYLOAD_BYTES / 1e6:.0f} MB limit")
        _emit(logging.ERROR, msg, log_path)
        raise RunpodError(msg)
    endpoint = (endpoint or os.getenv("RUNPOD_ENDPOINT_ID", "")).strip()
    started = time.monotonic()
    hooks = getattr(_hooks, "current", None)
    job_id = _start_job(payload, label, timeout_sec, endpoint, log_path)
    with _active_lock:
        _active[job_id] = endpoint
    _emit(logging.INFO, f"RunPod job {job_id}: started ({label}, task={payload.get('task')}, "
                        f"endpoint={endpoint}, input {size / 1e6:.2f} MB, timeout {int(timeout_sec)}s)", log_path)
    settled = False
    try:
        if hooks:
            hooks[0](job_id, endpoint)
        st = _wait_for_job(job_id, timeout_sec, endpoint, log_path)
        settled = True
    except _JobEnded as exc:
        settled = True
        _emit(logging.ERROR, str(exc), log_path)
        raise
    except RunpodCancelled as exc:
        settled = cancel(job_id, endpoint, str(exc), log_path=log_path)
        raise
    except BaseException as exc:
        reason = (str(exc) or type(exc).__name__).removeprefix(f"RunPod job {job_id}: ")
        _emit(logging.ERROR, f"RunPod job {job_id}: {reason}", log_path)
        settled = cancel(job_id, endpoint, reason, log_path=log_path)
        raise
    finally:
        with _active_lock:
            _active.pop(job_id, None)
        if settled and hooks:
            try:
                hooks[1](job_id)
            except Exception:  # noqa: BLE001
                logger.exception("RunPod job %s: could not clear it from the job record", job_id)
    output = st.get("output") or {}
    if not isinstance(output, dict) or not output.get("ok") or not output.get("result_b64"):
        msg = f"RunPod job {job_id}: completed without a usable result: {str(output)[-1500:]}"
        _emit(logging.ERROR, msg, log_path)
        raise RunpodError(msg)
    if output.get("log"):
        with open(log_path, "a") as f:
            f.write(output["log"].rstrip() + "\n")
    _emit(logging.INFO, f"RunPod job {job_id}: completed in {time.monotonic() - started:.1f}s "
                        f"(queue {st.get('delayTime', 0) / 1000:.1f}s, "
                        f"execution {st.get('executionTime', 0) / 1000:.1f}s)", log_path)
    return output


# ─── T + RH for one timestep ────────────────────────────────────────────────────
def predict_t_rh(
    raster_dir: Path,
    weather_dir: Path,
    t_out: Path,
    rh_out: Optional[Path],
    log_path: Path,
    label: str = "",
) -> Dict[str, Any]:
    """
    run_predict_T.py (+ run_predict_RH.py when rh_out is given) on the worker: afterwards t_out / rh_out hold
    predictions_summary.json / predictions_RH_summary.json / PNGs exactly as those scripts write them.
    """
    payload = {
        "task": "rht",
        "bundle_b64": base64.b64encode(_bundle(raster_dir, weather_dir)).decode("ascii"),
        "run_rh": rh_out is not None,
        "return_npy": RETURN_NPY,
        "label": label,
    }
    output = call(payload, log_path, TIMEOUT_SEC, f"T+RH {label or raster_dir.name}")
    _place_results(output, t_out, rh_out)
    return output
