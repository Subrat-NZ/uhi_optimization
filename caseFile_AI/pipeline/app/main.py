"""
RH_T_API (UHI Instant) — FastAPI service: scene JSON + EPW + wind OBJ -> T and RH maps of the hottest day.

Inputs (POST /rhtAPI, multipart):
  * scene_json      : .json file upload            (or scene_json_str text field)
  * epw_file        : .epw file upload
  * wind_obj        : wind-model .obj file upload
  * wind_direction  : Summer wind direction, deg (0-360)
  * wind_speed      : Summer wind speed, m/s (> 0)
  * userid          : text                         (required)
  * project_id      : text                         (optional)
  * model_id        : text                         (optional)
  * X-API-Key       : header                       (required)

Flow:
  POST /rhtAPI                              -> validates, saves inputs, returns 202 {job_id, status:"pending"}
  GET  /rhtAPI/status/{job_id}              -> pending | running | completed | failed
  POST /rhtAPI/cancel/{job_id}              -> stop a pending/running job (status -> failed, RunPod job cancelled)
  GET  /rhtAPI/result/{job_id}/image/{name} -> <HHMM>_T.png / <HHMM>_RH.png
  GET  /rhtAPI/project/{userid}/{project_id}/inference -> T/RH inference for the latest project run

The job itself is app/pipeline.py: CPU preparation here, wind + T + RH models on one RunPod endpoint.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fcntl
import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader

from app import pipeline, runpod_client
from app.job_safety import cleanup_old_runs, owner_alive, process_identity
from app.log_context import clear_log_context, set_log_context, setup_logging
from inference_generator import generate_report as generate_inference_report


# ─── Config (from environment / .env loaded by deploy_aws.sh or start_api.sh) ───
API_KEY = os.getenv("API_KEY", "")
APP_ROOT = Path(__file__).resolve().parent.parent
RUNS_BASE_DIR = Path(os.getenv("RUNS_BASE_DIR", str(APP_ROOT / "runs")))
MAX_PARALLEL_JOBS = int(os.getenv("MAX_PARALLEL_JOBS", "2"))
# A job still pending/running after this long is marked failed (backend stops waiting after 3 h as well)
JOB_MAX_SEC = int(os.getenv("JOB_MAX_SEC", "10800"))
PREPARE_TIMEOUT_SEC = int(os.getenv("PREPARE_TIMEOUT_SEC", "1800"))
HEALTH_MIN_FREE_GB = float(os.getenv("HEALTH_MIN_FREE_GB", "2"))

RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)

setup_logging()  # job/user/project/model/status on every line, errors in red (app/log_context.py)
logger = logging.getLogger("rht_api")

app = FastAPI(title="RH_T_API", version="1.0.0")

# Optional CORS: set CORS_ORIGINS="http://host1:3000,http://host2" in .env if a
# browser frontend calls this API directly. Off by default.
_cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors,
        allow_methods=["*"],
        allow_headers=["*"],
    )

_JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_PARALLEL_JOBS, thread_name_prefix="rht-job"
)

# ─── Authentication (same as UHI API: X-API-Key header) ─────────────────────────
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(x_api_key: Optional[str] = Depends(_api_key_scheme)) -> str:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ─── Input validation (copied from the UHI API) ─────────────────────────────────
_EPW_MAX_BYTES = int(float(os.getenv("EPW_MAX_MB", "10")) * 1024 * 1024)
_JSON_MAX_BYTES = int(float(os.getenv("JSON_MAX_MB", "20")) * 1024 * 1024)
_OBJ_MAX_BYTES = int(float(os.getenv("OBJ_MAX_MB", "100")) * 1024 * 1024)
_USERID_MAX_LEN = 64
_USERID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_ENTITY_ID_MAX_LEN = 128
_ENTITY_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_userid(userid: str) -> Optional[str]:
    s = userid.strip()
    if not s:
        return "userid must not be empty"
    if len(s) > _USERID_MAX_LEN:
        return f"userid must be at most {_USERID_MAX_LEN} characters"
    if not _USERID_RE.match(s):
        return "userid may only contain alphanumeric characters, hyphens and underscores"
    return None


def _normalize_optional_entity_id(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if len(s) > _ENTITY_ID_MAX_LEN:
        raise ValueError(f"{field_name} must be at most {_ENTITY_ID_MAX_LEN} characters")
    if not _ENTITY_ID_RE.match(s):
        raise ValueError(
            f"{field_name} may only contain alphanumeric characters, hyphens and underscores"
        )
    return s


def _validate_epw(contents: bytes, filename: str) -> Optional[str]:
    if not filename.lower().endswith(".epw"):
        return "epw_file must be an EnergyPlus Weather (.epw) file"
    if len(contents) == 0:
        return "epw_file is empty"
    if len(contents) > _EPW_MAX_BYTES:
        return f"epw_file exceeds the {_EPW_MAX_BYTES // 1_048_576} MB size limit"
    try:
        first_line = contents.split(b"\n")[0].decode("utf-8", errors="replace").strip()
    except Exception:
        return "epw_file could not be decoded as text"
    if not first_line.upper().startswith("LOCATION"):
        return (
            "epw_file does not appear to be a valid EnergyPlus Weather file "
            "(expected LOCATION record on first line)"
        )
    return None


def _normalize_epw_encoding(contents: bytes) -> bytes:
    """Return the EPW as UTF-8.

    Many EPW files are Latin-1/cp1252 (e.g. a location name like "Lomé"). _validate_epw decodes with
    errors="replace", so they pass validation, but pvlib.read_epw in extractWeatherData.py opens the
    file as strict UTF-8 and crashes. Valid UTF-8 is returned unchanged (a BOM is dropped).
    """
    if contents.startswith(b"\xef\xbb\xbf"):
        contents = contents[3:]
    try:
        contents.decode("utf-8")
        return contents
    except UnicodeDecodeError:
        pass
    for encoding in ("cp1252", "latin-1"):
        try:
            return contents.decode(encoding).encode("utf-8")
        except UnicodeDecodeError:
            continue
    return contents


def _validate_scene_json(raw: str, data: Any) -> Optional[str]:
    if len(raw.encode("utf-8", errors="replace")) > _JSON_MAX_BYTES:
        return f"scene_json exceeds the {_JSON_MAX_BYTES // (1024 * 1024)} MB size limit"
    if not isinstance(data, dict):
        return "scene_json must be a JSON object"

    # Supports both the flat format and the wrapped 'siteOnly_json' envelope
    payload = data.get("siteOnly_json", data)
    if not isinstance(payload, dict):
        return "scene_json 'siteOnly_json' must be an object"

    buildings = payload.get("buildings")
    if buildings is None:
        return "scene_json must contain a 'buildings' key"
    if not isinstance(buildings, list):
        return "scene_json 'buildings' must be an array"
    if len(buildings) == 0:
        return "scene_json 'buildings' array must not be empty"
    return None


# ─── Response helpers / error handlers (same shape as the UHI API) ──────────────
def _err(message: str, status_code: int = 500) -> JSONResponse:
    if status_code >= 500:
        logger.error("request failed (%s): %s", status_code, message)
    else:
        logger.warning("request rejected (%s): %s", status_code, message)
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    log = logger.error if exc.status_code >= 500 else logger.warning
    log("%s %s -> %s: %s", request.method, request.url.path, exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    message = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    logger.warning("%s %s -> 422: %s", request.method, request.url.path, message)
    return JSONResponse(status_code=422, content={"status": "error", "message": message})


# ─── Job records (JSON file per job, so it works with multiple uvicorn workers) ─
_job_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str, userid: str, project_id: Optional[str]) -> Path:
    return RUNS_BASE_DIR / userid / (project_id or "default") / job_id


def _job_index_path(job_id: str) -> Path:
    # Small index so status lookups only need the job_id
    return RUNS_BASE_DIR / "_jobs" / f"{job_id}.json"


def _write_job(record: Dict[str, Any]) -> None:
    path = _job_index_path(record["job_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with _job_lock:
        tmp.write_text(json.dumps(record, default=str))
        tmp.replace(path)  # atomic


def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_index_path(job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


_TERMINAL = {"completed", "failed"}


class _job_file_lock:
    """Lock one job record across all uvicorn workers (read-modify-write of the JSON file)."""

    def __init__(self, job_id: str) -> None:
        self.path = _job_index_path(job_id).with_suffix(".lock")

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()  # file is kept: unlinking it would let two workers lock different inodes


def _update_job(job_id: str, status: str, **fields: Any) -> None:
    with _job_file_lock(job_id):
        record = _read_job(job_id) or {"job_id": job_id}
        if record.get("status") in _TERMINAL:
            # e.g. marked failed by the timeout / dead-worker check while this thread was still running
            set_log_context(job_id=job_id, status=record.get("status"))
            logger.warning("job already %s, ignoring status -> %s", record.get("status"), status)
            return
        record.update(status=status, updated_at=_now(), **fields)
        _write_job(record)
    set_log_context(job_id=job_id, userid=record.get("userid"), project_id=record.get("project_id"),
                    model_id=record.get("model_id"), status=status)
    if status == "failed":
        logger.error("job status -> failed: %s", fields.get("error"))
        # whoever failed it (timeout, dead worker, cancel request), stop its RunPod work too
        first_line = (str(fields.get("error") or "").strip().splitlines() or ["no error given"])[0][:200]
        _cancel_runpod_jobs(record, reason=f"API job failed: {first_line}")
    else:
        logger.info("job status -> %s", status)


# ─── RunPod jobs of an API job (record["runpod_jobs"] = {runpod_id: endpoint}) ──
def _set_runpod_job(job_id: str, runpod_id: str, endpoint: Optional[str]) -> None:
    """Add (endpoint given) or remove (None) one in-flight RunPod job, whatever the API job's status."""
    with _job_file_lock(job_id):
        record = _read_job(job_id)
        if record is None:
            return
        jobs = dict(record.get("runpod_jobs") or {})
        if endpoint is None:
            jobs.pop(runpod_id, None)
        else:
            jobs[runpod_id] = endpoint
        record["runpod_jobs"] = jobs
        _write_job(record)


def _cancel_runpod_jobs(record: Dict[str, Any], reason: str) -> int:
    """Cancel the RunPod jobs left on a record; ones RunPod would not confirm stay for the next startup.
    Returns how many are still left."""
    left = 0
    for runpod_id, endpoint in (record.get("runpod_jobs") or {}).items():
        try:
            if runpod_client.cancel(runpod_id, endpoint, reason):
                _set_runpod_job(record["job_id"], runpod_id, None)
                continue
        except Exception:  # noqa: BLE001
            logger.exception("RunPod job %s: unexpected error while cancelling", runpod_id)
        left += 1
    return left


def _stop_reason(job_id: str) -> Optional[str]:
    """For the pipeline thread: stop if the job was failed/cancelled from elsewhere meanwhile."""
    record = _read_job(job_id) or {}
    if record.get("status") in _TERMINAL:
        return f"job is already {record['status']}: {record.get('error') or ''}".rstrip(": ")
    return None


def _stale_reason(record: Dict[str, Any]) -> Optional[str]:
    """Why a pending/running job can never finish (worker gone / over JOB_MAX_SEC), else None."""
    if record.get("status") in _TERMINAL:
        return None
    if not owner_alive(record):
        return "server restarted while the job was running, please resubmit"
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["created_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None
    if age > JOB_MAX_SEC:
        return f"job timed out after {JOB_MAX_SEC // 60} minutes"
    return None


def _reconcile_job(record: Dict[str, Any]) -> Dict[str, Any]:
    """Mark a stuck job failed so pollers get an answer instead of waiting forever."""
    reason = _stale_reason(record)
    if reason:
        _update_job(record["job_id"], "failed", error=reason)
        record = _read_job(record["job_id"]) or record
    return record


def _client_error(record: Dict[str, Any]) -> str:
    """Short error for API clients. Full error + log tail stays in the job record and the server log."""
    first_line = str(record.get("error") or "Job failed").strip().splitlines()[0].rstrip(":")
    return f"{first_line[:300]} (job_id={record.get('job_id')}, see server logs)"


def _job_status_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "job_id": record.get("job_id"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }
    if record.get("status") == "completed":
        payload["result"] = record.get("result")
    elif record.get("status") == "failed":
        payload["error"] = _client_error(record)
    return payload


def _run_job(
    job_id: str,
    run_dir: Path,
    scene_json_path: Path,
    epw_path: Path,
    scene_data: Dict[str, Any],
    userid: str,
    project_id: Optional[str],
    model_id: Optional[str],
    wind: Dict[str, Any],
) -> None:
    """Background thread: app/pipeline.py, then the job record -> completed (with result) or failed."""
    # worker threads do not inherit the request's log context -> set it again for this job
    clear_log_context()
    set_log_context(job_id=job_id, userid=userid, project_id=project_id, model_id=model_id)
    started = time.monotonic()
    try:
        _update_job(job_id, "running")
        if (_read_job(job_id) or {}).get("status") != "running":
            return  # already failed (timed out in the queue) -> do not start the pipeline
        with runpod_client.job_hooks(
            on_submit=lambda rid, ep: _set_runpod_job(job_id, rid, ep),
            on_settled=lambda rid: _set_runpod_job(job_id, rid, None),
            should_stop=lambda: _stop_reason(job_id),
        ):
            rht = pipeline.run(
                run_dir=run_dir, scene_json_path=scene_json_path, epw_path=epw_path,
                wind_obj_path=Path(wind["obj_path"]), wind_direction=wind["wind_direction"],
                wind_speed=wind["wind_speed"], app_root=APP_ROOT, prepare_timeout=PREPARE_TIMEOUT_SEC,
            )
        payload = scene_data.get("siteOnly_json", scene_data)
        result = {
            "message": "multi-season wind -> T/RH completed",
            "mode": "multi_season",  # unchanged result schema (the frontend reads rht.times[*])
            "building_count": len(payload.get("buildings", [])),
            "rht": rht,
            "run_dir": str(run_dir),
        }
        _update_job(job_id, "completed", result=result)
        logger.info("job completed in %.1fs (run_dir=%s)", time.monotonic() - started, run_dir)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        try:
            _update_job(job_id, "failed", error=str(exc))
        except Exception:  # noqa: BLE001
            set_log_context(status="failed")
            logger.exception("could not write failed status for job")
        logger.error("job failed after %.1fs (run_dir=%s): %s\n%s",
                     time.monotonic() - started, run_dir, exc, tb)
    finally:
        clear_log_context()


# ─── Startup: fail jobs whose worker died, delete old runs ───────────────────────
def _startup_sweep() -> None:
    stuck = leftover = 0
    for path in sorted((RUNS_BASE_DIR / "_jobs").glob("*.json")):
        try:
            record = json.loads(path.read_text())
            if _stale_reason(record):
                _reconcile_job(record)  # -> failed, which also cancels its RunPod jobs
                stuck += 1
            elif record.get("status") in _TERMINAL and record.get("runpod_jobs"):
                # a cancel RunPod did not confirm last time
                set_log_context(job_id=record.get("job_id"), status=record.get("status"))
                logger.warning("retrying cancel of %d RunPod job(s) left on a %s job",
                               len(record["runpod_jobs"]), record.get("status"))
                leftover += _cancel_runpod_jobs(record, reason="left over from a previous run")
        except Exception as exc:  # noqa: BLE001
            logger.warning("startup check skipped %s: %s", path.name, exc)
        finally:
            clear_log_context()
    if stuck:
        logger.warning("marked %d stuck job(s) as failed at startup", stuck)
    if leftover:
        logger.error("%d RunPod job(s) could not be cancelled at startup; RunPod stops them at their "
                     "execution timeout", leftover)
    cleanup_old_runs(RUNS_BASE_DIR)


def _daily_cleanup() -> None:
    while True:
        time.sleep(86400)
        try:
            cleanup_old_runs(RUNS_BASE_DIR)
        except Exception:  # noqa: BLE001
            logger.exception("daily run cleanup failed")


@app.on_event("startup")
async def _on_startup() -> None:
    threading.Thread(target=_startup_sweep, name="rht-startup-sweep", daemon=True).start()
    threading.Thread(target=_daily_cleanup, name="rht-cleanup", daemon=True).start()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    # PM2 stop/restart, deploy, SIGTERM: stop the job threads' waits and cancel their RunPod jobs now
    runpod_client.shutdown()


# ─── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """ok -> 200. Missing RunPod settings, unwritable or full runs dir -> 503 (PM2 / deploy / monitoring see it)."""
    problems = list(runpod_client.config_problems())  # the models run on RunPod, not on this host
    if not os.access(RUNS_BASE_DIR, os.W_OK):
        problems.append("runs dir is not writable")
    free_gb = shutil.disk_usage(RUNS_BASE_DIR).free / 1e9
    if free_gb < HEALTH_MIN_FREE_GB:
        problems.append(f"only {free_gb:.1f} GB disk free")
    if problems:
        logger.error("health check failed: %s", "; ".join(problems))
        return JSONResponse(status_code=503, content={"status": "error", "problems": problems})
    return {"status": "ok", "disk_free_gb": round(free_gb, 1)}


@app.post("/rhtAPI")
async def rht_api(
    scene_json: Optional[UploadFile] = File(
        None,
        description="Scene JSON (.json) file with buildings geometry.",
    ),
    scene_json_str_field: Optional[str] = Form(
        None,
        alias="scene_json_str",
        description="Scene JSON as a raw JSON string (alternative to uploading a file).",
    ),
    epw_file: UploadFile = File(..., description="EnergyPlus Weather (.epw) file"),
    wind_obj: Optional[UploadFile] = File(None, description="Wind-model OBJ (.obj) of the scene."),
    wind_direction: Optional[float] = Form(None, description="Summer wind direction, deg (0-360)."),
    wind_speed: Optional[float] = Form(None, description="Summer wind speed, m/s (> 0)."),
    userid: str = Form(..., description="User identifier"),
    project_id: Optional[str] = Form(None, description="Optional project identifier."),
    model_id: Optional[str] = Form(None, description="Optional model identifier."),
    _key: str = Depends(_require_api_key),
):
    # ── userid / project_id / model_id ────────────────────────────────────────
    clear_log_context()
    set_log_context(userid=(userid or "").strip() or None, project_id=project_id, model_id=model_id)
    err = _validate_userid(userid)
    if err:
        return _err(err, 400)
    userid = userid.strip()
    try:
        project_id = _normalize_optional_entity_id(project_id, "project_id")
        model_id = _normalize_optional_entity_id(model_id, "model_id")
    except ValueError as exc:
        return _err(str(exc), 400)

    # ── wind: OBJ + Summer wind (the wind model runs on RunPod, app/wind_ai.py) ─
    if wind_obj is None:
        return _err("wind_obj is required (wind_request_id / vtk_s3_key / wind_requests are no longer "
                    "supported: the wind model runs inside this API's pipeline)", 400)
    if not (wind_obj.filename or "").lower().endswith(".obj"):
        return _err("wind_obj must be an .obj file", 400)
    wind_obj_bytes = await wind_obj.read()
    if len(wind_obj_bytes) == 0:
        return _err("wind_obj is empty", 400)
    if len(wind_obj_bytes) > _OBJ_MAX_BYTES:
        return _err(f"wind_obj exceeds the {_OBJ_MAX_BYTES // (1024 * 1024)} MB size limit", 400)
    if wind_direction is None or wind_speed is None:
        return _err("wind_direction and wind_speed are required", 400)
    if not (0 <= wind_direction <= 360):
        return _err("wind_direction must be between 0 and 360", 400)
    if wind_speed <= 0:
        return _err("wind_speed must be greater than 0", 400)

    # ── scene_json: file upload OR raw string field ───────────────────────────
    if scene_json is not None:
        scene_json_filename = scene_json.filename or "scene.json"
        if not scene_json_filename.lower().endswith(".json"):
            return _err("scene_json must be a JSON (.json) file", 400)
        scene_json_bytes = await scene_json.read()
        if len(scene_json_bytes) == 0:
            return _err("scene_json file is empty", 400)
        if len(scene_json_bytes) > _JSON_MAX_BYTES:
            return _err(
                f"scene_json exceeds the {_JSON_MAX_BYTES // (1024 * 1024)} MB size limit", 400
            )
        try:
            scene_json_str = scene_json_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return _err("scene_json file could not be decoded as UTF-8", 400)
    elif scene_json_str_field is not None:
        scene_json_str = scene_json_str_field
        if len(scene_json_str.encode("utf-8")) > _JSON_MAX_BYTES:
            return _err(
                f"scene_json exceeds the {_JSON_MAX_BYTES // (1024 * 1024)} MB size limit", 400
            )
    else:
        return _err(
            "scene_json is required: provide it as a file upload or as the 'scene_json_str' form field",
            400,
        )

    try:
        scene_data = json.loads(scene_json_str)
    except json.JSONDecodeError as exc:
        return _err(f"scene_json is not valid JSON: {exc}", 400)
    err = _validate_scene_json(scene_json_str, scene_data)
    if err:
        return _err(err, 400)

    # ── epw_file ──────────────────────────────────────────────────────────────
    epw_contents = await epw_file.read()
    epw_filename = epw_file.filename or "weather.epw"
    err = _validate_epw(epw_contents, epw_filename)
    if err:
        return _err(err, 400)
    epw_contents = _normalize_epw_encoding(epw_contents)

    # ── Save inputs + create job record ───────────────────────────────────────
    job_id = str(uuid.uuid4())
    run_dir = _job_dir(job_id, userid, project_id)
    input_dir = run_dir / "input"
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        scene_json_path = input_dir / "scene.json"
        epw_path = input_dir / "weather.epw"
        scene_json_path.write_text(scene_json_str, encoding="utf-8")
        epw_path.write_bytes(epw_contents)
        wind_obj_path = input_dir / "wind.obj"
        wind_obj_path.write_bytes(wind_obj_bytes)
        wind = {
            "obj_path": str(wind_obj_path),
            "wind_direction": float(wind_direction),
            "wind_speed": float(wind_speed),
        }

        created = _now()
        _write_job(
            {
                "job_id": job_id,
                "status": "pending",
                "userid": userid,
                "project_id": project_id,
                "model_id": model_id,
                "epw_filename": epw_filename,
                "summer_wind": wind,
                "run_dir": str(run_dir),
                **process_identity(),
                "created_at": created,
                "updated_at": created,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"Unable to create job: {exc}", 503)

    # ── Hand off to background thread ─────────────────────────────────────────
    try:
        _JOB_EXECUTOR.submit(
            _run_job,
            job_id, run_dir, scene_json_path, epw_path, scene_data, userid, project_id, model_id, wind,
        )
    except Exception as exc:  # noqa: BLE001
        _update_job(job_id, "failed", error=f"Unable to submit job: {exc}")
        return _err(f"Unable to submit job: {exc}", 503)

    set_log_context(job_id=job_id, status="pending")
    logger.info("job submitted (wind dir=%s speed=%s)", wind_direction, wind_speed)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending"})


@app.get("/rhtAPI/status/{job_id}", summary="Get job status")
async def get_job_status(job_id: str, _key: str = Depends(_require_api_key)):
    job_id = job_id.strip()
    set_log_context(job_id=job_id or None)
    if not job_id:
        return _err("job_id must not be empty", 400)
    # job_id is a uuid; reject anything else so it can't be used to build odd paths
    try:
        uuid.UUID(job_id)
    except ValueError:
        return _err(f"No job found for job_id {job_id}", 404)
    try:
        record = _read_job(job_id)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Unable to read job status: {exc}", 503)
    if not record:
        return _err(f"No job found for job_id {job_id}", 404)
    try:
        record = _reconcile_job(record)
    except Exception:  # noqa: BLE001
        logger.exception("could not check job for timeout / dead worker")
    return JSONResponse(status_code=200, content=_job_status_payload(record))


@app.post("/rhtAPI/cancel/{job_id}", summary="Cancel a pending / running job")
async def cancel_job(job_id: str, _key: str = Depends(_require_api_key)):
    job_id = job_id.strip()
    set_log_context(job_id=job_id or None)
    try:
        uuid.UUID(job_id)
    except ValueError:
        return _err(f"No job found for job_id {job_id}", 404)
    record = _read_job(job_id)
    if not record:
        return _err(f"No job found for job_id {job_id}", 404)
    if record.get("status") in _TERMINAL:
        return _err(f"Job is already {record.get('status')}", 409)
    set_log_context(userid=record.get("userid"), project_id=record.get("project_id"),
                    model_id=record.get("model_id"), status=record.get("status"))
    logger.warning("cancel requested by API client (%d RunPod job(s) in flight: %s)",
                   len(record.get("runpod_jobs") or {}), ", ".join(record.get("runpod_jobs") or {}) or "none")
    # -> failed + RunPod cancel; the job thread sees it on its next poll and submits nothing more
    await asyncio.to_thread(_update_job, job_id, "failed", error="cancelled by request")
    return JSONResponse(status_code=200, content=_job_status_payload(_read_job(job_id) or record))


@app.get("/rhtAPI/result/{job_id}/image/{name}", summary="Download a T / RH result image")
async def get_result_image(job_id: str, name: str, _key: str = Depends(_require_api_key)):
    set_log_context(job_id=job_id.strip() or None)
    try:
        uuid.UUID(job_id.strip())
    except ValueError:
        return _err("No job found", 404)
    if not pipeline.IMAGE_NAME_RE.match(name):
        return _err("Invalid image name", 400)
    record = _read_job(job_id.strip())
    if not record or record.get("status") != "completed" or not record.get("run_dir"):
        return _err("No completed job found", 404)
    path = Path(record["run_dir"]) / "output" / "rht_results" / name
    if not path.is_file():
        return _err("Image not found", 404)
    return FileResponse(str(path), media_type="image/png")


@app.get(
    "/rhtAPI/project/{userid}/{project_id}/inference",
    summary="Get T / RH inference for the latest project run",
)
async def get_project_inference(
    userid: str,
    project_id: str,
    _key: str = Depends(_require_api_key),
):
    userid_error = _validate_userid(userid)
    if userid_error:
        return _err(userid_error, 400)
    try:
        project_id = _normalize_optional_entity_id(project_id, "project_id")
    except ValueError as exc:
        return _err(str(exc), 400)
    if project_id is None:
        return _err("project_id must not be empty", 400)

    userid = userid.strip()
    set_log_context(userid=userid, project_id=project_id)
    project_dir = RUNS_BASE_DIR / userid / project_id
    if not project_dir.is_dir():
        return _err("No job found for this project", 404)

    try:
        records = []
        for job_dir in project_dir.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                uuid.UUID(job_dir.name)
            except ValueError:
                continue
            record = _read_job(job_dir.name)
            if (record and record.get("userid") == userid
                    and record.get("project_id") == project_id):
                records.append(record)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("could not look up project inference")
        return _err(f"Unable to read project jobs: {exc}", 503)

    if not records:
        return _err("No job found for this project", 404)
    latest = max(records, key=lambda record: record.get("created_at") or "")
    set_log_context(job_id=latest.get("job_id"), status=latest.get("status"))
    if latest.get("status") != "completed":
        return JSONResponse(
            status_code=409,
            content={
                "status": latest.get("status"),
                "message": "The latest project run is not completed yet.",
                "job_id": latest.get("job_id"),
            },
        )

    summary_path = (
        _job_dir(latest["job_id"], userid, project_id) / "output" / "rht_results" / "summary.json"
    )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("Project summary must be a JSON object")
        inference = generate_inference_report(summary)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.exception("could not generate project inference")
        return _err(f"Unable to generate inference: {exc}", 503)

    if inference.startswith("No valid temperature"):
        return _err("Completed project summary contains no valid T/RH prediction data", 503)
    return {
        "status": "completed",
        "userid": userid,
        "project_id": project_id,
        "job_id": latest.get("job_id"),
        "inference": inference,
    }