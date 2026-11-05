"""
Production guards for main.py.

  * process_identity()  -- pid + /proc start time of this worker; stored on each job so a job whose worker died
    owner_alive()          (PM2 restart, deploy, crash) can be detected and marked failed instead of hanging.
  * cleanup_old_runs()  -- delete run folders older than RUNS_RETENTION_DAYS (0 = keep forever, the default).

.env:
    RUNS_RETENTION_DAYS=0      delete runs/<user>/... older than N days (0 = never)
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("rht_api.safety")

RUNS_RETENTION_DAYS = float(os.getenv("RUNS_RETENTION_DAYS", "0"))


# ─── job owner (which worker process runs the job) ──────────────────────────────
def _proc_start_time(pid: int) -> Optional[str]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # field 22 (starttime); comm (field 2) may contain spaces, so split after the closing ')'
    return stat.rsplit(")", 1)[1].split()[19]


def process_identity() -> Dict[str, Any]:
    pid = os.getpid()
    return {"owner_host": socket.gethostname(), "owner_pid": pid, "owner_start": _proc_start_time(pid)}


def owner_alive(record: Dict[str, Any]) -> bool:
    """False only when we are sure the worker that owns this job is gone (same host, pid dead or reused)."""
    pid = record.get("owner_pid")
    if not pid or record.get("owner_host") != socket.gethostname():
        return True  # old record or another machine: cannot tell, assume alive
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    start = record.get("owner_start")
    return not start or _proc_start_time(int(pid)) == start


# ─── retention ──────────────────────────────────────────────────────────────────
def cleanup_old_runs(runs_base: Path, retention_days: float = RUNS_RETENTION_DAYS) -> int:
    """Remove runs/<user>/<project>/<job> folders + job records older than retention_days. Returns count."""
    if retention_days <= 0 or not runs_base.is_dir():
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for job_dir in runs_base.glob("*/*/*"):
        if job_dir.parts[-3].startswith("_") or not job_dir.is_dir():
            continue
        try:
            if job_dir.stat().st_mtime < cutoff:
                shutil.rmtree(job_dir)
                for suffix in (".json", ".lock"):
                    (runs_base / "_jobs" / f"{job_dir.name}{suffix}").unlink(missing_ok=True)
                removed += 1
        except OSError as exc:
            logger.warning("could not remove old run %s: %s", job_dir, exc)
    if removed:
        logger.info("removed %d run folder(s) older than %s days", removed, retention_days)
    return removed
