"""
Production logging for RH_T_API.

Every log line carries the job it belongs to, so an issue can be traced back to the user / project / model:

    2026-10-21 10:12:03 ERROR rht_api [job=ab12... user=42 project=7 model=19 status=failed] job failed: ...

  * set_log_context(...) stores job_id / userid / project_id / model_id / status for the current request or
    job thread (contextvars). All loggers ("rht_api", "rht_api.wind_ai", "rht_api.multi", uvicorn) pick it up.
    Threads do not inherit it, so the job thread sets it again in main._run_job.
  * ERROR / CRITICAL lines are red, WARNING lines yellow. LOG_COLOR=0 in .env turns colours off.
  * LOG_LEVEL in .env (default INFO).
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
from typing import Any, Dict

_CONTEXT_KEYS = (("job_id", "job"), ("userid", "user"), ("project_id", "project"),
                 ("model_id", "model"), ("status", "status"))

_log_ctx: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("rht_log_ctx", default={})

_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def set_log_context(**fields: Any) -> None:
    """Merge fields (job_id, userid, project_id, model_id, status) into the current log context."""
    ctx = dict(_log_ctx.get())
    ctx.update({k: v for k, v in fields.items() if v is not None})
    _log_ctx.set(ctx)


def clear_log_context() -> None:
    _log_ctx.set({})


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _log_ctx.get()
        parts = [f"{label}={ctx[key]}" for key, label in _CONTEXT_KEYS if ctx.get(key) is not None]
        record.ctx = f" [{' '.join(parts)}]" if parts else ""
        return True


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, use_color: bool) -> None:
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        if not self.use_color:
            return line
        if record.levelno >= logging.ERROR:
            return f"{_RED}{line}{_RESET}"
        if record.levelno >= logging.WARNING:
            return f"{_YELLOW}{line}{_RESET}"
        return line


def setup_logging() -> None:
    """Send root + uvicorn loggers through one handler with job context and coloured errors."""
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    use_color = os.getenv("LOG_COLOR", "1").strip().lower() not in ("0", "false", "no", "off")

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(_ColorFormatter("%(asctime)s %(levelname)s %(name)s%(ctx)s %(message)s", use_color))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers before the app is imported; route them through ours instead
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
