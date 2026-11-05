#!/usr/bin/env python3
"""
backend.py  --  where the three models (wind, T, RH) run.   UHI_AI_BACKEND = runpod | local | mock | auto (default)

    runpod   the pipeline's own RunPod client (pipeline/app/runpod_client.py).  Needs RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in the environment
             (or in pipeline/.env / ai/.env).  Scales with the NSGA batch size, nothing runs on this computer.
    local    the pipeline's worker/handler.py in ONE persistent process on this computer's GPU (ai/serve.py, started by the first case,
             models loaded once, requests served one at a time, so 4 parallel cases do not load the weights 4 times).  Needs torch + the weights.
    mock     ai/mock_models.py in this process: numpy only, fake but plausible fields, for testing the plumbing.  NOT a prediction.
    auto     runpod if RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID are set, else local if the weights are in pipeline/, else an error that says what is missing.

Both real back ends take and give exactly what the pipeline's app/pipeline.py takes and gives (same payloads), so a change of back end changes nothing else.
"""
import base64
import fcntl
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text()) if (HERE / "config.json").is_file() else {}


class AIError(RuntimeError):
    pass


def pipeline_dir(case_dir=None):
    """The folder with app/ scripts/ worker/ T_model/ RH_model/ wind_api/: $UHI_CASE_SOURCE/pipeline (the original caseFile, so the 1.2 GB of
    weights are not copied into every candidate), else <this case>/pipeline."""
    cands = []
    if os.environ.get("UHI_PIPELINE_DIR"):
        cands.append(Path(os.environ["UHI_PIPELINE_DIR"]))
    if os.environ.get("UHI_CASE_SOURCE"):
        cands.append(Path(os.environ["UHI_CASE_SOURCE"]) / "pipeline")
    cands.append(HERE.parent / "pipeline")
    for c in cands:
        if (c / "app" / "runpod_client.py").is_file():
            return c.resolve()
    raise AIError("the AI pipeline folder (app/, scripts/, worker/, T_model/, RH_model/, wind_api/) was not found; looked in "
                  + ", ".join(str(c) for c in cands))


def weights_missing(pipe):
    need = {"T model": pipe / "T_model" / "latest_net_G.pth", "RH model": next(iter(sorted((pipe / "RH_model").glob("*_net_G.pth"))), pipe / "RH_model" / "<n>_net_G.pth")}
    wind = list((pipe / "wind_api" / "model").rglob("*.ckpt"))
    miss = [f"{k}: {v}" for k, v in need.items() if not v.is_file() or v.stat().st_size < 1_000_000]
    if not wind or max(p.stat().st_size for p in wind) < 1_000_000:
        miss.append(f"wind model: no real *.ckpt under {pipe / 'wind_api' / 'model'}")
    return miss


def _load_env_files(pipe):
    for f in (HERE / ".env", pipe / ".env"):
        if f.is_file():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def choose(name=None, pipe=None):
    name = (name or os.environ.get("UHI_AI_BACKEND") or CONFIG.get("backend") or "auto").lower()
    pipe = pipe or pipeline_dir()
    _load_env_files(pipe)
    if name == "auto":
        if os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_ENDPOINT_ID"):
            return "runpod"
        if not weights_missing(pipe):
            return "local"
        raise AIError("UHI_AI_BACKEND=auto found no model to run: set RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID (RunPod), or put the weights into the "
                      "pipeline folder (local GPU): missing " + "; ".join(weights_missing(pipe)) + ".  UHI_AI_BACKEND=mock runs fake fields for testing only.")
    if name not in ("runpod", "local", "mock"):
        raise AIError(f"unknown backend {name!r} (runpod | local | mock | auto)")
    return name


class Backend:
    def __init__(self, name=None, case_dir=None):
        self.pipe = pipeline_dir(case_dir)
        self.name = choose(name, self.pipe)
        os.environ.setdefault("RUNPOD_RETURN_NPY", "1")      # the UTCI step needs the T and RH fields, not only the PNGs
        os.environ["RUNPOD_RETURN_NPY"] = "1"
        sys.path.insert(0, str(self.pipe))
        from app import runpod_client                         # stdlib only
        self.rc = runpod_client
        self.rc.RETURN_NPY = True
        self._server = None
        if self.name == "runpod":
            probs = self.rc.config_problems()
            if probs:
                raise AIError("RunPod is not configured: " + "; ".join(probs))
        elif self.name == "local" and not os.environ.get("UHI_AI_SERVER_MOCK"):
            miss = weights_missing(self.pipe)
            if miss:
                raise AIError("local back end: the model weights are missing: " + "; ".join(miss))

    # ---- one worker job: payload in, worker output out ------------------------------------------------------------
    def _call(self, payload, log_path, timeout, label):
        if self.name == "runpod":
            return self.rc.call(payload, Path(log_path), timeout, label)
        if self.name == "mock":
            sys.path.insert(0, str(HERE))
            import mock_models
            out = mock_models.handler({"input": payload})
        else:
            out = self._post_local({"input": payload}, timeout)
        if "error" in out or not out.get("ok"):
            raise AIError(f"{label}: {str(out.get('error') or out)[-1500:]}")
        if out.get("log"):
            with open(log_path, "a") as f:
                f.write(out["log"].rstrip() + "\n")
        return out

    # ---- local persistent server ----------------------------------------------------------------------------------
    def _state_dir(self):
        d = Path(os.environ.get("UHI_AI_STATE") or (Path(os.environ.get("UHI_CASE_SOURCE") or HERE.parent) / ".ai_server"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _health(self, port):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _ensure_server(self):
        sd = self._state_dir()
        state = sd / "state.json"

        def alive():
            try:
                port = json.loads(state.read_text())["port"]
            except (OSError, ValueError, KeyError):
                return None
            return port if self._health(port) else None
        port = alive()
        if port:
            return port
        with open(sd / "start.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)               # one case starts the server, the others wait here
            port = alive()
            if port:
                return port
            state.unlink(missing_ok=True)
            log = open(sd / "server.log", "a")
            extra = ["--mock"] if os.environ.get("UHI_AI_SERVER_MOCK") else []        # test hook: the server answers with ai/mock_models.py
            subprocess.Popen([sys.executable, str(HERE / "serve.py"), "--pipe", str(self.pipe), "--state", str(state), *extra],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(self.pipe))
            t0 = time.time()
            limit = float(CONFIG.get("server_start_timeout_s", 900))
            while time.time() - t0 < limit:
                port = alive()
                if port:
                    return port
                time.sleep(2)
        raise AIError(f"the local model server did not start within {limit:.0f} s; see {sd / 'server.log'}")

    def _post_local(self, event, timeout):
        port = self._ensure_server()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/run", data=json.dumps(event).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError) as exc:
            raise AIError(f"local model server: {exc}") from exc

    # ---- the two jobs ---------------------------------------------------------------------------------------------
    def wind(self, obj_path, ux, uy, uz, out_dir, log_path, label="wind"):
        """wind OBJ + inlet vector -> out_dir/{U_2d.npy, U_flow_direction.json, wind_extent.txt, facade_cp_summary.txt}"""
        payload = {"task": "wind", "obj_gz_b64": base64.b64encode(gzip.compress(Path(obj_path).read_bytes())).decode("ascii"),
                   "ux": float(ux), "uy": float(uy), "uz": float(uz), "res": int(CONFIG.get("raster_res", 512)),
                   "z_index": int(CONFIG.get("z_index", 1)), "label": label}
        out = self._call(payload, log_path, float(CONFIG.get("wind_timeout_s", 1800)), "wind model")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.rc.safe_extract(base64.b64decode(out["result_b64"]), out_dir)
        miss = [n for n in ("U_2d.npy", "U_flow_direction.json", "wind_extent.txt") if not (out_dir / n).is_file()]
        if miss:
            raise AIError(f"wind model result is missing {', '.join(miss)}")
        return out

    def t_rh(self, raster_dir, weather_dir, t_out, rh_out, log_path, label="T+RH"):
        """T model, then RH model on that T (same worker call).  Afterwards t_out / rh_out hold the .npy fields + summaries."""
        payload = {"task": "rht", "bundle_b64": base64.b64encode(self.rc._bundle(Path(raster_dir), Path(weather_dir))).decode("ascii"),
                   "run_rh": True, "return_npy": True, "label": label}
        out = self._call(payload, log_path, float(CONFIG.get("rht_timeout_s", 900)), f"T+RH {label}")
        self.rc._place_results(out, Path(t_out), Path(rh_out))
        return out
