#!/usr/bin/env python3
"""case_type.py  --  what kind of case folder is this?   ("openfoam" | "ai")

The optimiser only needs ONE thing of a case folder: an executable `Allrun` that, given  <NAME>.json + date_and_time.xyz + ../epw/*.epw,
leaves `UTCI_details.txt` with the line  "Site area-averaged: UTCI=xx.xx C".  The OpenFOAM case (momentum + airTqFoam) and the AI case (wind -> T -> RH
models) both do, so  caseFile/  can be either; python/ asks this module only for the few things that differ:

    detect(dir)      "ai" if the folder has case_type.cfg (JSON text) with "type": "ai", or has ai/run_ai_case.py and no OpenFOAM momentum case; else "openfoam"
    no_copy(dir)     top-level names run_candidates.py must NOT copy into every candidate (the AI case's pipeline/ holds 1.2 GB of weights; the
                     candidates use it from the original folder through $UHI_CASE_SOURCE)
    preflight(dir, root, mock)   (bad, warnings) for the pre-flight of optimize.py
Nothing here runs a simulation.
"""
import importlib
import json
import os
import shutil
import sys
from pathlib import Path


def meta(case_dir):
    f = Path(case_dir) / "case_type.cfg"          # not *.json: the case folder must contain no *.json
    try:
        return json.loads(f.read_text()) if f.is_file() else {}
    except (OSError, ValueError):
        return {}


def detect(case_dir):
    d = Path(case_dir)
    t = str(meta(d).get("type", "")).lower()
    if t in ("ai", "openfoam"):
        return t
    if (d / "ai" / "run_ai_case.py").is_file() and not ((d / "momentum" / "of_Allrun").is_file() or (d / "momentum" / "Allrun").is_file()):
        return "ai"
    return "openfoam"


def no_copy(case_dir):
    return set(meta(case_dir).get("no_copy", [])) if detect(case_dir) == "ai" else set()


def _openfoam_preflight(case, bad, warn):
    if not (case / "Allrun").exists():
        bad.append(f"{case.name}/Allrun is missing")
    if not os.environ.get("WM_PROJECT_DIR"):
        bad.append("OpenFOAM is not sourced (source .../OpenFOAM-8/etc/bashrc first)")
    elif not shutil.which("simpleFoam"):
        bad.append("simpleFoam not found on PATH")
    if not shutil.which("mpirun"):
        bad.append("mpirun not found on PATH")
    mom = case / "momentum"
    if not ((mom / "of_Allrun").exists() or (mom / "Allrun").exists()):
        bad.append(f"{case.name}/momentum has neither of_Allrun nor Allrun")


def _ai_preflight(case, bad, warn):
    if not (case / "Allrun").exists():
        bad.append(f"{case.name}/Allrun is missing")
    for need in ("ai/run_ai_case.py", "ai/ai_utci.py", "ai/backend.py", "momentum/json_to_obj.py", "momentum/gen_ground_obj.py",
                 "temperature_humidity/tools/run_thermal.py", "surface_temperature/solar_thermal_model.py"):
        if not (case / need).is_file():
            bad.append(f"{case.name}/{need} is missing")
    for m in ("pandas", "PIL", "pythermalcomfort"):
        try:
            importlib.import_module(m)
        except Exception:
            bad.append(f"python module '{m}' is not installed (the AI case needs it)")
    if bad:
        return
    sys.path.insert(0, str(case / "ai"))
    try:
        import backend as B
        sys.modules.pop("backend", None)
        os.environ["UHI_CASE_SOURCE"] = str(case.resolve())
        name = B.choose(None, B.pipeline_dir())
        pipe = B.pipeline_dir()
        if name == "local":
            miss = B.weights_missing(pipe)
            if miss:
                bad.append("UHI_AI_BACKEND=local but the weights are missing: " + "; ".join(miss))
            try:
                importlib.import_module("torch")
            except Exception:
                bad.append("UHI_AI_BACKEND=local needs torch (pip install -r pipeline/worker/requirements.txt)")
        elif name == "runpod":
            if not (os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_ENDPOINT_ID")):
                bad.append("UHI_AI_BACKEND=runpod: set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
        elif name == "mock":
            warn.append("UHI_AI_BACKEND=mock: the wind / T / RH fields are FAKE (plumbing test only); do not use the result")
        warn.append(f"AI case: back end '{name}'.  The objective is a surrogate; confirm the best layouts with the OpenFOAM case "
                    f"(--verify-case-file caseFile --verify-run-dir run_verify)")
    except Exception as exc:  # noqa: BLE001
        bad.append(f"AI back end: {exc}")
    finally:
        if str(case / "ai") in sys.path:
            sys.path.remove(str(case / "ai"))


def preflight(case_dir, root=None, mock=False):
    """(bad, warn) for one case folder (not run when the optimiser's evaluator is the mock)."""
    case = Path(case_dir)
    bad, warn = [], []
    if mock:
        return bad, warn
    if not case.is_dir():
        return [f"case folder {case} not found"], warn
    kind = detect(case)
    if list(case.glob("*.json")):
        bad.append(f"{case.name}/ must contain no *.json (Allrun uses the only one it is given)")
    (_ai_preflight if kind == "ai" else _openfoam_preflight)(case, bad, warn)
    return bad, warn
