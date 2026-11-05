#!/usr/bin/env python3
"""
run_candidates.py  --  run candidate layouts (heat case + cold case) through ./Allrun

Project folders (all next to python/):
    caseFile/   template OpenFOAM case that contains Allrun
    epw/        exactly ONE *.epw  (Allrun copies it with  cp ../epw/*.epw weather.epw)
    input/      candidate layout JSONs: individual_X.json, offspring_X.json, ...
    jsons/      <epw-stem>_cases.json from select_epw_cases.py (the heat and cold hours)
    run/        simulations are created here
    output/     objectives.csv and run_summary_<time>.json

For every candidate NAME and each season (heat, cold) this script
    1. copies   caseFile/                 ->  run/NAME/<season>/
    2. copies   input/NAME.json           ->  run/NAME/<season>/NAME.json   (the ONLY *.json in the case)
    3. writes   run/NAME/<season>/date_and_time.xyz   (the peak-heat / peak-cold date and hour)
    4. runs     ./Allrun  in that folder  (log: log.Allrun)
and waits for all of them.  Then it reads UTCI_details.txt from each case:
    UTCI of the heat case  -> f1 = UTCI_heat          (minimize: lower heat stress)
    UTCI of the cold case  -> f2 = -UTCI_cold         (minimize: i.e. raise the cold-case UTCI)
Each simulation uses 4 cores; --max-parallel 16 simulations run at once (64 cores).
One candidate = 2 simulations = 8 cores, so 8 candidates run at the same time.

Usage (from anywhere):
    python python/run_candidates.py individual_1 individual_2 offspring_1
    python python/run_candidates.py                  # every unfinished individual_*/offspring_* JSON in input/
    python python/run_candidates.py --max-parallel 16 --timeout-min 180

From the NSGA-II driver:
    from run_candidates import run_batch
    out = run_batch(["individual_1", "individual_2"])   # blocks until done
    out["objectives"]      # {name: {"f1": ..., "f2": ..., "status": ...}}
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # folder that contains python/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_type as CT  # noqa: E402  (the case folder may be the OpenFOAM case or the AI case; see case_type.py)
SEASONS = ("heat", "cold")
STATUS_FILE = ".uhi_status"        # no .json extension: Allrun globs *.json
UTCI_FILE = "UTCI_details.txt"
DEFAULT_PATTERNS = ("individual_*", "offspring_*")
PENALTY = 1.0e6

_live = {}
_live_lock = threading.Lock()


# ----------------------------------------------------------------------------- helpers
def _kill_group(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def single_epw(epw_dir):
    """Allrun copies ../epw/*.epw, so there must be exactly one EPW."""
    f = sorted(Path(epw_dir).glob("*.epw"))
    if len(f) != 1:
        raise SystemExit(f"epw/ must contain exactly one .epw file, found {len(f)} in {epw_dir}")
    return f[0]


def load_cases(cases_path=None, jsons_dir=None):
    if cases_path is None:
        c = sorted(Path(jsons_dir or ROOT / "jsons").glob("*_cases.json"))
        if len(c) != 1:
            raise SystemExit(f"expected exactly one *_cases.json in jsons/ (run select_epw_cases.py), found {len(c)}")
        cases_path = c[0]
    return json.loads(Path(cases_path).read_text())


def xyz_text(case):
    """Content of date_and_time.xyz for one case. Hour = EPW hour_end - 1 (hour START),
    which is what extractWeatherData.py (pvlib index) matches against."""
    t = case["timestamp"]
    return f"#MM,DD HH\n{t['month']:02d},{t['day']:02d}  {t['hour_start']:02d}\n"


UTCI_REGEX = r"Site area-averaged:\s*UTCI\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def parse_utci(case_dir, filename=UTCI_FILE, regex=None):
    """Site area-averaged UTCI (deg C) from UTCI_details.txt, or None if it is missing.

    The file has one block per simulated hour, e.g.
        === Processing t = 3600 s (1AM) ===
        ...
        Site area-averaged: UTCI=-63.21 C (Extreme cold stress), Delta UTCI ...
    With one hour in date_and_time.xyz there is exactly one block.  If there are several, the LAST
    one is used (the final simulated hour).  --utci-regex (one capture group) overrides the pattern."""
    try:
        text = (Path(case_dir) / filename).read_text(errors="ignore")
    except OSError:
        return None
    hits = re.findall(regex or UTCI_REGEX, text, re.S if regex else 0)
    for h in reversed(hits):
        try:
            v = float(h)
        except ValueError:
            continue
        if v == v and abs(v) < 1e3:          # reject nan / absurd values
            return v
    return None


def _read_status(job_dir):
    try:
        return json.loads((job_dir / STATUS_FILE).read_text())
    except (OSError, ValueError):
        return None


def _ensure_epw_links(case_dir, epw_dir):
    """Allrun does `cp ../epw/*.epw`. In run/NAME/season that is run/NAME/epw; make that
    (and run/epw for ../../epw) point at the real epw/ folder."""
    for parent in (case_dir.parent, case_dir.parent.parent):
        link = parent / "epw"
        try:
            if not link.exists() and not link.is_symlink():
                link.symlink_to(Path(epw_dir).resolve(), target_is_directory=True)
        except FileExistsError:
            pass
        except OSError:
            # file systems without symlinks (e.g. [Errno 95] on some network mounts): the EPW is small, copy it
            try:
                if not link.exists():
                    shutil.copytree(Path(epw_dir).resolve(), link)
            except FileExistsError:
                pass


# ----------------------------------------------------------------------------- one simulation
_RESULT_DIR = re.compile(r"^(processor\d+|[1-9][0-9.eE+-]*|postProcessing|VTK|__pycache__)$")


def _ignore_results(directory, names):
    """Do not copy the results of earlier runs into every candidate (processor* and time directories, logs, videos,
    vtk frames, polyMesh): the case recreates all of them, and surface_temperature/ can hold hundreds of MB of frames."""
    skip = set()
    for n in names:
        p = Path(directory) / n
        if p.is_dir() and (_RESULT_DIR.match(n) or n in ("polyMesh", "extendedFeatureEdgeMesh")):
            skip.add(n)
        elif n.startswith("log.") or n.endswith((".gif", ".jpeg", ".vtk", ".pyc", ".png", ".log")):
            skip.add(n)
    return skip


def prepare_job(name, season, case, case_file, run_dir, cand_dir, epw_dir, force):
    src_json = cand_dir / f"{name}.json"
    if not src_json.is_file():
        raise FileNotFoundError(f"candidate JSON not found: {src_json}")
    stray = sorted(p.name for p in Path(case_file).glob("*.json"))
    if stray:
        raise RuntimeError(f"caseFile must not contain any *.json (Allrun takes the only *.json): {stray}")
    dst = run_dir / name / season
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} already exists (use --force)")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    top, skip_top = Path(case_file).resolve(), CT.no_copy(case_file)     # AI case: pipeline/ (weights) stays in the original folder

    def ignore(directory, names):
        skip = _ignore_results(directory, names)
        if skip_top and Path(directory).resolve() == top:
            skip |= skip_top & set(names)
        return skip
    try:
        shutil.copytree(case_file, dst, symlinks=True, ignore=ignore)
    except (OSError, shutil.Error):          # no symlink support on this file system: copy what they point to
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(case_file, dst, symlinks=False, ignore=ignore)
    shutil.copy2(src_json, dst / f"{name}.json")
    (dst / "date_and_time.xyz").write_text(xyz_text(case))
    jsons = sorted(p.name for p in dst.glob("*.json"))
    if jsons != [f"{name}.json"]:
        raise RuntimeError(f"case folder must contain exactly one *.json, found {jsons}")
    allrun = dst / "Allrun"
    if not allrun.is_file():
        raise FileNotFoundError(f"no Allrun in {case_file}")
    if not os.access(allrun, os.X_OK):
        allrun.chmod(allrun.stat().st_mode | 0o111)
    _ensure_epw_links(dst, epw_dir)
    return dst


def _why_failed(job_dir):
    """The reason Allrun gave (its last 'Allrun: ERROR' line), so the console shows it without opening the logs."""
    try:
        lines = [l.strip() for l in (Path(job_dir) / "log.Allrun").read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return None
    err = [l for l in lines if "ERROR" in l]
    return (err or lines or [None])[-1]


def run_job(name, season, case, a):
    """Prepare and run one simulation. Never raises; returns a result dict."""
    t0 = time.time()
    res = {"name": name, "season": season, "status": "failed", "returncode": None,
           "seconds": 0.0, "utci": None}
    job_dir = a["run_dir"] / name / season
    try:
        done = _read_status(job_dir)
        kind = CT.detect(a["case_file"])
        # a result made by the other kind of case (OpenFOAM <-> AI) in the same run/ folder is not a result of this one
        if done and done.get("status") == "ok" and done.get("case_type", "openfoam") != kind:
            done = None
        if done and done.get("status") == "ok" and not a["force"]:
            res.update(status="ok", skipped="already finished", seconds=done.get("seconds", 0.0))
        else:
            job_dir = prepare_job(name, season, case, a["case_file"], a["run_dir"],
                                  a["cand_dir"], a["epw_dir"], force=a["force"] or job_dir.exists())
            log = open(job_dir / "log.Allrun", "w")
            env = dict(os.environ, UHI_CASE_SOURCE=str(Path(a["case_file"]).resolve()), UHI_CASE_TYPE=kind)
            p = subprocess.Popen(["./Allrun"], cwd=job_dir, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
            with _live_lock:
                _live[(name, season)] = p
            try:
                p.wait(timeout=a["timeout_s"])
                res["returncode"] = p.returncode
                res["status"] = "ok" if p.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                _kill_group(p)
                res.update(status="timeout", returncode=-9)
            finally:
                with _live_lock:
                    _live.pop((name, season), None)
                log.close()
            res["seconds"] = round(time.time() - t0, 1)
            if res["status"] == "failed":
                res["note"] = _why_failed(job_dir)
        if res["status"] == "ok":
            res["utci"] = parse_utci(job_dir, a["utci_file"], a["utci_regex"])
            if res["utci"] is None:
                res["status"] = "no_utci"
                res["note"] = f"could not read a UTCI value from {a['utci_file']}"
        if not res.get("skipped"):
            (job_dir / STATUS_FILE).write_text(json.dumps(
                {**{k: res.get(k) for k in ("status", "returncode", "seconds")}, "case_type": kind}, indent=2))
    except (OSError, RuntimeError) as e:
        res.update(status="failed", error=str(e), seconds=round(time.time() - t0, 1))
    return res


# ----------------------------------------------------------------------------- batch
def objectives_from_jobs(name, jobs, cases, penalty=PENALTY):
    """f1 = UTCI of the heat case, f2 = -UTCI of the cold case (both minimized).
    A season with no stress in this climate contributes 0. Any failed job -> penalty."""
    out = {"name": name, "utci_heat": None, "utci_cold": None}
    ok = True
    for season in SEASONS:
        if cases.get(f"{season}_case") is None:
            continue
        j = jobs.get(season)
        if j is None or j["status"] != "ok":
            ok = False
        else:
            out[f"utci_{season}"] = j["utci"]
    if ok:
        out["status"] = "ok"
        out["f1"] = out["utci_heat"] if cases.get("heat_case") is not None else 0.0
        out["f2"] = -out["utci_cold"] if cases.get("cold_case") is not None else 0.0
    else:
        out.update(status="failed", f1=penalty, f2=penalty)
    return out


def discover(cand_dir, run_dir, cases, patterns=DEFAULT_PATTERNS, include_done=False):
    names = sorted({p.stem for pat in patterns for p in Path(cand_dir).glob(pat + ".json")})
    if include_done:
        return names
    need = [s for s in SEASONS if cases.get(f"{s}_case") is not None]
    return [n for n in names
            if not all((_read_status(Path(run_dir) / n / s) or {}).get("status") == "ok" for s in need)]


def run_batch(names, case_file=None, run_dir=None, cand_dir=None, epw_dir=None, cases_path=None,
              max_parallel=16, timeout_min=None, force=False, utci_file=UTCI_FILE, utci_regex=None,
              summary_dir=None, penalty=PENALTY, quiet=False):
    """Run every candidate (heat + cold simulations), max_parallel simulations at a time, wait for all."""
    a = {"case_file": Path(case_file or ROOT / "caseFile"), "run_dir": Path(run_dir or ROOT / "run"),
         "cand_dir": Path(cand_dir or ROOT / "input"), "epw_dir": Path(epw_dir or ROOT / "epw"),
         "timeout_s": timeout_min * 60 if timeout_min else None, "force": force,
         "utci_file": utci_file, "utci_regex": utci_regex}
    single_epw(a["epw_dir"])
    cases = load_cases(cases_path)
    names = list(names)
    if not names:
        return {"results": [], "objectives": {}}
    a["run_dir"].mkdir(parents=True, exist_ok=True)
    jobs = [(n, s) for n in names for s in SEASONS if cases.get(f"{s}_case") is not None]
    started = time.time()
    results = {}
    ex = ThreadPoolExecutor(max_workers=max(1, min(max_parallel, len(jobs))))
    try:
        futs = [(n, s, ex.submit(run_job, n, s, cases[f"{s}_case"], a)) for n, s in jobs]
        for n, s, f in futs:
            r = f.result()
            results[(n, s)] = r
            if not quiet:
                extra = (f"  rc={r['returncode']}" if r["returncode"] not in (0, None) else "")
                extra += f"  UTCI {r['utci']:.2f}" if r["utci"] is not None else ""
                msg = r.get("error") or r.get("note") or r.get("skipped")
                print(f"  {n:<20s} {s:<5s} {r['status']:<8s} {r['seconds']:>8.1f} s{extra}"
                      + (f"  ({msg})" if msg else ""), flush=True)
    except KeyboardInterrupt:
        print("Interrupted: stopping all running simulations ...", file=sys.stderr)
        with _live_lock:
            for p in list(_live.values()):
                _kill_group(p)
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)

    objectives = {n: objectives_from_jobs(n, {s: results[(n, s)] for s in SEASONS if (n, s) in results},
                                          cases, penalty) for n in names}
    summary_dir = Path(summary_dir or ROOT / "output")
    summary_dir.mkdir(parents=True, exist_ok=True)
    _update_csv(summary_dir / "objectives.csv", objectives)
    summary = {"time": datetime.now().isoformat(timespec="seconds"),
               "wall_seconds": round(time.time() - started, 1),
               "objective_definition": {"f1": "site-averaged UTCI, peak-heat case (minimize)",
                                        "f2": "-(site-averaged UTCI, peak-cold case) (minimize)"},
               "cases_used": {s: (cases.get(f"{s}_case") or {}).get("timestamp") for s in SEASONS},
               "objectives": objectives, "results": list(results.values())}
    (summary_dir / f"run_summary_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(summary, indent=2))
    return summary


def _update_csv(path, objectives):
    cols = ["name", "status", "utci_heat", "utci_cold", "f1", "f2"]
    rows = {}
    if path.exists():
        with path.open() as f:
            rows = {r["name"]: r for r in csv.DictReader(f)}
    for n, o in objectives.items():
        rows[n] = {c: ("" if o.get(c) is None else o.get(c)) for c in cols}
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


def main():
    ap = argparse.ArgumentParser(description="Run heat + cold simulations of candidate layouts via ./Allrun.")
    ap.add_argument("names", nargs="*", help="candidate names without .json (individual_1 offspring_2 ...)")
    ap.add_argument("--pattern", action="append", help="glob in input/ (repeatable), e.g. 'individual_*'")
    ap.add_argument("--case-file", type=Path, default=ROOT / "caseFile")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "run")
    ap.add_argument("--candidates-dir", type=Path, default=ROOT / "input")
    ap.add_argument("--cases", type=Path, default=None, help="<epw>_cases.json (default: the only one in jsons/)")
    ap.add_argument("--max-parallel", type=int, default=16, help="simulations at once, 4 cores each (default 16)")
    ap.add_argument("--timeout-min", type=float, default=None, help="kill a simulation after this many minutes")
    ap.add_argument("--utci-file", default=UTCI_FILE)
    ap.add_argument("--utci-regex", default=None,
                    help="regex with ONE group that captures the site-averaged UTCI in UTCI_details.txt")
    ap.add_argument("--force", action="store_true", help="rerun simulations that already finished OK")
    ap.add_argument("--list", action="store_true", help="only show what would run")
    a = ap.parse_args()

    single_epw(ROOT / "epw")
    cases = load_cases(a.cases)
    names = a.names or discover(a.candidates_dir, a.run_dir, cases,
                                tuple(a.pattern) if a.pattern else DEFAULT_PATTERNS, include_done=a.force)
    for s in SEASONS:
        c = cases.get(f"{s}_case")
        print(f"{s:>4s} case: " + (f"{c['timestamp']['label']}  -> date_and_time.xyz "
                                    f"{c['timestamp']['month']:02d},{c['timestamp']['day']:02d}  "
                                    f"{c['timestamp']['hour_start']:02d}" if c else "none (no stress in this climate)"))
    print(f"{len(names)} candidate(s): {', '.join(names) if names else '-'}")
    if a.list or not names:
        return
    s = run_batch(names, a.case_file, a.run_dir, a.candidates_dir, None, a.cases, a.max_parallel,
                  a.timeout_min, a.force, a.utci_file, a.utci_regex)
    bad = [n for n, o in s["objectives"].items() if o["status"] != "ok"]
    print(f"Done in {s['wall_seconds']} s: {len(names) - len(bad)}/{len(names)} candidates ok"
          + (f"; failed: {', '.join(bad)}" if bad else ""))
    sys.exit(0 if not bad else 2)


if __name__ == "__main__":
    main()
