#!/usr/bin/env python3
"""One command: input/base.json (+ constraints.json)  ->  NSGA-II  ->  Pareto plot and the best 5.

    python python/optimize.py                       # defaults: 10 individuals + base, 4 children per generation, 5 generations
    python python/optimize.py --pop 12 --gens 8 --offspring 4 --pm 0.25
    python python/optimize.py --check               # pre-flight only (nothing is run)

What it does, in order (every step is skipped when its result already exists, so re-running resumes):
  0. pre-flight : base.json, exactly one EPW, OpenFOAM + python modules, free disk, caseFile/Allrun
  1. select_epw_cases.py -> jsons/<epw>_cases.json   (peak-heat and peak-cold hour; only if missing)
  2. nsga2.py run        -> initial individuals + base -> simulate -> rank -> offspring (crossover, mutation) -> select -> repeat
  3. nsga2.py top5, plot -> output/top5.json, input/top5_1..5.json, output/nsga_report.png (Pareto front), output/overview.png
Anything not recognised here is passed on to nsga2.py (--pop, --gens, --offspring, --batch, --pc, --pm, --comfort, --seed, --retry-failed, --evaluator mock ...).
"""
import argparse, importlib, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = Path(__file__).resolve().parent


def preflight(a, mock):
    bad, warn = [], []
    inp = ROOT / a.input_dir
    if not (inp / "base.json").exists():
        bad.append(f"{inp}/base.json is missing")
    if not (inp / "constraints.json").exists():
        warn.append("input/constraints.json not found: the default rules are used")
    epws = sorted((ROOT / "epw").glob("*.epw"))
    if len(epws) != 1:
        bad.append(f"epw/ must contain exactly one .epw (found {len(epws)})")
    for m in ("numpy", "shapely", "matplotlib", "scipy"):
        try:
            importlib.import_module(m)
        except Exception:
            bad.append(f"python module '{m}' is not installed (activate your venv)")
    if not mock:
        for m in ("pythermalcomfort",):
            try:
                importlib.import_module(m)
            except Exception:
                bad.append(f"python module '{m}' is not installed (activate your venv)")
        if not (ROOT / "caseFile" / "Allrun").exists():
            bad.append("caseFile/Allrun is missing")
        if not os.environ.get("WM_PROJECT_DIR"):
            bad.append("OpenFOAM is not sourced (source .../OpenFOAM-8/etc/bashrc first)")
        elif not shutil.which("simpleFoam"):
            bad.append("simpleFoam not found on PATH")
        if not shutil.which("mpirun"):
            bad.append("mpirun not found on PATH")
        mom = ROOT / "caseFile" / "momentum"
        if not ((mom / "of_Allrun").exists() or (mom / "Allrun").exists()):
            bad.append("caseFile/momentum has neither of_Allrun nor Allrun")
        if list((ROOT / "caseFile").glob("*.json")):
            bad.append("caseFile/ must contain no *.json (Allrun uses the only one it is given)")
        free = shutil.disk_usage(ROOT).free / 2**30
        need = 0.5 * 40 + 2   # rough: ~0.5 GB per simulation kept (about 40 simulations)
        if free < need:
            warn.append(f"only {free:.0f} GB free; this run keeps roughly {need:.0f} GB")
    return bad, warn


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    return subprocess.call(list(map(str, cmd)), cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="pre-flight only")
    ap.add_argument("--occupied", type=float, nargs=2, default=[1, 24], metavar=("START", "END"),
                    help="hours of the day used to pick the peak-heat / peak-cold hour (default 1 24)")
    ap.add_argument("--north-deg", type=float, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--evaluator", default="real")
    a, rest = ap.parse_known_args()
    mock = a.evaluator == "mock"

    bad, warn = preflight(a, mock)
    for w in warn:
        print("WARNING:", w)
    if bad:
        print("\nCannot start:")
        for b in bad:
            print("  -", b)
        return 2
    print("pre-flight OK")
    if a.check:
        return 0

    # 1. peak-heat / peak-cold hours, once per EPW
    if not list((ROOT / "jsons").glob("*_cases.json")):
        cmd = [sys.executable, PY / "select_epw_cases.py", "--occupied", *a.occupied]
        if a.north_deg is not None:
            cmd += ["--north-deg", a.north_deg]
        if run(cmd) != 0 or not list((ROOT / "jsons").glob("*_cases.json")):
            print("select_epw_cases.py failed"); return 3
    else:
        print("jsons/*_cases.json exists: peak hours not re-selected")

    # 2. NSGA-II (resumable); 3. Pareto + decision from whatever finished
    common = ["--batch", a.batch, "--input-dir", a.input_dir, "--evaluator", a.evaluator, *rest]
    rc = run([sys.executable, PY / "nsga2.py", "run", *common])
    if rc == 0:
        run([sys.executable, PY / "nsga2.py", "top5", *common])
        run([sys.executable, PY / "nsga2.py", "plot", *common])
    return rc


if __name__ == "__main__":
    sys.exit(main())
