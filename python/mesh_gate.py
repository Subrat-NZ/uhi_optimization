#!/usr/bin/env python3
"""mesh_gate.py  --  can the surfaces of this layout be meshed?  (run BEFORE a layout is simulated)

Runs the SAME scripts the momentum case runs (caseFile/momentum/json_to_obj.py and gen_ground_obj.py) on the layout in a temporary folder and
reports a failure when
    * a pond / lawn / road outline cannot be triangulated ("constrained triangulation failed", "Ran out of precision" ...),
    * the ground mesh needs the Delaunay fall-back or fails, or an OBJ has non-finite coordinates (objBoundingDim then reports y = +-inf).
A layout that fails here would waste a simulation slot (momentum stops at gen_ground_obj / cubic_block), so nsga2.py does not accept it as a child.

    python python/mesh_gate.py input/offspring_17.json [more.json ...]       (case folder: caseFile/ next to python/; --case-file DIR to change)
Takes a few seconds per layout.  nsga2.py: on by default when caseFile/momentum/json_to_obj.py exists; --no-mesh-gate switches it off.
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BAD = ("constrained triangulation failed", "ran out of precision", "triangulation failed", "falling back", "did not finish",
       "crashed", "traceback", "error:", "cubic_block")


def available(case_file):
    m = Path(case_file) / "momentum"
    return (m / "json_to_obj.py").exists() and (m / "gen_ground_obj.py").exists()


def _obj_bounds(files):
    xs, ys, bad = [], [], []
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.startswith("v "):
                p = line.split()[1:4]
                try:
                    x, y, z = (float(v) for v in p)
                except ValueError:
                    bad.append(f.name)
                    break
                if not all(math.isfinite(v) for v in (x, y, z)):
                    bad.append(f.name)
                    break
                xs.append(x)
                ys.append(y)
    return (min(xs), max(xs), min(ys), max(ys)) if xs else None, sorted(set(bad))


def check(layout, case_file, timeout=240):
    """(ok, reason).  layout: dict or path."""
    mom = Path(case_file) / "momentum"
    with tempfile.TemporaryDirectory(prefix="meshgate_") as td:
        td = Path(td)
        (td / "constant" / "triSurface").mkdir(parents=True)
        lay = td / "layout.json"
        if isinstance(layout, (str, Path)):
            shutil.copy(layout, lay)
        else:
            lay.write_text(json.dumps(layout))
        for f in ("json_to_obj.py", "gen_ground_obj.py"):
            shutil.copy(mom / f, td / f)

        def run(cmd):
            try:
                r = subprocess.run([sys.executable] + cmd, cwd=td, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return None, f"{cmd[0]} did not finish in {timeout} s"
            return r, (r.stdout or "") + (r.stderr or "")

        r, out = run(["json_to_obj.py", "layout.json", "constant/triSurface"])
        if r is None:
            return False, out
        low = out.lower()
        hit = next((k for k in BAD if k in low), None)
        if r.returncode != 0 or hit:
            line = next((l for l in out.splitlines() if hit and hit in l.lower()), out.strip().splitlines()[-1] if out.strip() else "")
            return False, f"json_to_obj.py: {line.strip()[:200]}"
        objs = sorted((td / "constant" / "triSurface").glob("*.obj"))
        b, bad = _obj_bounds(objs)
        if bad:
            return False, f"non-finite coordinates in {', '.join(bad)}"
        if b is None:
            return False, "json_to_obj.py wrote no surface"
        xmin, xmax, ymin, ymax = b
        mx, my = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
        (td / "blockParameters").write_text(f"xMin {xmin - mx};\nxMax {xmax + mx};\nyMin {ymin - my};\nyMax {ymax + my};\nzMin 0;\nzMax 60;\n")
        r, out = run(["gen_ground_obj.py", "blockParameters", "--json", "layout.json", "--workers", "1"])
        if r is None:
            return False, out
        low = out.lower()
        hit = next((k for k in BAD if k in low), None)
        gnd = td / "constant" / "triSurface" / "ground.obj"
        if r.returncode != 0 or hit or not gnd.exists() or gnd.stat().st_size < 100:
            line = next((l for l in out.splitlines() if hit and hit in l.lower()), out.strip().splitlines()[-1] if out.strip() else "no ground.obj")
            return False, f"gen_ground_obj.py: {line.strip()[:200]}"
    return True, "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-file", default=str(Path(__file__).resolve().parent.parent / "caseFile"))
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    if not available(a.case_file):
        sys.exit(f"no caseFile/momentum/json_to_obj.py + gen_ground_obj.py in {a.case_file}")
    bad = 0
    for f in a.files:
        ok, why = check(f, a.case_file)
        print(f"{Path(f).name:<24}{'OK' if ok else 'FAIL  ' + why}")
        bad += not ok
    sys.exit(1 if bad else 0)
