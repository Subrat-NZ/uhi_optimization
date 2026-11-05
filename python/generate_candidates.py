#!/usr/bin/env python3
"""
generate_candidates.py  --  create individual_X.json (initial NSGA-II individuals) / offspring_X.json (mutated children) from the base layout.

    python generate_candidates.py individual --count 10            # 10 random feasible variations of the base (individual_1 ...)
    python generate_candidates.py individual --count 10 --include-base   # also writes individual_0.json = base.json exactly
    python generate_candidates.py offspring --parent input/individual_3.json --count 9   # mutated children (offspring_1 ...)
    python generate_candidates.py template                             # the two seeds: cool wind catcher on every building (heat wind) + wind blocker belt along the windward edge(s) (cold wind), both with dense pathway trees
    python generate_candidates.py template --kind all                  # ... plus the earlier variants (tree wind catcher, tree wind blocker wall, cool wind catcher on one building)
    python generate_candidates.py validate input/individual_3.json     # report broken rules
    python generate_candidates.py plot input/individual_3.json         # base vs candidate picture (more: plot_candidates.py)
    python generate_candidates.py individual --count 10 --plot         # ... and draw them straight away
    (the old names `generation` / `mutation` still work and write individual_X / offspring_X)
    python generate_candidates.py constraints                          # write default input/constraints.json

Folders (relative to the project root, the parent of python/):
    input/base.json          base layout (never modified)
    input/constraints.json   optional, overrides the defaults in layout_ops.DEFAULTS
    input/individual_X.json  / offspring_X.json   written here (what run_candidates.py picks up)
    output/candidates_log.jsonl   one line per file written: operations, seed, parent, violations

Numbering continues from the highest existing X (individual_* and offspring_* are counted separately)
and files are never overwritten.  Every candidate has its own seed (base seed + index), so
`--seed 7` always gives the same set.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout_ops as L  # noqa: E402
import layout_reflow as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- helpers
# internal recipe name -> file prefix: the full recipe makes the INDIVIDUALS of the initial population,
# small perturbations of a parent make OFFSPRING
PREFIX = {"generation": "individual", "mutation": "offspring"}


def next_index(input_dir, prefix):
    pat = re.compile(rf"^{prefix}_(\d+)\.json$")
    nums = [int(m.group(1)) for p in Path(input_dir).glob(f"{prefix}_*.json") if (m := pat.match(p.name))]
    return max(nums) + 1 if nums else 1


def fingerprint(data):
    """Hash of everything the solver sees except bookkeeping (timestamps, counts)."""
    data = L.ground_view(data)             # what stands on a building moves with it: not part of the identity of a layout
    d = {k: data[k] for k in ("buildings", "trees", "ponds", "grassAreas") if k in data}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def build_candidate(parent, base, cfg, seed, n_ops, strength, base_map, kind="mutation"):
    """generation: the full recipe (R.make_generation) + n_ops extra random operations.
    mutation: n_ops small random operations on the parent.  Returns (data, op_list) or None."""
    rng = np.random.default_rng(seed)
    if kind == "generation":
        res = R.make_generation(parent, base, cfg, rng, base_map, n_extra_ops=n_ops)
        if res:
            L.finalize_patches(res[0], cfg)
        return res
    cur, ops = L.working_copy(parent), []
    R.fix_trees(cur, base, cfg, rng)                    # a parent that still breaks a tree rule (e.g. the raw base)
    fixed = L.fit_patch_area(cur, base, cfg, rng)      # e.g. lawn above the water budget -> shrink first
    if fixed:
        if L.new_violations(L.Scene(cur, cfg, base.osm_polys), base, base_map):
            return None
        ops.append({"op": "fit_area_to_budget", "categories": fixed})
    for _ in range(n_ops):
        r = L.try_op(cur, base, cfg, rng, strength, base_map)
        if r is None:
            continue
        cur, info = r
        ops.append(info)
    if not ops:
        return None
    L.finalize_patches(cur, cfg)
    return cur, ops


def make_base_copy(base_data, base, cfg, seed=0):
    """generation_0: the base layout with every rule enforced - lawn / pond totals fitted to the band, trees that
    break a rule (e.g. crowns on a road) moved to the nearest legal place.  Returns (data, notes)."""
    g0 = L.working_copy(base_data)
    notes = {}
    fitted = L.fit_patch_area(g0, base, cfg)
    if fitted:
        notes["area_rescaled"] = fitted
    moved = R.fix_trees(g0, base, cfg, np.random.default_rng(seed))
    if moved:
        notes["trees_moved_off_rule_breaks"] = len(moved)
    L.finalize_patches(g0, cfg)
    L.refresh_metadata(g0)
    return g0, notes


def write_candidate(data, ops, path, log_path, meta, base=None):
    """Write a layout.  `base` (the base Scene): trees / lawns / ponds that stand on a building of the base are put back, each moved
    with its building (layout_ops.attach_hosted) - the optimiser itself never sees them."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if base is not None:
        data = L.attach_hosted(data, base)
    path.write_text(json.dumps(data, indent=2))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(dict(meta, file=path.name, ops=ops,
                                time=time.strftime("%Y-%m-%d %H:%M:%S")), default=str) + "\n")


# ----------------------------------------------------------------------------- commands
def cmd_make(a, kind):
    in_dir, out_dir = ROOT / a.input_dir, ROOT / a.output_dir
    in_dir.mkdir(exist_ok=True)
    base_path = Path(a.base) if a.base else in_dir / "base.json"
    base_data = load(base_path)
    cfg = L.load_constraints(a.constraints or in_dir / "constraints.json")
    base = L.Scene(base_data, cfg, split_hosted=True)
    base_map = L.violation_map(L.check(base, base))
    log = out_dir / "candidates_log.jsonl"
    a0_lawn = L.base_area0(base, cfg, "lawn")
    q = L.lawn_water_l_per_day(cfg, a0_lawn)
    lo_l, hi_l = L.area_band(cfg, "lawn", a0_lawn)
    print(f"lawn irrigation budget: {q:,.0f} L/day at 5 mm/day -> lawn area {lo_l:,.1f}-{hi_l:,.1f} m2 "
          f"(base {a0_lawn:,.1f} m2){'  [from base lawn]' if not cfg['lawn'].get('water_available_l_per_day') else ''}")
    a0_pond = L.base_area0(base, cfg, "pond")
    lo_p, hi_p = L.area_band(cfg, "pond", a0_pond)
    print(f"ponds: {len(base.ponds)} in base -> allowed total {lo_p:,.1f}-{hi_p:,.1f} m2"
          f"{'  [none in base: 10 % of open space]' if not base.ponds else ''}")
    max_shift = cfg["building"]["max_shift_frac_of_site_diameter"] * base.site_diameter
    mins = [min(cfg["building"]["min_shift_frac_of_length"] * ln, 0.98 * max_shift) for ln in base.group_len]
    print(f"building shift ring: {min(mins):.1f}-{max(mins):.1f} m (0.5 x longest span) up to {max_shift:.1f} m, "
          f"{len(base.groups)} buildings; "
          + (f"trees {len(base.tx)} -> +{cfg['generation']['tree_count_increase_frac']:.0%} in generations" if len(base.tx) else
             f"trees: none in base -> {L.tree_count0(base, cfg)} are created ({cfg['tree']['canopy_cover_frac_if_missing']:.0%} canopy cover of the open space)"))
    if not base.lawns:
        print(f"lawns: none in base -> A0 = {a0_lawn:,.0f} m2 ({cfg['lawn']['open_space_frac_if_missing']:.0%} of the open space), placed in the largest open areas")

    parents = []
    if kind == "mutation":
        plist = list(a.parent or [])
        if not plist:
            sys.exit("offspring needs --parent FILE (repeat it, or give several)")
        parents = [(Path(p), load(p)) for p in plist]
    seen = {fingerprint(base_data)}
    for p, d in parents:
        seen.add(fingerprint(d))
    for p in in_dir.glob("*_*.json"):                 # do not re-create something that already exists
        if re.match(r"^(individual|offspring|generation|mutation)_\d+\.json$", p.name):
            seen.add(fingerprint(load(p)))

    idx = next_index(in_dir, PREFIX[kind])
    written, attempts, made = 0, 0, []
    if kind == "generation" and a.include_base and not (in_dir / "individual_0.json").exists():
        (in_dir / "individual_0.json").write_bytes(base_path.read_bytes())
        with open(log, "a") as f:
            f.write(json.dumps({"file": "individual_0.json", "kind": "base (exact copy)"}) + "\n")
        print("wrote individual_0.json (exact copy of the base layout)")

    while written < a.count and attempts < a.count * 10:
        attempts += 1
        seed = a.seed * 1_000_003 + (idx + attempts) * 97
        rng = np.random.default_rng(seed)
        if kind == "generation":
            parent_name, parent = base_path.name, base_data
            n_ops = int(rng.integers(a.min_ops, a.max_ops + 1))          # extra random ops on top of the recipe
            strength = float(rng.uniform(0.6, 1.4)) * a.strength
        else:
            pp, parent = parents[(written + attempts) % len(parents)]
            parent_name = pp.name
            n_ops = int(rng.integers(1, 4))
            strength = float(rng.uniform(0.3, 0.8)) * a.strength
        res = build_candidate(parent, base, cfg, seed, n_ops, strength, base_map, kind)
        if res is None:
            continue
        data, ops = res
        fp = fingerprint(data)
        if fp in seen:
            continue
        new, _inh = L.validate(data, base, cfg, stripped=True)
        if new:                                        # should never happen: try_op already checks
            print(f"  skipped (violations): {[(t, i) for t, i, _ in new][:3]}")
            continue
        seen.add(fp)
        name = f"{PREFIX[kind]}_{idx}.json"
        write_candidate(data, ops, in_dir / name, log,
                        {"kind": kind, "parent": parent_name, "seed": seed, "strength": round(strength, 3),
                         "lawn_water_l_per_day": round(q, 1)}, base=base)
        kinds = ", ".join(sorted({o["op"] for o in ops}))
        print(f"wrote {name:<18} {len(ops)} ops ({kinds})")
        made.append(in_dir / name)
        idx += 1
        written += 1
    if written < a.count:
        print(f"only {written} of {a.count} candidates could be made (constraints too tight? "
              f"try --strength smaller, or raise --count attempts)")
    if getattr(a, "plot", False) and made:
        import plot_candidates as PC
        PC.plot_files(made, base_path, cfg, out_dir, diff=a.diff)
        if len(made) > 1:
            PC.plot_grid(made, base_path, cfg, out_dir)


def cmd_validate(a):
    cfg = L.load_constraints(a.constraints or ROOT / a.input_dir / "constraints.json")
    base_path = Path(a.base) if a.base else ROOT / a.input_dir / "base.json"
    base = L.Scene(load(base_path), cfg, split_hosted=True)
    bad = 0
    for f in a.files:
        new, inh = L.validate(load(f), base, cfg)
        print(f"{Path(f).name}: {'OK' if not new else 'VIOLATIONS'}  new={len(new)}  inherited-from-base={len(inh)}")
        for t, ids, sev in new:
            print(f"    NEW  {t:<26} {', '.join(map(str, ids))}  (by {sev:.3g})")
        if a.show_inherited:
            for t, ids, sev in inh:
                print(f"    base {t:<26} {', '.join(map(str, ids))}  (by {sev:.3g})")
        bad += bool(new)
    sys.exit(1 if bad else 0)


def cmd_constraints(a):
    p = ROOT / a.input_dir / "constraints.json"
    if p.exists() and not a.force:
        sys.exit(f"{p} exists (use --force to overwrite)")
    p.write_text(json.dumps(L.DEFAULTS, indent=2))
    print(f"wrote {p}  - edit any value; missing keys fall back to the defaults")


def cmd_template(a):
    """individual_K.json from the tree wind catcher / wind blocker template (heat-case / cold-case wind)."""
    import layout_templates as TPL
    import run_candidates as RC
    in_dir, out_dir = ROOT / a.input_dir, ROOT / a.output_dir
    base_path = Path(a.base) if a.base else in_dir / "base.json"
    base_data = load(base_path)
    cfg = L.load_constraints(a.constraints or in_dir / "constraints.json")
    base = L.Scene(base_data, cfg, split_hosted=True)
    base_map = L.violation_map(L.check(base, base))
    cases = RC.load_cases(a.cases)
    jdir = Path(a.jsons_dir) if a.jsons_dir else ROOT / "jsons"
    try:
        kinds = TPL.kind_list(a.kind)
    except ValueError as e:
        sys.exit(str(e))
    for kind in kinds:
        spec = TPL.KINDS[kind]
        wind = TPL.season_wind(cases, spec["season"])
        if wind is None:
            print(f"{spec['label']}: not made - the climate has no {spec['season']}-stress case")
            continue
        path = TPL.find_template(kind, jdir, a.template if len(kinds) == 1 else None)
        if path is None:
            print(f"{spec['label']}: not made - template file not found in {jdir} ({' or '.join(spec['files'])})")
            continue
        res, why = TPL.make_individual(base_data, base, cfg, base_map, kind, TPL.load_template(kind, path), wind, seed=a.seed,
                                       ref_building=a.ref_building)
        if res is None:
            print(f"{spec['label']}: not made - {why}")
            continue
        data, ops = res
        name = f"individual_{next_index(in_dir, 'individual')}"
        write_candidate(data, ops, in_dir / f"{name}.json", out_dir / "candidates_log.jsonl",
                        {"kind": "template", "template": path.name, "season": spec["season"]}, base=base)
        o = ops[0]
        import layout_seeds as SD
        txt = SD.describe(kind, f"{name}.json", path, spec, o)
        if txt:
            print(txt)
        elif kind == "cool_wind_catcher":
            print(f"{name}.json  {spec['label']}: {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), reference building "
                  f"{o['reference_building']}, {'mirrored, ' if o['mirrored_about_west_east_axis'] else ''}pond on the {o['side_of_wind']} of the wind, "
                  f"scale {o['scale']:g}, pond {o['pond_area_m2']:g} m2, {o['trees_placed']}/{o['trees_in_template']} trees")
        else:
            print(f"{name}.json  {spec['label']}: {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), rotated "
                  f"{o['rotation_deg']:g} deg, scale {o['scale']:g}, {o['trees_placed']}/{o['trees_in_template']} trees, "
                  f"{o['distance_from_upwind_site_edge_m']:g} m from the upwind site edge")


def cmd_plot(a):
    import plot_candidates as PC
    cfg = L.load_constraints(a.constraints or ROOT / a.input_dir / "constraints.json")
    base_path = Path(a.base) if a.base else ROOT / a.input_dir / "base.json"
    out_dir = ROOT / a.output_dir
    obj = a.objectives or (str(out_dir / "objectives.csv") if (out_dir / "objectives.csv").exists() else None)
    if a.grid:
        PC.plot_grid(a.files, base_path, cfg, out_dir, obj, a.cols)
    else:
        PC.plot_files(a.files, base_path, cfg, out_dir, obj, a.diff)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--base", help="base layout (default input/base.json)")
    ap.add_argument("--constraints", help="constraints JSON (default input/constraints.json if present)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for k in ("individual", "offspring", "generation", "mutation"):
        s = sub.add_parser(k)
        s.add_argument("--count", type=int, default=10)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--strength", type=float, default=1.0, help="scale of all moves (1 = default)")
        s.add_argument("--min-ops", type=int, default=1, help="individual: extra random operations after the recipe")
        s.add_argument("--max-ops", type=int, default=4)
        s.add_argument("--plot", action="store_true", help="also write base-vs-candidate PNGs (+ overview.png) to the output dir")
        s.add_argument("--diff", action="store_true", help="with --plot: add the 'what changed' panel")
        if k in ("individual", "generation"):
            s.add_argument("--include-base", action="store_true", help="also write individual_0.json = base")
        else:
            s.add_argument("--parent", action="append", help="parent layout file (repeatable)")
    s = sub.add_parser("template", help="individual from the tree wind catcher (heat wind) / wind blocker (cold wind) template")
    s.add_argument("--kind", default="default",
                   help="'default' (cool_wind_catcher_all + wind_blocker_edge), 'all' (every kind incl. the earlier variants) or a comma list of: "
                        "cool_wind_catcher_all, wind_blocker_edge, wind_catcher, wind_blocker, cool_wind_catcher")
    s.add_argument("--ref-building", default=None, help="only for --kind cool_wind_catcher: building id to hang it on (default: best fit)")
    s.add_argument("--template", default=None, help="template file (only with a single --kind; default jsons/tree_wind_*.json)")
    s.add_argument("--jsons-dir", default=None)
    s.add_argument("--cases", default=None, help="*_cases.json (default: the one in jsons/)")
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("validate")
    s.add_argument("files", nargs="+")
    s.add_argument("--show-inherited", action="store_true")
    s = sub.add_parser("plot", help="base-vs-candidate pictures (see plot_candidates.py for all options)")
    s.add_argument("files", nargs="+")
    s.add_argument("--diff", action="store_true", help="add a 'what changed' panel")
    s.add_argument("--grid", action="store_true", help="one overview.png with all files")
    s.add_argument("--cols", type=int, default=3)
    s.add_argument("--objectives", help="objectives.csv to print f1/f2 in the titles")
    s = sub.add_parser("constraints")
    s.add_argument("--force", action="store_true")

    a = ap.parse_args()
    if a.cmd in ("individual", "offspring", "generation", "mutation"):
        cmd_make(a, {"individual": "generation", "offspring": "mutation"}.get(a.cmd, a.cmd))
    elif a.cmd == "template":
        cmd_template(a)
    elif a.cmd == "validate":
        cmd_validate(a)
    elif a.cmd == "plot":
        cmd_plot(a)
    else:
        cmd_constraints(a)


if __name__ == "__main__":
    main()
