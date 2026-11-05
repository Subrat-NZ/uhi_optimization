#!/usr/bin/env python3
"""
nsga2.py  --  NSGA-II for the UHI layout optimisation (resumable).

    python python/nsga2.py run                      # population -> simulate -> offspring -> select -> repeat
    python python/nsga2.py run --gens 8             # continue / resume up to 8 generations (state: output/nsga_state.json)
    python python/nsga2.py status                   # generation, front size, hypervolume per generation
    python python/nsga2.py top5                     # the five picks from the Pareto front -> output/top5.json, input/top5_1..5.json
    python python/nsga2.py plot                     # output/nsga_report.png (Pareto front, hypervolume) + output/overview.png
    python python/nsga2.py exclude offspring_20     # drop a layout whose result you do not trust (it is treated as failed)
    python python/nsga2.py run --evaluator mock     # synthetic objectives (NOT physics) to test the loop

Objectives (both minimised): how far the site-averaged UTCI is from the comfortable band (default 9 .. 26 C)
    f1 = max(0, UTCI_heat - 26)          peak-heat hour: heat stress above the band
    f2 = max(0, 9 - UTCI_cold)           peak-cold hour: cold stress below the band
(0 = inside the comfortable band.)

Algorithm
    initial population   the base layout exactly as in base.json (individual_0) + --pop INDIVIDUALS (individual_1 ...):
                         - a COOL WIND CATCHER individual (jsons/cool_wind_catcher.json: a pond + curved tree rows on EVERY building,
                           aligned to the HEAT-case wind; the pond area of the layout is shared between the buildings)
                         - a WIND BLOCKER individual (jsons/tree_wind_breaker.json: a belt of staggered tree rows along the COMPLETE
                           windward edge(s) of the site for the COLD-case wind)
                         - both with dense trees on BOTH sides of every pathway / road (layout_seeds.py)
                           (only when that season exists in the climate; they count within --pop; --no-templates switches them off;
                           --template-kinds all adds the earlier single-building / wall variants)
                         - the rest made by the generator
                         every one is simulated for the heat and the cold hour (8 simulations at a time = --batch 4 layouts)
    each generation      1. rank the population: fast non-dominated sorting, then crowding distance
                         2. --offspring children: binary tournament (lower rank, then larger crowding) picks two parents,
                            layout crossover (probability --pc), then mutation (1-3 small operations) with probability --pm
                            (default 0.25; a child that was not crossed is always mutated); all Stage 1 rules hold for every child
                         3. the children (input/offspring_K.json) are simulated
                         4. parents + children are ranked together and the best N survive (elitist, mu + lambda)
    result               Pareto front of the final population, 5 picks: best heat, best cold, knee, two spread along the front

Files:  input/individual_K.json   initial individuals           input/offspring_K.json   children (individual_0 = base)
        output/nsga_state.json    resumable state               output/nsga_history.csv, nsga_gen_XX.csv   per generation
        output/top5.json          the five picks (copies: input/top5_K.json)
A layout that failed to simulate gets objective 1e6 (it is dominated and dies out); it is simulated again only with --retry-failed.
Simulations that finished earlier are never repeated (run_candidates.py skips them).
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_candidates as G  # noqa: E402
import layout_cross as X  # noqa: E402
import layout_ops as L  # noqa: E402
import layout_templates as TPL  # noqa: E402
import layout_seeds as SD  # noqa: E402
import mesh_gate as MG  # noqa: E402
import run_candidates as RC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PENALTY = RC.PENALTY

# ----------------------------------------------------------------------------- --workers: layouts are made in parallel processes
# The worker processes are forked from the main process AFTER the base layout is loaded, so they share it (nothing big is pickled).
_W = {}


def n_workers(a):
    """--workers: 1 = everything in this process (default, the old behaviour); 0 = one worker per CPU core; N = N processes."""
    w = int(getattr(a, "workers", 1))
    return max(1, os.cpu_count() or 1) if w <= 0 else w


def _job_individual(seed, n_ops, strength):
    return G.build_candidate(_W["base_data"], _W["base"], _W["cfg"], seed, n_ops, strength, _W["base_map"], "generation")


def _job_template(kind, tpl_path, wind, seed, ref):
    try:
        return TPL.make_individual(_W["base_data"], _W["base"], _W["cfg"], _W["base_map"], kind, TPL.load_template(kind, tpl_path), wind,
                                   seed=seed, ref_building=ref)
    except Exception as e:                                   # never stop the optimisation for a seed
        return None, f"{type(e).__name__}: {e}"


def _job_child(pa, pb, rng_state, n_mut, strength):
    """One child: crossover of pa and pb (or a copy of pa when pb is None), then n_mut mutations; returns (data, ops) or None.
    The rule check of the child is done here too (layout_ops.new_violations)."""
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    base, cfg, bm = _W["base"], _W["cfg"], _W["base_map"]
    ops = []
    if pb is not None:
        r = X.crossover(pa, pb, base, cfg, rng, bm)
        if r is None:
            return None
        data, info = r
        ops = [info]
    else:
        data = L.working_copy(pa)
    if n_mut:
        data, mops = X.mutate(data, base, cfg, rng, bm, n_mut, strength)
        ops += mops
    if not ops:
        return None
    if L.new_violations(L.Scene(data, cfg, base.osm_polys), base, bm):
        return None
    return data, ops


def _make_pool(run, workers):
    """A pool of `workers` forked processes that share this run's base layout (see _W)."""
    _W.update(base_data=run.base_data, base=run.base, cfg=run.cfg, base_map=run.base_map)
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"))


# ----------------------------------------------------------------------------- NSGA-II core
def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_sort(F):
    """Fronts (lists of row indices), front 0 = Pareto optimal."""
    n = len(F)
    dom_by = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                dom_by[i].append(j)
                cnt[j] += 1
            elif dominates(F[j], F[i]):
                dom_by[j].append(i)
                cnt[i] += 1
    fronts, cur = [], [i for i in range(n) if cnt[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in dom_by[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def crowding_distance(F, front):
    d = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for m in range(F.shape[1]):
        order = sorted(front, key=lambda i: F[i, m])
        lo, hi = F[order[0], m], F[order[-1], m]
        d[order[0]] = d[order[-1]] = math.inf
        if hi - lo <= 0:
            continue
        for k in range(1, len(order) - 1):
            d[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / (hi - lo)
    return d


def rank_and_crowd(F):
    fronts = nondominated_sort(F)
    rank, crowd = np.zeros(len(F), int), np.zeros(len(F))
    for r, fr in enumerate(fronts):
        cd = crowding_distance(F, fr)
        for i in fr:
            rank[i], crowd[i] = r, cd[i]
    return fronts, rank, crowd


def select_survivors(F, n):
    fronts, _, _ = rank_and_crowd(F)
    keep = []
    for fr in fronts:
        if len(keep) + len(fr) <= n:
            keep += fr
        else:
            cd = crowding_distance(F, fr)
            keep += sorted(fr, key=lambda i: -cd[i])[:n - len(keep)]
            break
    return keep


def hypervolume_2d(F, ref):
    pts = sorted((p for p in F if p[0] < ref[0] and p[1] < ref[1]), key=lambda p: p[0])
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def pick_top_roles(F, front, k=5):
    """Picks from the Pareto front, ALWAYS in this order:  best f1 (heat), best f2 (cold), the knee (farthest from the line between
    the two extremes), then farthest-point sampling in objective space normalised to [0, 1].  A front with fewer than k members
    is returned complete, in the same order.  Returns (indices into F, role names)."""
    front = list(front)
    if not front:
        return [], []
    P = F[front]
    lo, hi = P.min(0), P.max(0)
    Z = (P - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    k = min(k, len(front))
    sel, roles = [int(np.argmin(Z[:, 0]))], ["best heat (min f1)"]
    j = int(np.argmin(Z[:, 1]))
    if j not in sel:
        sel.append(j)
        roles.append("best cold (min f2)")
    a, b = Z[sel[0]], Z[sel[-1]]
    ab = b - a
    if len(sel) > 1 and np.linalg.norm(ab) > 1e-12 and len(sel) < k:
        dist = np.abs(ab[0] * (Z[:, 1] - a[1]) - ab[1] * (Z[:, 0] - a[0])) / np.linalg.norm(ab)
        dist[sel] = -1
        kn = int(np.argmax(dist))
        if dist[kn] > 0:
            sel.append(kn)
            roles.append("knee")
    while len(sel) < k:
        d = np.min(np.linalg.norm(Z[:, None, :] - Z[None, sel, :], axis=2), axis=1)
        d[sel] = -1
        sel.append(int(np.argmax(d)))
        roles.append("spread along the front")
    return [front[i] for i in sel], roles


def pick_top(F, front, k=5):
    return pick_top_roles(F, front, k)[0]


# ----------------------------------------------------------------------------- evaluators
def mock_objectives(data, base, cfg):
    """SYNTHETIC objectives with a heat/cold trade-off - only to test the loop, NOT a physical model.
    f1 falls with canopy and ponds and rises with compactness; f2 rises with canopy (it blocks winter sun)."""
    sc = L.Scene(data, cfg, base.osm_polys)
    canopy = float(np.sum(math.pi * sc.tR ** 2))
    pond = sum(p.area for p in sc.ponds)
    cents = np.array([sc.centroid(g) for g in range(len(sc.groups))])
    d = np.hypot(cents[:, None, 0] - cents[None, :, 0], cents[:, None, 1] - cents[None, :, 1]) + np.eye(len(cents)) * 1e9
    gap = float(np.mean(d.min(axis=1)))
    f1 = 38.0 - 0.0010 * canopy - 0.0004 * pond + 0.03 * max(0.0, 60.0 - gap)
    utci_cold = -4.0 - 0.0006 * canopy + 0.04 * max(0.0, 60.0 - gap)
    return {"status": "ok", "utci_heat": f1, "utci_cold": utci_cold, "f1": f1, "f2": -utci_cold}


# ----------------------------------------------------------------------------- driver
def migrate_legacy(in_dir, run_dir):
    """Earlier versions called the files generation_K / mutation_K.  Rename them (and their finished simulations in run/)
    to individual_K / offspring_K so nothing that already ran has to be simulated again."""
    ren = {"generation": "individual", "mutation": "offspring"}
    n = 0
    for p in sorted(Path(in_dir).glob("*_*.json")):
        old, _, k = p.stem.rpartition("_")
        if old not in ren or not k.isdigit():
            continue
        new = f"{ren[old]}_{k}"
        if (Path(in_dir) / f"{new}.json").exists():
            continue
        p.rename(Path(in_dir) / f"{new}.json")
        d = Path(run_dir) / p.stem
        if d.is_dir() and not (Path(run_dir) / new).exists():
            d.rename(Path(run_dir) / new)
            for j in (Path(run_dir) / new).glob(f"*/{p.stem}.json"):
                j.rename(j.with_name(f"{new}.json"))
        n += 1
    if n:
        print(f"renamed {n} earlier files generation_K / mutation_K -> individual_K / offspring_K (input/ and run/)")


class Run:
    def __init__(self, a):
        self.a = a
        self.in_dir = Path(a.input_dir) if os.path.isabs(a.input_dir) else ROOT / a.input_dir
        self.out_dir = Path(a.output_dir) if os.path.isabs(a.output_dir) else ROOT / a.output_dir
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if a.cmd == "run" and not (self.out_dir / "nsga_state.json").exists():
            migrate_legacy(self.in_dir, Path(a.run_dir) if a.run_dir else ROOT / "run")
        self.cfg = L.load_constraints(a.constraints or self.in_dir / "constraints.json")
        self.base_path = Path(a.base) if a.base else self.in_dir / "base.json"
        self.base_data = G.load(self.base_path)
        self.base = L.Scene(self.base_data, self.cfg, split_hosted=True)
        self.base_map = L.violation_map(L.check(self.base, self.base))
        self.state_path = self.out_dir / "nsga_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else None
        self.cache = {}
        self.feat = {}                        # novelty features of every individual / offspring (filled when needed)
        self.log = self.out_dir / "candidates_log.jsonl"
        self.seen = {G.fingerprint(self.base_data)}
        for p in self.in_dir.glob("*_*.json"):
            if p.stem.split("_")[0] in ("individual", "offspring", "generation", "mutation") and p.stem.split("_")[-1].isdigit():
                self.seen.add(G.fingerprint(G.load(p)))

    # ---- persistence
    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def near_copy(self, data):
        """Name of an existing individual / offspring that `data` is practically a copy of (layout_ops.too_similar), else None."""
        if not self.cfg.get("novelty", {}).get("enabled", True):
            return None
        if not self.feat:
            for p in sorted(self.in_dir.glob("*_*.json")):
                if p.stem.split("_")[0] in ("individual", "offspring") and p.stem.split("_")[-1].isdigit():
                    self.feat[p.stem] = L.novelty_features(self.layout(p.stem))
        f = L.novelty_features(data)
        for name, g in self.feat.items():
            if L.too_similar(f, g, self.cfg):
                return name
        return None

    def layout(self, name):
        if name not in self.cache:
            self.cache[name] = G.load(self.in_dir / f"{name}.json")
        return self.cache[name]

    def F(self, names):
        out = []
        for n in names:
            e = self.state["evals"].get(n)
            out.append([e["f1"], e["f2"]] if e and e.get("f1") is not None else [PENALTY, PENALTY])
        return np.array(out, float)

    # ---- evaluation
    def comfort(self):
        c = (self.state or {}).get("comfort") or list(self.a.comfort)
        return float(c[0]), float(c[1])

    def score(self, utci_heat, utci_cold):
        """Distance of the two site-averaged UTCI values from the comfortable band (0 = inside the band)."""
        lo, hi = self.comfort()
        f1 = max(0.0, float(utci_heat) - hi) if utci_heat is not None else 0.0
        f2 = max(0.0, lo - float(utci_cold)) if utci_cold is not None else 0.0
        return {"status": "ok", "utci_heat": utci_heat, "utci_cold": utci_cold, "f1": round(f1, 4), "f2": round(f2, 4)}

    def evaluate(self, names):
        a, ev = self.a, self.state["evals"]
        todo = [n for n in names if n not in ev or (ev[n].get("status") != "ok" and a.retry_failed)]
        if not todo:
            return
        print(f"simulating {len(todo)} layout(s) ({a.max_parallel or 2 * a.batch} simulations at a time, evaluator '{a.evaluator}') ...", flush=True)
        t0 = time.time()
        if a.evaluator == "mock":
            for n in todo:
                m = mock_objectives(self.layout(n), self.base, self.cfg)
                ev[n] = self.score(m["utci_heat"], m["utci_cold"])
        else:
            out = RC.run_batch(todo, case_file=a.case_file, run_dir=a.run_dir, cand_dir=self.in_dir,
                               cases_path=a.cases, max_parallel=a.max_parallel or 2 * a.batch,
                               timeout_min=a.timeout_min, summary_dir=self.out_dir, force=False)
            for n in todo:
                o = out["objectives"].get(n)
                ev[n] = (self.score(o.get("utci_heat"), o.get("utci_cold")) if o and o.get("status") == "ok"
                         else {"status": "failed", "f1": PENALTY, "f2": PENALTY})
        bad = [n for n in todo if ev[n].get("status") != "ok"]
        print(f"  done in {time.time() - t0:.0f} s" + (f"; failed: {', '.join(bad)}" if bad else ""), flush=True)
        self.save()

    # ---- candidates
    def _write(self, data, ops, name, meta):
        G.write_candidate(data, ops, self.in_dir / f"{name}.json", self.log, meta, base=self.base)
        self.seen.add(G.fingerprint(data))

    def install_true_base(self):
        """individual_0.json = base.json exactly as you gave it (no lawn / pond / tree added, nothing moved).
        Versions before this one wrote a copy with the rules enforced (lawns and ponds created for a base that has none);
        such a file and its simulation are set aside, so the real base is simulated.  Returns True if the file was replaced."""
        p = self.in_dir / "individual_0.json"
        if p.exists():
            try:
                if G.load(p) == self.base_data:
                    return False
            except Exception:
                pass
            k, old = 0, self.in_dir / "individual_0_rules_enforced.json.old"
            while old.exists():
                k += 1
                old = self.in_dir / f"individual_0_rules_enforced_{k}.json.old"
            p.rename(old)
            rd = Path(self.a.run_dir) if getattr(self.a, "run_dir", None) else ROOT / "run"
            moved = ""
            if (rd / "individual_0").exists():
                dst = rd / "individual_0_rules_enforced"
                while dst.exists():
                    dst = Path(str(dst) + "_")
                (rd / "individual_0").rename(dst)
                moved = f"; its simulation folder is now {dst}"
            print(f"individual_0 was a copy of the base with lawns / ponds added: set aside as input/{old.name}{moved}")
        shutil.copyfile(self.base_path, p)
        self.cache.pop("individual_0", None)
        with open(self.log, "a") as f:
            f.write(json.dumps({"file": "individual_0.json", "kind": "base (exact copy of " + self.base_path.name + ")"}) + "\n")
        print(f"individual_0 = {self.base_path.name} exactly as it is (the real base layout)")
        return True

    def refresh_base(self):
        """Resumed run: make sure the REAL base is in the population (see install_true_base) and simulated."""
        st = self.state
        changed = self.install_true_base()
        known = "individual_0" in st["evals"] or "individual_0" in st["origin"]
        if changed:
            st["evals"].pop("individual_0", None)
        if not changed and known:
            return
        st["origin"]["individual_0"] = {"gen": 0, "kind": "base"}
        if not st["population"]:                       # initial layouts still waiting: the base is simulated with them
            if "individual_0" not in st["pending"]:
                st["pending"].insert(0, "individual_0")
            self.save()
            return
        self.evaluate(["individual_0"])
        pop = st["population"]
        if "individual_0" not in pop:                  # let the base compete for a place
            allnames = pop + ["individual_0"]
            keep = select_survivors(self.F(allnames), st.get("size", len(pop)))
            st["population"] = [allnames[i] for i in keep]
        e = st["evals"].get("individual_0", {})
        print(f"base layout: UTCI heat {e.get('utci_heat')}, cold {e.get('utci_cold')}  (f1 {e.get('f1')}, f2 {e.get('f2')})"
              + ("  - in the population" if "individual_0" in st["population"] else "  - not among the best N, kept as reference"))
        self.record(st["gen"])

    def _logged_template(self, name):
        """kind ('wind_catcher' / 'wind_blocker') if candidates_log.jsonl says that `name` was made from a template."""
        if not self.log.exists():
            return None
        kind = None
        for line in self.log.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("file") == f"{name}.json":
                ops = r.get("ops") or []
                kind = ops[0].get("kind") if ops and ops[0].get("op") == "template" else None
        return kind

    def add_template_individuals(self, names, limit):
        """Cool wind catcher on every building (heat-case wind) and wind blocker belt (cold-case wind) individuals, both with dense
        pathway trees (--template-kinds picks others).  Appends their names to
        `names` (at most `limit` of them) and records their origin.  A season that is absent from the climate, a missing
        template file or a pattern that cannot be placed only skips that individual (with a message)."""
        a, st = self.a, self.state
        if not a.templates or limit <= 0:
            return
        try:
            cases = RC.load_cases(a.cases)
        except SystemExit as e:
            print(f"  template individuals skipped: {e}")
            return
        jdir = Path(a.jsons_dir) if a.jsons_dir else ROOT / "jsons"
        have = {}
        for p in self.in_dir.glob("individual_*.json"):
            k = self._logged_template(p.stem)
            if k:
                have[k] = p.stem
        idx = G.next_index(self.in_dir, "individual")
        made = 0
        try:
            kinds = TPL.kind_list(a.template_kinds)
        except ValueError as e:
            print(f"  template individuals skipped: {e}")
            return
        W = n_workers(a)
        pool, fut = getattr(self, "_pool", None), {}
        if W > 1 and pool is not None:              # --workers: the kinds that will be made are computed at the same time
            want, k = [], 0
            for kind in kinds:
                spec = TPL.KINDS[kind]
                if k >= limit:
                    break
                wind = TPL.season_wind(cases, spec["season"])
                if wind is None:
                    continue
                if kind in have and have[kind] not in names:
                    k += 1
                    continue
                path = TPL.find_template(kind, jdir, getattr(a, TPL.ARG_DEST.get(kind, kind + "_template"), None))
                if path is not None:
                    want.append((kind, path, wind))
                    k += 1
            fut = {kind: pool.submit(_job_template, kind, path, wind, a.seed, a.cool_reference_building) for kind, path, wind in want}
            # the workers that have no template to compute start on random individuals right away (consumed by _individuals_parallel)
            for _ in range(max(0, W - len(fut))):
                self._submit_individual(pool, len(self._pre) + 1)
        for kind in kinds:
            spec = TPL.KINDS[kind]
            if made >= limit:
                break
            wind = TPL.season_wind(cases, spec["season"])
            if wind is None:
                print(f"  {spec['label']}: not made - the climate has no {spec['season']}-stress case (status "
                      f"{cases.get(spec['season'] + '_status')!r})")
                continue
            if kind in have and have[kind] not in names:
                names.append(have[kind])
                st["origin"][have[kind]] = {"gen": 0, "kind": f"template:{kind}", "season": spec["season"]}
                print(f"  {spec['label']}: {have[kind]} (already there)")
                made += 1
                continue
            path = TPL.find_template(kind, jdir, getattr(a, TPL.ARG_DEST.get(kind, kind + "_template"), None))
            if path is None:
                opt = TPL.OPTION[kind]
                print(f"  {spec['label']}: not made - template file not found in {jdir} (expected "
                      f"{' or '.join(spec['files'])}; or give --{opt}-template)")
                continue
            if kind in fut:
                res, why = fut[kind].result()
            else:
                try:
                    res, why = TPL.make_individual(self.base_data, self.base, self.cfg, self.base_map, kind,
                                                   TPL.load_template(kind, path), wind, seed=a.seed,
                                                   ref_building=a.cool_reference_building)
                except Exception as e:                                   # never stop the optimisation for a seed
                    res, why = None, f"{type(e).__name__}: {e}"
            if res is None:
                print(f"  {spec['label']}: not made - {why}")
                continue
            data, ops = res
            name = f"individual_{idx}"
            idx += 1
            self._write(data, ops, name, {"kind": "template", "template": path.name, "season": spec["season"]})
            names.append(name)
            st["origin"][name] = {"gen": 0, "kind": f"template:{kind}", "season": spec["season"],
                                  "wind_from_deg_model": ops[0]["wind_from_deg_model"], "scale": ops[0]["scale"]}
            o = ops[0]
            txt = SD.describe(kind, name, path, spec, o)
            if txt:
                print(f"  {txt}")
            elif kind == "cool_wind_catcher":
                print(f"  {spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), "
                      f"reference building {o['reference_building']} ({o['reference_building_size_m'][0]:g} x {o['reference_building_size_m'][1]:g} m), "
                      f"{'mirrored, ' if o['mirrored_about_west_east_axis'] else ''}pond on the {o['side_of_wind']} of the wind, "
                      f"scale {o['scale']:g}, pond {o['pond_area_m2']:g} m2, {o['trees_placed']} of {o['trees_in_template']} trees")
            else:
                print(f"  {spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg "
                      f"(model), rotated {o['rotation_deg']:g} deg, scale {o['scale']:g}, {o['trees_placed']} of "
                      f"{o['trees_in_template']} trees, {o['distance_from_upwind_site_edge_m']:g} m from the upwind site edge"
                      f"{'' if o['close_to_inlet'] else '  (NOT close to the inlet - no room there)'}")
            made += 1

    def init_population(self):
        a = self.a
        st = {"pop": a.pop, "seed": a.seed, "gen": 0, "population": [], "evals": {}, "origin": {}, "pending": [],
              "ref": None, "history": [], "evaluator": a.evaluator, "comfort": list(a.comfort)}
        self.state = st
        names = []
        if a.include_base:
            self.install_true_base()
            names.append("individual_0")
            st["origin"]["individual_0"] = {"gen": 0, "kind": "base"}
        n_ind = a.pop + len(names)          # a.pop individuals in addition to the base layout
        W = n_workers(a)
        self._pool, self._pre = (_make_pool(self, W) if W > 1 else None), []
        try:
            self._init_individuals(names, n_ind, st)
        finally:
            if self._pool is not None:
                self._pool.shutdown(cancel_futures=True)
                self._pool = None
        st["pending"] = names
        st["size"] = len(names)             # population size N (individuals + base)
        self.save()
        nt = sum(1 for n in names if str(st["origin"].get(n, {}).get("kind", "")).startswith("template:"))
        print(f"initial population: {len(names)} layouts = {'the base layout + ' if a.include_base else ''}{len(names) - int(a.include_base)} individuals "
              f"({nt} from tree templates; {', '.join(names[:4])} ...)")
        return names

    def _init_individuals(self, names, n_ind, st):
        a = self.a
        self.add_template_individuals(names, n_ind - len(names))   # wind catcher (heat wind) + wind blocker (cold wind)
        existing = sorted((p.stem for p in self.in_dir.glob("individual_*.json") if p.stem != "individual_0"),
                          key=lambda s: int(s.split("_")[1]))
        for n in existing:
            if len(names) < n_ind and n not in names:
                names.append(n)
                st["origin"][n] = {"gen": 0, "kind": "individual"}
        idx, attempts = G.next_index(self.in_dir, "individual"), 0
        W = n_workers(a)
        if W > 1 and len(names) < n_ind:
            idx, attempts = self._individuals_parallel(names, n_ind, idx, W)
        while len(names) < n_ind and attempts < a.pop * 20:
            attempts += 1
            seed = a.seed * 1_000_003 + (idx + attempts) * 97
            rng = np.random.default_rng(seed)
            res = G.build_candidate(self.base_data, self.base, self.cfg, seed, int(rng.integers(1, 5)),
                                    float(rng.uniform(0.6, 1.4)), self.base_map, "generation")
            if res is None or G.fingerprint(res[0]) in self.seen:
                continue
            name = f"individual_{idx}"
            self._write(res[0], res[1], name, {"kind": "individual", "parent": self.base_path.name, "seed": seed})
            names.append(name)
            st["origin"][name] = {"gen": 0, "kind": "individual", "seed": seed}
            idx += 1

    def _submit_individual(self, pool, t):
        """Random-individual attempt t (seed independent of how many layouts exist, so it can start before the templates are done)."""
        a = self.a
        seed = a.seed * 1_000_003 + t * 97
        rng = np.random.default_rng(seed)
        self._pre.append((seed, pool.submit(_job_individual, seed, int(rng.integers(1, 5)), float(rng.uniform(0.6, 1.4)))))

    def _individuals_parallel(self, names, n_ind, idx, W):
        """--workers: random individuals made by W processes at a time.  Attempt t uses seed  seed*1000003 + t*97  (another seed scheme than the
        one-process run, so the layouts differ from --workers 1, but they are the same for every --workers > 1)."""
        a, st, pool = self.a, self.state, self._pool
        t, limit = len(self._pre), a.pop * 20
        batch, self._pre = self._pre, []
        while len(names) < n_ind and (batch or t < limit):
            while len(batch) < min(W, max(1, n_ind - len(names))) and t < limit:
                t += 1
                self._submit_individual(pool, t)
                batch += self._pre
                self._pre = []
            for seed, f in batch:
                res = f.result()
                if res is None or G.fingerprint(res[0]) in self.seen or len(names) >= n_ind:
                    continue
                name = f"individual_{idx}"
                self._write(res[0], res[1], name, {"kind": "individual", "parent": self.base_path.name, "seed": seed})
                names.append(name)
                st["origin"][name] = {"gen": 0, "kind": "individual", "seed": seed}
                idx += 1
            batch = []
        return idx, t

    def _children_parallel(self, gen, pop, rank, crowd, n_kids, idx):
        """--workers: children made by W processes at a time (crossover / mutation / rule check in the workers; novelty check, mesh gate and
        writing here, in attempt order).  Attempt t has its own random stream default_rng([seed, gen, t])."""
        a, st = self.a, self.state
        W = n_workers(a)
        kids, attempts, n_copies, n_unmeshable, t = [], 0, 0, 0, 0
        gate = getattr(a, "mesh_gate", True) and MG.available(a.case_file or ROOT / "caseFile")
        with _make_pool(self, W) as pool:
            while len(kids) < n_kids and t < n_kids * 40:
                batch = []
                for _ in range(min(W, max(1, n_kids - len(kids)))):
                    t += 1
                    rng = np.random.default_rng([a.seed, gen, t])
                    tourn = lambda: min(rng.choice(len(pop), 2, replace=False), key=lambda i: (rank[i], -crowd[i])) if len(pop) > 1 else 0
                    ia, ib = int(tourn()), int(tourn())
                    pa, pb = self.layout(pop[ia]), self.layout(pop[ib])
                    if ia != ib and rng.random() < a.pc:
                        parents, cross = [pop[ia], pop[ib]], True
                        n_mut = int(rng.integers(a.mut_ops[0], a.mut_ops[1] + 1)) if rng.random() < a.pm else 0
                    else:
                        parents, cross = [pop[ia]], False
                        n_mut = int(rng.integers(max(1, a.mut_ops[0]), max(1, a.mut_ops[1]) + 1))
                    strength = float(rng.uniform(0.3, 0.8))
                    f = pool.submit(_job_child, pa, pb if cross else None, rng.bit_generator.state, n_mut, strength)
                    batch.append((parents, cross, n_mut, f))
                for parents, cross, n_mut, f in batch:
                    res = f.result()
                    if res is None or len(kids) >= n_kids:
                        continue
                    data, ops = res
                    if G.fingerprint(data) in self.seen:
                        continue
                    if self.near_copy(data):
                        n_copies += 1
                        continue
                    if gate:
                        ok_mesh, why = MG.check(L.attach_hosted(data, self.base), a.case_file or ROOT / "caseFile")
                        if not ok_mesh:
                            n_unmeshable += 1
                            print(f"  child rejected by the mesh gate: {why}", flush=True)
                            continue
                    name = f"offspring_{idx}"
                    self._write(data, ops, name, {"kind": "nsga_child", "parents": parents, "gen": gen})
                    self.cache[name] = data
                    self.feat[name] = L.novelty_features(data)
                    st["origin"][name] = {"gen": gen, "kind": ("crossover+mutation" if n_mut else "crossover") if cross else "mutation", "parents": parents}
                    kids.append(name)
                    idx += 1
        return kids, n_copies, n_unmeshable

    def make_children(self, gen):
        a, st = self.a, self.state
        pop = st["population"]
        F = self.F(pop)
        fronts, rank, crowd = rank_and_crowd(F)
        rng = np.random.default_rng([a.seed, gen])
        idx = G.next_index(self.in_dir, "offspring")
        kids, attempts, n_copies, n_unmeshable = [], 0, 0, 0
        gate = getattr(a, "mesh_gate", True) and MG.available(a.case_file or ROOT / "caseFile")
        tourn = lambda: min(rng.choice(len(pop), 2, replace=False), key=lambda i: (rank[i], -crowd[i])) if len(pop) > 1 else 0
        n_kids = a.offspring or a.batch
        if n_workers(a) > 1:
            kids, n_copies, n_unmeshable = self._children_parallel(gen, pop, rank, crowd, n_kids, idx)
            attempts = n_kids * 40                    # (the loop below is the one-process version)
        while len(kids) < n_kids and attempts < n_kids * 40:
            attempts += 1
            ia, ib = int(tourn()), int(tourn())
            pa, pb = self.layout(pop[ia]), self.layout(pop[ib])
            ops, parents = [], [pop[ia]]
            if ia != ib and rng.random() < a.pc:
                r = X.crossover(pa, pb, self.base, self.cfg, rng, self.base_map)
                if r is None:
                    continue
                data, info = r
                ops, parents = [info], [pop[ia], pop[ib]]
                n_mut = int(rng.integers(a.mut_ops[0], a.mut_ops[1] + 1)) if rng.random() < a.pm else 0
            else:
                data = L.working_copy(pa)
                n_mut = int(rng.integers(max(1, a.mut_ops[0]), max(1, a.mut_ops[1]) + 1))
            if n_mut:
                data, mops = X.mutate(data, self.base, self.cfg, rng, self.base_map, n_mut, float(rng.uniform(0.3, 0.8)))
                ops += mops
            if not ops or G.fingerprint(data) in self.seen:
                continue
            twin = self.near_copy(data)
            if twin:                                      # e.g. a 5 cm lawn change: simulating it would only add noise
                n_copies += 1
                continue
            if L.new_violations(L.Scene(data, self.cfg, self.base.osm_polys), self.base, self.base_map):
                continue
            if gate:                                      # the case's own json_to_obj.py / gen_ground_obj.py must be able to mesh it
                ok_mesh, why = MG.check(L.attach_hosted(data, self.base), a.case_file or ROOT / "caseFile")
                if not ok_mesh:
                    n_unmeshable += 1
                    print(f"  child rejected by the mesh gate: {why}", flush=True)
                    continue
            name = f"offspring_{idx}"
            self._write(data, ops, name, {"kind": "nsga_child", "parents": parents, "gen": gen})
            self.cache[name] = data
            self.feat[name] = L.novelty_features(data)
            st["origin"][name] = {"gen": gen, "kind": ("crossover+mutation" if n_mut else "crossover") if len(parents) == 2 else "mutation", "parents": parents}
            kids.append(name)
            idx += 1
        if n_copies:
            print(f"  {n_copies} child(ren) rejected as practically identical to an existing individual (novelty rule, constraints.json 'novelty')")
        if n_unmeshable:
            print(f"  {n_unmeshable} child(ren) rejected because the ground / pond / lawn surfaces could not be meshed (mesh gate, mesh_gate.py)")
        if len(kids) < n_kids:
            print(f"  only {len(kids)} of {n_kids} children could be made (constraints tight)")
        by = {}
        for k in kids:
            by[st["origin"][k]["kind"]] = by.get(st["origin"][k]["kind"], 0) + 1
        print(f"generation {gen}: {len(kids)} offspring ({', '.join(f'{v} {k}' for k, v in by.items())}) from parents "
              f"{', '.join(sorted({p for k in kids for p in st['origin'][k]['parents']}))}", flush=True)
        return kids

    # ---- bookkeeping
    def record(self, gen):
        st = self.state
        pop = st["population"]
        F = self.F(pop)
        ok = [i for i, n in enumerate(pop) if st["evals"].get(n, {}).get("status") == "ok"]
        fronts = nondominated_sort(F)
        if st["ref"] is None and ok:
            P = F[ok]
            span = np.maximum(P.max(0) - P.min(0), 1e-6)
            st["ref"] = [float(v) for v in P.max(0) + 0.1 * span]
        hv = hypervolume_2d([F[i] for i in fronts[0] if i in ok], st["ref"]) if st["ref"] and ok else 0.0
        row = {"gen": gen, "evaluated": len(st["evals"]), "front_size": len(fronts[0]), "hypervolume": round(hv, 6),
               "best_f1": float(F[ok, 0].min()) if ok else None, "best_f2": float(F[ok, 1].min()) if ok else None}
        st["history"] = [h for h in st["history"] if h["gen"] != gen] + [row]
        rank = np.zeros(len(pop), int)
        for r, fr in enumerate(fronts):
            for i in fr:
                rank[i] = r
        with open(self.out_dir / f"nsga_gen_{gen:02d}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "rank", "f1", "f2", "utci_heat", "utci_cold", "origin"])
            for i, n in enumerate(pop):
                e = st["evals"].get(n, {})
                w.writerow([n, rank[i], e.get("f1"), e.get("f2"), e.get("utci_heat"), e.get("utci_cold"),
                            st["origin"].get(n, {}).get("kind")])
        with open(self.out_dir / "nsga_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(sorted(st["history"], key=lambda h: h["gen"]))
        self.save()
        print(f"gen {gen}: front {row['front_size']}/{len(pop)}  hypervolume {hv:.4g}  best f1 {row['best_f1']}  best f2 {row['best_f2']}", flush=True)

    # ---- main loop
    def run(self):
        a = self.a
        if self.state is None:
            self.init_population()
        elif a.include_base:
            self.refresh_base()
        st = self.state
        if st["pending"] and st["gen"] == 0 and not st["population"]:
            self.evaluate(st["pending"])
            if not any(st["evals"].get(n, {}).get("status") == "ok" for n in st["pending"]):
                print("\nSTOPPED: none of the initial layouts completed both its heat and cold simulation, so no offspring can be made.\n"
                      "Simulations that did finish are kept. Read the reasons printed above (and run/<layout>/<season>/log.*), fix the cause\n"
                      "and run the same command again with --retry-failed.")
                sys.exit(4)
            st["population"] = list(st["pending"])
            st["pending"] = []
            self.record(0)
        while st["gen"] < a.gens:
            gen = st["gen"] + 1
            if not st["pending"]:
                st["pending"] = self.make_children(gen)
                self.save()
            self.evaluate(st["pending"])
            allnames = st["population"] + [n for n in st["pending"] if n not in st["population"]]
            keep = select_survivors(self.F(allnames), st.get("size", len(st["population"])))
            st["population"] = [allnames[i] for i in keep]
            st["pending"], st["gen"] = [], gen
            self.record(gen)
        print(f"done: generation {st['gen']}.  Next: python python/nsga2.py top5  and  plot")

    # ---- results
    def front(self):
        st = self.state
        pop = [n for n in st["population"] if st["evals"].get(n, {}).get("status") == "ok"]
        F = self.F(pop)
        return pop, F, nondominated_sort(F)[0]

    def top5(self):
        pop, F, fr = self.front()
        sel, roles = pick_top_roles(F, fr, 5)
        if len(sel) < 5:                     # tiny front: fill with the next best by rank / crowding
            rest = [i for i in select_survivors(F, min(len(pop), 5)) if i not in sel]
            add = rest[:5 - len(sel)]
            sel += add
            roles += ["not on the front (next best)"] * len(add)
        out = []
        for k, i in enumerate(sel, 1):
            e = self.state["evals"][pop[i]]
            out.append({"rank_in_list": k, "name": pop[i], "role": roles[k - 1], "f1": e["f1"], "f2": e["f2"],
                        "utci_heat": e.get("utci_heat"), "utci_cold": e.get("utci_cold"), "origin": self.state["origin"].get(pop[i], {})})
            src = self.in_dir / f"{pop[i]}.json"
            (self.in_dir / f"top5_{k}.json").write_text(src.read_text())
        (self.out_dir / "top5.json").write_text(json.dumps(out, indent=2))
        print("The list is NOT a quality ranking: every layout on the Pareto front is a different heat / cold trade-off.  "
              "Order: best heat, best cold, knee, then spread.")
        print(f"{'#':>2} {'layout':<16}{'UTCI heat':>10}{'UTCI cold':>10}{'f1 heat':>9}{'f2 cold':>9}   role")
        for r in out:
            uh = r["utci_heat"] if r["utci_heat"] is not None else float("nan")
            uc = r["utci_cold"] if r["utci_cold"] is not None else float("nan")
            print(f"{r['rank_in_list']:>2} {r['name']:<16}{uh:>10.2f}{uc:>10.2f}{r['f1']:>9.2f}{r['f2']:>9.2f}   {r['role']}")
        b = self.state["evals"].get("individual_0", {})
        if b.get("status") == "ok":
            better = sum(1 for r in out if r["f1"] <= b["f1"] and r["f2"] <= b["f2"] and (r["f1"] < b["f1"] or r["f2"] < b["f2"]))
            print(f"    base layout (individual_0): UTCI heat {b['utci_heat']:.2f}, cold {b['utci_cold']:.2f}, f1 {b['f1']:.2f}, f2 {b['f2']:.2f}"
                  f"   -> {better} of these {len(out)} are better in at least one objective and not worse in the other")
        print(f"wrote {self.out_dir / 'top5.json'} and input/top5_1..5.json")
        return out

    def verify(self):
        a = self.a
        top = json.loads((self.out_dir / "top5.json").read_text())
        names = [t["name"] for t in top]
        vdir = self.out_dir / "verify"
        out = RC.run_batch(names, case_file=a.verify_case_file or a.case_file, run_dir=a.verify_run_dir,
                           cand_dir=self.in_dir, cases_path=a.cases, max_parallel=a.max_parallel or 2 * a.batch,
                           timeout_min=a.timeout_min, summary_dir=vdir)
        rows = []
        for t in top:
            o = out["objectives"][t["name"]]
            sc = self.score(o.get("utci_heat"), o.get("utci_cold")) if o.get("status") == "ok" else {}
            rows.append({"name": t["name"], "f1_search": t["f1"], "f2_search": t["f2"], "f1_full": sc.get("f1"),
                         "f2_full": sc.get("f2"), "status": o.get("status")})
        with open(self.out_dir / "top5_verified.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("verification (search evaluator -> full simulation):")
        for r in rows:
            print(f"  {r['name']:<16} f1 {r['f1_search']:.3f} -> {r['f1_full']}   f2 {r['f2_search']:.3f} -> {r['f2_full']}   {r['status']}")

    def exclude(self, names):
        """Treat layouts as failed (objective 1e6): they leave the Pareto front, the top 5 and, at the next survivor selection, the population."""
        ev = self.state["evals"]
        for n in names:
            if n not in ev:
                print(f"  {n}: not in the state, nothing to do")
                continue
            ev[n] = {"status": "excluded", "f1": PENALTY, "f2": PENALTY, "was": {k: ev[n].get(k) for k in ("utci_heat", "utci_cold", "f1", "f2")}}
            print(f"  {n}: excluded (was UTCI heat {ev[n]['was']['utci_heat']}, cold {ev[n]['was']['utci_cold']})")
        self.save()
        print("run  python python/nsga2.py top5  and  plot  again (or  run --gens N  to continue with a new generation)")

    def status(self):
        st = self.state
        if st is None:
            print("no run yet (python nsga2.py run ...)")
            return
        print(f"generation {st['gen']}, population {len(st['population'])}, simulated {len(st['evals'])}, waiting {len(st['pending'])}")
        for h in sorted(st["history"], key=lambda h: h["gen"]):
            print(f"  gen {h['gen']:>3}  front {h['front_size']:>3}  hypervolume {h['hypervolume']:.5g}  best f1 {h['best_f1']:.3f}  best f2 {h['best_f2']:.3f}")

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import plot_candidates as PC
        st = self.state
        ev = st["evals"]
        names = [n for n in ev if ev[n].get("status") == "ok"]
        pts = {n: (ev[n]["f1"], ev[n]["f2"]) for n in names}
        gens = {n: st["origin"].get(n, {}).get("gen", 0) for n in names}
        pop, F, fr = self.front()
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
        ax = axs[0]
        sc = ax.scatter([pts[n][0] for n in names], [pts[n][1] for n in names], c=[gens[n] for n in names], cmap="viridis",
                        s=22, alpha=.8, label="simulated layouts (colour = generation created)")
        plt.colorbar(sc, ax=ax, label="generation")
        fp = sorted((F[i] for i in fr), key=lambda p: p[0])
        ax.plot([p[0] for p in fp], [p[1] for p in fp], "-o", color="#c0392b", ms=5, lw=1.2, label="Pareto front (current population)")
        if "individual_0" in pts:
            ax.plot(*pts["individual_0"], marker="*", ms=16, color="gold", mec="k", ls="", label="base layout")
        for kind, mk, col in (("wind_catcher", "^", "#e67e22"), ("wind_blocker", "v", "#8e44ad"), ("cool_wind_catcher", "s", "#16a085"),
                              ("cool_wind_catcher_all", "P", "#16a085"), ("wind_blocker_edge", "X", "#8e44ad")):
            tn = [n for n in names if st["origin"].get(n, {}).get("kind") == f"template:{kind}"]
            if tn:
                ax.plot([pts[n][0] for n in tn], [pts[n][1] for n in tn], marker=mk, ms=10, color=col, mec="k", ls="",
                        label=TPL.KINDS[kind]["label"] + " individual")
        t5 = (self.out_dir / "top5.json")
        if t5.exists():
            for t in json.loads(t5.read_text()):
                ax.annotate(str(t["rank_in_list"]), (t["f1"], t["f2"]), textcoords="offset points", xytext=(6, 6),
                            fontsize=10, fontweight="bold", color="#c0392b")
        lo, hi = self.comfort()
        ax.set_xlabel(f"f1 = heat stress above {hi:g} °C UTCI (peak-heat hour, °C)  - lower is better")
        ax.set_ylabel(f"f2 = cold stress below {lo:g} °C UTCI (peak-cold hour, °C)  - lower is better")
        ax.set_title("objective space" + ("   [MOCK evaluator]" if st.get("evaluator") == "mock" else ""), fontsize=10)
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(alpha=.25)
        h = sorted(st["history"], key=lambda h: h["gen"])
        axs[1].plot([x["gen"] for x in h], [x["hypervolume"] for x in h], "-o", color="#1f6fb0")
        axs[1].set_xlabel("generation")
        axs[1].set_ylabel("hypervolume (fixed reference point)")
        axs[1].set_title("convergence: stop when the hypervolume stops growing", fontsize=10)
        axs[1].grid(alpha=.25)
        fig.tight_layout()
        p = self.out_dir / "nsga_report.png"
        fig.savefig(p, dpi=110)
        print("wrote", p)
        if t5.exists():
            top = json.loads(t5.read_text())
            files = [self.in_dir / f"{t['name']}.json" for t in top]
            obj = self.out_dir / "objectives.csv"
            if st.get("evaluator") == "mock":                     # titles carry the objectives of the mock run
                obj = self.out_dir / "objectives_mock.csv"
                with open(obj, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["name", "status", "utci_heat", "utci_cold", "f1", "f2"])
                    for n in ev:
                        w.writerow([n, ev[n].get("status"), ev[n].get("utci_heat"), ev[n].get("utci_cold"), ev[n].get("f1"), ev[n].get("f2")])
            PC.plot_grid(files, self.base_path, self.cfg, self.out_dir, str(obj), cols=3)
            print("wrote", self.out_dir / "overview.png", "(base + the five picks)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "top5", "verify", "plot", "exclude"])
    ap.add_argument("names", nargs="*", help="layouts for `exclude`")
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--base")
    ap.add_argument("--constraints")
    ap.add_argument("--pop", type=int, default=10, help="number of initial individuals (the base layout is added to them) (default 10)")
    ap.add_argument("--offspring", type=int, default=None, help="children per generation (default = --batch)")
    ap.add_argument("--gens", type=int, default=5, help="generations to run (resume continues up to this number) (default 5)")
    ap.add_argument("--batch", type=int, default=4, help="layouts simulated at the same time (default 4 = 8 simulations = 32 cores)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pc", type=float, default=0.9, help="crossover probability")
    ap.add_argument("--pm", type=float, default=0.25, help="probability that a child is mutated (default 0.25)")
    ap.add_argument("--mut-ops", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"), help="operations per mutation")
    ap.add_argument("--comfort", type=float, nargs=2, default=[9.0, 26.0], metavar=("LOW", "HIGH"), help="comfortable UTCI band (default 9 26)")
    ap.add_argument("--no-base", dest="include_base", action="store_false", help="do not put the base layout in the initial population")
    ap.add_argument("--no-mesh-gate", dest="mesh_gate", action="store_false",
                    help="do not test every child with the case's json_to_obj.py / gen_ground_obj.py before accepting it (mesh_gate.py)")
    ap.add_argument("--no-templates", dest="templates", action="store_false",
                    help="do not make the tree wind catcher / wind blocker individuals (jsons/tree_wind_*.json)")
    ap.add_argument("--template-kinds", default="default",
                    help="seed individuals to make: 'default' (cool_wind_catcher_all, wind_blocker_edge), 'all', or a comma list of "
                         + ", ".join(TPL.KINDS))
    ap.add_argument("--jsons-dir", default=None, help="folder with the tree templates (default jsons/)")
    ap.add_argument("--catcher-template", dest="wind_catcher_template", default=None, help="tree wind catcher template (kind wind_catcher; default jsons/tree_wind_catcher.json)")
    ap.add_argument("--blocker-template", dest="wind_blocker_template", default=None, help="wind blocker template, belt or wall (default jsons/tree_wind_breaker.json)")
    ap.add_argument("--cool-catcher-template", dest="cool_wind_catcher_template", default=None, help="cool wind catcher (pond + trees at a building) template (default jsons/cool_wind_catcher.json)")
    ap.add_argument("--cool-reference-building", default=None, metavar="ID", help="only for --template-kinds cool_wind_catcher: building (id, e.g. 'Building 2') the single catcher hangs on")
    ap.add_argument("--evaluator", choices=["real", "mock"], default="real", help="real = ./Allrun via run_candidates.py")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--case-file", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--max-parallel", type=int, default=None, help="simulations at once (default 2 x --batch)")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="processes that make layouts (seed individuals, random individuals, children) at the same time; 1 = one process (default), "
                         "0 = one per CPU core. Layouts differ from a --workers 1 run (other random streams) but are valid and reproducible for a given N>1")
    ap.add_argument("--timeout-min", type=float, default=None)
    ap.add_argument("--verify-case-file", default=None, help="caseFile of the FULL simulation used for the top 5")
    ap.add_argument("--verify-run-dir", default=str(ROOT / "run_full"))
    a = ap.parse_args()
    r = Run(a)
    if a.cmd == "run":
        a.case_file = a.case_file or ROOT / "caseFile"
        a.run_dir = a.run_dir or ROOT / "run"
        r.run()
    elif a.cmd == "status":
        r.status()
    elif a.cmd == "exclude":
        r.exclude(a.names)
    elif a.cmd == "top5":
        r.top5()
    elif a.cmd == "verify":
        a.case_file = a.case_file or ROOT / "caseFile"
        a.run_dir = a.run_dir or ROOT / "run"
        r.verify()
    else:
        r.plot()


if __name__ == "__main__":
    main()
