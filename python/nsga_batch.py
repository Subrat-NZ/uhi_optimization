#!/usr/bin/env python3
"""
nsga_batch.py  --  NSGA-II in batches:  generate -> run 4 -> mutate 25 % -> repeat -> Pareto front -> top 5

    python python/nsga_batch.py run                              # 10 layouts, 4 cases at a time, 25 % mutations
    python python/nsga_batch.py run --generations 10 --batch 4 --mutation-fraction 0.25 --seed 1
    python python/nsga_batch.py status                           # what has been run, front so far
    python python/nsga_batch.py report                           # Pareto plot + top 5 from what is evaluated
    python python/nsga_batch.py run --evaluator mock             # synthetic objectives (NOT physics): tests the loop

The loop (all numbers are options)
    1  make --generations (10) layouts  input/generation_1.json ... with generate_candidates' recipe (every Stage-1 rule holds)
    2  take the next --batch (4) layouts of the queue and run them: each layout = a heat and a cold simulation,
       i.e. 4 layouts = 8 simulations of 4 cores = 32 cores at the same time (run_candidates.run_batch, waits for all)
    3  from the layouts that were just run, the best ceil(--mutation-fraction (0.25) x layouts) are mutated
       (best = lowest Pareto rank among everything evaluated so far, then largest crowding distance):
       a mutation = 1..3 small random operations on a parent  ->  input/mutation_K.json, appended to the queue
    4  repeat 2-3 until the queue is empty.  Mutations are only made from layouts of depth < --mutation-depth
       (default 1: from the generated layouts, not from mutations of mutations), so the loop ends by itself.
       With 10 layouts and batch 4:  batches 4, 4, 4 (2 new + 2 mutations), 1  ->  3 mutations, 13 layouts, 26 simulations.
    5  Pareto front of every successfully simulated layout (non-dominated sorting on f1, f2), plot, top 5

Objectives (both minimised):  f1 = site-averaged UTCI of the peak-heat hour,  f2 = -(site-averaged UTCI of the peak-cold hour)
Top 5: best f1, best f2, the knee of the front, then two more spread along the front (farthest-point sampling).

Files   output/nsga_batch_state.json   resumable state (queue, evaluations, parents)   -> rerun the same command to resume
        output/pareto.png              objective space, Pareto front, top 5 numbered, convergence
        output/pareto.csv, top5.json, input/top5_1..5.json, output/overview.png (base + the five layouts)
        output/nsga_batch_history.csv  one row per batch
A layout whose simulation failed gets objectives 1e6, never enters the front and is not mutated.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_candidates as G  # noqa: E402
import layout_ops as L  # noqa: E402
import nsga2 as N  # noqa: E402  (non-dominated sorting, crowding, top-k picking, hypervolume, mock objectives)
import run_candidates as RC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PENALTY = RC.PENALTY


class Batch:
    def __init__(self, a):
        self.a = a
        self.in_dir = Path(a.input_dir) if Path(a.input_dir).is_absolute() else ROOT / a.input_dir
        self.out_dir = Path(a.output_dir) if Path(a.output_dir).is_absolute() else ROOT / a.output_dir
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = L.load_constraints(a.constraints or self.in_dir / "constraints.json")
        self.base_path = Path(a.base) if a.base else self.in_dir / "base.json"
        self.base_data = G.load(self.base_path)
        self.base = L.Scene(self.base_data, self.cfg, split_hosted=True)
        self.base_map = L.violation_map(L.check(self.base, self.base))
        self.state_path = self.out_dir / "nsga_batch_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else None
        self.log = self.out_dir / "candidates_log.jsonl"
        self.cache = {}
        self.seen = {G.fingerprint(self.base_data)}
        for p in self.in_dir.glob("*_*.json"):
            if p.stem.split("_")[0] in ("generation", "mutation") and p.stem.split("_")[-1].isdigit():
                self.seen.add(G.fingerprint(G.load(p)))

    # ------------------------------------------------------------------ state
    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def layout(self, name):
        if name not in self.cache:
            self.cache[name] = G.load(self.in_dir / f"{name}.json")
        return self.cache[name]

    def ok_names(self):
        ev = self.state["evals"]
        return [n for n, e in ev.items() if e.get("status") == "ok" and e.get("f1") is not None]

    def F(self, names):
        ev = self.state["evals"]
        return np.array([[ev[n]["f1"], ev[n]["f2"]] for n in names], float)

    # ------------------------------------------------------------------ step 1: the generated layouts
    def make_initial(self):
        a = self.a
        names, origin = [], {}
        if a.include_base:
            p = self.in_dir / "generation_0.json"
            if not p.exists():
                g0, notes = G.make_base_copy(self.base_data, self.base, self.cfg)
                p.write_text(json.dumps(g0, indent=2))
                with open(self.log, "a") as f:
                    f.write(json.dumps({"file": "generation_0.json", "kind": "base copy", "notes": notes}) + "\n")
            names.append("generation_0")
            origin["generation_0"] = {"kind": "base", "depth": 1}          # the base is not mutated
        idx, attempts = G.next_index(self.in_dir, "generation"), 0
        n_new = 0
        while n_new < a.generations and attempts < a.generations * 30:
            attempts += 1
            seed = a.seed * 1_000_003 + (idx + attempts) * 97
            rng = np.random.default_rng(seed)
            res = G.build_candidate(self.base_data, self.base, self.cfg, seed, int(rng.integers(1, 5)),
                                    float(rng.uniform(0.6, 1.4)), self.base_map, "generation")
            if res is None or G.fingerprint(res[0]) in self.seen:
                continue
            name = f"generation_{idx}"
            G.write_candidate(res[0], res[1], self.in_dir / f"{name}.json", self.log,
                              {"kind": "generation", "parent": self.base_path.name, "seed": seed}, base=self.base)
            self.seen.add(G.fingerprint(res[0]))
            names.append(name)
            origin[name] = {"kind": "generation", "depth": 0, "seed": seed}
            idx += 1
            n_new += 1
        if n_new < a.generations:
            print(f"only {n_new} of {a.generations} generations could be made (constraints tight)")
        return names, origin

    # ------------------------------------------------------------------ step 2: run a batch
    def evaluate(self, names):
        a, ev = self.a, self.state["evals"]
        todo = [n for n in names if n not in ev or (ev[n].get("status") != "ok" and a.retry_failed)]
        if not todo:
            return
        t0 = time.time()
        if a.evaluator == "mock":
            for n in todo:
                ev[n] = N.mock_objectives(self.layout(n), self.base, self.cfg)
        else:
            out = RC.run_batch(todo, case_file=a.case_file, run_dir=a.run_dir, cand_dir=self.in_dir,
                               cases_path=a.cases, max_parallel=a.max_parallel or 2 * len(todo),
                               timeout_min=a.timeout_min, summary_dir=self.out_dir, force=False)
            for n in todo:
                o = out["objectives"].get(n)
                ev[n] = ({k: o.get(k) for k in ("status", "utci_heat", "utci_cold", "f1", "f2")} if o
                         else {"status": "failed", "f1": PENALTY, "f2": PENALTY})
        for n in todo:
            ev[n]["seconds"] = round((time.time() - t0), 1)
        bad = [n for n in todo if ev[n].get("status") != "ok"]
        print(f"  batch evaluated in {time.time() - t0:.0f} s" + (f"; FAILED: {', '.join(bad)}" if bad else ""), flush=True)

    # ------------------------------------------------------------------ step 3: mutate the best 25 % of the batch
    def parents_of(self, batch):
        st = self.state
        ok = self.ok_names()
        if not ok:
            return []
        F = self.F(ok)
        fronts, rank, crowd = N.rank_and_crowd(F)
        pos = {n: i for i, n in enumerate(ok)}
        cand = [n for n in batch if n in pos and st["origin"][n].get("depth", 0) < self.a.mutation_depth]
        if not cand:
            return []
        n_mut = int(math.ceil(self.a.mutation_fraction * len(cand) - 1e-9))
        cand.sort(key=lambda n: (rank[pos[n]], -crowd[pos[n]] if math.isfinite(crowd[pos[n]]) else -1e18))
        return cand[:n_mut]

    def make_mutants(self, parents, batch_no):
        a, st = self.a, self.state
        rng = np.random.default_rng([a.seed, 7919, batch_no])
        idx, kids = G.next_index(self.in_dir, "mutation"), []
        for par in parents:
            made = None
            for attempt in range(40):
                seed = int(rng.integers(1, 2 ** 31 - 1))
                r = np.random.default_rng(seed)
                res = G.build_candidate(self.layout(par), self.base, self.cfg, seed, int(r.integers(a.mut_ops[0], a.mut_ops[1] + 1)),
                                        float(r.uniform(0.3, 0.8)), self.base_map, "mutation")
                if res is None or G.fingerprint(res[0]) in self.seen:
                    continue
                if L.new_violations(L.Scene(res[0], self.cfg, self.base.osm_polys), self.base, self.base_map):
                    continue
                made = (res, seed)
                break
            if made is None:
                print(f"  could not mutate {par} (constraints tight)")
                continue
            (data, ops), seed = made
            name = f"mutation_{idx}"
            G.write_candidate(data, ops, self.in_dir / f"{name}.json", self.log,
                              {"kind": "nsga_mutation", "parent": par, "seed": seed, "batch": batch_no}, base=self.base)
            self.seen.add(G.fingerprint(data))
            self.cache[name] = data
            st["origin"][name] = {"kind": "mutation", "depth": st["origin"][par].get("depth", 0) + 1,
                                  "parents": [par], "seed": seed, "batch": batch_no}
            kids.append(name)
            idx += 1
        return kids

    # ------------------------------------------------------------------ bookkeeping
    def record(self, batch_no, batch, parents, kids):
        st = self.state
        ok = self.ok_names()
        row = {"batch": batch_no, "layouts": " ".join(batch), "mutated": " ".join(parents), "new_mutations": " ".join(kids),
               "evaluated_total": len(st["evals"]), "ok_total": len(ok)}
        if ok:
            F = self.F(ok)
            fr = N.nondominated_sort(F)[0]
            if st.get("ref") is None:
                span = np.maximum(F.max(0) - F.min(0), 1e-6)
                st["ref"] = [float(v) for v in F.max(0) + 0.1 * span]
            row.update(front_size=len(fr), hypervolume=round(N.hypervolume_2d([F[i] for i in fr], st["ref"]), 6),
                       best_f1=float(F[:, 0].min()), best_f2=float(F[:, 1].min()))
        st["batches"] = [b for b in st["batches"] if b["batch"] != batch_no] + [row]
        cols = ["batch", "layouts", "mutated", "new_mutations", "evaluated_total", "ok_total", "front_size", "hypervolume", "best_f1", "best_f2"]
        with open(self.out_dir / "nsga_batch_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(st["batches"], key=lambda b: b["batch"]))
        print(f"batch {batch_no}: ran {', '.join(batch)}"
              + (f"\n   front {row.get('front_size')} of {len(ok)}  best f1 {row.get('best_f1', float('nan')):.3f}  "
                 f"best f2 {row.get('best_f2', float('nan')):.3f}" if ok else "")
              + (f"\n   mutating {', '.join(parents)} -> {', '.join(kids)}" if parents else ""), flush=True)

    # ------------------------------------------------------------------ main loop
    def run(self):
        a = self.a
        if self.state is None:
            self.state = {"config": {k: getattr(a, k) for k in ("generations", "batch", "mutation_fraction", "mutation_depth",
                                                                 "seed", "include_base", "evaluator")},
                          "queue": [], "evals": {}, "origin": {}, "batches": [], "ref": None, "initialised": False}
        st = self.state
        if not st["initialised"]:
            names, origin = self.make_initial()
            st["queue"], st["initialised"] = names, True
            st["origin"].update(origin)
            self.save()
            print(f"{len(names)} layouts generated: {', '.join(names)}")
        if a.retry_failed:        # put the failed layouts back at the front of the queue and simulate them again
            failed = [n for n, e in st["evals"].items() if e.get("status") != "ok"]
            for n in failed:
                del st["evals"][n]
            st["queue"] = failed + [n for n in st["queue"] if n not in failed]
            if failed:
                self.save()
                print(f"--retry-failed: {len(failed)} layouts queued again: {', '.join(failed)}")
        total0 = len(st["queue"]) + len(st["evals"])
        while st["queue"]:
            batch_no = len(st["batches"]) + 1
            batch, st["queue"] = st["queue"][:a.batch], st["queue"][a.batch:]
            print(f"\n=== batch {batch_no}: {len(batch)} layouts ({len(st['queue'])} more waiting) ===", flush=True)
            self.save()
            try:
                self.evaluate(batch)
            except KeyboardInterrupt:
                st["queue"] = batch + st["queue"]
                self.save()
                print("interrupted: the batch goes back to the queue; run the same command to resume")
                raise
            if not any(st["evals"].get(n, {}).get("status") == "ok" for n in batch) and not self.ok_names():
                # nothing has ever worked: this is a set-up problem, not a bad layout. Do not burn through the queue.
                for n in batch:
                    st["evals"].pop(n, None)
                st["queue"] = batch + st["queue"]
                self.save()
                print("\nSTOPPED: every simulation of the first batch failed, so the rest was not started.\n"
                      "Look at run/<layout>/<season>/log.* and the messages above, fix the cause and run the same command again "
                      "(the layouts are still queued).")
                sys.exit(4)
            parents = self.parents_of(batch)
            kids = self.make_mutants(parents, batch_no) if parents else []
            st["queue"] += kids
            self.record(batch_no, batch, parents, kids)
            self.save()
        print(f"\nall {len(st['evals'])} layouts finished ({len(self.ok_names())} ok).")
        self.report()

    # ------------------------------------------------------------------ results
    def report(self):
        st = self.state
        if st is None:
            print("nothing run yet")
            return
        ok = self.ok_names()
        if len(ok) < 1:
            print("no successfully simulated layout yet")
            return
        F = self.F(ok)
        fronts = N.nondominated_sort(F)
        fr = fronts[0]
        sel, roles_sel = N.pick_top_roles(F, fr, 5)          # best heat, best cold, knee, spread - in this order
        if len(sel) < 5:                                    # tiny front: continue with the next fronts by crowding
            for i in N.select_survivors(F, min(len(ok), 5)):
                if i not in sel and len(sel) < 5:
                    sel.append(i)
        rank = np.zeros(len(ok), int)
        for r, f in enumerate(fronts):
            for i in f:
                rank[i] = r
        roles = ["best heat case (min f1)", "best cold case (min f2)", "knee of the front"]
        top = []
        for k, i in enumerate(sel, 1):
            n, e = ok[i], st["evals"][ok[i]]
            top.append({"rank_in_list": k, "name": n, "pareto_rank": int(rank[i]), "f1": e["f1"], "f2": e["f2"],
                        "utci_heat": e.get("utci_heat"), "utci_cold": e.get("utci_cold"),
                        "role": ("dominated, next best" if rank[i] > 0 else roles_sel[k - 1] if k <= len(roles_sel) else "spread along the front"),
                        "origin": st["origin"].get(n, {})})
            (self.in_dir / f"top5_{k}.json").write_text((self.in_dir / f"{n}.json").read_text())
        (self.out_dir / "top5.json").write_text(json.dumps(top, indent=2))
        with open(self.out_dir / "pareto.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "pareto_rank", "f1", "f2", "utci_heat", "utci_cold", "kind", "parent"])
            for i in sorted(range(len(ok)), key=lambda i: (rank[i], F[i, 0])):
                n, e = ok[i], st["evals"][ok[i]]
                o = st["origin"].get(n, {})
                w.writerow([n, rank[i], e["f1"], e["f2"], e.get("utci_heat"), e.get("utci_cold"), o.get("kind"),
                            ",".join(o.get("parents", []))])
        base = st["evals"].get("generation_0")
        print(f"\nPareto front: {len(fr)} of {len(ok)} layouts (f1 = UTCI heat, f2 = -UTCI cold, lower is better)")
        print(f"{'#':>2} {'layout':<16}{'UTCI heat':>10}{'UTCI cold':>10}{'f1':>9}{'f2':>9}   role")
        for t in top:
            uh, uc = t.get("utci_heat"), t.get("utci_cold")
            print(f"{t['rank_in_list']:>2} {t['name']:<16}{(uh if uh is not None else float('nan')):>10.2f}"
                  f"{(uc if uc is not None else float('nan')):>10.2f}{t['f1']:>9.2f}{t['f2']:>9.2f}   {t['role']}")
        if base and base.get("status") == "ok":
            print(f"   base layout: UTCI heat {base['utci_heat']:.2f}  UTCI cold {base['utci_cold']:.2f}")
        print(f"wrote {self.out_dir / 'top5.json'}, pareto.csv, input/top5_1..5.json")
        self.plot(ok, F, fr, top)

    def plot(self, ok, F, fr, top):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        st = self.state
        kind = np.array([st["origin"].get(n, {}).get("kind", "generation") for n in ok])
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
        ax = axs[0]
        dom = np.ones(len(ok), bool)
        dom[fr] = False
        for k, mk, col, lab in (("generation", "o", "#1f6fb0", "generated layout"), ("mutation", "^", "#e08a1e", "mutation")):
            m = (kind == k)
            if m.any():
                ax.scatter(F[m & dom, 0], F[m & dom, 1], marker=mk, s=46, facecolor="none", edgecolor=col, lw=1.4, label=lab)
                ax.scatter(F[m & ~dom, 0], F[m & ~dom, 1], marker=mk, s=70, color=col, edgecolor="k", lw=.8,
                           label=lab + " (on the Pareto front)")
        pf = sorted((F[i] for i in fr), key=lambda p: p[0])
        ax.step([p[0] for p in pf], [p[1] for p in pf], where="post", color="#c0392b", lw=1.2, alpha=.8, label="Pareto front")
        if "generation_0" in ok:
            i0 = ok.index("generation_0")
            ax.plot(*F[i0], marker="*", ms=17, color="gold", mec="k", ls="", label="base layout")
        for t in top:
            ax.annotate(str(t["rank_in_list"]), (t["f1"], t["f2"]), textcoords="offset points", xytext=(7, 6),
                        fontsize=11, fontweight="bold", color="#c0392b")
        ax.set_xlabel("f1 = site-averaged UTCI, peak-heat hour (°C)   - lower is better")
        ax.set_ylabel("f2 = -(site-averaged UTCI, peak-cold hour) (°C)   - lower is better")
        ax.set_title("Pareto front of the evaluated layouts   (numbers = top 5)"
                     + ("   [MOCK evaluator, not physics]" if st["config"].get("evaluator") == "mock" else ""), fontsize=10)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7.5, loc="best")
        b = sorted((x for x in st["batches"] if x.get("hypervolume") is not None), key=lambda x: x["batch"])
        if b:
            axs[1].plot([x["evaluated_total"] for x in b], [x["hypervolume"] for x in b], "-o", color="#1f6fb0")
        axs[1].set_xlabel("layouts evaluated")
        axs[1].set_ylabel("hypervolume (fixed reference point)")
        axs[1].set_title("progress after each batch", fontsize=10)
        axs[1].grid(alpha=.25)
        fig.tight_layout()
        p = self.out_dir / "pareto.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print("wrote", p)
        try:
            import plot_candidates as PC
            obj = self.out_dir / "objectives.csv"
            if st["config"].get("evaluator") == "mock":
                obj = self.out_dir / "objectives_mock.csv"
                with open(obj, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["name", "status", "utci_heat", "utci_cold", "f1", "f2"])
                    for n, e in st["evals"].items():
                        w.writerow([n, e.get("status"), e.get("utci_heat"), e.get("utci_cold"), e.get("f1"), e.get("f2")])
            PC.plot_grid([self.in_dir / f"{t['name']}.json" for t in top], self.base_path, self.cfg, self.out_dir,
                         str(obj), cols=3)
            print("wrote", self.out_dir / "overview.png", "(base + the five picks)")
        except Exception as exc:                                    # the picture is a bonus, never fail the run for it
            print("overview picture skipped:", exc)

    def status(self):
        st = self.state
        if st is None:
            print("nothing run yet")
            return
        ev = st["evals"]
        print(f"evaluated {len(ev)} ({len(self.ok_names())} ok), waiting in the queue: {len(st['queue'])} "
              f"({', '.join(st['queue'][:8])}{' ...' if len(st['queue']) > 8 else ''})")
        for b in sorted(st["batches"], key=lambda b: b["batch"]):
            print(f"  batch {b['batch']}: {b['layouts']}  | mutated {b['mutated'] or '-'} -> {b['new_mutations'] or '-'}"
                  + (f"  | front {b.get('front_size')}  best f1 {b.get('best_f1'):.3f} f2 {b.get('best_f2'):.3f}" if b.get("front_size") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "report"])
    ap.add_argument("--generations", type=int, default=10, help="number of generated layouts (default 10)")
    ap.add_argument("--batch", type=int, default=4, help="layouts simulated at the same time (default 4 = 8 simulations)")
    ap.add_argument("--mutation-fraction", type=float, default=0.25, help="share of each batch that is mutated (default 0.25)")
    ap.add_argument("--mutation-depth", type=int, default=1,
                    help="mutate generated layouts only (1, default); 2 also mutates mutations (longer run)")
    ap.add_argument("--mut-ops", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"), help="operations per mutation")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-base", action="store_true", help="also simulate the base layout (generation_0) as a reference")
    ap.add_argument("--evaluator", choices=["real", "mock"], default="real")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=None, help="simulations at once (default 2 x batch)")
    ap.add_argument("--timeout-min", type=float, default=None)
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--base")
    ap.add_argument("--constraints")
    ap.add_argument("--case-file", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--cases", default=None)
    a = ap.parse_args()
    a.case_file = a.case_file or ROOT / "caseFile"
    a.run_dir = a.run_dir or ROOT / "run"
    b = Batch(a)
    {"run": b.run, "status": b.status, "report": b.report}[a.cmd]()


if __name__ == "__main__":
    main()
