#!/usr/bin/env python3
"""
nsga_family_video.py
====================
Replays an NSGA-II urban-heat-island layout optimisation run.

Reads
  * <input-dir>/*.json            scene files (base / individual_N / offspring_N / top5_N)
  * <output-dir>/nsga_state.json  the run state: origin (gen, kind, parents) = the SEQUENCE,
                                  evals (UTCI heat/cold, f1, f2), history, reference point
  * <output-dir>/nsga_gen_NN.csv  (optional) survivors after each generation
  * <output-dir>/candidates_log.jsonl (optional) operator log -> "what changed" text
  * <output-dir>/top5.json        (optional) the picked layouts, starred in the tree

Writes (into --out)
  * family_tree.png / .svg        the complete genealogy (who was bred from whom)
  * frames/NNN_<name>.png         one keyframe per layout, in the order of nsga_state.json
  * frames/999_summary.png        closing frame with the picked layouts
  * sequence.csv                  the replay order that was used
  * evolution.mp4                 ffmpeg video of the keyframes (cross-faded)

Usage
  python3 nsga_family_video.py --input-dir input --output-dir output --out replay
  python3 nsga_family_video.py --input-dir input --output-dir output --out replay --seconds 1.5 --fade 0.4
  python3 nsga_family_video.py --input-dir input --output-dir output --out replay --no-video

Needs: python>=3.8, numpy, matplotlib, and ffmpeg on PATH (only for the video).
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects                                       # noqa: E402,F401
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.collections import PatchCollection                 # noqa: E402
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                PathPatch, Polygon, Rectangle)
from matplotlib.path import Path as MPath                          # noqa: E402

# --------------------------------------------------------------------------- theme
# same palette as the Scene Inspector so the frames feel like the same tool
BG, PANEL, INK, MUTED, LINE, ACCENT = "#f1eee6", "#f8f6f0", "#1c2830", "#6d7780", "#d5d0c3", "#d9481f"
GOLD = "#e8a800"
KIND_STYLE = {                                   # label, colour
    "base": ("Base layout", "#1c2830"),
    "individual": ("Random individual", "#7d8b97"),
    "mutation": ("Mutation", "#2f7ed8"),
    "crossover": ("Crossover", "#d9741a"),
    "crossover+mutation": ("Crossover + mutation", "#8e44ad"),
}
OP_NAMES = {
    "shift_building": "building shift", "resize_tree": "tree resize", "reflow_lawns_ponds": "lawn/pond reflow",
    "trees": "tree planting", "move_tree": "tree move", "crossover": "crossover", "stretch_lawn": "lawn stretch",
    "rotate_pond": "pond rotate", "rotate_lawn": "lawn rotate", "shift_lawn": "lawn shift",
    "add_tree": "tree add", "scale_lawn": "lawn scale", "scale_pond": "pond scale",
}
PARENT_COLORS = ["#2f7ed8", "#c2185b"]           # ghost outlines of parent A / parent B


def kind_color(kind):
    return KIND_STYLE.get(kind, ("", "#999999"))[1]


def kind_label(kind):
    return KIND_STYLE.get(kind, (kind, ""))[0]


def mix(c1, c2, w):
    """blend colour c1 towards c2 by weight w (0..1)"""
    a, b = np.array(matplotlib.colors.to_rgb(c1)), np.array(matplotlib.colors.to_rgb(c2))
    return tuple(a * (1 - w) + b * w)


# --------------------------------------------------------------------------- data
def num(v):
    """JSON value -> float; null, text and the 1e6 'failed' penalty become NaN (= not available)"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) and abs(v) < 1e5 else float("nan")


def fm(v, spec=".2f", unit=""):
    """formatted number, or a dash when the value is not available (e.g. a layout without a cold case)"""
    return "–" if v is None or math.isnan(v) else f"{v:{spec}}{unit}"


@dataclass
class Node:
    name: str
    idx: int                       # position in the replay sequence
    gen: int
    kind: str
    parents: list
    seeded: bool = False           # parent link is "derived from base", not a breeding event
    seed: object = None
    status: str = "ok"
    heat: float = float("nan")
    cold: float = float("nan")
    f1: float = float("nan")
    f2: float = float("nan")
    ops: str = ""
    short: str = ""

    @property
    def ok(self):
        """both objectives known: can be plotted, ranked and take part in the Pareto front"""
        return self.status == "ok" and not math.isnan(self.f1) and not math.isnan(self.f2)

    @property
    def failed(self):
        """nothing usable came out of the CFD evaluation"""
        return self.status != "ok" or all(math.isnan(v) for v in (self.heat, self.cold, self.f1, self.f2))

    @property
    def partial(self):
        """evaluated, but one case (heat or cold) is missing: shown in the tree and header, not in the objective plot"""
        return not self.failed and not self.ok


@dataclass
class Run:
    nodes: dict
    order: list
    state: dict
    scenes: dict
    alive_after_gen: dict          # gen -> set(names) from nsga_gen_NN.csv
    picks: list                    # [(rank, name)]
    pos: dict = field(default_factory=dict)
    f_labels: tuple = ("f1 (heat)", "f2 (cold)")

    @property
    def n(self):
        return len(self.order)


def short_name(name):
    m = re.fullmatch(r"(individual|offspring)_(\d+)", name)
    if not m:
        return name[:5]
    return ("i" if m.group(1) == "individual" else "o") + m.group(2)


def summarise_ops(ops):
    c = Counter(o.get("op", "?") for o in ops)
    parts = []
    for op, k in c.items():
        label = OP_NAMES.get(op, op.replace("_", " "))
        parts.append(label + (f" ×{k}" if k > 1 else ""))
    return " · ".join(parts)


def parse_scene(J):
    """reduce a scene JSON to what is needed for a plan view (same fields the Scene Inspector reads)"""
    def pt(p):
        """(x, y) of a vertex: {x,y}, {x,z} (OSM footprints in some files, as the Scene Inspector accepts) or [x,y,...]"""
        if isinstance(p, dict):
            return p["x"], (p["y"] if p.get("y") is not None else p["z"])
        return p[0], p[1]

    def xy(coords):
        return np.array([pt(p) for p in coords], float)

    def polys(items, key="coordinates"):
        return [xy(a[key]) for a in items or [] if len(a.get(key) or []) >= 3]

    site = xy(J["site"]["coordinates"]) if J.get("site", {}).get("coordinates") else None
    # buildings: the file stores every storey as its own "building"; merge identical footprints
    towers = {}
    for b in J.get("buildings") or []:
        vs = (b.get("dimensions") or {}).get("baseVertices") or (b.get("geometry") or {}).get("footprint")
        if not vs or len(vs) < 3:
            continue
        h = b.get("height") or (b.get("dimensions") or {}).get("height") or 3
        z0 = (vs[0].get("z") if isinstance(vs[0], dict) else None)
        z0 = (b.get("position") or {}).get("z", 0) if z0 is None else z0
        z0 = z0 or 0
        poly = xy(vs)
        key = tuple(np.round(poly[:, 0], 3)) + tuple(np.round(poly[:, 1], 3))
        t = towers.setdefault(key, {"poly": poly, "z0": z0, "z1": z0 + h, "storeys": 0})
        t["z0"], t["z1"], t["storeys"] = min(t["z0"], z0), max(t["z1"], z0 + h), t["storeys"] + 1
    osm = []
    for o in (J.get("osmBuildings") or J.get("osm_buildings") or []):
        vs = (o.get("dimensions") or {}).get("baseVertices")
        if vs and len(vs) >= 3:
            osm.append(xy(vs))
    trees = []
    for t in J.get("trees") or []:
        d, p = t.get("dimensions") or {}, t["position"]
        r = d.get("crownRadius") or (t.get("crownDiameter") or 4) / 2
        trees.append((p["x"], p["y"], r, t.get("height") or d.get("totalHeight") or 8, t.get("id", "")))
    return {
        "site": site, "grass": polys(J.get("grassAreas")), "ponds": polys(J.get("ponds")),
        "roads": polys(J.get("roads")), "osm": osm, "towers": list(towers.values()),
        "trees": trees,
    }


def poly_centroid(p):
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    if abs(a) < 1e-9:
        return p.mean(axis=0)
    return np.array([((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)])


def nondominated(points):
    """indices of the non-dominated points (both objectives minimised)"""
    idx = []
    for i, (a1, a2) in enumerate(points):
        dominated = any((b1 <= a1 and b2 <= a2) and (b1 < a1 or b2 < a2)
                        for j, (b1, b2) in enumerate(points) if j != i)
        if not dominated:
            idx.append(i)
    return idx


def load_run(input_dir, output_dir, state_path, limit=None):
    state = json.loads(Path(state_path).read_text())
    origin, evals = state["origin"], state.get("evals", {})
    # SEQUENCE = order of nsga_state.json "origin" (already chronological), grouped by generation
    order = sorted(origin.keys(), key=lambda n: origin[n].get("gen", 0))
    if limit:
        order = order[:limit]
    root = next((n for n in order if origin[n].get("kind") == "base"), None)

    ops_by_file = {}
    log = Path(output_dir) / "candidates_log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                ops_by_file[Path(r.get("file", "")).stem] = r

    nodes = {}
    nan = float("nan")
    for i, name in enumerate(order):
        o, ev = origin[name], evals.get(name, {})
        parents, seeded = list(o.get("parents") or []), False
        if not parents and o.get("kind") == "individual" and root and name != root:
            parents, seeded = [root], True            # random individuals are perturbations of the base layout
        rec = ops_by_file.get(name, {})
        ops = summarise_ops(rec.get("ops", [])) if rec else ""
        if o.get("kind") == "base":
            ops = "starting layout (base)"
        nodes[name] = Node(
            name=name, idx=i, gen=o.get("gen", 0), kind=o.get("kind", "?"), parents=parents, seeded=seeded,
            seed=o.get("seed"), status=ev.get("status", "missing"),
            heat=num(ev.get("utci_heat")), cold=num(ev.get("utci_cold")),
            # failed runs are stored with a 1e6 penalty value, and a missing heat/cold case may be null: never plot that
            f1=num(ev.get("f1")) if ev.get("status") == "ok" else nan,
            f2=num(ev.get("f2")) if ev.get("status") == "ok" else nan,
            ops=ops, short="base" if o.get("kind") == "base" else short_name(name))

    scenes = {}
    for name in order:
        p = Path(input_dir) / f"{name}.json"
        if p.exists():
            try:
                scenes[name] = parse_scene(json.loads(p.read_text()))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise SystemExit(f"cannot read the scene {p}: {type(exc).__name__} {exc} "
                                 f"(unexpected vertex or position format - send me this file)")
        else:
            print(f"  ! {p} not found - plan view for {name} will be blank", file=sys.stderr)
            scenes[name] = None

    alive = {}
    for p in sorted(Path(output_dir).glob("nsga_gen_*.csv")):
        g = int(re.search(r"nsga_gen_(\d+)", p.name).group(1))
        with open(p, newline="") as fh:
            alive[g] = {row["name"] for row in csv.DictReader(fh)}

    picks = []
    t5 = Path(output_dir) / "top5.json"
    if t5.exists():
        picks = [(r.get("rank_in_list", i + 1), r["name"]) for i, r in enumerate(json.loads(t5.read_text()))
                 if r["name"] in nodes]

    run = Run(nodes, order, state, scenes, alive, picks)
    # axis captions: f1/f2 are "distance from the comfort band" in this run, say so when the data confirm it
    lo, hi = (state.get("comfort") or [None, None])
    ok1 = [n for n in nodes.values() if n.status == "ok" and not math.isnan(n.f1) and not math.isnan(n.heat)]
    ok2 = [n for n in nodes.values() if n.status == "ok" and not math.isnan(n.f2) and not math.isnan(n.cold)]
    # f1/f2 = distance outside the comfort band; some runs clamp it at 0 once a layout is inside the band
    if ok1 and hi is not None:
        if all(abs(n.f1 - (n.heat - hi)) < 0.011 for n in ok1):
            run.f_labels = (f"f1 = UTCI heat − {hi:g} °C", run.f_labels[1])
        elif all(abs(n.f1 - max(0.0, n.heat - hi)) < 0.011 for n in ok1):
            run.f_labels = (f"f1 = max(0, UTCI heat − {hi:g} °C)", run.f_labels[1])
    if ok2 and lo is not None:
        if all(abs(n.f2 - (lo - n.cold)) < 0.011 for n in ok2):
            run.f_labels = (run.f_labels[0], f"f2 = {lo:g} °C − UTCI cold")
        elif all(abs(n.f2 - max(0.0, lo - n.cold)) < 0.011 for n in ok2):
            run.f_labels = (run.f_labels[0], f"f2 = max(0, {lo:g} °C − UTCI cold)")
    run.pos = layout_tree(run)
    return run


# --------------------------------------------------------------------------- genealogy helpers
def ancestors(run, name):
    seen, stack = set(), [name]
    while stack:
        n = stack.pop()
        if n in seen or n not in run.nodes:
            continue
        seen.add(n)
        stack.extend(run.nodes[n].parents)
    return seen


def layout_tree(run):
    """x = generation, y = barycentre of the parents (overlaps resolved), keeps crossings low"""
    cols = defaultdict(list)
    for n in run.order:
        cols[run.nodes[n].gen].append(n)
    pos = {}
    for g in sorted(cols):
        names = cols[g]
        if g == min(cols):
            for i, n in enumerate(names):
                pos[n] = (g, float(i))
            continue
        want = []
        for n in names:
            ps = [pos[p][1] for p in run.nodes[n].parents if p in pos]
            want.append(np.mean(ps) if ps else np.nan)
        fill = np.nanmean([w for w in want if not np.isnan(w)]) if any(not np.isnan(w) for w in want) else 0
        want = [fill if np.isnan(w) else w for w in want]
        order = np.argsort(want, kind="stable")
        ys, prev = {}, -1e9
        for i in order:
            ys[i] = max(want[i], prev + 1.0)
            prev = ys[i]
        shift = np.mean([want[i] for i in ys]) - np.mean(list(ys.values()))
        lo = min(p[1] for p in pos.values())
        hi = max(p[1] for p in pos.values())
        for i, n in enumerate(names):
            pos[n] = (g, float(np.clip(ys[i] + shift, lo, hi)))
        # clipping could re-create overlaps, push them apart again
        srt = sorted(names, key=lambda n: pos[n][1])
        for a, b in zip(srt, srt[1:]):
            if pos[b][1] - pos[a][1] < 1.0:
                pos[b] = (g, pos[a][1] + 1.0)
    return pos


def front_names(run, names):
    ok = [n for n in names if run.nodes[n].ok]
    pts = [(run.nodes[n].f1, run.nodes[n].f2) for n in ok]
    return {ok[i] for i in nondominated(pts)}


def dead_names(run, k):
    """layouts that had already been dropped by selection when child k was created"""
    nd = run.nodes[run.order[k]]
    c = nd.gen - 1
    if c < 0 or c not in run.alive_after_gen:
        return set()
    born = run.order[:k + 1]
    return {n for n in born if run.nodes[n].gen <= c and run.nodes[n].ok and n not in run.alive_after_gen[c]}


# --------------------------------------------------------------------------- drawing: scene
def draw_scene(ax, run, name, parents=None, compact=False):
    sc = run.scenes.get(name)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if sc is None:
        ax.text(0.5, 0.5, "scene file missing", transform=ax.transAxes, ha="center", color=MUTED)
        return
    ref = sc["site"]
    if ref is None:
        allp = np.vstack([t["poly"] for t in sc["towers"]] or [np.zeros((1, 2))])
        ref = allp
    mn, mx = ref.min(axis=0), ref.max(axis=0)
    pad = 0.07 * (mx - mn).max()
    ax.set_xlim(mn[0] - pad, mx[0] + pad)
    ax.set_ylim(mn[1] - pad, mx[1] + pad)

    if sc["site"] is not None:
        ax.add_patch(Polygon(sc["site"], closed=True, fc="#f4f2ea", ec=INK, lw=1.2, zorder=1))
    for p in sc["grass"]:
        ax.add_patch(Polygon(p, closed=True, fc="#c3e6a9", ec="#6aa84f", lw=.8, zorder=2))
    for p in sc["ponds"]:
        ax.add_patch(Polygon(p, closed=True, fc="#8fc4ee", ec="#3f86c9", lw=.9, alpha=.95, zorder=3))
    for p in sc["roads"]:
        ax.add_patch(Polygon(p, closed=True, fc="#8a8a8a", ec="none", zorder=3))
    for p in sc["osm"]:
        ax.add_patch(Polygon(p, closed=True, fc="#a9b3bd", ec="#7d8b97", lw=.6, zorder=3))

    # parents' building footprints as dashed ghosts + arrows to the child's building
    parents = parents or []
    child_t = sc["towers"]
    for pi, pn in enumerate(parents[:2]):
        psc = run.scenes.get(pn)
        if not psc:
            continue
        col = PARENT_COLORS[pi] if len(parents) > 1 else INK
        for t in psc["towers"]:
            ax.add_patch(Polygon(t["poly"], closed=True, fc="none", ec=col, lw=1.5, ls=(0, (4, 3)), zorder=4))
        if len(child_t) == len(psc["towers"]):           # same building order in both files: arrow per moved building
            for pt, ct in zip(psc["towers"], child_t):
                a, b = poly_centroid(pt["poly"]), poly_centroid(ct["poly"])
                if np.hypot(*(b - a)) > 1.0:
                    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=col,
                                                 zorder=9, shrinkA=0, shrinkB=0))

    # trees: dark outline = tree that is new / moved / resized relative to the parent(s)
    known = set()
    for pn in parents:
        psc = run.scenes.get(pn)
        if psc:
            known |= {(t[4], round(t[0], 3), round(t[1], 3), round(t[2], 3)) for t in psc["trees"]}
    circles, changed_flags = [], []
    for t in sorted(sc["trees"], key=lambda t: t[3]):
        circles.append(Circle((t[0], t[1]), t[2]))
        changed_flags.append(bool(known) and (t[4], round(t[0], 3), round(t[1], 3), round(t[2], 3)) not in known)
    if circles:
        fresh = [c for c, f in zip(circles, changed_flags) if f]
        old = [c for c, f in zip(circles, changed_flags) if not f]
        if old:
            ax.add_collection(PatchCollection(old, fc=(.37, .54, .16, .55), ec=(.18, .35, .12, .9), lw=.6, zorder=5))
        if fresh:
            ax.add_collection(PatchCollection([Circle(c.center, c.radius) for c in fresh],
                                              fc=(.37, .54, .16, .62), ec="#143d0c", lw=1.6, zorder=6))

    for t in child_t:
        ax.add_patch(Polygon(t["poly"] + np.array([1.8, -1.8]), closed=True, fc="black", ec="none", alpha=.18,
                             zorder=7))
        ax.add_patch(Polygon(t["poly"], closed=True, fc="#d98e65", ec="#6e3f26", lw=1.4, zorder=8))
        c = poly_centroid(t["poly"])
        if not compact:
            ax.text(c[0], c[1], f"{t['z1'] - t['z0']:g} m", ha="center", va="center", fontsize=9, color="#3a2012",
                    family="DejaVu Sans Mono", zorder=10, fontweight="bold",
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground="#d98e65")])
    if not compact:                              # 20 m scale bar
        x0, y0 = mn[0], mn[1] - pad * .55
        ax.plot([x0, x0 + 20], [y0, y0], color=INK, lw=2, solid_capstyle="butt", zorder=10, clip_on=False)
        ax.text(x0 + 10, y0 - pad * .12, "20 m", ha="center", va="top", fontsize=8, color=MUTED, clip_on=False)


# --------------------------------------------------------------------------- drawing: tree
def bezier(ax, p0, p1, **kw):
    dx = p1[0] - p0[0]
    dx = dx if dx > 0 else 0.5
    verts = [p0, (p0[0] + dx * .55, p0[1]), (p1[0] - dx * .55, p1[1]), p1]
    ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO] + [MPath.CURVE4] * 3), fc="none", **kw))


def draw_tree(ax, run, born, *, current=None, dead=(), front=(), picks=None, lineage=None,
              msize=480, fs=8.5, detail=False, show_cols=True):
    """born = names that exist at this point of the replay (others are drawn as faint placeholders)"""
    picks = picks or {}
    born = set(born)
    pos = run.pos
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - .55, max(xs) + .55)
    ax.set_ylim(max(ys) + .8, min(ys) - (1.35 if show_cols else .8))      # inverted: gen 0 top-down
    ax.axis("off")
    ax.patch.set_alpha(0)

    if show_cols:
        for g in sorted(set(xs)):
            ax.text(g, min(ys) - 1.0, "initial" if g == 0 else f"gen {g}", ha="center", va="center",
                    fontsize=fs + 1, color=MUTED, family="DejaVu Sans Mono")
    # edges first (under the nodes)
    lin_edges = set()
    if lineage:
        for n in lineage:
            for p in run.nodes[n].parents:
                if p in lineage:
                    lin_edges.add((p, n))
    for name in run.order:
        nd = run.nodes[name]
        if name not in born:
            continue
        for p in nd.parents:
            if p not in pos or p not in born:
                continue
            if nd.seeded:                      # drawn as one bracket below, not as 10 overlapping curves
                continue
            on = (p, name) in lin_edges
            col = ACCENT if on else kind_color(nd.kind)
            kw = dict(ec=col, lw=3.0 if on else 1.5, alpha=1 if on else (.75 if not lineage else .35),
                      zorder=3 if on else 1)
            bezier(ax, pos[p], pos[name], **kw)
    seeded = [n for n in run.order if run.nodes[n].seeded and n in born]
    if seeded:                                  # "i1..i10 are random perturbations of the base layout"
        x0 = pos[seeded[0]][0] - .32
        ya, yb = pos[run.nodes[seeded[0]].parents[0]][1], max(pos[n][1] for n in seeded)
        ax.plot([x0 + .06, x0, x0, x0 + .06], [ya, ya, yb, yb], color="#9aa3aa", lw=1.2, zorder=1,
                solid_capstyle="round")

    for name in run.order:
        x, y = pos[name]
        nd = run.nodes[name]
        if name not in born:                                                  # not created yet
            ax.scatter([x], [y], s=msize, fc="none", ec=LINE, lw=1, ls=":", zorder=4)
            ax.text(x, y, nd.short, ha="center", va="center", fontsize=fs, color=LINE, zorder=5)
            continue
        col = kind_color(nd.kind)
        is_dead = name in dead
        fc = mix(col, PANEL, .62) if is_dead else col
        if name == current:
            ax.scatter([x], [y], s=msize * 3.6, fc=ACCENT, ec="none", alpha=.22, zorder=4)
        ec, lw = (GOLD, 3.2) if name in front else (INK if name == current else mix(INK, PANEL, .45), 0.9)
        if name == current:
            lw = 3.2 if name not in front else 3.6
            ec = INK if name not in front else GOLD
        if nd.failed:                                                         # CFD evaluation failed
            ax.scatter([x], [y], s=msize, fc=[PANEL], ec=[ACCENT], lw=2.2, ls=(0, (2, 1.5)), zorder=6)
            ax.text(x, y, nd.short, ha="center", va="center", fontsize=fs, color=ACCENT, fontweight="bold", zorder=7)
            if detail:
                ax.text(x, y + .50, "eval failed", ha="center", va="center", fontsize=fs - 1.5, color=ACCENT,
                        family="DejaVu Sans Mono", zorder=7)
            continue
        ax.scatter([x], [y], s=msize, fc=[fc], ec=[ec], lw=lw, zorder=6)
        ax.text(x, y, nd.short, ha="center", va="center", fontsize=fs, color="white" if not is_dead else col,
                fontweight="bold", zorder=7)
        if detail and nd.ok:
            ax.text(x, y + .50, f"{nd.f1:.2f} | {nd.f2:.2f}", ha="center", va="center", fontsize=fs - 1.5,
                    color=MUTED, family="DejaVu Sans Mono", zorder=7)
        if name in picks:
            ax.annotate(f"★{picks[name]}", (x, y), xytext=(12, 12), textcoords="offset points", fontsize=fs + 2,
                        color=GOLD, fontweight="bold", zorder=8,
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground=INK)])


# --------------------------------------------------------------------------- drawing: objective space
def scatter_limits(run):
    ok = [n for n in run.nodes.values() if n.ok]
    f1 = [n.f1 for n in ok]
    f2 = [n.f2 for n in ok]
    ref = run.state.get("ref") or [max(f1), max(f2)]
    x0, x1 = min(f1), max(max(f1), ref[0])
    y0, y1 = min(0.0, min(f2)), ref[1] * 1.18
    xp = (x1 - x0) * .06
    return (x0 - xp, x1 + xp), (y0 - (y1 - y0) * .06, y1)


def draw_scatter(ax, run, born, *, current=None, dead=(), front=(), picks=None, fs=8.5, label_picks=False):
    picks = picks or {}
    xl, yl = scatter_limits(run)
    ref = run.state.get("ref")
    ax.set_xlim(*xl), ax.set_ylim(*yl)
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(labelsize=fs, colors=MUTED, length=3)
    ax.grid(color=LINE, lw=.5, alpha=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(run.f_labels[0] + "   (lower is better)", fontsize=fs, color=MUTED)
    ax.set_ylabel(run.f_labels[1] + "   (lower is better)", fontsize=fs, color=MUTED)
    if ref:
        ax.axvline(ref[0], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axhline(ref[1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(xl[0] + (xl[1] - xl[0]) * .01, ref[1], " hypervolume reference", fontsize=fs - 1.5, color=MUTED,
                va="bottom")
    nodes = [run.nodes[n] for n in run.order if n in born and run.nodes[n].ok]
    fr = sorted([run.nodes[n] for n in front if n in born and run.nodes[n].ok], key=lambda n: n.f1)
    inside = [n for n in fr if ref is None or (n.f1 < ref[0] and n.f2 < ref[1])]
    if inside and ref:                                   # dominated area (the hypervolume)
        xs = [n.f1 for n in inside] + [ref[0]]
        ys = [n.f2 for n in inside] + [inside[-1].f2]
        ax.fill_between(xs, ys, ref[1], step="post", color=ACCENT, alpha=.10, lw=0, zorder=1)
    if fr:
        ax.step([n.f1 for n in fr], [n.f2 for n in fr], where="post", color=GOLD, lw=1.8, zorder=2)
    for n in nodes:
        y, clipped = n.f2, n.f2 > yl[1]
        if clipped:
            y = yl[1] - (yl[1] - yl[0]) * .03
        isdead = n.name in dead
        mk = "^" if clipped else "o"
        ax.scatter([n.f1], [y], s=70 if n.name != current else 150, marker=mk,
                   fc=[mix(kind_color(n.kind), PANEL, .6) if isdead else kind_color(n.kind)],
                   ec=[GOLD if n.name in front else mix(INK, PANEL, .4)], lw=2 if n.name in front else .8, zorder=4)
        if clipped:
            ax.annotate(f"{n.short}: f2 = {n.f2:.2f} ↑", (n.f1, y), xytext=(8, -3), textcoords="offset points",
                        fontsize=fs - 1, color=INK, va="top")
    if label_picks and picks:
        draw_pick_inset(ax, run, born, dead, front, picks, fs)
    cur = run.nodes.get(current)
    if cur is not None and cur.ok:
        y = min(cur.f2, yl[1] - (yl[1] - yl[0]) * .03)
        ax.scatter([cur.f1], [y], s=420, fc="none", ec=ACCENT, lw=2.4, zorder=5)


def draw_pick_inset(ax, run, born, dead, front, picks, fs):
    """the selected layouts are usually packed within a few hundredths: zoom in on them with numbered stars"""
    pn = [run.nodes[n] for n in picks if run.nodes[n].ok]
    if not pn:
        return
    x0, x1 = min(n.f1 for n in pn), max(n.f1 for n in pn)
    y0, y1 = min(n.f2 for n in pn), max(n.f2 for n in pn)
    px, py = max((x1 - x0) * .3, .02), max((y1 - y0) * .3, .01)
    ins = ax.inset_axes([0.38, 0.40, 0.60, 0.52])
    ins.set_facecolor("white")
    ins.set_xlim(x0 - px, x1 + px), ins.set_ylim(max(y0 - py, -py * .5), y1 + py)
    ins.tick_params(labelsize=fs - 1.5, colors=MUTED, length=2)
    for sp in ins.spines.values():
        sp.set_color(GOLD)
    ins.grid(color=LINE, lw=.5, alpha=.7)
    ins.set_title("zoom on the selected layouts", fontsize=fs - 1.5, color=MUTED, pad=3)
    for n in (run.nodes[m] for m in run.order if m in born and run.nodes[m].ok):
        if x0 - px <= n.f1 <= x1 + px and ins.get_ylim()[0] <= n.f2 <= y1 + py:
            ins.scatter([n.f1], [n.f2], s=90 if n.name in picks else 45, fc=[kind_color(n.kind)],
                        ec=[GOLD if n.name in front else mix(INK, PANEL, .4)],
                        lw=2 if n.name in front else .6, zorder=3)
    for i, n in enumerate(sorted(pn, key=lambda n: n.f1)):
        ins.annotate(f"★{picks[n.name]}", (n.f1, n.f2), xytext=(0, 9 if i % 2 == 0 else -15),
                     textcoords="offset points", ha="center", fontsize=fs, color=INK, fontweight="bold")
    ax.indicate_inset_zoom(ins, edgecolor=GOLD, lw=1)


# --------------------------------------------------------------------------- frame composition
class Canvas:
    def __init__(self, W, H, dpi):
        self.W, self.H, self.dpi = W, H, dpi
        self.fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
        self.bg = self.fig.add_axes([0, 0, 1, 1], zorder=0)
        self.bg.set_xlim(0, W), self.bg.set_ylim(H, 0), self.bg.axis("off")

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], zorder=2, **kw)

    def card(self, x, y, w, h, title=None):
        self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=PANEL, ec=LINE,
                                         lw=1, zorder=1))
        if title:
            self.bg.text(x + 14, y + 22, title.upper(), fontsize=8.5, color=MUTED, family="DejaVu Sans Mono",
                         va="center", zorder=3)

    def text(self, x, y, s, size=10, color=INK, mono=True, weight="normal", ha="left", va="center", **kw):
        return self.bg.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5,
                            family="DejaVu Sans Mono" if mono else "DejaVu Sans", **kw)

    def chip(self, x, y, s, color, size=10, h=26, charw=None):
        w = (charw or size * .83) * len(s) + 20
        self.bg.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=4", fc=color,
                                         ec="none", zorder=4))
        self.text(x + w / 2, y, s, size=size, color="white", weight="bold", ha="center")
        return x + w

    def save(self, path):
        self.fig.savefig(path, facecolor=BG, dpi=self.dpi)
        plt.close(self.fig)


def tree_legend(cv, x, y, run, right=None):
    """legend row; with right=<px> the row is right-aligned to that x so it never leaves its card"""
    kinds = [k for k in KIND_STYLE if any(n.kind == k for n in run.nodes.values())]
    if right is not None:
        w = sum(24 + 6.1 * len(kind_label(k)) for k in kinds) + 24 + 6.1 * len("Pareto front")
        if any(n.failed for n in run.nodes.values()):
            w += 24 + 6.1 * len("evaluation failed")
        x = right - w
    for k in kinds:
        cv.bg.scatter([x], [y], s=70, fc=kind_color(k), ec="none", zorder=5)
        cv.text(x + 10, y, kind_label(k), size=8, color=MUTED, mono=False)
        x += 24 + 6.1 * len(kind_label(k))
    cv.bg.scatter([x], [y], s=70, fc="none", ec=GOLD, lw=2.4, zorder=5)
    cv.text(x + 10, y, "Pareto front", size=8, color=MUTED, mono=False)
    if any(n.failed for n in run.nodes.values()):
        x += 24 + 6.1 * len("Pareto front")
        cv.bg.scatter([x], [y], s=70, fc=PANEL, ec=ACCENT, lw=1.6, ls=(0, (2, 1.5)), zorder=5)
        cv.text(x + 10, y, "evaluation failed", size=8, color=MUTED, mono=False)


def header(cv, run, nd, k, N, title="UHI LAYOUT · NSGA-II REPLAY"):
    W = cv.W
    cv.bg.add_patch(Rectangle((0, 0), W, 96, fc=PANEL, ec="none", zorder=1))
    cv.bg.add_patch(Rectangle((0, 95), W, 1, fc=LINE, ec="none", zorder=2))
    cv.bg.add_patch(Rectangle((0, 92), W * (k + 1) / N, 4, fc=ACCENT, ec="none", zorder=3))
    cv.text(28, 24, title, size=9, color=MUTED)
    cv.text(28, 60, nd.name, size=25, weight="bold")
    x = 28 + 25 * .83 * len(nd.name) + 22
    x = cv.chip(x, 58, f"GEN {nd.gen}", INK, size=10) + 8
    cv.chip(x, 58, kind_label(nd.kind).upper(), kind_color(nd.kind), size=10)
    # parents / ops
    if nd.parents and not nd.seeded:
        par = " × ".join(nd.parents)
    elif nd.seeded:
        par = f"{nd.parents[0]} (random perturbation)"
    else:
        par = "–"
    cv.text(640, 24, "PARENTS", size=8, color=MUTED)
    cv.text(640, 44, par, size=11)
    cv.text(640, 64, "CHANGES", size=8, color=MUTED)
    ops = textwrap.shorten(nd.ops or "–", width=52, placeholder=" …")
    cv.text(640, 82, ops, size=9)
    # metrics
    if nd.failed:
        cv.text(1110, 28, "EVALUATION", size=8, color=MUTED)
        cv.text(1110, 60, "FAILED", size=21, weight="bold", color=ACCENT)
        cv.text(1290, 50, "CFD post-processing error: this layout is excluded", size=9, color=MUTED, mono=False)
        cv.text(1290, 68, "from selection, the front and the plots.", size=9, color=MUTED, mono=False)
        return
    cols = [("UTCI HEAT", fm(nd.heat, ".2f", " °C")), ("UTCI COLD", fm(nd.cold, ".2f", " °C")),
            ("f1", fm(nd.f1)), ("f2", fm(nd.f2))]
    for i, (lab, val) in enumerate(cols):
        xx = 1110 + i * 200
        cv.text(xx, 28, lab, size=8, color=MUTED)
        cv.text(xx, 60, val, size=21, weight="bold")


def fmt_state_line(run):
    s = run.state
    pop = f"population {s.get('pop')} + base" if s.get("pop") else f"population {s.get('size')}"
    return f"seed {s.get('seed')} · {pop} · {s.get('gen')} generations · evaluator '{s.get('evaluator')}'"


def scene_legend(cv, x, y, parents_n, max_x=None):
    """legend row; with max_x the row wraps onto a second line instead of leaving its card"""
    items = [("box", "#d98e65", "building"), ("box", "#c3e6a9", "lawn"), ("box", "#8fc4ee", "pond"),
             ("box", (.37, .54, .16, .6), "tree"), ("ring", None, "tree new/moved vs parent"),
             ("dash", None, "parent's building")]
    x0 = x
    for kind, col, lab in items:
        w = 38 + 6.2 * len(lab)
        if max_x is not None and x + w - 14 > max_x and x > x0:
            x, y = x0, y + 20
        if kind == "box":
            cv.bg.add_patch(Rectangle((x, y - 6), 14, 12, fc=col, ec=mix(INK, PANEL, .5), lw=.8, zorder=5))
        elif kind == "ring":
            cv.bg.add_patch(Circle((x + 7, y), 6.5, fc=(.37, .54, .16, .6), ec="#143d0c", lw=1.8, zorder=5))
        else:
            cv.bg.add_patch(Rectangle((x, y - 6), 14, 12, fc="none", ec=INK, lw=1.4, ls=(0, (3, 2)), zorder=5))
        cv.text(x + 20, y, lab, size=8, color=MUTED, mono=False)
        x += w


# --------------------------------------------------------------------------- UTCI maps
MAP_CASES = (("heat", "UTCI · heat case"), ("cold", "UTCI · cold case"))
# utci_plot.py draws on a fixed 10.5 x 8 in figure: title block above y=0.918, footer text below y=0.14.
# Those strips are small at video size, so they are cropped away (the header already carries the numbers).
MAP_CROP = (0.082, 0.14, 0.165)       # top, bottom, right (the stress-category names right of the colour bar)


def find_map(runs_dir, name, case, pattern):
    """path of the UTCI image of one layout and case, or None"""
    if runs_dir is None:
        return None
    p = Path(pattern.format(runs=runs_dir, name=name, case=case))
    if p.exists():
        return p
    base = Path(runs_dir) / name / case                      # fall back to any utci_map.png below <run>/<name>/<case>
    if base.is_dir():
        hits = sorted(base.rglob("utci_map.png"))
        if hits:
            return hits[0]
    return None


_MAP_CACHE = {}


def load_map(path, max_w, max_h, crop=MAP_CROP):
    """UTCI png -> RGB array (cropped to the map + colour bar) that fits max_w x max_h px, or None if unreadable"""
    key = (str(path), max_w, max_h, crop)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    arr = None
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if crop and abs(w / h - 10.5 / 8.0) < 0.03:          # only crop images that have utci_plot's proportions
            im = im.crop((0, int(round(crop[0] * h)), int(round((1 - crop[2]) * w)), int(round((1 - crop[1]) * h))))
        w, h = im.size
        f = min(max_w / w, max_h / h)
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS)
        arr = np.asarray(im)
    except Exception as exc:                                   # unreadable file: show a note instead of failing
        print(f"  ! cannot read {path}: {exc}", file=sys.stderr)
    if len(_MAP_CACHE) > 8:
        _MAP_CACHE.clear()
    _MAP_CACHE[key] = arr
    return arr


def draw_map_card(cv, x, y, w, h, title, nd, case, path, caption, crop):
    cv.card(x, y, w, h, title)
    if path is None:
        why = ("evaluation failed - no map" if nd.failed else
               f"no {case} case for this layout" if math.isnan(nd.heat if case == "heat" else nd.cold) else
               f"no {case} utci_map.png found")
        cv.text(x + w / 2, y + h / 2, why, size=10, color=MUTED, ha="center", mono=False)
        return False
    arr = load_map(path, w - 20, h - 36 - 36, crop)
    if arr is None:
        cv.text(x + w / 2, y + h / 2, "map unreadable", size=10, color=MUTED, ha="center", mono=False)
        return False
    ih, iw = arr.shape[:2]
    top = y + 36 + max(0, (h - 36 - 36 - ih) / 2)
    ax = cv.axes(x + (w - iw) / 2, top, iw, ih)
    ax.imshow(arr, interpolation="nearest", aspect="auto")
    ax.axis("off")
    cv.text(x + w / 2, y + h - 18, caption, size=9, color=INK, ha="center")
    return True


def draw_hv_spark(cv, run, completed_gen, x, y, w, h):
    hist = run.state.get("history") or []
    ax = cv.axes(x, y, w, h)
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(labelsize=8, colors=MUTED, length=2)
    gens = [r["gen"] for r in hist]
    hv = [r["hypervolume"] for r in hist]
    if gens:
        ax.set_xlim(min(gens) - .3, max(gens) + .3)
        pad = (max(hv) - min(hv)) * .15 or .05
        ax.set_ylim(min(hv) - pad, max(hv) + pad)
        ax.set_xticks(gens)
        ax.plot(gens, hv, color=LINE, lw=1.5, ls=":")
        done = [(g, v) for g, v in zip(gens, hv) if g <= completed_gen]
        if done:
            ax.plot(*zip(*done), color=ACCENT, lw=2.2, marker="o", ms=5)
            ax.annotate(f"{done[-1][1]:.3f}", done[-1], xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.grid(color=LINE, lw=.5, alpha=.7)
    ax.set_axisbelow(True)


def render_frame(run, k, path, W=1920, H=1080, dpi=100, maps=None):
    """maps = dict(runs_dir=..., pattern=..., crop=...) to add the heat/cold UTCI images, or None for the old layout"""
    name = run.order[k]
    nd = run.nodes[name]
    born = run.order[:k + 1]
    front = front_names(run, born)
    dead = dead_names(run, k)
    lineage = ancestors(run, name)
    cv = Canvas(W, H, dpi)
    header(cv, run, nd, k, run.n)
    ntree = len(run.scenes[name]["trees"]) if run.scenes.get(name) else 0
    note = (f"{ntree} trees" if len(nd.parents) != 2 else
            f"dashed: blue = {nd.parents[0]}, magenta = {nd.parents[1]}")
    found = {}

    if maps is None:                                           # ---- layout without UTCI images
        cv.card(24, 116, 860, 940, "Layout · plan view")
        cv.text(870, 138, note if len(nd.parents) != 2 else "parent's building, " + note,
                size=8.5, color=MUTED, ha="right", mono=len(nd.parents) != 2)
        ax = cv.axes(34, 150, 840, 860)
        draw_scene(ax, run, name, parents=nd.parents)
        scene_legend(cv, 40, 1034, len(nd.parents))
        tree_box, obj_box, prog_box = (904, 116, 992, 544), (904, 680, 496, 376), (1420, 680, 476, 376)
    else:                                                      # ---- layout with the two UTCI maps
        RA, HA = 116, 490                                       # row A: plan | heat map | cold map
        cv.card(24, RA, 572, HA, "Layout · plan view")
        cv.text(582, RA + 22, note, size=8.5, color=MUTED, ha="right", mono=len(nd.parents) != 2)
        ax = cv.axes(30, RA + 34, 560, HA - 34 - 50)
        draw_scene(ax, run, name, parents=nd.parents)
        scene_legend(cv, 38, RA + HA - 34, len(nd.parents), max_x=584)
        vals = {"heat": (nd.heat, nd.f1, "f1"), "cold": (nd.cold, nd.f2, "f2")}
        for i, (case, ttl) in enumerate(MAP_CASES):
            p = find_map(maps["runs_dir"], name, case, maps["pattern"])
            v, f, fname = vals[case]
            cap = "–" if nd.failed else f"site-mean UTCI {fm(v)} °C   ·   {fname} = {fm(f)}"
            found[case] = draw_map_card(cv, 616 + i * 650, RA, 630, HA, ttl, nd, case, p, cap, maps["crop"])
        tree_box, obj_box, prog_box = (24, 626, 980, 430), (1024, 626, 440, 430), (1484, 626, 412, 430)

    # tree
    x0, y0, w0, h0 = tree_box
    cv.card(x0, y0, w0, h0, "Family tree")
    tree_legend(cv, x0 + 246, y0 + 22, run, right=x0 + w0 - 16)
    ax = cv.axes(x0 + 10, y0 + 34, w0 - 20, h0 - 44)
    small = maps is not None
    draw_tree(ax, run, born, current=name, dead=dead, front=front, lineage=lineage,
              msize=400 if small else 520, fs=8 if small else 8.5)

    # objective space
    x0, y0, w0, h0 = obj_box
    cv.card(x0, y0, w0, h0, "Objective space")
    ax = cv.axes(x0 + 56, y0 + 38, w0 - 72, h0 - 38 - 42)
    draw_scatter(ax, run, born, current=name, dead=dead, front=front)

    # stats
    x0, y0, w0, h0 = prog_box
    cv.card(x0, y0, w0, h0, "Search progress")
    ok = [run.nodes[n] for n in born if run.nodes[n].ok]
    in_gen = [n for n in run.order if run.nodes[n].gen == nd.gen]
    pos_in_gen = in_gen.index(name) + 1
    stage = "initial population" if nd.gen == 0 else "offspring"
    lines = [
        ("step", f"{k + 1} / {run.n}"),
        ("generation", f"{nd.gen} · {stage} {pos_in_gen}/{len(in_gen)}"),
        ("evaluated", f"{len(ok)} layouts"),
        ("front size", f"{len(front)}"),
        ("best f1", (lambda b: f"{b.f1:.2f}  ({b.short})")(min(ok, key=lambda n: n.f1)) if ok else "–"),
        ("best f2", (lambda b: f"{b.f2:.2f}  ({b.short})")(min(ok, key=lambda n: n.f2)) if ok else "–"),
    ]
    for i, (a_, b_) in enumerate(lines):
        cv.text(x0 + 18, y0 + 46 + i * 24, a_.upper(), size=8, color=MUTED)
        cv.text(x0 + 140, y0 + 46 + i * 24, b_, size=10)
    ty = y0 + 46 + len(lines) * 24 + 22
    cv.text(x0 + 18, ty, "HYPERVOLUME BY COMPLETED GENERATION", size=8, color=MUTED)
    draw_hv_spark(cv, run, nd.gen - 1, x0 + 32, ty + 14, w0 - 56, y0 + h0 - 44 - (ty + 14))
    cv.save(path)
    return found


def render_summary(run, path, W=1920, H=1080, dpi=100):
    N = run.n
    nd_last = run.nodes[run.order[-1]]
    cv = Canvas(W, H, dpi)
    cv.bg.add_patch(Rectangle((0, 0), W, 96, fc=PANEL, ec="none", zorder=1))
    cv.bg.add_patch(Rectangle((0, 95), W, 1, fc=LINE, ec="none", zorder=2))
    cv.bg.add_patch(Rectangle((0, 92), W, 4, fc=ACCENT, ec="none", zorder=3))
    cv.text(28, 24, "UHI LAYOUT · NSGA-II REPLAY", size=9, color=MUTED)
    cv.text(28, 60, "Result", size=25, weight="bold")
    cv.text(640, 34, fmt_state_line(run).upper(), size=9, color=MUTED)
    hist = (run.state.get("history") or [{}])[-1]
    nfail = sum(n.failed for n in run.nodes.values())
    cv.text(640, 62, f"{N - nfail} layouts evaluated" + (f" ({nfail} failed)" if nfail else "") +
            f" · final front {hist.get('front_size', '?')} · "
            f"hypervolume {hist.get('hypervolume', float('nan')):.3f}", size=11)

    picks = {n: r for r, n in run.picks}
    root = next((n for n in run.order if run.nodes[n].kind == "base"), None)
    show = ([root] if root else []) + [n for _, n in sorted(run.picks)]
    if not run.picks:                                  # no top5.json: show the final Pareto front instead
        fr = sorted(front_names(run, run.order), key=lambda n: run.nodes[n].f1)[:5]
        show = ([root] if root else []) + fr
        picks = {n: i + 1 for i, n in enumerate(fr)}
    cv.card(24, 116, 860, 940, "Base layout and selected layouts")
    cols, rows_ = 3, 2
    cw, ch = 280, 440
    for i, n in enumerate(show[:6]):
        cx, cy = 34 + (i % cols) * cw + 6, 150 + (i // cols) * (ch + 20)
        ax = cv.axes(cx, cy + 30, cw - 12, ch - 80)
        draw_scene(ax, run, n, parents=[], compact=True)
        nd = run.nodes[n]
        tag = "BASE" if nd.kind == "base" else f"★ #{picks.get(n, '')}"
        cv.text(cx + 4, cy + 12, f"{tag}  {n}", size=10, weight="bold", color=GOLD if tag.startswith("★") else INK)
        cv.text(cx + 4, cy + ch - 34, f"UTCI heat {fm(nd.heat, '.1f')} · cold {fm(nd.cold, '.1f')} °C", size=8.5, color=MUTED)
        cv.text(cx + 4, cy + ch - 14, f"f1 {fm(nd.f1)}   f2 {fm(nd.f2)}", size=9.5)

    final_pop = set(run.state.get("population") or [])
    dead = {n for n in run.order if final_pop and n not in final_pop}
    front = front_names(run, run.order)
    lineage = set()
    for n in picks:
        lineage |= ancestors(run, n)
    cv.card(904, 116, 992, 544, "Family tree")
    tree_legend(cv, 0, 138, run, right=1880)
    ax = cv.axes(914, 150, 972, 500)
    draw_tree(ax, run, run.order, dead=dead, front=front, picks=picks, lineage=lineage, msize=520)
    cv.card(904, 680, 496, 376, "Objective space")
    ax = cv.axes(960, 718, 424, 296)
    draw_scatter(ax, run, run.order, dead=dead, front=front, picks=picks, label_picks=True)
    cv.card(1420, 680, 476, 376, "Hypervolume by generation")
    draw_hv_spark(cv, run, 10 ** 6, 1452, 716, 434, 300)
    cv.text(918, 644, "orange path = ancestors of the starred layouts · faded = dropped by selection", size=8.5,
            color=MUTED, mono=False)
    cv.save(path)


def render_family_tree(run, out_png, out_svg=None, W=2600, H=1500, dpi=100):
    cv = Canvas(W, H, dpi)
    cv.bg.add_patch(Rectangle((0, 0), W, H, fc=PANEL, ec="none", zorder=0))
    cv.text(50, 52, "Family tree", size=26, weight="bold", mono=False)
    cv.text(50, 90, fmt_state_line(run) + " · each node: f1 | f2", size=11, color=MUTED)
    picks = {n: r for r, n in run.picks}
    final_pop = set(run.state.get("population") or [])
    dead = {n for n in run.order if final_pop and n not in final_pop}
    front = front_names(run, run.order)
    lineage = set()
    for n in picks:
        lineage |= ancestors(run, n)
    tree_legend(cv, 1400, 56, run)
    cv.bg.plot([1400, 1430], [92, 92], color=ACCENT, lw=3, zorder=5)
    cv.text(1438, 92, "ancestry of the starred (selected) layouts", size=9.5, color=MUTED, mono=False)
    cv.bg.scatter([1850], [92], s=70, fc=mix("#7d8b97", PANEL, .62), ec="none", zorder=5)
    cv.text(1862, 92, "dropped by selection", size=9.5, color=MUTED, mono=False)
    ax = cv.axes(60, 130, W - 120, H - 170)
    draw_tree(ax, run, run.order, dead=dead, front=front, picks=picks, lineage=lineage or None,
              msize=1500, fs=13, detail=True)
    cv.fig.savefig(out_png, facecolor=PANEL, dpi=dpi)
    if out_svg:
        cv.fig.savefig(out_svg, facecolor=PANEL)
    plt.close(cv.fig)


# --------------------------------------------------------------------------- video
def make_video(frames, durations, out_mp4, fps, fade, size):
    """cross-faded slideshow of the keyframes with ffmpeg (xfade); falls back to a hard-cut concat"""
    if not shutil.which("ffmpeg"):
        print("  ! ffmpeg not found on PATH - frames are written, video skipped", file=sys.stderr)
        return False
    n = len(frames)
    scale = f"scale={size[0]}:{size[1]},format=yuv420p"
    if fade > 0 and n > 1:
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for f, d in zip(frames, durations):
            cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{d + fade:.3f}", "-i", str(f)]
        parts, last, acc = [], "[0:v]", 0.0
        for i in range(1, n):
            acc += durations[i - 1]
            out = f"[x{i}]"
            parts.append(f"{last}[{i}:v]xfade=transition=fade:duration={fade:.3f}:offset={acc:.3f}{out}")
            last = out
        parts.append(f"{last}{scale}[v]")
        cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-r", str(fps), "-c:v", "libx264",
                "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(out_mp4)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print("  ! xfade pipeline failed, falling back to hard cuts:\n" + r.stderr[-600:], file=sys.stderr)
    lst = Path(out_mp4).with_suffix(".txt")
    with open(lst, "w") as fh:
        for f, d in zip(frames, durations):
            fh.write(f"file '{Path(f).resolve()}'\nduration {d:.3f}\n")
        fh.write(f"file '{Path(frames[-1]).resolve()}'\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-vf", f"fps={fps},{scale}", "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart", str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lst.unlink(missing_ok=True)
    if r.returncode != 0:
        print("  ! ffmpeg failed:\n" + r.stderr[-800:], file=sys.stderr)
        return False
    return True


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="folder with base/individual_N/offspring_N .json scenes")
    ap.add_argument("--output-dir", help="folder with nsga_state.json (and optional csv/log/top5)")
    ap.add_argument("--state", help="explicit path to nsga_state.json (default: <output-dir>/nsga_state.json)")
    ap.add_argument("--out", default="replay", help="where to write tree, frames and video")
    ap.add_argument("--seconds", type=float, default=2.0, help="seconds per layout in the video")
    ap.add_argument("--final-seconds", type=float, default=6.0, help="seconds for the closing summary frame")
    ap.add_argument("--fade", type=float, default=0.4, help="cross-fade seconds between frames (0 = hard cuts)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080", help="video/frame size WxH")
    ap.add_argument("--limit", type=int, help="only the first N layouts (quick test)")
    ap.add_argument("--no-video", action="store_true", help="write tree + frames only")
    ap.add_argument("--no-frames", action="store_true", help="write the family tree only")
    ap.add_argument("--no-summary", action="store_true", help="no closing summary frame")
    ap.add_argument("--runs-dir", help="folder with the CFD runs (run/<name>/heat|cold/...); default: <input-dir>/../run")
    ap.add_argument("--map-pattern", default="{runs}/{name}/{case}/temperature_humidity/postProcessing/utci_map.png",
                    help="where one UTCI image lives; {runs} {name} {case} (case = heat or cold) are filled in")
    ap.add_argument("--map-crop", type=float, nargs=3, metavar=("TOP", "BOTTOM", "RIGHT"), default=MAP_CROP,
                    help="fractions of utci_map.png cut off the top (title), bottom (footer text) and right "
                         "(category names); 0 0 0 keeps the whole image")
    ap.add_argument("--no-maps", action="store_true", help="do not add the UTCI images (old layout)")
    a = ap.parse_args(argv)

    state_path = Path(a.state) if a.state else (Path(a.output_dir) / "nsga_state.json" if a.output_dir else None)
    if not state_path or not state_path.exists():
        ap.error("nsga_state.json not found - pass --output-dir or --state")
    out_dir = Path(a.output_dir) if a.output_dir else state_path.parent
    W, H = (int(v) for v in a.size.lower().split("x"))
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    print(f"reading {state_path} and scenes in {a.input_dir}")
    run = load_run(a.input_dir, out_dir, state_path, a.limit)
    print(f"  {run.n} layouts, generations 0..{max(n.gen for n in run.nodes.values())}, "
          f"{sum(len(n.parents) for n in run.nodes.values() if not n.seeded)} breeding links")

    with open(out / "sequence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "name", "gen", "kind", "parents", "utci_heat", "utci_cold", "f1", "f2", "ops"])
        for n in run.order:
            nd = run.nodes[n]
            w.writerow([nd.idx + 1, n, nd.gen, nd.kind, "+".join(nd.parents) if not nd.seeded else "",
                        nd.heat, nd.cold, nd.f1, nd.f2, nd.ops])

    render_family_tree(run, out / "family_tree.png", out / "family_tree.svg")
    print(f"  wrote {out / 'family_tree.png'}")
    if a.no_frames:
        return 0

    maps = None
    if not a.no_maps:
        if a.runs_dir:
            runs = Path(a.runs_dir)
        else:                                                  # <case>/input, <case>/output, <case>/run
            cands = [Path(os.path.abspath(Path(a.input_dir) / ".." / "run")), Path.cwd() / "run"]
            runs = next((c for c in cands if c.is_dir()), cands[0])
        if runs.is_dir():
            maps = dict(runs_dir=runs, pattern=a.map_pattern, crop=tuple(a.map_crop))
            print(f"  UTCI images from {runs}")
        else:
            print(f"  ! no run folder at {runs} - frames are made without UTCI images (pass --runs-dir)",
                  file=sys.stderr)

    frames, durs, missing = [], [], []
    for k, name in enumerate(run.order):
        p = out / "frames" / f"{k:03d}_{name}.png"
        got = render_frame(run, k, p, W, H, maps=maps)
        missing += [(name, c) for c, okc in got.items() if not okc]
        frames.append(p), durs.append(a.seconds)
        print(f"  frame {k + 1}/{run.n}  {name}", end="\r", flush=True)
    if missing:
        print(f"\n  ! {len(missing)} UTCI images missing or unreadable: "
              + ", ".join(f"{n}/{c}" for n, c in missing[:12]) + (" ..." if len(missing) > 12 else ""), file=sys.stderr)
    if not a.no_summary:
        p = out / "frames" / "999_summary.png"
        render_summary(run, p, W, H)
        frames.append(p), durs.append(a.final_seconds)
    print(f"\n  wrote {len(frames)} frames to {out / 'frames'}")

    if not a.no_video:
        mp4 = out / "evolution.mp4"
        if make_video(frames, durs, mp4, a.fps, a.fade, (W, H)):
            print(f"  wrote {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())