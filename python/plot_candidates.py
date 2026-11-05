#!/usr/bin/env python3
"""
plot_candidates.py  --  pictures that compare the BASE layout with individual_X / offspring_X candidates.

    python plot_candidates.py                                  # every individual_*/offspring_* in input/  -> output/<name>.png
    python plot_candidates.py input/individual_3.json          # one file (or several)
    python plot_candidates.py --diff input/individual_3.json   # 3 panels: base | candidate | what changed
    python plot_candidates.py --grid                           # one contact sheet: base + all candidates -> output/overview.png
    python plot_candidates.py --grid --cols 4 input/individual_*.json
    python plot_candidates.py --objectives output/objectives.csv ...   # put f1 / f2 (UTCI) in the titles

Panels
    base | candidate       buildings (dashed = original position, arrow = translation, label = shift in m),
                           lawns, ponds, pathways, OSM context, trees (new trees get a dark outline)
    --diff                 only what changed: building moves, lawn / pond gained (solid) and lost (hatched),
                           trees added (+), removed (x) and moved (line)
    --grid                 candidate-only thumbnails for a quick look at a whole population

Options: --base FILE  --constraints FILE  --input-dir DIR  --output-dir DIR  --dpi N  --no-labels
Also available as   python generate_candidates.py plot FILE   and   individual|offspring ... --plot
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout_ops as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

C = dict(site="#f4f4f0", osm="#bdbdbd", road="#6b6b6b", lawn="#b7e3a0", lawn_e="#4a8a2a", pond="#8ec5f0",
         pond_e="#1f6fb0", bld="#d98a5f", tree="#2e7d32", tree_e="#1b5e20", new="#0b3d0b", lost="#c0392b",
         move="#1a1a1a")


# ----------------------------------------------------------------------------- helpers
def _geoms(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [x for p in getattr(g, "geoms", []) for x in _geoms(p)]


def _fill(ax, g, **kw):
    for p in _geoms(g):
        ax.fill(*p.exterior.xy, **kw)
        for h in p.interiors:                       # carve holes so differences draw correctly
            ax.fill(*h.xy, fc=ax.get_facecolor(), ec="none", zorder=kw.get("zorder", 1) + 0.01)


def _outline(ax, g, **kw):
    for p in _geoms(g):
        ax.plot(*p.exterior.xy, **kw)


def load_objectives(path):
    out = {}
    if path and Path(path).exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    out[r["name"]] = (float(r["f1"]), float(r["f2"]), float(r.get("utci_heat", "nan")), float(r.get("utci_cold", "nan")))
                except (KeyError, ValueError):
                    pass
    return out


def label_of(sc, scb, name, obj):
    s = f"{name}   trees {len(sc.tx)} | ponds {len(sc.ponds)} ({sum(p.area for p in sc.ponds):,.0f} m²) | lawns {len(sc.lawns)} ({sum(p.area for p in sc.lawns):,.0f} m²)"
    o = obj.get(name) or (obj.get("base") if name == "base" else None) or (obj.get("individual_0") or obj.get("generation_0") if name == "base" else None)
    if o:
        s += f"\nUTCI heat {o[2]:.1f} °C  cold {o[3]:.1f} °C   (f1 {o[0]:.2f}, f2 {o[1]:.2f})"
    return s


def building_moves(sc, scb):
    """[(name, from_xy, to_xy, shift_m)] for every physical building."""
    out = []
    base_by_ids = {ids: i for i, ids in enumerate(scb.group_ids)}
    for gi, ids in enumerate(sc.group_ids):
        bi = base_by_ids.get(ids)
        if bi is None:
            continue
        a, b = scb.centroid(bi), sc.centroid(gi)
        out.append((sc.group_names[gi].replace("Building ", "B"), a, b, float(np.hypot(*(b - a)))))
    return out


def tree_changes(sc, scb):
    bi = {t: i for i, t in enumerate(scb.tree_ids)}
    ci = {t: i for i, t in enumerate(sc.tree_ids)}
    added = [i for t, i in ci.items() if t not in bi]
    removed = [i for t, i in bi.items() if t not in ci]
    moved = [(bi[t], ci[t]) for t in ci if t in bi and np.hypot(sc.tx[ci[t]] - scb.tx[bi[t]], sc.ty[ci[t]] - scb.ty[bi[t]]) > 1e-6]
    return added, removed, moved


def setup_axes(ax, scb, margin=0.06):
    minx, miny, maxx, maxy = scb.site.bounds
    mx, my = (maxx - minx) * margin, (maxy - miny) * margin
    ax.set_xlim(minx - mx, maxx + mx)
    ax.set_ylim(miny - my, maxy + my)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_facecolor("white")


def draw_scene(ax, sc, scb, ghost=False, moves=None, labels=True, mark_new_trees=True):
    setup_axes(ax, scb)
    ax.fill(*sc.site.exterior.xy, fc=C["site"], ec="k", lw=1, zorder=0)
    for p in sc.osm_polys:
        ax.fill(*p.exterior.xy, fc=C["osm"], ec="none", zorder=1)
    if sc.pathway is not None:
        _fill(ax, sc.pathway, fc=C["road"], ec="none", alpha=.55, zorder=1.5)
    for p in sc.lawns:
        ax.fill(*p.exterior.xy, fc=C["lawn"], ec=C["lawn_e"], alpha=.85, lw=.8, zorder=2)
    for p in sc.ponds:
        ax.fill(*p.exterior.xy, fc=C["pond"], ec=C["pond_e"], lw=.8, zorder=3)
    for gp in sc.group_polys:
        ax.fill(*gp.exterior.xy, fc=C["bld"], ec="k", alpha=.92, lw=.8, zorder=4)
    base_ids = set(scb.tree_ids)
    for x, y, r, tid in zip(sc.tx, sc.ty, sc.tR, sc.tree_ids):
        new = mark_new_trees and tid not in base_ids
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(
            (x, y), r, fc=C["tree"], ec=C["new"] if new else C["tree_e"], lw=1.3 if new else .5, alpha=.5, zorder=5))
    if ghost:
        for gp in scb.group_polys:
            ax.plot(*gp.exterior.xy, color="k", ls="--", lw=.7, zorder=6)
    if moves:
        for name, a, b, d in moves:
            if d > 1e-6:
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=C["move"], lw=1.1), zorder=7)
            if labels:
                ax.text(b[0], b[1], f"{name}\n{d:.0f} m" if d > 1e-6 else name, ha="center", va="center", fontsize=6.5,
                        zorder=8, color="k", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.7))


def draw_diff(ax, sc, scb, moves, labels=True):
    setup_axes(ax, scb)
    ax.fill(*sc.site.exterior.xy, fc=C["site"], ec="k", lw=1, zorder=0)
    for p in sc.osm_polys:
        ax.fill(*p.exterior.xy, fc="#e2e2e2", ec="none", zorder=1)
    if sc.pathway is not None:
        _fill(ax, sc.pathway, fc=C["road"], ec="none", alpha=.25, zorder=1.5)
    from shapely.ops import unary_union
    for cat, col, ece in (("lawns", C["lawn"], C["lawn_e"]), ("ponds", C["pond"], C["pond_e"])):
        new, old = unary_union(getattr(sc, cat)), unary_union(getattr(scb, cat))
        gained, lost, kept = new.difference(old), old.difference(new), new.intersection(old)
        _fill(ax, kept, fc=col, ec="none", alpha=.35, zorder=2)
        _fill(ax, gained, fc=col, ec=ece, lw=.8, alpha=.95, zorder=3)
        for p in _geoms(lost):
            ax.fill(*p.exterior.xy, fc="none", ec=C["lost"], hatch="////", lw=.0, zorder=3)
            ax.plot(*p.exterior.xy, color=C["lost"], lw=.6, zorder=3)
    for gp in scb.group_polys:
        ax.plot(*gp.exterior.xy, color="k", ls="--", lw=.7, zorder=5)
    for gp in sc.group_polys:
        ax.fill(*gp.exterior.xy, fc=C["bld"], ec="k", alpha=.9, lw=.8, zorder=4)
    for name, a, b, d in moves:
        if d > 1e-6:
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=C["move"], lw=1.1), zorder=7)
            if labels:
                ax.text(b[0], b[1], f"{name}\n{d:.0f} m", ha="center", va="center", fontsize=6.5, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.7))
    for x, y, r in zip(scb.tx, scb.ty, scb.tR):                     # context: all base trees, faint
        ax.plot(x, y, "o", ms=1.5, color=C["tree"], alpha=.25, zorder=5)
    added, removed, moved = tree_changes(sc, scb)
    for i in added:
        ax.plot(sc.tx[i], sc.ty[i], marker="P", ms=6, mfc=C["tree"], mec=C["new"], mew=.8, ls="", zorder=9)
    for i in removed:
        ax.plot(scb.tx[i], scb.ty[i], marker="x", ms=6, color=C["lost"], mew=1.4, ls="", zorder=9)
    for bi, ci in moved:
        ax.plot([scb.tx[bi], sc.tx[ci]], [scb.ty[bi], sc.ty[ci]], color=C["tree_e"], lw=1, zorder=8)
        ax.plot(sc.tx[ci], sc.ty[ci], "o", ms=3, mfc=C["tree"], mec=C["tree_e"], zorder=9)
    return len(added), len(removed), len(moved)


def legend(fig, diff=False, y=0.005):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(fc=C["bld"], ec="k", label="building (moved)"), Line2D([], [], color="k", ls="--", lw=.8, label="building, original position"),
         Patch(fc=C["lawn"], ec=C["lawn_e"], label="lawn"), Patch(fc=C["pond"], ec=C["pond_e"], label="pond"),
         Patch(fc=C["road"], alpha=.55, label="pathway"), Patch(fc=C["osm"], label="OSM context")]
    if diff:
        h += [Patch(fc="none", ec=C["lost"], hatch="////", label="lawn/pond removed"),
              Line2D([], [], marker="P", ls="", mfc=C["tree"], mec=C["new"], label="tree added"),
              Line2D([], [], marker="x", ls="", color=C["lost"], label="tree removed"),
              Line2D([], [], color=C["tree_e"], lw=1, label="tree moved")]
    else:
        h += [Patch(fc=C["tree"], ec=C["tree_e"], alpha=.5, label="tree"), Patch(fc=C["tree"], ec=C["new"], alpha=.5, label="tree (new, dark outline)")]
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 6), fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, y))


# ----------------------------------------------------------------------------- figures
def figure_pair(base, cand, name, obj, diff=False, labels=True):
    import matplotlib.pyplot as plt
    n = 3 if diff else 2
    fig, axs = plt.subplots(1, n, figsize=(7.2 * n, 5.4))
    moves = building_moves(cand, base)
    draw_scene(axs[0], base, base, labels=False)
    axs[0].set_title(label_of(base, base, "base", obj), fontsize=9)
    draw_scene(axs[1], cand, base, ghost=True, moves=moves, labels=labels)
    axs[1].set_title(label_of(cand, base, name, obj), fontsize=9)
    if diff:
        na, nr, nm = draw_diff(axs[2], cand, base, moves, labels)
        shifts = [m[3] for m in moves]
        axs[2].set_title(f"changes   buildings moved {sum(s > 1e-6 for s in shifts)}/{len(shifts)} "
                         f"(max {max(shifts, default=0):.0f} m) | trees +{na} −{nr} ↔{nm}", fontsize=9)
    legend(fig, diff)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def figure_grid(base, cands, obj, cols=3, labels=False):
    import matplotlib.pyplot as plt
    items = [("base", base)] + cands
    rows = int(np.ceil(len(items) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.4 * rows), squeeze=False)
    for k, ax in enumerate(axs.ravel()):
        if k >= len(items):
            ax.axis("off")
            continue
        name, sc = items[k]
        if name == "base":
            draw_scene(ax, sc, base, labels=False)
        else:
            draw_scene(ax, sc, base, ghost=True, moves=building_moves(sc, base), labels=labels)
        ax.set_title(label_of(sc, base, name, obj), fontsize=8)
    legend(fig, False, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ----------------------------------------------------------------------------- API used by generate_candidates
def plot_files(files, base_path, cfg, out_dir, objectives=None, diff=False, labels=True, dpi=110, quiet=False):
    """Write one PNG per file (base | candidate [| changes]).  Returns the list of PNG paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json
    base = L.Scene(json.loads(Path(base_path).read_text()), cfg, split_hosted=True)
    obj = load_objectives(objectives) if objectives else {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        cand = L.Scene(json.loads(Path(f).read_text()), cfg, base.osm_polys)
        fig = figure_pair(base, cand, Path(f).stem, obj, diff, labels)
        p = out_dir / f"{Path(f).stem}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        out.append(p)
        if not quiet:
            print("wrote", p)
    return out


def plot_grid(files, base_path, cfg, out_dir, objectives=None, cols=3, labels=False, dpi=100, quiet=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json
    base = L.Scene(json.loads(Path(base_path).read_text()), cfg, split_hosted=True)
    cands = [(Path(f).stem, L.Scene(json.loads(Path(f).read_text()), cfg, base.osm_polys)) for f in files]
    fig = figure_grid(base, cands, load_objectives(objectives) if objectives else {}, cols, labels)
    p = Path(out_dir) / "overview.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    if not quiet:
        print("wrote", p)
    return p


def _sort_key(p):
    import re
    m = re.match(r"([a-z]+)_(\d+)", p.stem)
    return (m.group(1), int(m.group(2))) if m else (p.stem, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="candidate JSONs (default: individual_*.json and offspring_*.json in --input-dir)")
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--base", help="base layout (default <input-dir>/base.json)")
    ap.add_argument("--constraints", help="constraints JSON (default <input-dir>/constraints.json if present)")
    ap.add_argument("--objectives", help="objectives.csv from run_candidates.py (default <output-dir>/objectives.csv if present)")
    ap.add_argument("--diff", action="store_true", help="add a third panel that shows only the changes")
    ap.add_argument("--grid", action="store_true", help="one overview.png (base + all candidates) instead of one PNG per file")
    ap.add_argument("--cols", type=int, default=3, help="columns of the --grid sheet")
    ap.add_argument("--no-labels", action="store_true", help="no building labels / shift distances")
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args()

    in_dir = Path(a.input_dir) if Path(a.input_dir).is_absolute() else ROOT / a.input_dir
    out_dir = Path(a.output_dir) if Path(a.output_dir).is_absolute() else ROOT / a.output_dir
    cfg = L.load_constraints(a.constraints or in_dir / "constraints.json")
    base_path = Path(a.base) if a.base else in_dir / "base.json"
    files = [Path(f) for f in a.files] or sorted([f for pat in ("individual_*.json", "offspring_*.json", "generation_*.json", "mutation_*.json") for f in in_dir.glob(pat)], key=_sort_key)
    if not files:
        sys.exit(f"no candidate files in {in_dir}")
    objs = a.objectives or (str(out_dir / "objectives.csv") if (out_dir / "objectives.csv").exists() else None)
    if a.grid:
        plot_grid(files, base_path, cfg, out_dir, objs, a.cols, labels=not a.no_labels and len(files) <= 4, dpi=a.dpi)
    else:
        plot_files(files, base_path, cfg, out_dir, objs, a.diff, not a.no_labels, a.dpi)


if __name__ == "__main__":
    main()
