#!/usr/bin/env python3
"""
layout_templates.py  --  seed individuals built from the TREE WIND CATCHER / TREE WIND BLOCKER templates.

The templates (jsons/tree_wind_catcher.json, jsons/tree_wind_breaker.json) are drawn for WIND FROM THE WEST
(the air flows towards +x).  They are patterns, not fixed layouts: every tree can be rescaled, thinned out
or dropped so that the individual obeys every Stage 1 rule (inside the site, clear of buildings / ponds /
pathways, canopy gap, crown radius, height band of its location, tree-count limit).

    kind           template                 wind used                         what it does
    wind_catcher   tree_wind_catcher.json   HEAT case (peak-heat hour)        funnel of two converging double rows, wide mouth facing the wind
    wind_blocker   tree_wind_breaker.json   COLD case (peak-cold hour)        wall of staggered rows across the wind
    cool_wind_catcher  cool_wind_catcher.json  HEAT case                      pond + curved tree rows beside a REFERENCE BUILDING that lead the
                                                                              (cooled) air into that building's wake region (make_cool_catcher_individual)

How an individual is made (make_template_individual):
  1. wind      the flow vector of the season (jsons/<epw>_cases.json, boundary.wind_flow_unit_vector_xy_model,
               model axes).  The template is rotated so that its +x axis points along this vector.
  2. layout    base layout with the rules enforced (lawn / pond totals fitted, trees that break a rule moved);
               buildings are NOT moved.  The base trees stay.
  3. place     the rotated pattern is slid to the UPWIND edge of the site (the inlet side).  For every position
               on a 2 m grid (distance from the upwind edge, cross-wind offset) the trees that satisfy all
               rules are counted; the position closest to the inlet that keeps (almost) the best tree count wins;
               among equals the one nearest the middle of the buildings.
  4. size      the whole pattern (spacing AND crown radius) is scaled 1.0, 5/6, 2/3, 1/2 of the template until
               at least 90 % of its trees fit; trees that cannot stand there are dropped, the number of trees is
               thinned evenly if the tree-count limit would be exceeded.  Heights follow the height band of
               each location (template height where allowed, otherwise the nearest allowed height).
  5. check     the result must have no new rule violation (like every other individual).

A season that does not exist in the climate (heat_case / cold_case = null in the cases file) gives no
individual of that kind.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

import layout_ops as L

# template frame: wind blows towards +x ("wind from the west"); +y is to the LEFT of the flow
TEMPLATE_FLOW = (1.0, 0.0)

KINDS = {
    # ---- the default seeds (see layout_seeds.py)
    "cool_wind_catcher_all": dict(season="heat", label="cool wind catcher (every building)", builder="cool_all", default=True,
                                  files=("cool_wind_catcher.json",)),
    "wind_blocker_edge": dict(season="cold", label="wind blocker (windward edge belt)", builder="edge", default=True,
                              files=("tree_wind_breaker.json", "tree_wind_blocker.json")),
    # ---- earlier variants, only made when asked for (--template-kinds / template --kind)
    "wind_catcher": dict(season="heat", label="tree wind catcher", default=False,
                         files=("tree_wind_catcher.json",)),
    "wind_blocker": dict(season="cold", label="tree wind blocker (wall)", default=False,
                         files=("tree_wind_breaker.json", "tree_wind_blocker.json")),
    "cool_wind_catcher": dict(season="heat", label="cool wind catcher (one reference building)", builder="cool", default=False,
                              files=("cool_wind_catcher.json",)),
}
OPTION = {"wind_catcher": "catcher", "wind_blocker": "blocker", "cool_wind_catcher": "cool-catcher",
          "cool_wind_catcher_all": "cool-catcher", "wind_blocker_edge": "blocker"}   # --<option>-template
ARG_DEST = {"cool_wind_catcher_all": "cool_wind_catcher_template", "wind_blocker_edge": "wind_blocker_template"}   # nsga2 option (dest) for the kinds that share a template file
DEFAULT_KINDS = tuple(k for k, v in KINDS.items() if v.get("default"))


def kind_list(spec):
    """'all' / 'default' / comma separated kinds -> list of kinds (unknown names raise ValueError)."""
    if spec in (None, "", "default", "both"):
        return list(DEFAULT_KINDS)
    if spec == "all":
        return list(KINDS)
    out = [k.strip() for k in str(spec).split(",") if k.strip()]
    bad = [k for k in out if k not in KINDS]
    if bad:
        raise ValueError(f"unknown template kind(s) {bad}; known: {', '.join(KINDS)}")
    return out


SCALES_OF_RADIUS = (1.0, 5.0 / 6.0, 2.0 / 3.0, 0.5, 0.4)     # relative crown radius (and spacing) tried in this order
MIN_KEEP = 0.60                                          # share of the (thinned) template's trees that must fit
GRID_M = 2.0
INLET_ZONE_FRAC = 0.25                                   # "close to the inlet": within this share of the site length along the wind
INLET_ZONE_MIN_M = 15.0                                  # ... but never stricter than this many metres
INLET_SLACK_M = 4.0
KEEP_FRACTIONS = (1.0, 0.75, 0.5)                        # share of the template's trees tried, per scale
BASE_TREES_REPLACEABLE = 0.5                             # at most this share of the base trees may give way
EPS = 0.02                                               # safety margin on every distance rule (m)


# ----------------------------------------------------------------------------- inputs
def find_template(kind, jsons_dir, explicit=None):
    """Path of the template file of `kind` (explicit path, or the usual names inside jsons/), or None."""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for n in KINDS[kind]["files"]:
        p = Path(jsons_dir) / n
        if p.is_file():
            return p
    return None


def read_template(path):
    """Trees of a template as (tree record, a, b): a = distance behind the upwind edge of the crowns,
    b = cross-flow offset from the middle of the pattern (template frame, flow towards +x)."""
    d = json.loads(Path(path).read_text())
    site = d["siteOnly_json"] if "siteOnly_json" in d else d
    trees = list(site.get("trees", []))
    if not trees:
        raise ValueError(f"{path}: no trees in the template")
    xs = np.array([t["position"]["x"] for t in trees], float)
    ys = np.array([t["position"]["y"] for t in trees], float)
    Rs = np.array([L._tree_R(t) for t in trees], float)
    a = xs - (xs - Rs).min()
    b = ys - 0.5 * (ys.min() + ys.max())
    return dict(trees=trees, a=a, b=b, R=Rs, path=str(path))


def season_wind(cases, season):
    """Wind of the heat / cold case in the MODEL frame: dict(flow=(fx, fy), from_deg, speed, label), or None
    when the climate has no such case (or the wind is undefined)."""
    case = (cases or {}).get(f"{season}_case")
    if not case:
        return None
    b = case.get("boundary", {})
    f = b.get("wind_flow_unit_vector_xy_model")
    ok = f is not None and len(f) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in f)
    if not ok:
        wd = b.get("wind_from_deg_model")
        if wd is None or not math.isfinite(float(wd)):
            return None
        f = [-math.sin(math.radians(wd)), -math.cos(math.radians(wd))]
    n = math.hypot(f[0], f[1])
    if n < 1e-9:
        return None
    f = (f[0] / n, f[1] / n)
    return dict(flow=f, from_deg=(math.degrees(math.atan2(-f[0], -f[1])) % 360.0),
                speed=b.get("wind_speed_10m_ms"), label=case.get("timestamp", {}).get("label", ""),
                status=(cases or {}).get(f"{season}_status"))


# ----------------------------------------------------------------------------- geometry
def _free_region(sc, cfg, r, ponds=None):
    """Where the centre of a tree of crown radius r may stand: exactly the distance rules of layout_ops.check
    (site margin, building clearance, ponds, pathways); the canopy gap is tested separately."""
    tc = cfg["tree"]
    kf = tc["building_clearance_factor"]
    extra = tc["building_clearance_extra_frac_of_width"] * sc.min_width
    g = sc.site.buffer(-(r + EPS))
    if sc.group_polys:
        g = g.difference(unary_union(sc.group_polys).buffer(kf * r + extra + EPS))
    if sc.osm_polys and tc["clearance_applies_to_osm"]:
        g = g.difference(unary_union(sc.osm_polys).buffer(kf * r + EPS))
    pl = sc.ponds if ponds is None else ponds
    if pl:
        g = g.difference(unary_union(pl).buffer(r + EPS))
    if sc.pathway is not None and not tc["tree_on_pathway_allowed"]:
        g = g.difference(sc.pathway.buffer(r + EPS))
    shapely.prepare(g)
    return g


def _frame(flow):
    f = np.array(flow, float)
    return f, np.array([-f[1], f[0]])                      # n = left of the flow (template +y)


def _positions(u0, du, vc, A, B, f, n):
    u = u0 + du + A
    v = vc + B
    return u[:, None] * f[None, :] + v[:, None] * n[None, :]


def _search(sc, cfg, tpl, idx, s, flow, free, v_pref, max_total, budget, min_keep):
    """Best (du, vc) of the pattern (trees `idx`, scale s) -> dict or None.  Everything is vectorised.
    Base trees are NOT obstacles: a base tree whose canopy would touch a template tree is replaced
    (removed); the number of removed base trees (those plus the ones removed to stay under the tree-count
    limit) must not exceed `budget`."""
    # free: cache {crown radius: shapely region}, filled on demand
    tc = cfg["tree"]
    lo_r, hi_r = tc["radius_range_m"]
    Rt = np.clip(tpl["R"][idx] * s, lo_r, hi_r)
    A, B = tpl["a"][idx] * s, tpl["b"][idx] * s
    f, n = _frame(flow)
    sv = np.array(L._xy(sc.d["site"]["coordinates"]), float)
    su, sv_ = sv @ f, sv @ n
    u0 = float(su.min())
    depth = float(A.max())
    du_max = max(0.0, float(su.max() - u0) - depth)
    dus = np.arange(0.0, du_max + 1e-9, GRID_M)
    half = 0.5 * float(sv_.max() - sv_.min())
    vmid = 0.5 * float(sv_.max() + sv_.min())
    vcs = np.arange(vmid - half, vmid + half + 1e-9, GRID_M)
    ex, ey, eR = sc.tx, sc.ty, sc.tR
    nb = len(ex)
    groups = []
    for rv in np.unique(np.round(Rt, 3)):
        groups.append((rv, np.where(np.isclose(Rt, rv, atol=1e-3))[0]))
        if rv not in free:
            free[rv] = _free_region(sc, cfg, float(rv))
    gap = tc["canopy_gap_factor"]
    best = []
    for du in dus:
        for vc in vcs:
            P = _positions(u0, du, vc, A, B, f, n)
            ok = np.zeros(len(P), bool)
            for key, sel in groups:
                ok[sel] = shapely.contains_xy(free[key], P[sel, 0], P[sel, 1])
            k = int(ok.sum())
            if not k:
                continue
            removed = 0
            if nb:
                d = np.hypot(P[ok, None, 0] - ex[None, :], P[ok, None, 1] - ey[None, :])
                conf = (d < gap * (Rt[ok, None] + eR[None, :]) + EPS).any(axis=0)
                removed = int(conf.sum())
                removed += max(0, nb - removed + k - max_total)
            elif k > max_total:
                removed = k - max_total
            if removed > budget:
                continue
            best.append((k - 0.25 * removed, k, float(du), abs(float(vc) - v_pref), float(vc), ok, removed))
    if not best:
        return None
    need = max(3, math.ceil(min_keep * len(idx)))
    full = [b for b in best if b[1] >= need]
    if full:       # closest to the inlet (a few metres of slack are traded for more trees), then the middle
        d0 = min(b[2] for b in full)
        near = [b for b in full if b[2] <= d0 + INLET_SLACK_M]
        sc_, k, du, dv, vc, ok, removed = min(near, key=lambda b: (-b[0], b[2], b[3]))
    else:          # the pattern is too big for this scale: the most trees, then closest to the inlet
        sc_, k, du, dv, vc, ok, removed = min(best, key=lambda b: (-b[0], b[2], b[3]))
    return dict(kept=k, du=du, vc=vc, mask=ok, u0=u0, s=s, f=f, n=n, A=A, B=B, Rt=Rt, removed=removed,
                full=bool(full), du_max=du_max)


def _thin(m, total):
    """Indices of m trees spread evenly over a pattern of `total` trees."""
    if m >= total:
        return np.arange(total)
    return np.unique(np.round(np.linspace(0, total - 1, m)).astype(int))


def _sync_vegetation(t, R, h):
    v = t.get("materials", {}).get("vegetation")
    if isinstance(v, dict):
        if "height" in v:
            v["height"] = float(h)
        if "crown_diameter" in v:
            v["crown_diameter"] = float(2 * R)


def load_template(kind, path):
    """Template of any kind, ready for make_individual."""
    if KINDS[kind].get("builder") in ("cool", "cool_all"):
        return read_cool_template(path)
    return read_template(path)


SEED_RETRIES = 5


def make_individual(base_data, base, cfg, base_map, kind, tpl, wind, seed=0, ref_building=None):
    """(layout, ops), or (None, reason), for any template kind."""
    b = KINDS[kind].get("builder")
    if b in ("cool_all", "edge"):
        import layout_seeds as SD
        mk = SD.make_cool_all_individual if b == "cool_all" else SD.make_edge_blocker_individual
        # the random part (pond / lawn fitting) fails now and then for one seed: try a few other internal seeds, same layout idea
        why = None
        for k in range(SEED_RETRIES):
            try:
                res, why = mk(base_data, base, cfg, base_map, tpl, wind, seed=seed + 7919 * k)
            except ValueError as e:
                res, why = None, str(e)
            if res is not None:
                if k:
                    res[1][0]["internal_seed_retry"] = k
                return res, why
        return None, why
    if b == "cool":
        return make_cool_catcher_individual(base_data, base, cfg, base_map, tpl, wind, seed=seed,
                                            ref_building=ref_building)
    return make_template_individual(base_data, base, cfg, base_map, kind, tpl, wind, seed=seed)


def make_template_individual(base_data, base, cfg, base_map, kind, tpl, wind, scales=SCALES_OF_RADIUS,
                             min_keep=MIN_KEEP, seed=0):
    """(layout dict, ops list) for one template individual, or (None, reason).

    Order of work (so that the pattern gets the inlet side and the ponds / lawns adapt to it, not the other way
    round): search the pattern position -> reserve its crowns -> fit the pond / lawn totals around them ->
    replace the base trees that give way -> plant the template trees -> move base trees that break a rule ->
    final rule check."""
    import layout_reflow as RF
    spec = KINDS[kind]
    rng = np.random.default_rng(seed)
    data = L.working_copy(base_data)
    todo = L.patches_out_of_band(data, base, cfg)                  # patches that fit_patch_area will rebuild
    sdata = dict(data, ponds=[]) if "pond" in todo else data       # a rebuilt pond is not an obstacle (yet)
    sc = L.Scene(sdata, cfg, base.osm_polys)
    tc = cfg["tree"]
    max_total = L.max_tree_count(cfg, L.tree_count0(base, cfg), sc.pathway is not None)
    nb = len(sc.tx)
    budget = int(math.floor(BASE_TREES_REPLACEABLE * nb))        # base trees that may give way to the pattern
    total = len(tpl["trees"])
    f, _ = _frame(wind["flow"])
    # middle of the buildings, across the flow: the pattern is centred there when it can be
    n_vec = np.array([-f[1], f[0]])
    cents = [np.array(p.centroid.coords[0]) for p in sc.group_polys]
    v_pref = float(np.mean([c @ n_vec for c in cents])) if cents else float(
        np.array(sc.site.centroid.coords[0]) @ n_vec)
    Rmean = float(tpl["R"].mean())
    chosen, tried, free = None, [], {}
    for rel in scales:
        Rt = Rmean * rel
        if Rt < tc["radius_range_m"][0] - 1e-9 or Rt > tc["radius_range_m"][1] + 1e-9:
            continue
        for frac in KEEP_FRACTIONS:
            idx = _thin(int(math.ceil(frac * total)), total)
            res = _search(sc, cfg, tpl, idx, rel, wind["flow"], free, v_pref, max_total, budget, min_keep)
            tried.append((rel, frac, 0 if res is None else res["kept"], None if res is None else res["du"]))
            if res is None or not res["full"]:
                continue
            res["idx"] = idx
            zone = max(INLET_ZONE_MIN_M, INLET_ZONE_FRAC * (res["du_max"] + float(res["A"].max())))
            res["in_zone"] = res["du"] <= zone
            # in the inlet zone the pattern with the largest tree-count x crown-size wins; outside it the closest
            rank = (res["in_zone"], res["kept"] * rel if res["in_zone"] else -res["du"])
            if chosen is None or rank > chosen["_rank"]:
                chosen = dict(res, _rank=rank)
    if chosen is None:
        return None, ("the pattern does not fit the site (tried scale/thinning: "
                      + ", ".join(f"{r:.2f}/{fr:.2f}: {k} trees" for r, fr, k, _ in tried) + ")")
    idx_all = chosen["idx"]
    P = _positions(chosen["u0"], chosen["du"], chosen["vc"], chosen["A"], chosen["B"], chosen["f"], chosen["n"])
    m = chosen["mask"]
    Pm, Rm = P[m], chosen["Rt"][m]

    # 1. ponds / lawns: totals fitted around the reserved crowns
    notes = {}
    reserve = [shapely.Point(float(x), float(y)).buffer(float(r) + EPS, 16) for (x, y), r in zip(Pm, Rm)]
    fitted = L.fit_patch_area(data, base, cfg, rng, pond_blockers=reserve)
    if fitted:
        notes["area_rescaled"] = fitted

    # 2. base trees that give way: those touching a template tree, then (if the limit needs it) the nearest ones
    gone = np.zeros(nb, bool)
    if nb:
        d = np.hypot(Pm[:, None, 0] - sc.tx[None, :], Pm[:, None, 1] - sc.ty[None, :])
        gone = (d < tc["canopy_gap_factor"] * (Rm[:, None] + sc.tR[None, :]) + EPS).any(axis=0)
        over = nb - int(gone.sum()) + int(m.sum()) - max_total
        if over > 0:
            for i in np.argsort(d.min(axis=0)):
                if over <= 0:
                    break
                if not gone[i]:
                    gone[i] = True
                    over -= 1
    removed_ids = {sc.tree_ids[i] for i in np.where(gone)[0]}
    data["trees"] = [t for t in data["trees"] if t["id"] not in removed_ids]

    # 3. the template trees (heights follow the height band of their location)
    ball = sc.ball_union
    nid = max([int(str(t["id"]).split()[-1]) for t in base_data["trees"] if str(t["id"]).split()[-1].isdigit()] + [0])
    added = []
    for j, ti in enumerate(idx_all):
        if not chosen["mask"][j]:
            continue
        t = copy.deepcopy(tpl["trees"][ti])
        x, y = float(P[j, 0]), float(P[j, 1])
        R = float(chosen["Rt"][j])
        nid += 1
        t["id"] = f"Tree {nid}"
        t["position"] = {"x": x, "y": y, "z": t.get("position", {}).get("z", 0)}
        d0 = float(shapely.distance(shapely.points(x, y), ball)) if ball is not None else math.inf
        lo, hi = L.height_band(cfg, d0, L.on_pathway(cfg, sc.pathway, x, y))
        h = float(np.clip(float(t.get("height", 12.0)), lo, hi))
        L.set_tree_geometry(t, R, h, tc)
        _sync_vegetation(t, R, h)
        added.append(t)
    data["trees"] = list(data["trees"]) + added
    mine = {t["id"] for t in added}

    # 4. base trees that the new ponds / lawns put in breach of a rule are moved (as for every individual)
    moved = RF.fix_trees(data, base, cfg, rng)
    if moved:
        notes["trees_moved_off_rule_breaks"] = len([i for i in moved if i not in mine])
    L.finalize_patches(data, cfg)
    L.refresh_metadata(data)

    # 5. final check: nothing new may be broken; a template tree that is, is dropped (at most 6 rounds)
    for _ in range(6):
        notes["pathway_shade"] = RF.ensure_pathway_shade(data, base, cfg, rng, protect=mine)     # road shade: mandatory
        fsc = L.Scene(data, cfg, base.osm_polys)
        nv = L.new_violations(fsc, base, base_map)
        if not nv:
            break
        drop = set()
        for typ, ids, _sev in nv:
            mm = [i for i in ids if i in mine and i not in drop]
            drop.update(mm[-1:] if typ == "tree_gap" else mm)       # two template trees too close: drop one
        if not drop:
            return None, "the layout has new rule violations that are not template trees: " + ", ".join(
                sorted({t for t, _, _ in nv}))
        data["trees"] = [t for t in data["trees"] if t["id"] not in drop]
        mine -= drop
        L.refresh_metadata(data)
    else:
        return None, "final rule check failed"
    added = [t for t in data["trees"] if t["id"] in mine]
    if len(added) < 3:
        return None, "fewer than 3 template trees survive the rules"
    info = {"op": "template", "kind": kind, "template": Path(tpl["path"]).name, "season": spec["season"],
            "wind_from_deg_model": round(wind["from_deg"], 1),
            "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
            "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
            "scale": round(chosen["s"], 3), "crown_radius_m": round(float(chosen["Rt"].mean()), 2),
            "trees_in_template": total, "trees_planned": int(len(idx_all)), "trees_placed": len(added),
            "distance_from_upwind_site_edge_m": round(chosen["du"], 1),
            "close_to_inlet": bool(chosen["in_zone"]),
            "base_trees_kept": len(data["trees"]) - len(added), "base_trees_replaced": len(removed_ids),
            "template_tree_ids": sorted(mine, key=lambda i: int(i.split()[-1])),
            "rules_enforced_on_base": notes}
    return (data, [info]), None


# ============================================================================= cool wind catcher
# Template (jsons/cool_wind_catcher.json): ONE reference building, a pond and curved rows of trees, drawn for wind from
# the west.  The pond lies beside the building (south of it as drawn), the trees wrap round the pond and end at the
# building's wake (east of it).  The wind crosses the pond (cooled, moistened) and the tree rows steer it into the wake.
# The pattern may be mirrored along the west-east axis through the building centre (pond / trees on the other side).
#
# The template is a SHAPE: tall large trees (8 m, R 4) next to the pond, medium trees (6 m, R 3) behind them and a fringe of
# short small trees (3 m, R 1.5) outside.  What is kept: the layered rows, the pond shape and its place at the building.
# What gives: size (one scale for pond and trees), crown radius (a few % smaller so that the 1.1 canopy-gap rule holds - the
# template itself breaks it), the pose (a little rotation / shift to find room), single trees (nudged a few metres out of the
# way of a pond edge, building or the site boundary instead of being dropped) and the heights (see _band_height).
COOL_K = (1.0, 0.85)                                     # tree scale relative to the pond scale (pond first, then smaller trees)
COOL_ROT_DEG = (-20.0, 0.0, 20.0)                        # pose search, coarse ...
COOL_SHIFT_M = (-8.0, 0.0, 8.0)                          # ... along / across the wind
COOL_ROT_FINE = (-6.0, 0.0, 6.0)                         # ... and fine, around the best poses
COOL_SHIFT_FINE = (-3.0, 0.0, 3.0)
COOL_KEEP = (1.0, 0.85, 0.7)                             # share of the template's trees tried if the tree-count limit needs thinning
COOL_MIN_TREES = 0.6                                     # at least this share of the template's trees must stand
COOL_BASE_REPLACEABLE = 0.9                              # share of the base trees that may give way to the (complete) pattern
COOL_CLIP_MAX = 0.2                                      # pond: at most this share of its area may be cut off by buildings / site edge
COOL_NUDGE_M = 5.0                                       # a tree may be moved this far to clear an obstacle
COOL_MAX_BUILDINGS = 6                                   # reference candidates (the largest buildings) when none is named


def read_cool_template(path):
    """Reference building, pond and trees of the cool wind catcher in the template frame (flow towards +x)."""
    d = json.loads(Path(path).read_text())
    site = d["siteOnly_json"] if "siteOnly_json" in d else d
    if not site.get("buildings") or not site.get("ponds") or not site.get("trees"):
        raise ValueError(f"{path}: the cool wind catcher template needs a building, a pond and trees")
    b = site["buildings"][0]
    fp = np.array(L._xy(b["geometry"]["footprint"]), float)
    pond = site["ponds"][0]
    pxy = np.array(L._xy(pond["coordinates"]), float)
    trees = list(site["trees"])
    xy = np.array([[t["position"]["x"], t["position"]["y"]] for t in trees], float)
    R = np.array([L._tree_R(t) for t in trees], float)
    H = np.array([float(t["height"]) for t in trees], float)
    lo, hi = fp.min(0), fp.max(0)
    yc = 0.5 * (lo[1] + hi[1])
    side = -1.0 if np.concatenate([xy[:, 1], pxy[:, 1]]).mean() < yc else 1.0      # which face of the building the pattern lies at
    anchor = np.array([hi[0], lo[1] if side < 0 else hi[1]])                          # downwind face x, pattern-side face y
    D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    need = 1.1 * (R[:, None] + R[None, :])                      # the canopy-gap rule at its default; the real value is applied later
    np.fill_diagonal(D, np.inf)
    ratio = float((D / need).min())                             # < 1: the template crowns are tighter than the rule allows
    hn = (H - H.min()) / max(H.max() - H.min(), 1e-9)           # 0 = shortest tree class, 1 = tallest
    return dict(path=str(path), building=b["id"], bld_lo=lo, bld_hi=hi, centre=(lo + hi) / 2,
                half=(hi - lo) / 2, side=side, anchor=anchor, pond=pond, pond_xy=pxy,
                pond_area=float(Polygon(pxy).area), trees=trees, xy=xy, R=R, H=H, hn=hn, pair_ratio=ratio)


def _building_frame(poly, f, n):
    """Extent of a building in the flow frame: (u_min, u_max, v_min, v_max)."""
    xy = np.array(poly.exterior.coords)
    u, v = xy @ f, xy @ n
    return float(u.min()), float(u.max()), float(v.min()), float(v.max())


def _largest(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type == "Polygon":
        return g
    parts = [q for q in getattr(g, "geoms", []) if q.geom_type == "Polygon"]
    return max(parts, key=lambda q: q.area) if parts else None


def _pose(tpl, f, n, frame, mirror, rot, du, dv):
    """map(xy_template, scale) -> xy_layout.  Rigid, uniformly scaled, hung on the reference building like the template:
    the template building's downwind face maps to the real building's downwind face, and the face the pond / trees lie
    at maps to its face on that side of the wind.  mirror = -1: mirror image along the west-east axis through the building
    centre.  rot (rad) turns the pattern about that corner, (du, dv) shifts it along / across the wind."""
    u_min, u_max, v_min, v_max = frame
    side = tpl["side"]
    out = side * mirror                        # layout: -1 = right of the wind (-n), +1 = left (+n)
    v_face = v_max if out > 0 else v_min
    ax, ay = tpl["anchor"]
    c, sn = math.cos(rot), math.sin(rot)

    def mp(xy, s):
        xy = np.asarray(xy, float)
        au = s * (xy[:, 0] - ax)
        av = out * s * (side * (xy[:, 1] - ay))             # side*(y-ay) = distance away from the building face
        u = u_max + c * au - sn * av + du
        v = v_face + sn * au + c * av + dv
        return u[:, None] * f[None, :] + v[:, None] * n[None, :]
    return mp


def _pond_at(tpl, mp, s0, region, lo, hi):
    """The template pond at scale s (about the building corner, so it keeps its place at the building) with its area inside
    [lo, hi] after clipping to `region`.  Returns (polygon, s, share of area clipped away) or None."""
    target, tol, s = 0.5 * (lo + hi), 0.01 * (hi - lo) + 1e-9, s0
    for _ in range(6):
        g = Polygon(mp(tpl["pond_xy"], s)).buffer(0)
        q = _largest(g.intersection(region))
        if q is None or q.area < 1.0 or g.area <= 0:
            return None
        if lo + tol <= q.area <= hi - tol:
            loss = 1.0 - q.area / g.area
            return (q, s, loss) if loss <= COOL_CLIP_MAX else None
        s *= math.sqrt(target / q.area)
        if not 0.3 < s < 3.0:
            return None
    return None


def _thin(m, total):
    """Indices of m trees spread evenly over a pattern of `total` trees."""
    if m >= total:
        return np.arange(total)
    return np.unique(np.round(np.linspace(0, total - 1, m)).astype(int))


def _nudge(P, Rt, ok, free0, pond, gap, step=0.5):
    """Move every tree that is not ok to the nearest spot within COOL_NUDGE_M that is (clear of the obstacles, the pond and the
    trees already standing).  Largest crowns first.  Returns (positions, ok)."""
    P, ok = P.copy(), ok.copy()
    angs = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    rings = np.arange(step, COOL_NUDGE_M + 1e-9, step)
    O = np.array([[r * math.cos(a), r * math.sin(a)] for r in rings for a in angs])
    standing = list(np.where(ok)[0])
    for i in np.argsort(-Rt):
        if ok[i]:
            continue
        C = P[i] + O
        m = shapely.contains_xy(free0[round(float(Rt[i]), 2)], C[:, 0], C[:, 1])
        if pond is not None:
            m &= shapely.distance(shapely.points(C[:, 0], C[:, 1]), pond) >= Rt[i] + EPS
        if standing and m.any():
            d = np.hypot(C[:, None, 0] - P[standing, 0][None, :], C[:, None, 1] - P[standing, 1][None, :])
            m &= (d >= gap * (Rt[i] + Rt[standing])[None, :] + EPS).all(axis=1)
        if m.any():
            P[i] = C[int(np.argmax(m))]                    # rings are in increasing distance: the first hit is the nearest
            ok[i] = True
            standing.append(i)
    return P, ok


def _band_height(cfg, hn, x, y, ball, pathway):
    """Template height class (0 = shortest ... 1 = tallest) -> a height inside the height band of the location, keeping the
    order: the short class gets the lowest allowed height, the tall class the highest."""
    d0 = float(shapely.distance(shapely.points(x, y), ball)) if ball is not None else math.inf
    lo, hi = L.height_band(cfg, d0, L.on_pathway(cfg, pathway, x, y))
    step = cfg["tree"]["size_step_m"]
    return float(np.clip(L._rt(lo + hn * (hi - lo), step), lo, hi))


def _static_pair(static):
    """static = geometry (same keep-out for ponds and lawns) or (pond keep-out, lawn keep-out)."""
    return static if isinstance(static, tuple) else (static, static)


def _separate_lawns_from_pond(data, cfg, static, step):
    """The pond / lawn reflow can leave a sliver (< 0.1 m2) where a lawn touches the pond or a building; cut the lawns back
    by a hair and re-discretize pond and lawns together so that they do not overlap."""
    static_p, static_l = _static_pair(static)
    pond = L._poly(data["ponds"][0]["coordinates"]).buffer(0)
    lawns = [L._poly(g["coordinates"]).buffer(0) for g in data["grassAreas"]]
    if not any(q.intersection(pond).area > 0 or q.intersection(static_l).area > 0 for q in lawns):
        return
    cut = [_largest(q.difference(pond.buffer(0.15)).difference(static_l.buffer(0.05))) for q in lawns]
    pond_new = L.discretize_patches([pond], step, static_p)[0]
    polys = [pond_new] + L.discretize_patches([q for q in cut if q is not None], step, unary_union([static_l, pond_new]))
    L.patch_set_polygon(data["ponds"][0], polys[0], True, 0)
    kept, it = [], iter(polys[1:])
    for g, q in zip(data["grassAreas"], cut):
        if q is not None and q.area >= cfg["lawn"]["min_area_m2"]:
            L.patch_set_polygon(g, next(it), False, 0)
            kept.append(g)
        elif q is not None:
            next(it)
    data["grassAreas"] = kept


def make_cool_catcher_individual(base_data, base, cfg, base_map, tpl, wind, seed=0, ref_building=None,
                                 min_keep=COOL_MIN_TREES):
    """(layout, ops) or (None, reason).  The reference building is chosen by trying the buildings of the layout (or only
    `ref_building`, a building id such as "Building 2") on both sides of the wind (the template as drawn and its mirror
    image along the west-east axis through the building centre), with a small search over scale, rotation and shift."""
    import layout_reflow as RF
    rng = np.random.default_rng(seed)
    tc, pc = cfg["tree"], cfg["pond"]
    data = L.working_copy(base_data)
    sc = L.Scene(data, cfg, base.osm_polys)
    if not sc.group_polys:
        return None, "the layout has no building to use as reference"
    f, n = _frame(wind["flow"])
    a0 = L.base_area0(base, cfg, "pond")
    lo_a, hi_a = L.area_band(cfg, "pond", a0)
    max_total = L.max_tree_count(cfg, L.tree_count0(base, cfg), sc.pathway is not None)
    nb = len(sc.tx)
    budget = int(math.floor(COOL_BASE_REPLACEABLE * nb))
    N = len(tpl["trees"])
    gap = tc["canopy_gap_factor"]
    lo_r, hi_r = tc["radius_range_m"]
    rho = min(1.0, 0.995 * tpl["pair_ratio"] * 1.1 / gap)      # crown radii shrink a little so that the template spacing obeys the gap rule
    s_t_min = lo_r / (float(tpl["R"].min()) * rho)             # below this scale the smallest crown would be under the minimum radius
    s_t_max = hi_r / (float(tpl["R"].max()) * rho)
    margin = cfg["patch"].get("edge_margin_m", 0.25)
    static = (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn"))   # buildings grown by 2 m (ponds) / 1 m (lawns), pathway
    pond_region = sc.site.buffer(-margin).difference(static[0].buffer(0.1))
    shapely.prepare(pond_region)
    s_area = math.sqrt(0.5 * (lo_a + hi_a) / tpl["pond_area"])
    free0 = {}

    def free(r):
        r = round(float(r), 2)
        if r not in free0:
            free0[r] = _free_region(sc, cfg, r, ponds=[])
        return free0[r]

    areas = [p.area for p in sc.group_polys]
    amax = max(areas)
    tp = Polygon(tpl["pond_xy"]).buffer(0)
    sol_t = tp.area / tp.convex_hull.area                       # how compact the template pond is
    names = sc.group_names
    cands = [i for i in range(len(names)) if ref_building in (None, names[i])]
    if not cands:
        return None, f"reference building {ref_building!r} not found (buildings: {', '.join(names)})"
    if ref_building is None and len(cands) > COOL_MAX_BUILDINGS:
        cands = sorted(cands, key=lambda i: -areas[i])[:COOL_MAX_BUILDINGS]

    def evaluate(bi, mirror, k, rot_deg, du, dv, frac_list=COOL_KEEP):
        frame = _building_frame(sc.group_polys[bi], f, n)
        mp = _pose(tpl, f, n, frame, mirror, math.radians(rot_deg), du, dv)
        got = _pond_at(tpl, mp, s_area, pond_region, lo_a, hi_a)
        if got is None:
            return None
        pond, s_p, loss = got
        s_t = float(np.clip(s_p * k, s_t_min, s_t_max))
        P = mp(tpl["xy"], s_t)
        Rt = np.clip(rho * s_t * tpl["R"], lo_r, hi_r)
        ok = np.zeros(N, bool)
        for rv in np.unique(np.round(Rt, 2)):
            sel = np.where(np.isclose(np.round(Rt, 2), rv))[0]
            ok[sel] = shapely.contains_xy(free(rv), P[sel, 0], P[sel, 1])
        shapely.prepare(pond)
        ok &= shapely.distance(shapely.points(P[:, 0], P[:, 1]), pond) >= Rt + EPS
        notch = max(0.0, sol_t - pond.area / pond.convex_hull.area)          # a notch / bite taken out of the pond outline
        shape_pen = 1.0 * loss + 1.5 * notch
        return dict(bi=bi, mirror=mirror, k=k, rot=rot_deg, du=du, dv=dv, frame=frame, pond=pond, s_p=s_p, s_t=s_t,
                    loss=loss, notch=notch, shape_pen=shape_pen, P=P, Rt=Rt, ok=ok, n_ok=int(ok.sum()),
                    score=ok.mean() - shape_pen + 0.1 * areas[bi] / amax)

    # ---- stage 1: coarse pose search over buildings, sides, tree scale, rotation and shift
    pool = []
    for bi in cands:
        for mirror in (1.0, -1.0):
            for k in COOL_K:
                for rot in COOL_ROT_DEG:
                    for du in COOL_SHIFT_M:
                        for dv in COOL_SHIFT_M:
                            e = evaluate(bi, mirror, k, rot, du, dv)
                            if e is not None:
                                pool.append(e)
    if not pool:
        return None, ("no position for the pond (area %.0f-%.0f m2, at most %d%% cut off) next to any building"
                      % (lo_a, hi_a, round(100 * COOL_CLIP_MAX)))
    # ---- stage 2: fine search around the best few
    pool.sort(key=lambda e: -e["score"])
    fine = []
    for e0 in pool[:4]:
        for rot in COOL_ROT_FINE:
            for du in COOL_SHIFT_FINE:
                for dv in COOL_SHIFT_FINE:
                    e = evaluate(e0["bi"], e0["mirror"], e0["k"], e0["rot"] + rot, e0["du"] + du, e0["dv"] + dv)
                    if e is not None:
                        fine.append(e)
    pool = sorted(pool + fine, key=lambda e: -e["score"])
    # ---- stage 3: nudge the trees that stand in the way, tree-count limit, choose
    best = None
    for e in pool[:6]:
        pond = e["pond"]
        shapely.prepare(pond)
        P, ok = _nudge(e["P"], e["Rt"], e["ok"], free0, pond, gap)
        order = np.where(ok)[0]
        for frac in frac_list if False else COOL_KEEP:
            take = order[_thin(int(math.ceil(frac * len(order))), len(order))] if len(order) else order
            if len(take) < max(3, min_keep * N):
                continue
            conf = np.zeros(nb, bool)
            if nb:
                d = np.hypot(P[take, None, 0] - sc.tx[None, :], P[take, None, 1] - sc.ty[None, :])
                conf = (d < gap * (e["Rt"][take, None] + sc.tR[None, :]) + EPS).any(axis=0)
            removed = int(conf.sum()) + max(0, nb - int(conf.sum()) + len(take) - max_total)
            if removed > budget:
                continue
            sc_ = len(take) / N - e["shape_pen"] + 0.1 * areas[e["bi"]] / amax - 0.003 * removed
            if best is None or sc_ > best["score"]:
                best = dict(e, score=sc_, P=P, keep=take, removed=removed)
            break                                              # the fullest pattern that fits the limit
    if best is None:
        return None, ("fewer than %d%% of the template trees fit next to any building within the tree-count limit "
                      "(pond area must be %.0f-%.0f m2)" % (round(100 * min_keep), lo_a, hi_a))

    # ---- build the layout --------------------------------------------------------------------------------------
    pond, P, Rt, keep = best["pond"], best["P"], best["Rt"], best["keep"]
    # 1. the pond replaces the base ponds (total area stays in its band); lawns give way to it
    pat = copy.deepcopy(tpl["pond"])
    pat.setdefault("dimensions", {}).setdefault("depth", pc["default_depth_m"])
    step = cfg["patch"].get("discretization_m", 1.0)
    used = {p_["id"] for p_ in data["ponds"]}
    pat["id"] = next(f"pond_{k}" for k in range(1, 1000) if f"pond_{k}" not in used)
    cut = []
    for g in data["grassAreas"]:
        q = _largest(L._poly(g["coordinates"]).buffer(0).difference(pond.buffer(0.3)).difference(static[1]))
        cut.append(q if (q is not None and q.area >= cfg["lawn"]["min_area_m2"]) else None)
    pond_new = L.discretize_patches([pond], step, static[0])[0]
    polys = [pond_new] + L.discretize_patches([q for q in cut if q is not None], step, unary_union([static[1], pond_new]))
    L.patch_set_polygon(pat, polys[0], True, 0)
    kept, it = [], iter(polys[1:])
    for g, q in zip(data["grassAreas"], cut):
        if q is not None:
            L.patch_set_polygon(g, next(it), False, 0)
            kept.append(g)
    data["ponds"] = [pat]
    data["grassAreas"] = kept
    notes = {}
    reserve = [shapely.Point(float(P[i, 0]), float(P[i, 1])).buffer(float(Rt[i]) + EPS, 16) for i in keep]
    fitted = L.fit_patch_area(data, base, cfg, rng, pond_blockers=reserve)
    if fitted:
        notes["area_rescaled"] = fitted
        _separate_lawns_from_pond(data, cfg, static, step)

    # 2. base trees that give way (touching a template tree, then the nearest ones if the count limit needs it)
    gone = np.zeros(nb, bool)
    if nb:
        d = np.hypot(P[keep, None, 0] - sc.tx[None, :], P[keep, None, 1] - sc.ty[None, :])
        gone = (d < gap * (Rt[keep, None] + sc.tR[None, :]) + EPS).any(axis=0)
        over = nb - int(gone.sum()) + len(keep) - max_total
        if over > 0:
            for i in np.argsort(d.min(axis=0)):
                if over <= 0:
                    break
                if not gone[i]:
                    gone[i] = True
                    over -= 1
    removed_ids = {sc.tree_ids[i] for i in np.where(gone)[0]}
    data["trees"] = [t for t in data["trees"] if t["id"] not in removed_ids]

    # 3. the template trees; heights keep the tall / short order inside the height band of their location
    ball = sc.ball_union
    nid = max([int(str(t["id"]).split()[-1]) for t in base_data["trees"] if str(t["id"]).split()[-1].isdigit()] + [0])
    added = []
    for i in keep:
        t = copy.deepcopy(tpl["trees"][i])
        x, y, R = float(P[i, 0]), float(P[i, 1]), float(Rt[i])
        nid += 1
        t["id"] = f"Tree {nid}"
        t["position"] = {"x": x, "y": y, "z": t.get("position", {}).get("z", 0)}
        h = _band_height(cfg, float(tpl["hn"][i]), x, y, ball, sc.pathway)
        L.set_tree_geometry(t, R, h, tc)
        _sync_vegetation(t, R, h)
        added.append(t)
    data["trees"] = list(data["trees"]) + added
    mine = {t["id"] for t in added}

    moved = RF.fix_trees(data, base, cfg, rng)
    if moved:
        notes["trees_moved_off_rule_breaks"] = len([i for i in moved if i not in mine])
    L.finalize_patches(data, cfg)
    L.refresh_metadata(data)

    # 4. final check: a template tree that breaks a rule is dropped; anything else that is new makes the seed invalid
    for _ in range(6):
        notes["pathway_shade"] = RF.ensure_pathway_shade(data, base, cfg, rng, protect=mine)     # road shade: mandatory
        fsc = L.Scene(data, cfg, base.osm_polys)
        nv = L.new_violations(fsc, base, base_map)
        if not nv:
            break
        drop = set()
        for typ, ids_, _sev in nv:
            mm = [i for i in ids_ if i in mine and i not in drop]
            drop.update(mm[-1:] if typ == "tree_gap" else mm)
        if not drop:
            return None, "the layout has new rule violations that are not template trees: " + ", ".join(
                sorted({t for t, _, _ in nv}))
        data["trees"] = [t for t in data["trees"] if t["id"] not in drop]
        mine -= drop
        L.refresh_metadata(data)
    else:
        return None, "final rule check failed"
    if len(mine) < max(3, min_keep * N * 0.9):
        return None, "too few template trees survive the rules"
    fu0, fu1, fv0, fv1 = best["frame"]
    nud = int(len(keep) - best["n_ok"]) if best["n_ok"] <= len(keep) else 0
    info = {"op": "template", "kind": "cool_wind_catcher", "template": Path(tpl["path"]).name, "season": "heat",
            "wind_from_deg_model": round(wind["from_deg"], 1),
            "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
            "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
            "reference_building": names[best["bi"]],
            "reference_building_size_m": [round(fu1 - fu0, 1), round(fv1 - fv0, 1)],
            "mirrored_about_west_east_axis": best["mirror"] < 0,
            "side_of_wind": "left" if tpl["side"] * best["mirror"] > 0 else "right",
            "pose": {"rotate_deg": round(best["rot"], 1), "shift_along_wind_m": round(best["du"], 1),
                     "shift_across_wind_m": round(best["dv"], 1)},
            "scale": round(best["s_t"], 3), "pond_scale": round(best["s_p"], 3), "pond_area_cut_off_share": round(best["loss"], 3), "pond_notch": round(best["notch"], 3),
            "crown_radius_factor": round(rho, 3),
            "pond_area_m2": round(float(L._poly(data["ponds"][0]["coordinates"]).area), 1),
            "pond_area_allowed_m2": [round(lo_a, 1), round(hi_a, 1)],
            "trees_in_template": N, "trees_placed": len(mine), "trees_nudged": nud,
            "base_trees_replaced": len(removed_ids),
            "template_tree_ids": sorted(mine, key=lambda i: int(i.split()[-1])),
            "rules_enforced_on_base": notes,
            "distance_from_upwind_site_edge_m": None, "close_to_inlet": None}
    return (data, [info]), None
