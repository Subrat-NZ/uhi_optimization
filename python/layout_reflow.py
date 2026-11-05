#!/usr/bin/env python3
"""
layout_reflow.py  --  what has to happen around a moved building, and the full 'generation' recipe.

  reflow_patches   lawns / ponds make ROOM for moved buildings (they are cut back), then the lost area is
                   grown back into free space (or new patches are added) so each category's total area is
                   back in its allowed band.  If the base has no ponds, ponds are CREATED with a total area
                   of 10 % of the open space.
  fix_trees        trees that no longer satisfy a rule (new building position, pond grew, ...) are moved to
                   the nearest feasible spot; heights are re-derived from the location zone.
  make_generation  one complete 'generation' candidate from the base layout:
                     1. every physical building is translated by 0.5 x (its longest projected span) up to the
                        10 % limit (all stacked parts together)
                     2. lawns / ponds make room and re-grow, ponds are created if missing
                     3. trees: repaired, a share relocated / resized, count increased by 10 %, heights set by zone
                     4. extra random pond / lawn operations, then a final check (no new violation allowed)
All numbers live in input/constraints.json (building.min_shift_frac_of_length, generation.*, pond.*).
"""

import copy
import os
import math
import time

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polylabel, unary_union

import layout_ops as L

_ID_COUNTER = [0]
LAST_FAIL = [""]


def _fail(msg):
    LAST_FAIL[0] = msg


def _new_id(cat):
    _ID_COUNTER[0] += 1
    return f"{'pond' if cat == 'pond' else 'grass'}_{int(time.time() * 1000)}{_ID_COUNTER[0]}"


# ----------------------------------------------------------------------------- polygon helpers
def _polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for x in g.geoms:
            out += _polys(x)
        return out
    return []


def no_holes(poly):
    """A layout patch is a single outline, it cannot have holes.  A hole (a building inside the patch) is joined
    to the outside by a 1 m wide channel, so the patch stays one valid polygon and the building is OUTSIDE the
    outline even for a reader that only looks at the outer ring."""
    out = [poly]
    for _ in range(12):
        nxt, changed = [], False
        for p in out:
            if not p.interiors:
                nxt.append(p)
                continue
            changed = True
            h = Polygon(p.interiors[0])
            c = h.representative_point()
            q = p.exterior.interpolate(p.exterior.project(c))
            r = p.difference(LineString([(c.x, c.y), (q.x, q.y)]).buffer(0.5, cap_style=2))
            nxt += _polys(r)
        out = nxt
        if not changed:
            break
    return out


def clean(g, min_area):
    out = []
    for p in _polys(g):
        if not p.is_valid:
            p = p.buffer(0)
        for q in _polys(p):
            for r in no_holes(q):
                if r.is_valid and r.area >= min_area:
                    out.append(r)
    return out


def static_blockers(scene, cat=None, margin=0.0):
    """What a lawn / pond can never cover: the buildings (generated footprints and OSM) and the pathway.
    With cat='pond' / 'lawn' the buildings are grown by that category's clearance (patch.building_clearance_m:
    ponds 2 m, lawns 1 m), so a patch that stays out of this area keeps its distance from every building."""
    cl = L.clearance_m(scene.cfg, cat) if cat in ("pond", "lawn") else 0.0
    z = L.building_zone(scene.group_polys, scene.osm_polys, cl + max(margin, 0.0))      # (grown once by clearance + margin: same as growing twice)
    parts = [z] if z is not None else []
    if scene.pathway is not None:
        parts.append(scene.pathway)
    return unary_union(parts)


# ----------------------------------------------------------------------------- lawns and ponds
def _default_patch(cat, cfg):
    pc = cfg[cat]
    if cat == "pond":
        return {"id": "", "coordinates": [], "dimensions": {"area": 0.0, "depth": pc["default_depth_m"], "volume": 0.0},
                "material": copy.deepcopy(pc["template"]["material"]), "elevation": 0}
    return {"id": "", "coordinates": [], "dimensions": {"area": 0.0},
            "material": copy.deepcopy(pc["template"]["material"]), "elevation": 0}


def _poles(free, min_area, k, margin):
    """The k most open spots of the free space, best first: (point, clearance radius, free component).
    Spot = pole of inaccessibility (the point with the most room around it) of a free component that is
    eroded by `margin`; after a spot is taken, a disc around it is removed and the next spot is searched."""
    out, work = [], free
    for _ in range(k):
        best = None
        for comp in _polys(work):
            if comp.area < min_area:
                continue
            for q in _polys(comp.buffer(-margin)):
                if q.area < min_area:
                    continue
                pt = polylabel(q, tolerance=0.5)
                r = q.boundary.distance(pt)
                if best is None or r > best[1]:
                    best = (pt, r, q)
        if best is None:
            break
        out.append(best)
        work = work.difference(best[0].buffer(1.5 * best[1] + 1.0))
    return out


def _seed_patch(free_rest, need, min_area, rng, cfg):
    """A NEW lawn / pond in the largest open area: seed at the most open spot of the free space (rank 1; rank
    2, 3 ... with the probabilities patch.largest_open_rank_probs) and grow the part of a disc that is VISIBLE
    from the seed until the area is `need`.  Such a patch is star-shaped: it never has a hole and never
    encloses a building.  If the visible part is smaller than `need` the patch is smaller (the caller adds
    another patch in the next open area)."""
    pc = cfg["patch"]
    probs = np.array(pc["largest_open_rank_probs"], float)
    spots = _poles(free_rest, min_area, len(probs), pc["edge_margin_m"])
    if not spots:
        return None
    pr = probs[:len(spots)]
    pt, _r, q = spots[int(rng.choice(len(spots), p=pr / pr.sum()))]
    n = int(pc["rays"])
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rmax = 2.0 * float(np.hypot(q.bounds[2] - q.bounds[0], q.bounds[3] - q.bounds[1]))
    ends = np.stack([pt.x + rmax * np.cos(ang), pt.y + rmax * np.sin(ang)], axis=1)
    lines = shapely.linestrings(np.stack([np.tile([pt.x, pt.y], (n, 1)), ends], axis=1))
    hit = shapely.distance(shapely.intersection(lines, q.boundary), pt)
    dvis = np.where(np.isnan(hit), 0.0, hit)                    # distance to the first obstacle along each ray

    w = 0.5 * float(pc.get("min_width_m", 4.0))

    def shape(r):
        d = np.minimum(dvis, r)
        g = Polygon(np.stack([pt.x + d * np.cos(ang), pt.y + d * np.sin(ang)], axis=1)).buffer(0)
        g = g.buffer(-w, quad_segs=6).buffer(w, quad_segs=6).intersection(q)       # opening: no spikes thinner than min_width_m
        ps = [x for x in _polys(g) if x.buffer(1e-6).contains(pt) or x.distance(pt) < 1e-6]
        return max(ps, key=lambda x: x.area) if ps else None

    def area(r):
        g = shape(r)
        return g.area if g is not None else 0.0

    top = area(float(dvis.max()))
    need = min(need, 0.98 * top)
    if need < min_area:
        return None
    lo, hi = 0.0, float(dvis.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if area(mid) < need:
            lo = mid
        else:
            hi = mid
    g = shape(hi)
    cl = clean(g, min_area) if g is not None else []
    return max(cl, key=lambda x: x.area) if cl else None


def _others_of(items):
    """For every patch: the union of all the OTHER patches' original outlines (None for a single patch)."""
    originals = [it["poly"] for it in items]
    return [unary_union([o for j, o in enumerate(originals) if j != i]) if len(originals) > 1 else None for i in range(len(originals))]


def _expand(items, free, d, weights, oth=None):
    out, taken = [], None
    oth = _others_of(items) if oth is None else oth
    for i, (it, w) in enumerate(zip(items, weights)):
        others = oth[i]
        buf = it["poly"].buffer(d * w, quad_segs=6)
        try:                                                              # only the part of `free` inside the grown patch's box matters
            g = buf.intersection(shapely.clip_by_rect(free, *buf.bounds))
        except shapely.errors.GEOSException:
            g = buf.intersection(free)
        if others is not None:
            g = g.difference(others)
        if taken is not None:
            g = g.difference(taken)
        # growth only reaches what is connected to the patch (not the far side of a road or a building)
        keep = [q for q in _polys(g) if q.intersects(it["poly"])]
        g = unary_union(keep) if keep else it["poly"]
        out.append(g)
        taken = g if taken is None else taken.union(g)
    return out


def _grow(cat, items, free, site, target, cfg, rng, maxp=None):
    min_area = cfg[cat]["min_area_m2"]
    maxp = cfg[cat]["max_patches"] if maxp is None else maxp
    area = lambda gs: sum(g.area for g in gs)
    tol = 2e-4 * max(target, 1.0)
    if items:
        w = rng.uniform(0.3, 1.0, len(items))
        dmax = 0.5 * max(site.bounds[2] - site.bounds[0], site.bounds[3] - site.bounds[1])
        oth = _others_of(items)                                  # the originals do not change during the search
        if area(_expand(items, free, dmax, w, oth)) >= target:
            lo, hi = 0.0, dmax
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                if area(_expand(items, free, mid, w, oth)) < target:
                    lo = mid
                else:
                    hi = mid
            gs = _expand(items, free, 0.5 * (lo + hi), w, oth)
        else:
            gs = _expand(items, free, dmax, w, oth)
        new = []
        for it, g in zip(items, gs):
            pieces = sorted(clean(g, min_area), key=lambda q: -q.area)
            for k, q in enumerate(pieces):
                new.append({"id": it["id"] if k == 0 else _new_id(cat), "poly": q, "src": it["src"]})
        items = new[:maxp] if len(new) > maxp else new
    tot = sum(i["poly"].area for i in items)
    tries = 0
    while target - tot > tol and len(items) < maxp and tries < 6:
        tries += 1
        rest = free.difference(unary_union([i["poly"] for i in items])) if items else free
        need = target - tot
        if not items and tries == 1 and need > 4 * min_area:          # creating from nothing: 1-2 patches
            need *= rng.uniform(0.5, 1.0) if rng.random() < 0.5 else 1.0
        poly = _seed_patch(rest, need, min_area, rng, cfg)
        if poly is None:
            break
        src = items[0]["src"] if items else _default_patch(cat, cfg)
        items.append({"id": _new_id(cat), "poly": poly, "src": src})
        tot += poly.area
    return items


def _shrink(cat, items, target, cfg):
    min_area = cfg[cat]["min_area_m2"]
    polys = [i["poly"] for i in items]

    def shrunk(d):
        return [p.buffer(-d, quad_segs=6) for p in polys]

    lo, hi = 0.0, max(math.sqrt(p.area) for p in polys)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if sum(g.area for g in shrunk(mid)) > target:
            lo = mid
        else:
            hi = mid
    new = []
    for it, g in zip(items, shrunk(0.5 * (lo + hi))):
        for k, q in enumerate(sorted(clean(g, min_area), key=lambda q: -q.area)):
            new.append({"id": it["id"] if k == 0 else _new_id(cat), "poly": q, "src": it["src"]})
    return new


def _target(lo, hi, aim=0.1):
    return 0.5 * (lo + hi) if hi - lo < 0.004 * max(hi, 1.0) else lo + aim * (hi - lo)


def _dbg(msg):
    if os.environ.get("REFLOW_DEBUG"):
        print("reflow_patches:", msg, flush=True)


CUT_MARGIN_M = 0.1       # patches are cut this much beyond the clearance (the clearance itself is checked with patch.clearance_tol_m)
REFLOW_AIMS = (0.1, 0.3, 0.55, 0.8)      # where in the band the reflow aims (share of the band width above its lower limit)


def reflow_patches(data, base, cfg, rng, pond_blockers=None):
    """Cut lawns / ponds back around buildings (grown by the clearance of the category: ponds 2 m, lawns 1 m) and
    pathways, bring their totals back into the allowed band (grow into free space, add patches, or create ponds when
    the base has none).  Edits `data`.  Returns a dict describing what changed, or None if the totals could not be restored.
    Re-discretizing the outlines against the buildings / each other costs some area; if that pushes a total out of its band
    the reflow is repeated aiming higher in the band (REFLOW_AIMS).
    pond_blockers: optional shapely polygons (e.g. tree crowns reserved by a template seed) that ponds must avoid."""
    sc = L.Scene(data, cfg, base.osm_polys)
    site = sc.site
    statics = {c: static_blockers(sc, c) for c in ("lawn", "pond")}      # buildings grown by the clearance of the category
    # patches are CUT a little further back (CUT_MARGIN_M) than the clearance: re-discretizing an outline (1 m chords) then never touches the
    # exact keep-out, so the clean-up does not shave tenths of a metre off every boundary (hundreds of m2 in total)
    cuts = {c: static_blockers(sc, c, CUT_MARGIN_M) for c in ("lawn", "pond")}
    pb = unary_union(pond_blockers) if pond_blockers else None
    state0, report0 = {}, {}
    nbld = len(L.building_groups(data))
    limit = {c: L.patch_limit(cfg, c, nbld) for c in ("lawn", "pond")}
    for cat, key in (("lawn", "grassAreas"), ("pond", "ponds")):
        blocked = cuts["lawn"] if cat == "lawn" else unary_union([cuts["pond"]] + [i["poly"] for i in state0["lawn"]] + ([pb] if pb is not None else []))
        items = []
        for p in data[key]:
            g = L._poly(p["coordinates"]).buffer(0).intersection(site).difference(blocked)
            for k, q in enumerate(sorted(clean(g, cfg[cat]["min_area_m2"]), key=lambda q: -q.area)):
                items.append({"id": p["id"] if k == 0 else _new_id(cat), "poly": q, "src": p})
        if len(items) > limit[cat]:
            items = sorted(items, key=lambda i: -i["poly"].area)[:limit[cat]]
        state0[cat] = items
        report0[cat] = {"area_before": sum(L._poly(p["coordinates"]).area for p in data[key])}
    # both missing: the pond takes the largest open area first, the lawns the next ones
    order = ("pond", "lawn") if not state0["lawn"] and not state0["pond"] else ("lawn", "pond")
    stp = cfg["patch"].get("discretization_m", 1.0)
    bands = {}
    for cat in ("lawn", "pond"):
        a0 = L.base_area0(base, cfg, cat)
        bands[cat] = (L.area_band(cfg, cat, a0), 1e-3 * max(a0, 1.0))
    for aim in REFLOW_AIMS:
        state = {c: [dict(i) for i in v] for c, v in state0.items()}
        report = {c: dict(v) for c, v in report0.items()}
        ok = True
        for cat in order:
            other = "pond" if cat == "lawn" else "lawn"
            free = site.difference(unary_union([cuts[cat]] + [i["poly"] for i in state[other]]
                                               + ([pb] if (pb is not None and cat == "pond") else [])))
            (lo, hi), slack = bands[cat]
            for _ in range(3):
                tot = sum(i["poly"].area for i in state[cat])
                if lo - slack <= tot <= hi + slack:
                    break
                tgt = _target(lo, hi, aim)
                state[cat] = _shrink(cat, state[cat], tgt, cfg) if tot > hi else _grow(cat, state[cat], free, site, tgt, cfg, rng, limit[cat])
            tot = sum(i["poly"].area for i in state[cat])
            if not (lo - slack <= tot <= hi + slack):
                _dbg(f"aim {aim}: {cat} total {tot:.0f} after grow/shrink outside band {lo:.0f}-{hi:.0f}")
                ok = False
                break
            report[cat].update(area_after=tot, patches=len(state[cat]))
        if not ok:
            continue
        fixed, out_d, tot_ok = [], {}, True
        for cat, key in (("pond", "ponds"), ("lawn", "grassAreas")):        # ponds first: lawns are pulled back from them
            ko = unary_union([statics[cat]] + fixed + ([pb] if (pb is not None and cat == "pond") else []))
            polys = [L.unslit(q) for q in L.discretize_patches([it["poly"] for it in state[cat]], stp, ko)]
            (lo, hi), slack = bands[cat]
            tot = sum(q.area for q in polys)
            if not (lo - slack <= tot <= hi + slack):
                _dbg(f"aim {aim}: {cat} total {tot:.0f} after discretization outside band {lo:.0f}-{hi:.0f}")
                tot_ok = False
                break
            out_d[cat] = polys
            fixed.extend(polys)
        if not tot_ok:
            continue
        for cat, key in (("pond", "ponds"), ("lawn", "grassAreas")):
            out = []
            for it, poly in zip(state[cat], out_d[cat]):
                p = copy.deepcopy(it["src"])
                p["id"] = it["id"]
                L.patch_set_polygon(p, poly, cat == "pond", 0)
                out.append(p)
            data[key] = out
            report[cat]["area_after"] = sum(q.area for q in out_d[cat])
        L.refresh_metadata(data)
        return report
    return None


# ----------------------------------------------------------------------------- trees
class TreePlacer:
    """Fast 'can a tree of radius R stand here' test against everything except lawns."""

    def __init__(self, sc, cfg, data):
        self.cfg, self.tc = cfg, cfg["tree"]
        self.site = sc.site
        self.bg = unary_union(sc.group_polys) if sc.group_polys else None
        self.bo = unary_union(sc.osm_polys) if sc.osm_polys else None
        self.ball = unary_union(list(sc.group_polys) + list(sc.osm_polys)) if (sc.group_polys or sc.osm_polys) else None
        self.ponds = unary_union(sc.ponds) if sc.ponds else None
        self.pathway = sc.pathway
        self.extra = self.tc["building_clearance_extra_frac_of_width"] * sc.min_width
        self.xs, self.ys, self.Rs = list(sc.tx), list(sc.ty), list(sc.tR)

    def ok(self, x, y, R, ignore=None):
        tc = self.tc
        p = Point(x, y)
        if not self.site.contains(p) or self.site.exterior.distance(p) < R:
            return False
        kf = tc["building_clearance_factor"]
        if self.bg is not None and self.bg.distance(p) < kf * R + self.extra:
            return False
        if self.bo is not None and tc["clearance_applies_to_osm"] and self.bo.distance(p) < kf * R:
            return False
        if self.ponds is not None and self.ponds.distance(p) < R:
            return False
        if self.pathway is not None and not tc["tree_on_pathway_allowed"] and self.pathway.distance(p) < R:
            return False
        if self.xs:
            xs, ys, Rs = np.array(self.xs), np.array(self.ys), np.array(self.Rs)
            d = np.hypot(xs - x, ys - y)
            need = tc["canopy_gap_factor"] * (Rs + R)
            if ignore is not None:
                d[ignore] = np.inf
            if np.any(d < need):
                return False
        return True

    def place(self, rng, R0, near=None, tries=500):
        minx, miny, maxx, maxy = self.site.bounds
        for t in range(tries):
            if near is not None and t < 150:
                s = 4.0 + t / 8.0
                x, y = near[0] + rng.normal(0, s), near[1] + rng.normal(0, s)
            else:
                x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            R = R0 if t < 300 else max(self.tc["radius_range_m"][0], R0 - 0.5 * ((t - 300) // 40 + 1))
            if self.ok(x, y, R):
                return float(x), float(y), float(R)
        return None


def _set_pos(t, x, y):
    t["position"]["x"], t["position"]["y"] = float(x), float(y)


def fix_trees(data, base, cfg, rng, pl=None, sc=None):
    """Move trees that break a rule to the nearest feasible place; re-derive heights that fall outside the
    zone band.  Returns the list of tree ids that were moved (trees that cannot be placed are removed)."""
    sc = sc or L.Scene(data, cfg, base.osm_polys)
    pl = pl or TreePlacer(sc, cfg, data)
    T, moved, drop = data["trees"], [], []
    for i, t in enumerate(T):
        x, y, R = pl.xs[i], pl.ys[i], pl.Rs[i]
        if pl.ok(x, y, R, ignore=i):
            continue
        pl.xs[i], pl.ys[i] = 1e9, 1e9                           # park far away while searching
        spot = pl.place(rng, R, near=(x, y))
        if spot is None:
            drop.append(i)
            pl.xs[i], pl.ys[i] = 1e9, 1e9
            continue
        pl.xs[i], pl.ys[i] = spot[0], spot[1]
        pl.Rs[i] = spot[2]
        _set_pos(t, spot[0], spot[1])
        L.set_tree_geometry(t, spot[2], float(t["height"]), cfg["tree"])
        moved.append(t["id"])
    if drop:
        data["trees"] = [t for i, t in enumerate(T) if i not in set(drop)]
    rezone_heights(data, sc, cfg, rng)
    return moved


def rezone_heights(data, sc, cfg, rng, all_trees=False):
    """Set the height of every tree that is outside its location band (or of all trees)."""
    for t in data["trees"]:
        x, y = t["position"]["x"], t["position"]["y"]
        p = Point(x, y)
        d = sc.ball_union.distance(p) if sc.ball_union is not None else np.inf
        lo, hi = L.height_band(cfg, d, L.on_pathway(cfg, sc.pathway, x, y))
        h = float(t["height"])
        if all_trees or h < lo - 1e-6 or h > hi + 1e-6:
            h = float(np.clip(L._rt(rng.uniform(lo, hi), cfg["tree"]["size_step_m"]), lo, hi))
            L.set_tree_geometry(t, L._tree_R(t), h, cfg["tree"])


def _side_point(ring, s, offset, pathway):
    """Point `offset` m beside the pathway edge `ring` at arc length s, on the side that is NOT pathway."""
    ln = ring.length
    p, a, b = ring.interpolate(s % ln), ring.interpolate((s - 0.5) % ln), ring.interpolate((s + 0.5) % ln)
    tx, ty = b.x - a.x, b.y - a.y
    n = math.hypot(tx, ty)
    if n < 1e-9:
        return None
    nx, ny = -ty / n, tx / n
    if pathway.contains(Point(p.x + 0.3 * nx, p.y + 0.3 * ny)):
        nx, ny = -nx, -ny
        if pathway.contains(Point(p.x + 0.3 * nx, p.y + 0.3 * ny)):
            return None
    return p.x + offset * nx, p.y + offset * ny


def plant_pathways(data, cfg, rng, pl, sc, nid):
    """Plant trees along BOTH sides of every pathway wherever the tree rules leave room (slots without room
    are skipped).  Slots lie on the pathway outline (= both edges of a road, inner and outer edge of a loop)
    every ~spacing m; a random share of them is used, at most pathway_planting.max_new_trees.
    Returns (number planted, number of slots, next free tree number)."""
    pp, tc = cfg["tree"]["pathway_planting"], cfg["tree"]
    T = data["trees"]
    if not pp["enabled"] or sc.pathway is None:
        return 0, 0, nid
    rings = [r for poly in _polys(sc.pathway) for r in [poly.exterior, *poly.interiors] if r.length > 2.0]
    total = sum(r.length for r in rings)
    if total <= 0:
        return 0, 0, nid
    frac = float(rng.uniform(*pp["fill_fraction_range"]))
    n_slots = int(round(total / pp["spacing_m"]))
    n_target = max(1, min(int(pp["max_new_trees"]), int(round(frac * n_slots))))
    step = total / n_target                                   # even spacing over all pathway edges
    cum = np.cumsum([0.0] + [r.length for r in rings])
    phase = float(rng.uniform(0, step))
    rlo, rhi = pp["radius_range_m"]
    lo_r = tc["radius_range_m"][0]
    planted = 0
    for k in range(n_target):
        t = (phase + k * step) % total
        ri = int(np.searchsorted(cum, t, side="right") - 1)
        ring, s = rings[ri], t - cum[ri]
        R0 = float(np.clip(L._rt(rng.uniform(rlo, rhi), tc["size_step_m"]), max(rlo, lo_r), rhi))
        done = False
        for slide in pp["slide_tries_m"]:
            R = R0
            while R >= max(1.5, lo_r) - 1e-9 and not done:
                c = _side_point(ring, s + slide, R + pp["canopy_gap_to_path_m"], sc.pathway)
                if c is not None and sc.pathway.distance(Point(*c)) >= R + pp["canopy_gap_to_path_m"] - 1e-6 and pl.ok(c[0], c[1], R):
                    t_new = L.new_tree(cfg, T, rng)
                    nid += 1
                    t_new["id"] = f"Tree {nid}"
                    _set_pos(t_new, c[0], c[1])
                    L.set_tree_geometry(t_new, R, float(t_new["height"]), tc)
                    T.append(t_new)
                    pl.xs.append(c[0]); pl.ys.append(c[1]); pl.Rs.append(R)
                    planted += 1
                    done = True
                R -= tc["size_step_m"]
            if done:
                break
    return planted, n_target, nid


def ensure_pathway_shade(data, base, cfg, rng, protect=(), force=False):
    """ROAD SHADE repair, applied to EVERY individual (see layout_ops.shade_coverage for the rule).
    When fewer than min_coverage of the plantable road-edge samples have a crown edge within reach_m (or force=True), up to
    target_coverage is restored by
      1. planting trees beside the road: the crown radius is drawn from shade_radius_range_m and reduced step by step until it fits
         (so the SIZE adapts to the room that is there: big crowns where there is space, small ones between buildings), the
         centre is slid forward along the edge so that one crown covers as much road as possible;
      2. enlarging crowns of nearby trees (up to grow_max_radius_m) when a slot has no room for a new tree;
      3. when the tree-count limit is reached, moving trees that stand far from every road (never the ids in `protect`).
    Edits `data` in place.  Returns an info dict (coverage before / after, trees planted / enlarged / relocated)."""
    tc, pp = cfg["tree"], cfg["tree"]["pathway_planting"]
    info = {"before": None, "after": None, "planted": 0, "enlarged": 0, "relocated": 0}
    if not pp["enabled"] or not pp.get("shade_required", True):
        return info
    sc = L.Scene(data, cfg, base.osm_polys)
    if sc.pathway is None:
        return info
    cov0, _sh, S = L.shade_coverage(sc)
    info["before"] = round(cov0, 3)
    if cov0 >= pp["min_coverage"] and not force:
        info["after"] = info["before"]
        return info
    P, N, Tg, plant = S["P"], S["N"], S["Tg"], S["plantable"]
    n_pl = int(plant.sum())
    if n_pl == 0:
        info["after"] = 1.0
        return info
    protect = set(protect)
    T = data["trees"]
    pl = TreePlacer(sc, cfg, data)
    reach, gap, step = float(pp["reach_m"]), float(pp["canopy_gap_to_path_m"]), float(tc["size_step_m"])
    rlo, rhi = pp["shade_radius_range_m"]
    r_min = max(float(pp["min_radius_m"]), float(tc["radius_range_m"][0]))
    r_grow = min(float(pp["grow_max_radius_m"]), float(tc["radius_range_m"][1]))
    cap = L.max_tree_count(cfg, L.tree_count0(base, cfg), True)
    goal = float(pp["target_coverage"])
    dm = L.shade_dmin(P, pl.xs, pl.ys, pl.Rs)
    nid = max([int(str(t["id"]).split()[-1]) for t in T if str(t["id"]).split()[-1].isdigit()] + [0])
    pw = sc.pathway
    new_ids = []

    def coverage():
        return float((dm[plant] <= reach + 1e-9).mean())

    def fits(x, y, R, ignore=None):
        return pl.ok(x, y, R, ignore=ignore) and pw.distance(Point(x, y)) >= R + gap - 1e-6

    def donors():
        """indices of trees that may be moved: not protected, far from every road, farthest first."""
        out = []
        for i, t in enumerate(T):
            if t["id"] in protect or t["id"] in new_ids or pl.xs[i] > 1e8:
                continue
            d = pw.distance(Point(pl.xs[i], pl.ys[i]))
            if d > tc["pathway_zone_distance_m"] + pl.Rs[i]:
                out.append((d, i))
        return [i for _, i in sorted(out, reverse=True)]

    max_reloc = int(pp["max_relocated_frac"] * len(T))
    donor_list = None
    for i in np.roll(np.arange(len(P)), -int(rng.integers(len(P)))):
        if coverage() >= goal - 1e-9:
            break
        if not plant[i] or dm[i] <= reach:
            continue
        R0 = float(np.clip(L._rt(rng.uniform(rlo, rhi), step), rlo, rhi))
        spot = None
        R = R0
        while R >= r_min - 1e-9 and spot is None:
            half = math.sqrt(max((R + reach) ** 2 - (R + gap) ** 2, 0.0))
            for f in (0.8, 0.4, 0.0, -0.4):
                c = P[i] + (R + gap) * N[i] + f * half * Tg[i]
                if fits(c[0], c[1], R):
                    spot = (float(c[0]), float(c[1]), R)
                    break
            R -= step
        donor = None
        if spot is not None and len(T) >= cap:
            # at the tree-count limit: move the tree that stands farthest from every road
            if donor_list is None:
                donor_list = donors()
            if info["relocated"] >= max_reloc or not donor_list:
                spot = None
            else:
                donor = donor_list.pop(0)
        if spot is None:
            # no room for a new crown: enlarge a nearby tree instead
            best = None
            for j, t in enumerate(T):
                if t["id"] in protect or pl.xs[j] > 1e8:
                    continue
                d0 = math.hypot(pl.xs[j] - P[i, 0], pl.ys[j] - P[i, 1])
                if d0 > 14.0 or pw.distance(Point(pl.xs[j], pl.ys[j])) > pl.Rs[j] + 6.0:
                    continue
                Rn = pl.Rs[j] + step
                while Rn <= r_grow + 1e-9:
                    if fits(pl.xs[j], pl.ys[j], Rn, ignore=j):
                        if d0 - Rn <= reach + 1e-9:
                            if best is None or d0 - Rn < best[0]:
                                best = (d0 - Rn, j, Rn)
                            break
                        Rn += step
                    else:
                        break
            if best is not None:
                _, j, Rn = best
                pl.Rs[j] = Rn
                L.set_tree_geometry(T[j], Rn, float(T[j]["height"]), tc)
                tm = T[j].get("materials", {}).get("vegetation")
                if isinstance(tm, dict) and "crown_diameter" in tm:
                    tm["crown_diameter"] = float(2 * Rn)
                dm = np.minimum(dm, np.hypot(P[:, 0] - pl.xs[j], P[:, 1] - pl.ys[j]) - Rn)
                info["enlarged"] += 1
            continue
        if donor is not None:
            ox, oy = pl.xs[donor], pl.ys[donor]
            pl.xs[donor], pl.ys[donor] = 1e9, 1e9
            if not fits(spot[0], spot[1], spot[2]):          # the donor's own crown was in the way: put it back
                pl.xs[donor], pl.ys[donor] = ox, oy
                continue
            t = T[donor]
            _set_pos(t, spot[0], spot[1])
            pl.xs[donor], pl.ys[donor], pl.Rs[donor] = spot[0], spot[1], spot[2]
            L.set_tree_geometry(t, spot[2], float(t["height"]), tc)
            new_ids.append(t["id"])
            info["relocated"] += 1
            dm = np.minimum(dm, np.hypot(P[:, 0] - spot[0], P[:, 1] - spot[1]) - spot[2])
            # the moved tree may have been the only shade elsewhere
            dm = L.shade_dmin(P, pl.xs, pl.ys, pl.Rs)
            continue
        t = L.new_tree(cfg, T, rng)
        nid += 1
        t["id"] = f"Tree {nid}"
        _set_pos(t, spot[0], spot[1])
        L.set_tree_geometry(t, spot[2], float(t["height"]), tc)
        T.append(t)
        new_ids.append(t["id"])
        pl.xs.append(spot[0]); pl.ys.append(spot[1]); pl.Rs.append(spot[2])
        dm = np.minimum(dm, np.hypot(P[:, 0] - spot[0], P[:, 1] - spot[1]) - spot[2])
        info["planted"] += 1
    # heights: new / moved street trees stand in the upper half of their band (longer shadows), others stay in band
    sc2 = L.Scene(data, cfg, base.osm_polys)
    ids = set(new_ids)
    for t in T:
        x, y = t["position"]["x"], t["position"]["y"]
        d = sc2.ball_union.distance(Point(x, y)) if sc2.ball_union is not None else np.inf
        lo, hi = L.height_band(cfg, d, L.on_pathway(cfg, sc2.pathway, x, y))
        h = float(t["height"])
        if t["id"] in ids:
            h = float(np.clip(L._rt(rng.uniform(0.5 * (lo + hi), hi), step), lo, hi))
        elif h < lo - 1e-6 or h > hi + 1e-6:
            h = float(np.clip(L._rt(rng.uniform(lo, hi), step), lo, hi))
        else:
            continue
        L.set_tree_geometry(t, L._tree_R(t), h, tc)
        v = t.get("materials", {}).get("vegetation")
        if isinstance(v, dict) and "height" in v:
            v["height"] = float(h)
    L.refresh_metadata(data)
    info["after"] = round(L.shade_coverage(L.Scene(data, cfg, base.osm_polys))[0], 3)
    return info


def tree_stage(data, base, cfg, rng):
    """Generation recipe for trees.  Returns an info dict."""
    gc, tc = cfg["generation"], cfg["tree"]
    sc = L.Scene(data, cfg, base.osm_polys)
    pl = TreePlacer(sc, cfg, data)
    info = {}
    n0 = L.tree_count0(base, cfg)
    info["repaired"] = len(fix_trees(data, base, cfg, rng, pl=pl, sc=sc))
    T = data["trees"]
    pl.xs, pl.ys, pl.Rs = [t["position"]["x"] for t in T], [t["position"]["y"] for t in T], [L._tree_R(t) for t in T]
    # share of trees relocated / resized
    idx = list(rng.permutation(len(T)))
    n_move = int(round(gc["tree_relocate_fraction"] * len(T)))
    n_size = int(round(gc["tree_resize_fraction"] * len(T)))
    moved = resized = 0
    for i in idx[:n_move]:
        R = pl.Rs[i]
        old = (pl.xs[i], pl.ys[i])
        pl.xs[i], pl.ys[i] = 1e9, 1e9
        spot = pl.place(rng, R)
        if spot is None:
            pl.xs[i], pl.ys[i] = old
            continue
        pl.xs[i], pl.ys[i] = spot[0], spot[1]
        _set_pos(T[i], spot[0], spot[1])
        moved += 1
    for i in rng.permutation(len(T))[:n_size]:
        R0 = pl.Rs[i]
        R = float(np.clip(L._rt(R0 + rng.normal(0, tc["resize_sigma_m"]), tc["size_step_m"]), *tc["radius_range_m"]))
        if abs(R - R0) < 1e-9 or not pl.ok(pl.xs[i], pl.ys[i], R, ignore=i):
            continue
        pl.Rs[i] = R
        L.set_tree_geometry(T[i], R, float(T[i]["height"]), tc)
        resized += 1
    # count: +10 % of the base count
    target = n0 + math.ceil(gc["tree_count_increase_frac"] * n0) if len(base.tx) else n0   # no trees in the base: create n0
    added = 0
    nid = max([int(str(t["id"]).split()[-1]) for t in T if str(t["id"]).split()[-1].isdigit()] + [0])
    while len(T) < target:
        R0 = float(np.clip(L._rt(rng.uniform(*tc["new_tree_radius_range_m"]), tc["size_step_m"]), *tc["radius_range_m"]))
        spot = pl.place(rng, R0)
        if spot is None:
            break
        t = L.new_tree(cfg, T, rng)
        nid += 1
        t["id"] = f"Tree {nid}"
        _set_pos(t, spot[0], spot[1])
        L.set_tree_geometry(t, spot[2], float(t["height"]), tc)
        T.append(t)
        pl.xs.append(spot[0]); pl.ys.append(spot[1]); pl.Rs.append(spot[2])
        added += 1
    planted, slots, nid = plant_pathways(data, cfg, rng, pl, sc, nid)
    info.update(pathway_planted=planted, pathway_slots=slots)
    for i, t in enumerate(T):                                  # radius changes live in pl.Rs for moved/resized trees
        L.set_tree_geometry(t, pl.Rs[i], float(t["height"]), tc)
    if gc["normalize_tree_heights"]:
        rezone_heights(data, sc, cfg, rng, all_trees=True)
    else:
        rezone_heights(data, sc, cfg, rng)
    info["pathway_shade"] = ensure_pathway_shade(data, base, cfg, rng)       # road shade is mandatory for every individual
    info.update(relocated=moved, resized=resized, added=added, count=len(T), target=target)
    return info


# ----------------------------------------------------------------------------- the generation recipe
def make_generation(base_data, base, cfg, rng, base_map, n_extra_ops=2):
    """One complete generation candidate.  Returns (data, ops) or None when a rule could not be met."""
    gc = cfg["generation"]
    data, ops = L.working_copy(base_data), []
    # 1. every physical building moves by 0.5 L .. 10 % of the site diameter.  If the full minimum is
    #    physically impossible (boxed in by roads, other buildings, the site edge) it is relaxed in steps and
    #    the relaxation is written to the log.  Several random building orders are tried; the one that needs
    #    the least relaxation is kept (hard-to-move buildings are easier to place when they go first).
    best = None
    for _ in range(int(gc.get("building_attempts", 6))):
        trial, tops, shortfall = L.working_copy(base_data), [], 0.0
        for gi in rng.permutation(len(base.groups)):
            gi = int(gi)
            info = None
            for scale in gc["min_shift_relax_steps"]:
                sc = L.Scene(trial, cfg, base.osm_polys)
                info = L.op_shift_building(trial, base, cfg, rng, 1.0, sc, gi=gi, fresh=True, min_scale=scale)
                if info is not None:
                    info["min_shift_scale"] = scale
                    break
            full = min(cfg["building"]["min_shift_frac_of_length"] * base.group_len[gi],
                       0.98 * cfg["building"]["max_shift_frac_of_site_diameter"] * base.site_diameter)
            got = info["shift_m"] if info is not None else 0.0
            shortfall += max(0.0, full - got) / max(base.group_len[gi], 1e-9)
            tops.append(info or {"op": "building_could_not_move", "building": base.group_names[gi]})
        if best is None or shortfall < best[0]:
            best = (shortfall, trial, tops)
        if shortfall < 1e-9:
            break
    _, data, tops = best
    ops.extend(tops)
    # 2. lawns and ponds make room, totals restored (ponds created if the base has none)
    rep = reflow_patches(data, base, cfg, rng)
    if rep is None:
        _fail("reflow: lawn / pond totals could not be restored")
        return None
    ops.append({"op": "reflow_lawns_ponds", **rep})
    # 3. trees
    ops.append({"op": "trees", **tree_stage(data, base, cfg, rng)})
    # 4. extra variety: pond / lawn reshaping, then random operations
    # a category the base did not have (created above in the largest open area) is not shuffled afterwards
    present = {"pond": bool(base.ponds), "lawn": bool(base.lawns), "building": True, "tree": True}
    cw = {c: w for c, w in cfg["ops"]["category_weights"].items() if present.get(c, True)}
    cats, pw = list(cw), np.array(list(cw.values()), float)
    for cat, n in gc["min_extra_ops"].items():
        if not present.get(cat, True):
            continue
        for _ in range(int(n)):
            r = L.try_op(data, base, cfg, rng, 1.0, base_map, category=cat, tries=30)
            if r is not None:
                data, info = r
                ops.append(info)
    for _ in range(n_extra_ops):
        r = L.try_op(data, base, cfg, rng, 1.0, base_map, category=str(rng.choice(cats, p=pw / pw.sum())), tries=30)
        if r is not None:
            data, info = r
            ops.append(info)
    L.refresh_metadata(data)
    fsc = L.Scene(data, cfg, base.osm_polys)
    nv = L.new_violations(fsc, base, base_map)
    nv += [v for v in L.check(fsc, base) if v[0] == "tree_pathway_overlap" and v not in nv]   # generations: never inherited
    if nv:
        _fail("final check: " + ", ".join(sorted({t for t, _, _ in nv})))
        return None
    return data, ops
