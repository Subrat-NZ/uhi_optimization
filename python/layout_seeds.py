#!/usr/bin/env python3
"""
layout_seeds.py  --  seed individuals, version 3.

    cool_wind_catcher_all   HEAT-case wind.  A COOL WIND CATCHER (pond + curved tree rows, jsons/cool_wind_catcher.json) is
                            hung on EVERY building and aligned to the wind: pond and trees stand at the building's
                            downwind face, on the side of the wind that has room.  The pond area of the layout (the
                            allowed band 1.0 - 1.1 x A0) is shared between the buildings in proportion to their footprints.
    wind_blocker_edge       COLD-case wind.  A BELT of staggered tree rows (built from jsons/tree_wind_breaker.json) along
                            the COMPLETE windward edge or edges of the site, so that the wind is blocked where it enters.
    both                    DENSE trees on BOTH sides of every pathway / road (crown radius 2.5 m, centre-to-centre
                            >= 1.1 x the crown diameters, as close as the tree-count limit allows).

Every individual obeys all the rules of layout_ops (inside the site, clear of buildings / ponds / pathways, canopy gap,
crown radius, height band of the location, tree-count limit, pond / lawn totals, pond patches <= number of buildings).
The tree-count limit is shared out: pathway trees first (seeds.path_share, default 45 % of the limit), then the catchers / the belt
(seeds.struct_share, default 45 %, plus whatever the pathways leave unused); base trees fill what is left and give way where they are in
the way.  A pattern that does not fit is thinned (smaller number, same shape) - it is never allowed to break a rule.
"""

import copy
import math
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import layout_ops as L
import layout_templates as T

EPS = T.EPS

# ---- tree-count budget (shares of the limit  floor(max_count_factor x n0) + max_new_trees)
PATH_SHARE = 0.45
STRUCT_SHARE = 0.45
# ---- dense pathway planting
PATH_R = 2.5                      # crown radius of a pathway tree (m)
PATH_GAP_TO_EDGE = 0.5            # crown edge to pathway edge (m)  (pathway_planting.canopy_gap_to_path_m when present)
PATH_SLIDE_M = (0.0, 1.0, -1.0, 2.0, -2.0)
PATH_HN = 0.5                     # height class: 0 = lowest allowed height of the location, 1 = highest
# ---- cool wind catcher on every building
CATCH_SHARE_FLOOR = 0.012        # each pond's area is its target +- this share (so the total stays inside the band)
CATCH_MIN_TREES = 3               # fewest trees a catcher may be left with
# (share of the pond target, share of the template pond that buildings / roads / the site edge may cut off  -or-  "reshape": the pond keeps its place at the building,
# but its shape adapts to the room there is)
CATCH_ATTEMPTS = ((1.0, 0.2), (1.0, 0.45), (1.0, "reshape"), (0.7, "reshape"), (0.5, "reshape"), (0.35, "reshape"), (0.22, "reshape"), (0.12, "reshape"))
RESHAPE_MIN_START = 0.15          # a reshaped pond starts from at least this share of the target area of free template pond
CATCH_ATTEMPTS_SIMPLE = ((1.0, "reshape"), (0.5, "reshape"), (0.22, "reshape"))     # cool_mode "simple": the pond target of a building, then smaller ones
CATCH_ROUNDS = 5                  # passes to bring the pond areas to the middle of the band
CATCH_REGION_M = 90.0             # a catcher is searched within this distance of its building
# ---- wind blocker belt
BELT_WINDWARD_MIN = 0.38          # a site edge is windward when -(outward normal . flow) >= this (about 68 deg off head-on)
BELT_CONFIGS = ((3.0, 3), (4.0, 3), (3.0, 2), (4.0, 2), (5.0, 2), (6.0, 2), (6.0, 1))     # (largest crown radius m, rows)
BELT_FIRST_ROW_GAP = 0.5          # crown edge to the site boundary (m)
BELT_COVER_TARGET = 0.80          # the belt should shade at least this share of the windward edge, else the pathways give up tree budget (to seeds.path_share_min)
BELT_HN = 1.0                     # belt trees are as tall as their location allows


# ============================================================================= helpers
class Occ:
    """Trees that already stand (x, y, crown radius)."""

    def __init__(self):
        self.x, self.y, self.R = [], [], []

    def add(self, x, y, R):
        self.x.append(float(x)); self.y.append(float(y)); self.R.append(float(R))

    def arrays(self):
        return np.array(self.x, float), np.array(self.y, float), np.array(self.R, float)

    def free_of(self, P, R, gap):
        """bool array: tree centres P (n,2) with crown radii R (n,) keep the canopy gap to every standing tree."""
        P, R = np.atleast_2d(P), np.atleast_1d(R)
        if not self.x:
            return np.ones(len(P), bool)
        x, y, r = self.arrays()
        d = np.hypot(P[:, None, 0] - x[None, :], P[:, None, 1] - y[None, :])
        return (d >= gap * (R[:, None] + r[None, :]) + EPS).all(axis=1)


class FreeCache:
    """Region where the centre of a tree of crown radius r may stand (layout_templates._free_region), per radius."""

    def __init__(self, sc, cfg, ponds):
        self.sc, self.cfg, self.ponds, self.c = sc, cfg, ponds, {}

    def __call__(self, r):
        k = round(float(r), 2)
        if k not in self.c:
            self.c[k] = T._free_region(self.sc, self.cfg, k, ponds=self.ponds)
        return self.c[k]


SEED_DEFAULTS = {"catcher_style": "arc", "air_corridor": True, "path_share": PATH_SHARE, "struct_share": STRUCT_SHARE, "path_share_min": 0.30, "path_crown_radius_m": PATH_R,
                 "belt_cover_target": BELT_COVER_TARGET,
                 # "simple" (default): individual_1 = the base layout WITHOUT its ponds, lawns and trees (buildings and pathways stay), one cool wind catcher
                 # on every building aligned to the heat-case wind, one pose per building, no search and no pond-area fitting.  "search": the older
                 # pose / pond-area search with dense pathway planting (slow on a site with many buildings).
                 "cool_mode": "simple"}


def sd(cfg, key):
    """Setting of the seed individuals: constraints.json "seeds" block, else the default above."""
    return (cfg.get("seeds") or {}).get(key, SEED_DEFAULTS[key])


def budgets(cfg, cap, has_path, path_share=None):
    """Tree budgets: pathway trees and the structure (catchers / belt).  The two add up to seeds.path_share + seeds.struct_share of the
    limit (default 90 %); base trees get the rest."""
    ps0, ss = float(sd(cfg, "path_share")), float(sd(cfg, "struct_share"))
    ps = ps0 if path_share is None else path_share
    if not has_path:
        return dict(path=0, struct=int(math.floor((ps0 + ss) * cap)))
    return dict(path=int(math.floor(ps * cap)), struct=int(math.floor((ps0 + ss - ps) * cap)))


def _next_id(base_data, data):
    ids = [int(str(t["id"]).split()[-1]) for t in list(base_data["trees"]) + list(data["trees"])
           if str(t["id"]).split()[-1].isdigit()]
    return max(ids + [0])


def _closed_fill(discs, extra=None, k=1.5):
    """Obstacle made of tree discs (and `extra` geometry): neighbouring discs are merged (closing by k m) and enclosed holes are filled.  A pond
    outline cannot have holes (a patch is one ring), so ponds are kept out of the whole planted stretch, not just off each crown."""
    parts = list(discs) + ([extra] if extra is not None else [])
    if not parts:
        return None
    u = unary_union(parts).buffer(k, quad_segs=4).buffer(-k, quad_segs=4)
    polys = [u] if u.geom_type == "Polygon" else [g for g in getattr(u, "geoms", []) if g.geom_type == "Polygon"]
    return unary_union([Polygon(g.exterior) for g in polys]) if polys else None


def _disc(x, y, r):
    return Point(float(x), float(y)).buffer(float(r), 16)


# ============================================================================= dense pathway trees
def plant_pathway_dense(sc, cfg, occ, budget, free, base_trees):
    """Trees on BOTH sides of every pathway.  Slots lie on the pathway outline (every edge of every road) at a spacing that starts at the
    closest the canopy-gap rule allows and grows until the number of trees fits `budget`.  Trees are appended to `occ`.
    Returns (list of dict(x, y, R, rec, prio, kind), info dict)."""
    import layout_reflow as RF
    tc = cfg["tree"]
    gap = tc["canopy_gap_factor"]
    if sc.pathway is None or budget <= 0 or not cfg["tree"]["pathway_planting"]["enabled"]:
        return [], dict(enabled=False)
    R = float(np.clip(sd(cfg, "path_crown_radius_m"), *tc["radius_range_m"]))
    off = R + float(cfg["tree"]["pathway_planting"].get("canopy_gap_to_path_m", PATH_GAP_TO_EDGE))
    rings = [r for poly in RF._polys(sc.pathway) for r in [poly.exterior, *poly.interiors] if r.length > 2.0]
    total = float(sum(r.length for r in rings))
    if total <= 0:
        return [], dict(enabled=False)
    s_min = gap * 2 * R + 0.1

    def run(spacing):
        loc = Occ()
        loc.x, loc.y, loc.R = list(occ.x), list(occ.y), list(occ.R)
        out = []
        for ring in rings:
            n_slot = max(1, int(round(ring.length / spacing)))
            step = ring.length / n_slot
            for k in range(n_slot):
                done = False
                for slide in PATH_SLIDE_M:
                    c = RF._side_point(ring, k * step + slide, off, sc.pathway)
                    if c is None:
                        continue
                    p = np.array([c])
                    if (shapely.contains_xy(free(R), c[0], c[1]) and loc.free_of(p, np.array([R]), gap)[0]
                            and sc.pathway.distance(Point(*c)) >= R + EPS):
                        loc.add(c[0], c[1], R)
                        out.append((float(c[0]), float(c[1])))
                        done = True
                        break
        return out

    spacing = s_min
    pts = run(spacing)
    for _ in range(12):
        if len(pts) <= budget:
            break
        spacing *= max(1.05, 1.02 * len(pts) / budget)
        pts = run(spacing)
    if len(pts) > budget:                                   # last resort: even thinning
        pts = [pts[i] for i in T._thin(budget, len(pts))]
    out = []
    for x, y in pts:
        occ.add(x, y, R)
        out.append(dict(x=x, y=y, R=R, rec=None, hn=PATH_HN, prio=0, kind="pathway"))
    info = dict(enabled=True, trees=len(out), crown_radius_m=R, spacing_asked_m=round(spacing, 2),
                min_spacing_m=round(s_min, 2), pathway_edge_length_m=round(total, 1),
                mean_spacing_m=round(total / max(len(out), 1), 2), budget=int(budget))
    return out, info


# ============================================================================= assembling the layout
def _tree_record(rec, cfg, base_trees, rng):
    if rec is not None:
        return copy.deepcopy(rec)
    return L.new_tree(cfg, base_trees, rng)


def assemble(data, base_data, base, cfg, base_map, sc, added, max_total, rng, notes, static=None, clear_zone=None):
    """Put the `added` trees (dicts x, y, R, hn, rec, prio, kind) into `data`:  base trees that touch them give way (then, if
    the tree-count limit needs it, the nearest base trees), heights follow the height band of each location, base trees that the
    new ponds / lawns put in breach of a rule are moved, and the result must have no NEW rule violation (an added tree that causes one is
    dropped, the lowest priority first).  Returns ((data, mine ids, removed base tree ids, kept added list), None) or (None, reason)."""
    import layout_reflow as RF
    tc = cfg["tree"]
    gap = tc["canopy_gap_factor"]
    nb = len(sc.tx)
    added = sorted(added, key=lambda a: a["prio"])
    if len(added) > max_total:
        added = added[:max_total]
    px = np.array([a["x"] for a in added]); py = np.array([a["y"] for a in added]); pr = np.array([a["R"] for a in added])
    gone = np.zeros(nb, bool)
    if nb and len(added):
        d = np.hypot(px[:, None] - sc.tx[None, :], py[:, None] - sc.ty[None, :])
        gone = (d < gap * (pr[:, None] + sc.tR[None, :]) + EPS).any(axis=0)
        over = nb - int(gone.sum()) + len(added) - max_total
        if over > 0:
            for i in np.argsort(d.min(axis=0)):
                if over <= 0:
                    break
                if not gone[i]:
                    gone[i] = True
                    over -= 1
    if clear_zone is not None and nb:                       # base trees whose crown reaches the clear air corridor of a catcher give way
        inz = shapely.distance(shapely.points(sc.tx, sc.ty), clear_zone) < sc.tR
        notes["base_trees_removed_from_air_corridor"] = int((inz & ~gone).sum())
        gone |= inz
    removed_ids = {sc.tree_ids[i] for i in np.where(gone)[0]}
    data["trees"] = [t for t in data["trees"] if t["id"] not in removed_ids]
    ball = sc.ball_union
    nid = _next_id(base_data, data)
    base_trees = list(base_data["trees"])
    new, prio_of, kind_of = [], {}, {}
    for a in added:
        t = _tree_record(a["rec"], cfg, base_trees, rng)
        nid += 1
        t["id"] = f"Tree {nid}"
        t["position"] = {"x": a["x"], "y": a["y"], "z": t.get("position", {}).get("z", 0)}
        h = T._band_height(cfg, a["hn"], a["x"], a["y"], ball, sc.pathway)
        L.set_tree_geometry(t, a["R"], h, tc)
        T._sync_vegetation(t, a["R"], h)
        new.append(t)
        prio_of[t["id"]] = a["prio"]
        kind_of[t["id"]] = a["kind"]
    data["trees"] = list(data["trees"]) + new
    mine = {t["id"] for t in new}
    moved = RF.fix_trees(data, base, cfg, rng)
    if moved:
        notes["trees_moved_off_rule_breaks"] = len([i for i in moved if i not in mine])
    L.finalize_patches(data, cfg)
    _separate_lawns_from_ponds(data, cfg, static if static is not None else (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn")), 1.0)
    _fix_slits(data)
    topped = _topup_totals(data, base, cfg, sc, static if static is not None else (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn")))
    if topped:
        notes["area_topped_up"] = topped
    L.refresh_metadata(data)
    for _ in range(8):
        # road shade is mandatory for every individual: plant / enlarge / move trees (never the seed trees) until it holds
        notes["pathway_shade"] = RF.ensure_pathway_shade(data, base, cfg, rng, protect=mine)
        fsc = L.Scene(data, cfg, base.osm_polys)
        nv = L.new_violations(fsc, base, base_map)
        if not nv:
            break
        drop = set()
        for typ, ids, _sev in nv:
            mm = [i for i in ids if i in mine and i not in drop]
            if typ == "tree_gap":
                mm = sorted(mm, key=lambda i: (prio_of.get(i, 9), int(i.split()[-1])))
                drop.update(mm[-1:])
            else:
                drop.update(mm)
        if not drop:
            return None, "the layout has new rule violations that are not seed trees: " + ", ".join(sorted({t for t, _, _ in nv}))
        data["trees"] = [t for t in data["trees"] if t["id"] not in drop]
        mine -= drop
        L.refresh_metadata(data)
    else:
        return None, "final rule check failed"
    counts = {}
    for i in mine:
        counts[kind_of[i]] = counts.get(kind_of[i], 0) + 1
    return (data, mine, removed_ids, counts), None


FIT_FRACTIONS = (0.45, 0.7, 0.9)      # where in the allowed band the reflow aims (share of the band width above its lower limit)


def fit_patches(data, base, cfg, rng, reserve, static, step):
    """Bring the pond / lawn totals back into their bands when they are out (L.fit_patch_area) and clean up the outlines (overlap slivers,
    short edges - re-discretizing shared pond / lawn boundaries pulls them back by up to 0.4 m, which can cost 100 m2 or more).  The reflow
    aims at 45 % of the band; if the clean-up pushes a total out of the band again, the attempt is repeated aiming higher (70 %, 90 %).
    `data` is edited in place.  Returns the list of categories that were rebuilt (empty: nothing was out of band)."""
    todo = L.patches_out_of_band(data, base, cfg)
    if not todo:
        return []
    last = None
    for frac in FIT_FRACTIONS:
        c2 = copy.deepcopy(cfg)
        for cat in todo:
            a0 = L.base_area0(base, cfg, cat)
            lo, hi = L.area_band(cfg, cat, a0)
            if hi - lo >= 0.03 * a0:
                c2[cat]["area_factor_range"][0] = (lo + (frac - 0.1) / 0.9 * (hi - lo)) / a0
        d = L.working_copy(data)
        L.fit_patch_area(d, base, c2, rng, pond_blockers=reserve)
        _separate_lawns_from_ponds(d, cfg, static, step)
        L.finalize_patches(d, cfg)
        _fix_slits(d)
        last = d
        if not L.patches_out_of_band(d, base, cfg):
            break
    for k in ("ponds", "grassAreas"):
        data[k] = last[k]
    return todo


def _topup_totals(data, base, cfg, sc, static, rounds=6):
    """Narrow bands (a lawn that sits exactly at its irrigation cap: band = A0 +- 0.1 %) are missed by the clean-up of the outlines by a few
    tens of m2.  Bring each total back to the middle of its band: the largest patch grows (buffer outwards, clipped to the free room) or shrinks.
    Ponds keep off the crowns that stand, lawns off the ponds.  Returns the categories that were corrected."""
    fixed = []
    margin = cfg["patch"].get("edge_margin_m", 0.25)
    for cat, key in (("pond", "ponds"), ("lawn", "grassAreas")):
        a0 = L.base_area0(base, cfg, cat)
        lo, hi = L.area_band(cfg, cat, a0)
        slack = 1e-3 * a0
        aim = 0.5 * (lo + hi)
        for _ in range(rounds):
            polys = [L._poly(q["coordinates"]).buffer(0) for q in data[key]]
            if not polys:
                break
            tot = sum(q.area for q in polys)
            if lo - 0.5 * slack <= tot <= hi + 0.5 * slack:
                break
            i = int(np.argmax([q.area for q in polys]))
            own = [q for j, q in enumerate(polys) if j != i]
            ko = [static[0] if cat == "pond" else static[1]]
            if cat == "lawn":
                ko += [L._poly(q["coordinates"]).buffer(0) for q in data["ponds"]]
            else:
                ko += [_disc(tr["position"]["x"], tr["position"]["y"], L._tree_R(tr) + EPS) for tr in data["trees"]]
            ko += own
            free = sc.site.buffer(-margin).difference(unary_union(ko))
            want = aim - sum(q.area for q in own)
            q0 = polys[i]
            if tot < lo:
                a_, b_ = 0.0, 6.0 * math.sqrt(max(want, 1.0))
                sgn = 1.0
            else:
                a_, b_ = 0.0, 0.5 * math.sqrt(q0.area)
                sgn = -1.0
            best = None
            for _k in range(26):
                d = 0.5 * (a_ + b_)
                q = T._largest((q0.buffer(sgn * d, quad_segs=6).intersection(free)) if sgn > 0 else q0.buffer(-d, quad_segs=6))
                ar = 0.0 if q is None else q.area
                best = q if q is not None else best
                if abs(ar - want) < 0.2 * slack:
                    best = q
                    break
                if (ar < want) == (sgn > 0):
                    a_ = d
                else:
                    b_ = d
            if best is None or best.is_empty:
                break
            L.patch_set_polygon(data[key][i], best, cat == "pond", 0)
            L.finalize_patches(data, cfg)
            _fix_slits(data)
            if cat not in fixed:
                fixed.append(cat)
            tot2 = sum(L._poly(q["coordinates"]).area for q in data[key])
            aim += (0.5 * (lo + hi) - tot2)                  # the clean-up took / gave a little: aim a bit off the middle next time
    return fixed


SLIT_M = 0.2           # an outline whose own 0.1 m buffer closes a gap (gap < 0.2 m) "encloses something through a hairline slit"


def _fix_slits(data):
    """A pond / lawn outline that wraps round an obstacle through a channel that discretization pulled in to under 0.2 m (the audit calls this a
    hairline slit; meshers cannot resolve it) is eroded by 0.11 m, which widens every such channel by 0.22 m.  Other outlines are left alone."""
    n = 0
    for key in ("ponds", "grassAreas"):
        for p in data[key]:
            poly = L._poly(p["coordinates"]).buffer(0)
            c = poly.buffer(0.5 * SLIT_M, join_style=2)
            if c.geom_type == "Polygon" and not c.interiors:
                continue
            q = T._largest(poly.buffer(-0.55 * SLIT_M, join_style=2))
            if q is not None and q.area > 0.5 * poly.area:
                L.patch_set_polygon(p, q, key == "ponds", 0)
                n += 1
    return n


def _separate_lawns_from_ponds(data, cfg, static, step):
    """Discretization and the pond / lawn reflow leave slivers (a fraction of a m2) where two patches, or a patch and a building, touch.  Ponds
    are cut back against the buildings / pathways and the ponds before them, lawns against all of these and the lawns before them.  Nothing is
    re-discretized (that would shave metres off every shared boundary and could push a total under its band); finalize_patches() takes care of
    vertices that end up closer than the discretization step."""
    static_p, static_l = static if isinstance(static, tuple) else (static, static)       # (pond keep-out, lawn keep-out): buildings grown by 2 m / 1 m
    taken = static_p                                            # ponds first (against buildings / pathways and each other), then the lawns
    pond_polys = []
    for p in data["ponds"]:
        q = L._poly(p["coordinates"]).buffer(0)
        if q.intersection(taken).area > 0:
            q2 = T._largest(q.difference(taken))
            if q2 is not None and q2.area >= 0.5 * q.area:
                L.patch_set_polygon(p, q2, True, 0)
                q = q2
        taken = unary_union([taken, q])
        pond_polys.append(q)
    taken = unary_union([static_l] + pond_polys)                # lawns: the (smaller) lawn clearance from the buildings, the ponds, the lawns before
    kept = []
    for g in data["grassAreas"]:
        q = L._poly(g["coordinates"]).buffer(0)
        if q.intersection(taken).area > 0:
            q = T._largest(q.difference(taken))
            if q is None or q.area < cfg["lawn"]["min_area_m2"]:
                continue
            L.patch_set_polygon(g, q, False, 0)
        kept.append(g)
        taken = unary_union([taken, L._poly(g["coordinates"]).buffer(0)])
    data["grassAreas"] = kept


# ============================================================================= cool wind catcher on every building
def _nudge(P, Rt, ok, free, ponds, gap, occ, nudge_m=T.COOL_NUDGE_M, step=0.5):
    """Tree.not ok -> nearest spot within nudge_m that is clear of the obstacles, the ponds, the trees that stand (occ) and the
    pattern's own trees.  Largest crowns first."""
    P, ok = P.copy(), ok.copy()
    angs = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    rings = np.arange(step, nudge_m + 1e-9, step)
    O = np.array([[r * math.cos(a), r * math.sin(a)] for r in rings for a in angs])
    standing = list(np.where(ok)[0])
    for i in np.argsort(-Rt):
        if ok[i]:
            continue
        C = P[i] + O
        m = shapely.contains_xy(free(Rt[i]), C[:, 0], C[:, 1])
        if ponds is not None:
            m &= shapely.distance(shapely.points(C[:, 0], C[:, 1]), ponds) >= Rt[i] + EPS
        if m.any():
            m &= occ.free_of(C, np.full(len(C), Rt[i]), gap)
        if standing and m.any():
            d = np.hypot(C[:, None, 0] - P[standing, 0][None, :], C[:, None, 1] - P[standing, 1][None, :])
            m &= (d >= gap * (Rt[i] + Rt[standing])[None, :] + EPS).all(axis=1)
        if m.any():
            P[i] = C[int(np.argmax(m))]
            ok[i] = True
            standing.append(i)
    return P, ok


def _reshape(tpl, mp, s0, region, lo, hi):
    """The pond's SHAPE adapts to the room there is: start from the part of the template pond (scale s0, pose mp) that is free, then grow it
    (buffer outwards, clipped to the free region) or shrink it (buffer inwards) until its area is inside [lo, hi].  The pond keeps its place at the
    building and as much of the template outline as the room allows.  Returns (polygon, s0, share of the pond that is not template shape) or None."""
    target, tol = 0.5 * (lo + hi), 0.01 * (hi - lo) + 1e-9
    g = Polygon(mp(tpl["pond_xy"], s0)).buffer(0)
    q0 = T._largest(g.intersection(region))
    if q0 is None or q0.area < RESHAPE_MIN_START * target:
        return None

    def shape(d):
        q = q0.buffer(d, quad_segs=6) if abs(d) > 1e-9 else q0
        return T._largest(q.intersection(region)) if not q.is_empty else None

    def ok(q):
        return q is not None and lo + tol <= q.area <= hi - tol

    q = q0
    if not ok(q0):
        if q0.area < target:
            a, b = 0.0, 4.0 * math.sqrt(target)
        else:
            a, b = -0.5 * math.sqrt(q0.area), 0.0
        q = None
        for _ in range(26):
            d = 0.5 * (a + b)
            qd = shape(d)
            ar = 0.0 if qd is None else qd.area
            if ok(qd):
                q = qd
                break
            if ar < target:                                  # the area grows with d: too small -> go outwards
                a = d
            else:
                b = d
        if q is None:
            return None
    q = T._largest(q.buffer(0))
    if q is None or q.area < 1.0:
        return None
    iou = q.intersection(g).area / max(q.union(g).area, 1e-9)
    return q, s0, 0.6 * (1.0 - iou)


def _pond_at(tpl, mp, s0, region, lo, hi, clip_max):
    """layout_templates._pond_at with the allowed share of cut-off area as a parameter; clip_max == "reshape": the shape adapts (see _reshape)."""
    if clip_max == "reshape":
        return _reshape(tpl, mp, s0, region, lo, hi)
    target, tol, s = 0.5 * (lo + hi), 0.01 * (hi - lo) + 1e-9, s0
    for _ in range(6):
        g = Polygon(mp(tpl["pond_xy"], s)).buffer(0)
        q = T._largest(g.intersection(region))
        if q is None or q.area < 1.0 or g.area <= 0:
            return None
        if lo + tol <= q.area <= hi - tol:
            loss = 1.0 - q.area / g.area
            return (q, s, loss) if loss <= clip_max else None
        s *= math.sqrt(target / q.area)
        if not 0.3 < s < 3.0:
            return None
    return None


def _select(idx, Rtpl, n):
    """n of the trees `idx`: the big-crown classes first, a class that does not fit completely is thinned evenly."""
    idx = np.asarray(idx, int)
    if len(idx) <= n:
        return idx
    cls = sorted(set(np.round(Rtpl[idx], 1)), reverse=True)
    out = []
    for c in cls:
        room = n - len(out)
        if room <= 0:
            break
        m = idx[np.isclose(np.round(Rtpl[idx], 1), c)]
        out.extend(m if len(m) <= room else m[T._thin(room, len(m))])
    return np.array(out, int)


# ============================================================================= arc-shaped cool wind catcher (built for each building)
# The catcher the stored template shows is one fixed picture.  The "arc" catcher is BUILT for every building from a few rules, so it adapts to the
# building, the wind and the room there is (frame: u along the wind, d = distance from the building's lateral face, away from the building):
#   * the POND is a long, leaf-shaped water body on the lateral side of the building, in the open area.  It starts upwind of the building (the
#     wind crosses it before it reaches the building) and its downwind end leans in towards the building, so that the cooled air is guided to it;
#   * the TREES stand in one or two rows on the pond's FAR side and round its DOWNWIND END, starting just upwind of the widest part of the pond and
#     hooking back towards the building.  The wind enters between the pond and the building, crosses the water, and the tree arc stops it from
#     leaving sideways / downwind, so the cooled air is held at the building.  Crowns are largest in the middle of the arc and smaller at its ends.
ARC_VARIANTS = ((6.0, 0.5, 2.6), (12.0, 0.3, 2.8), (25.0, 0.5, 3.2), (25.0, 0.2, 3.2), (45.0, 0.5, 3.4), (60.0, 0.4, 3.6))   # (angle of the pond axis to the wind (deg), downwind end of the pond past the building's downwind face as a share of the building length, pond length / widest width)
ARC_LADDER = ((2, 5.0), (2, 5.5), (2, 6.0))      # (rows, largest crown radius m): densest arc first; the first one inside the tree budget is used
ARC_R_MIN = 2.5                   # smallest crown radius at the ends of the arc (m)
ARC_ROW_SHRINK = 0.15             # an outer row's crowns are this much smaller than the row inside it
ARC_POND_GAP = 1.6                # crown edge to the pond edge (m)
ARC_POND_CLEAR = 2.6              # pond edge to the building's lateral face (m); a little more than the 2 m pond clearance
ARC_NORMAL_BUILDING = 0.0         # the arc covers the pond's outline where the outward normal points at most this much (cosine) towards the building ...
ARC_NORMAL_UPWIND = 0.0           # ... and where it points at most this much towards the wind (the upwind end of the pond stays open)
ARC_END_TRIM = 0.06               # share of the arc's length left bare at each end (the openings: the wind enters / the cooled air leaves there)
ARC_POND_SEPARATION_M = 4.0       # the pond of a later building keeps this far from the ponds before it (a passage for the wind)
ARC_OTHER_POND_CLEAR_M = 4.0      # crown edge to a pond of another building (m)
AIR_CORRIDOR_MARGIN_M = 18.0     # the clear air corridor is this much wider than the pond + building pair (m, each side)
ARC_K = (1.0,)
ARC_ROT_DEG = (0.0,)
ARC_ROT_FINE = (0.0,)
ARC_BOTH_ORDERS_MAX = 3          # up to this many buildings the placement is tried in two orders (largest first / smallest first)
ARC_SHIFT_U = (-6.0, 0.0, 6.0)
ARC_SHIFT_V = (0.0, 6.0)
ARC_WIDTH_MAX_M = 40.0            # widest part of the pond (m), unless the length limit makes it wider
ARC_LENGTH_MAX_M = 130.0          # longest pond (m)
ARC_CENTRE_FREE_M = 25.0          # distance from the building to the pond's centre that costs nothing; beyond it the pose is penalised
ARC_DIST_FREE_M = 12.0            # the pond may be this far from the building without penalty (distance between the outlines)


def _arc_pond(area, phi_deg, shift_u, Lb, aspect=3.2):
    """Leaf-shaped pond (template frame a, d) of the given area: its axis rises away from the building going upwind at phi_deg to the wind, its
    downwind end is at a = shift_u * Lb, its lowest point is ARC_POND_CLEAR from the building's lateral face (d = 0)."""
    phi = math.radians(phi_deg)
    e = np.array([math.cos(phi), -math.sin(phi)])           # upwind end -> downwind end (the axis leans in towards the building)
    nrm = np.array([math.sin(phi), math.cos(phi)])           # across the axis, away from the building
    k = 0.70                                                 # area of the leaf / (length x width)
    Wp = math.sqrt(area / (k * aspect))
    if Wp > ARC_WIDTH_MAX_M:                                  # a big pond gets longer, not wider ...
        Wp = ARC_WIDTH_MAX_M
    Lp = area / (k * Wp)
    if Lp > ARC_LENGTH_MAX_M:                                 # ... up to a length, then wider
        Lp = ARC_LENGTH_MAX_M
        Wp = area / (k * Lp)
    t = np.linspace(0.0, 1.0, 41)
    tw = t ** 0.756                                           # widest part at 40 % of the length from the upwind end
    p = 2.4
    g = np.clip(1.0 - np.abs(2 * tw - 1.0) ** p, 0.0, 1.0) ** (1.0 / p)
    c = np.outer(t * Lp, e)
    top = c + np.outer(0.5 * Wp * g, nrm)
    bot = c - np.outer(0.5 * Wp * g, nrm)
    xy = np.vstack([top, bot[::-1][1:-1]])
    poly = Polygon(xy).buffer(0)
    poly = orient(T._largest(poly), 1.0)
    xy = np.array(poly.exterior.coords)[:-1]
    xy = xy * math.sqrt(area / poly.area)                    # exact area (the leaf's area factor is only approximate)
    c = c * math.sqrt(area / poly.area)
    # the downwind end of the axis (t = 1) goes to a = shift_u * Lb;  the lowest point goes to d = ARC_POND_CLEAR
    xy = xy + np.array([shift_u * Lb - c[-1, 0], 0.0])
    xy[:, 1] += ARC_POND_CLEAR - xy[:, 1].min()
    return xy


def _smooth_periodic(a, w):
    k = np.ones(2 * w + 1) / (2 * w + 1)
    pad = np.concatenate([a[-w:], a, a[:w]], axis=0)
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(a.shape[1])], axis=1)


def _arc_trees(pond_xy, rows, Rmax, gap_eff, lo_r, hi_r, toward=(0.0, -1.0), upwind=(-1.0, 0.0), valid=None, quant=None, ds=0.25):
    """Tree arc round the far side and the downwind end of a pond (see above).  Returns (xy (n,2), R (n,), row (n,)) in the order of the arc.
    toward: direction from the pond to the building, or the building polygon (then the direction is taken point by point - the arc stays on the far
    side of a pond that wraps round a building);  upwind: direction the wind comes from;  valid(q, R) -> bool: is a crown of radius R allowed at q
    (None: always);  quant: crown radii are rounded to this step (m)."""
    poly = orient(Polygon(pond_xy).buffer(0), 1.0)
    ring = poly.exterior
    Lr = ring.length
    s_grid = np.arange(0.0, Lr, ds)
    pts = np.array([ring.interpolate(s).coords[0] for s in s_grid])
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tang = _smooth_periodic(tang, max(2, int(2.0 / ds)))
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
    nor = np.stack([tang[:, 1], -tang[:, 0]], axis=1)         # outward (counter-clockwise ring)
    if hasattr(toward, "geom_type"):
        ln = shapely.get_coordinates(shapely.shortest_line(shapely.points(pts[:, 0], pts[:, 1]), toward)).reshape(-1, 2, 2)
        tv = ln[:, 1, :] - ln[:, 0, :]
        tv /= np.maximum(np.hypot(tv[:, 0], tv[:, 1]), 1e-9)[:, None]
    else:
        tv = np.tile(np.asarray(toward, float), (len(pts), 1))
    up = np.asarray(upwind, float)
    keep = ((nor * tv).sum(1) < ARC_NORMAL_BUILDING) & ((nor @ up) < ARC_NORMAL_UPWIND)
    if keep.sum() < 8:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
    # longest circular run of kept points
    n = len(keep)
    best, run0, run = (0, 0), None, 0
    kk = np.concatenate([keep, keep])
    for i, v in enumerate(kk):
        if v:
            if run == 0:
                run0 = i
            run += 1
            if run > best[0] and run <= n:
                best = (run, run0)
        else:
            run = 0
    cnt, start = best
    idx = (start + np.arange(cnt)) % n
    cut_ = int(ARC_END_TRIM * len(idx))
    if cut_ and len(idx) > 2 * cut_ + 8:
        idx = idx[cut_:len(idx) - cut_]
    P0, N0 = pts[idx], nor[idx]
    lo = max(lo_r, 1.0)
    out_xy, out_R, out_row = [], [], []
    placed_xy, placed_R = np.zeros((0, 2)), np.zeros(0)
    for row in range(rows):
        shrink = (1.0 - ARC_ROW_SHRINK) ** row
        i = 0
        while i < len(P0):
            t = i / max(len(P0) - 1, 1)
            Rc = (ARC_R_MIN + (Rmax - ARC_R_MIN) * math.sin(math.pi * t) ** 0.8) * shrink
            if quant:
                Rc = math.floor(Rc / quant) * quant
            Rc = float(np.clip(Rc, lo, hi_r))
            e0 = Rc + ARC_POND_GAP
            ok_pos = None
            for extra in ((0.0,) if row == 0 else np.arange(0.0, 3.0 * Rmax * (row + 1) + 1e-9, 0.5)):
                q = P0[i] + N0[i] * (e0 + extra)
                if valid is not None and not valid(q, Rc):
                    continue
                if len(placed_R):
                    d = np.hypot(placed_xy[:, 0] - q[0], placed_xy[:, 1] - q[1])
                    if (d >= gap_eff * (placed_R + Rc)).all():
                        ok_pos = q
                        break
                else:
                    ok_pos = q
                    break
            if ok_pos is None:
                i += 2
                continue
            placed_xy = np.vstack([placed_xy, ok_pos])
            placed_R = np.append(placed_R, Rc)
            out_xy.append(ok_pos); out_R.append(Rc); out_row.append(row)
            i += max(1, int(1.5 * Rc / ds))
    return np.array(out_xy).reshape(-1, 2), np.array(out_R), np.array(out_row, int)


def arc_on_pond(pond, bpoly, f, free, occ, ponds_u, gap, lo_r, hi_r, n_target, rec, others=None):
    """The tree arc of a pond that has been placed (its real outline, so the arc follows whatever shape the pond got): the densest step of ARC_LADDER
    that stays inside n_target trees, crowns only where the rules allow (free region, canopy gap to the trees that stand, clear of the ponds).
    Returns (P, Rt, hn, row) or None."""
    def valid(q, R):
        if not shapely.contains_xy(free(R), q[0], q[1]):
            return False
        if ponds_u is not None and float(shapely.distance(shapely.points(q[0], q[1]), ponds_u)) < R + EPS:
            return False
        if others is not None and float(shapely.distance(shapely.points(q[0], q[1]), others)) < R + ARC_OTHER_POND_CLEAR_M:
            return False                                             # the passage between two ponds stays open
        return bool(occ.free_of(np.array([q]), np.array([R]), gap)[0])
    pxy = np.array(pond.exterior.coords)[:-1]
    got = None
    for rows, Rmax in ARC_LADDER:
        xy, R, row = _arc_trees(pxy, rows, min(Rmax, hi_r), gap + 0.03, lo_r, hi_r, toward=bpoly, upwind=(-f[0], -f[1]), valid=valid, quant=0.25)
        got = (xy, R, row)
        if 0 < len(R) <= n_target:
            break
    xy, R, row = got
    if len(R) > n_target:
        keep = _thin_to(row, n_target)
        xy, R, row = xy[keep], R[keep], row[keep]
    if len(R) < CATCH_MIN_TREES:
        return None
    hn = 0.5 + 0.5 * (R - R.min()) / max(R.max() - R.min(), 1e-9)
    return xy, R, hn, row


def _thin_to(row, n):
    """Indices of the n trees that stay when an arc of trees (row number per tree, in the order of the arc) is over budget: the outer rows are
    thinned first (evenly), then the inner one."""
    idx = np.arange(len(row))
    excess = len(row) - n
    drop = np.zeros(len(row), bool)
    for r in sorted(set(row.tolist()), reverse=True):
        if excess <= 0:
            break
        m = idx[row == r]
        k = min(excess, len(m))
        if r == 0 and k < len(m):
            drop[m[T._thin(k, len(m))]] = True
        else:
            drop[m[np.unique(np.round(np.linspace(0, len(m) - 1, k)).astype(int))] if k < len(m) else m] = True
        excess -= k
    return idx[~drop]


def arc_variants(base_tpl, area, n_target, Lb, gap, lo_r, hi_r):
    """Template-like dicts (see layout_templates.read_cool_template) of the arc catcher, one per shape variant (ARC_VARIANTS), for a pond of about
    `area` m2 at a building of length Lb along the wind.  The tree arc is the densest one of ARC_LADDER that stays inside n_target trees."""
    out = []
    for phi, shift, asp in ARC_VARIANTS:
        pxy = _arc_pond(area, phi, shift, Lb, asp)
        got = None
        for rows, Rmax in ARC_LADDER:
            xy, R, row = _arc_trees(pxy, rows, min(Rmax, hi_r), gap + 0.03, lo_r, hi_r)
            got = (xy, R, row, rows, Rmax)
            if 0 < len(R) <= n_target:
                break
        xy, R, row, rows, Rmax = got
        if len(R) > n_target:                                  # even the sparsest ladder step is over budget: thin the outer row, then the arc
            keep = _thin_to(row, n_target)
            xy, R, row = xy[keep], R[keep], row[keep]
        if len(R) < CATCH_MIN_TREES:
            continue
        D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(D, np.inf)
        ratio = float((D / (1.1 * (R[:, None] + R[None, :]))).min())
        hn = 0.5 + 0.5 * (R - R.min()) / max(R.max() - R.min(), 1e-9)
        tv = dict(base_tpl)
        tv.update(side=1.0, anchor=np.array([0.0, 0.0]), pond_xy=pxy, pond_area=float(Polygon(pxy).area),
                  trees=[base_tpl["trees"][0]] * len(R), xy=xy, R=R, H=R, hn=hn, pair_ratio=ratio,
                  arc=dict(angle_deg=phi, downwind_end_share=shift, aspect=asp, rows=int(rows), crown_radius_max_m=float(R.max()), trees=int(len(R))))
        out.append(tv)
    return out


GEOMETRY_RULES = ("tree_building_clearance", "tree_outside_site", "tree_pathway_overlap", "tree_pond_overlap", "tree_gap",
                  "pond_building_overlap", "pond_building_clearance", "pond_pathway_overlap", "pond_outside_site", "patch_overlap", "pond_slit")


def make_simple_catchers_individual(base_data, base, cfg, base_map, tpl, wind):
    """individual_1 in "simple" mode: take the base layout, remove every pond, lawn and tree INSIDE the site (only buildings and pathways stay), then
    hang ONE cool wind catcher (pond + tree arc) on EVERY building, turned to the wind of the heat case.  One pose per building (the template's own:
    no shift, no rotation; the mirror image - other side of the wind - only if more of the catcher then fits), pond area shared in proportion to the
    footprints, NO search for poses or for the pond total.
    Geometry is never broken: the pond is cut to the free ground (inside the site, 2 m off every building, off the pathways, off the earlier ponds and
    trees), trees keep to the same rules as the checker (crown inside the site, clear of buildings / pathways / ponds, canopy gap) and are moved up to
    5 m or dropped, and a final check removes whatever still breaks one (GEOMETRY_RULES).  Not enforced: the pond total area, the number of ponds, the
    tree count, the road shade, tree heights.  Returns ((data, [info]), None)."""
    import layout_reflow as RF
    tc = cfg["tree"]
    rng = np.random.default_rng(0)
    data = L.working_copy(base_data)
    for k_ in ("trees", "ponds", "grassAreas"):
        data[k_] = []
    L.refresh_metadata(data)
    sc = L.Scene(data, cfg, base.osm_polys)
    nbld = len(sc.group_polys)
    if not nbld:
        return None, "the layout has no building"
    f, n = T._frame(wind["flow"])
    gap = tc["canopy_gap_factor"]
    lo_r, hi_r = tc["radius_range_m"]
    pc = cfg["pond"]
    a0 = L.base_area0(base, cfg, "pond")
    lo_a, hi_a = L.area_band(cfg, "pond", a0)
    mid = 0.5 * (lo_a + hi_a)                               # total pond area wanted (the middle of the allowed band), shared by footprint
    areas = np.array([p.area for p in sc.group_polys])
    style = str(sd(cfg, "catcher_style"))
    margin = cfg["patch"].get("edge_margin_m", 0.25)
    step = cfg["patch"].get("discretization_m", 1.0)
    static = RF.static_blockers(sc, "pond")                 # buildings grown by the pond clearance (2 m) and the pathways
    region_cur = sc.site.buffer(-margin).difference(static.buffer(0.1))
    free = FreeCache(sc, cfg, [])                           # where the centre of a tree of radius r may stand
    occ = Occ()
    ponds, trees_added, per_b = [], [], []
    n_flipped = n_cut = n_moved = n_dropped = 0

    def tree_ok(P, Rt, pond, pu):
        ok = np.zeros(len(Rt), bool)
        for rv in np.unique(np.round(Rt, 2)):
            sel = np.where(np.isclose(np.round(Rt, 2), rv))[0]
            ok[sel] = shapely.contains_xy(free(rv), P[sel, 0], P[sel, 1])
        pts = shapely.points(P[:, 0], P[:, 1])
        ok &= shapely.distance(pts, pond) >= Rt + EPS
        if pu is not None:
            ok &= shapely.distance(pts, pu) >= Rt + EPS
        ok &= occ.free_of(P, Rt, gap)
        return ok

    for bi in np.argsort(-areas):
        poly = sc.group_polys[bi]
        frame = T._building_frame(poly, f, n)
        a_b = float(max(mid * areas[bi] / areas.sum(), 1.5 * pc["min_area_m2"]))
        tv = tpl
        if style == "arc":
            vs = arc_variants(tpl, a_b, 10 ** 6, frame[1] - frame[0], gap, lo_r, hi_r)
            tv = vs[0] if vs else tpl
        rho = min(1.0, 0.995 * tv["pair_ratio"] * 1.1 / gap)
        s_p = math.sqrt(a_b / tv["pond_area"])
        Rt = np.clip(rho * s_p * tv["R"], lo_r, hi_r)
        pu = unary_union(ponds) if ponds else None
        if pu is not None:
            shapely.prepare(pu)
        tried = []
        for mirror in (1.0, -1.0):                          # the template side first; the other side of the wind only if more fits there
            mp = T._pose(tv, f, n, frame, mirror, 0.0, 0.0, 0.0)
            g = Polygon(mp(tv["pond_xy"], s_p)).buffer(0)
            q = T._largest(g.intersection(region_cur))
            if q is not None and q.interiors:
                q = Polygon(q.exterior)                     # a pond is one ring: no holes
            if q is not None and L.has_slit(q):
                q = T._largest(L.unslit(q))
            if q is None or q.area < pc["min_area_m2"] or g.area <= 0:
                continue
            shapely.prepare(q)
            P_m = mp(tv["xy"], s_p)
            ok_m = tree_ok(P_m, Rt, q, pu)
            tried.append((q.area / g.area + float(ok_m.mean()), mirror, g, q, P_m, ok_m))
            if tried[-1][0] >= 2.0 - 1e-9:
                break
        if not tried:
            per_b.append({"building": sc.group_names[bi], "pond_area_m2": 0.0, "trees_planned": 0, "note": "no free ground for its pond on either side"})
            continue
        best_t = tried[0]
        for t_ in tried[1:]:
            if t_[0] > best_t[0] + 1e-9:
                best_t = t_
        _, mirror, g, pond, P, ok = best_t
        n_flipped += int(mirror < 0)
        cut = pond.area < g.area - 1e-6
        n_cut += int(cut)
        ponds_u = pond if pu is None else unary_union([pond, pu])
        shapely.prepare(ponds_u)
        k0 = int(ok.sum())
        P, ok = _nudge(P, Rt, ok, free, ponds_u, gap, occ)
        n_moved += int(ok.sum()) - k0
        n_dropped += int((~ok).sum())
        ponds.append(pond)
        region_cur = region_cur.difference(pond.buffer(0.3))
        kept = np.where(ok)[0]
        for i in kept:
            occ.add(P[i, 0], P[i, 1], Rt[i])
            trees_added.append(dict(x=float(P[i, 0]), y=float(P[i, 1]), R=float(Rt[i]), rec=tv["trees"][i], hn=float(tv["hn"][i])))
        if len(kept):                                       # the ponds of the buildings after this one keep off its trees
            fill = _closed_fill([_disc(P[i, 0], P[i, 1], Rt[i] + 0.3) for i in kept])
            if fill is not None:
                region_cur = region_cur.difference(fill)
        per_b.append({"building": sc.group_names[bi], "size_m": [round(frame[1] - frame[0], 1), round(frame[3] - frame[2], 1)],
                      "style": "arc" if tv is not tpl else "template", "side": "mirrored (other side of the wind)" if mirror < 0 else "template side",
                      "pond_area_m2": round(float(pond.area), 1), "pond_cut_to_free_ground": cut, "pond_share_kept": round(float(pond.area / g.area), 3),
                      "trees_planned": int(len(kept)), "trees_dropped": int((~ok).sum())})
    if not ponds:
        return None, "no pond could be drawn"
    polys = L.discretize_patches(ponds, step, static)
    recs = []
    for k, poly in enumerate(polys, 1):
        rec = copy.deepcopy(tpl["pond"])
        rec.setdefault("dimensions", {}).setdefault("depth", pc["default_depth_m"])
        rec["id"] = f"pond_{k}"
        L.patch_set_polygon(rec, poly, True, 0)
        recs.append(rec)
    data["ponds"] = recs
    nid = _next_id(base_data, data)
    base_trees = list(base_data["trees"])
    ball = sc.ball_union
    new = []
    for a in trees_added:
        t = _tree_record(a["rec"], cfg, base_trees, rng)
        nid += 1
        t["id"] = f"Tree {nid}"
        t["position"] = {"x": a["x"], "y": a["y"], "z": t.get("position", {}).get("z", 0)}
        h = T._band_height(cfg, a["hn"], a["x"], a["y"], ball, sc.pathway)
        L.set_tree_geometry(t, a["R"], h, tc)
        T._sync_vegetation(t, a["R"], h)
        new.append(t)
    data["trees"] = new
    L.refresh_metadata(data)
    # final guarantee: nothing may break a geometry rule.  Whatever still does (rounding of the pond outlines ...) is removed: trees first, then ponds.
    removed_trees = removed_ponds = 0
    left = {}
    for _ in range(12):
        fsc = L.Scene(data, cfg, base.osm_polys)
        bad = [(t_, ids) for t_, ids, _sev in L.check(fsc, base) if t_ in GEOMETRY_RULES]
        if not bad:
            break
        drop_t, drop_p = set(), set()
        pid = set(fsc.pond_ids)
        for t_, ids in bad:
            if t_.startswith("tree_"):
                drop_t.add(max(ids[:2], key=lambda i: int(str(i).split()[-1])) if t_ == "tree_gap" else ids[0])
            else:
                hit = [i for i in ids if i in pid]
                if hit:
                    drop_p.add(hit[-1] if t_ == "patch_overlap" else hit[0])
        if drop_p:
            data["ponds"] = [p_ for p_ in data["ponds"] if p_["id"] not in drop_p]
            removed_ponds += len(drop_p)
        if drop_t or not drop_p:
            data["trees"] = [t_ for t_ in data["trees"] if t_["id"] not in drop_t]
            removed_trees += len(drop_t)
        L.refresh_metadata(data)
        if not drop_t and not drop_p:
            left = {t_: 1 for t_, _ in bad}
            break
    fsc = L.Scene(data, cfg, base.osm_polys)
    allv = {}
    for t_, ids, _sev in L.check(fsc, base):
        allv[t_] = allv.get(t_, 0) + 1
    geom_left = {k_: v for k_, v in allv.items() if k_ in GEOMETRY_RULES}
    others = {k_: v for k_, v in allv.items() if k_ not in GEOMETRY_RULES}
    pond_total = float(sum(L._poly(p["coordinates"]).area for p in data["ponds"]))
    n_tr = len(data["trees"])
    info = {"op": "template", "kind": "cool_wind_catcher_all", "mode": "simple", "template": Path(tpl["path"]).name, "season": "heat",
            "wind_from_deg_model": round(wind["from_deg"], 1),
            "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
            "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
            "buildings": nbld, "buildings_with_catcher": len(data["ponds"]), "buildings_without_catcher": {p_["building"]: p_["note"] for p_ in per_b if p_.get("note")},
            "catchers": per_b, "scale": 1.0,
            "pond_area_m2": round(pond_total, 1), "pond_area_allowed_m2": [round(lo_a, 1), round(hi_a, 1)],
            "ponds": len(data["ponds"]), "trees_placed": n_tr, "trees_catcher": n_tr, "trees_pathway": 0,
            "side_flipped_buildings": n_flipped, "ponds_cut_to_free_ground": n_cut, "trees_moved": n_moved, "trees_dropped": n_dropped + removed_trees,
            "ponds_removed_by_final_check": removed_ponds,
            "pathway_planting": {"enabled": False, "skipped": "simple mode: only the catchers are planted"},
            "base_trees_replaced": len(base_data.get("trees", [])),
            "geometry_rules_broken": geom_left, "rule_violations_of_the_result": others,
            "template_tree_ids": sorted((t["id"] for t in data["trees"]), key=lambda i: int(i.split()[-1])),
            "distance_from_upwind_site_edge_m": None, "close_to_inlet": None}
    if geom_left:
        return None, "the catchers still break geometry rules: " + ", ".join(f"{k_} x{v}" for k_, v in geom_left.items())
    return (data, [info]), None


def make_cool_all_individual(base_data, base, cfg, base_map, tpl, wind, seed=0):
    """(layout, ops) or (None, reason): a cool wind catcher on every building, dense trees on both sides of the pathways."""
    import layout_reflow as RF
    if str(sd(cfg, "cool_mode")) == "simple":
        return make_simple_catchers_individual(base_data, base, cfg, base_map, tpl, wind)
    rng = np.random.default_rng(seed)
    tc, pc = cfg["tree"], cfg["pond"]
    simple = False
    data = L.working_copy(base_data)
    if simple:                                             # only buildings and pathways stay: no ponds, lawns or trees (also none on roofs)
        for k_ in ("trees", "ponds", "grassAreas"):
            data[k_] = []
        L.refresh_metadata(data)
    sc = L.Scene(data, cfg, base.osm_polys)
    nbld = len(sc.group_polys)
    if not nbld:
        return None, "the layout has no building"
    f, n = T._frame(wind["flow"])
    a0 = L.base_area0(base, cfg, "pond")
    lo_a, hi_a = L.area_band(cfg, "pond", a0)
    mid = 0.5 * (lo_a + hi_a)
    gap = tc["canopy_gap_factor"]
    lo_r, hi_r = tc["radius_range_m"]
    cap = L.max_tree_count(cfg, L.tree_count0(base, cfg), sc.pathway is not None)
    bud = budgets(cfg, cap, sc.pathway is not None)
    pond_limit = L.patch_limit(cfg, "pond", nbld)
    static = (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn"))     # buildings grown by 2 m (ponds) / 1 m (lawns), pathway
    step = cfg["patch"].get("discretization_m", 1.0)
    margin = cfg["patch"].get("edge_margin_m", 0.25)
    free = FreeCache(sc, cfg, [])                              # the base ponds are replaced
    occ = Occ()

    # ---- 1. dense trees on both sides of the pathways
    if simple:
        ptrees, pinfo = [], {"skipped": "cool_mode simple: no pathway planting"}
        struct_budget = int(cap)
    else:
        ptrees, pinfo = plant_pathway_dense(sc, cfg, occ, bud["path"], free, base_data["trees"])
        struct_budget = bud["struct"] + max(0, bud["path"] - len(ptrees))
    path_discs = [_disc(a["x"], a["y"], a["R"] + 0.3) for a in ptrees]
    region_all = sc.site.buffer(-margin).difference(static[0].buffer(0.1))
    if path_discs:
        region_all = region_all.difference(_closed_fill(path_discs, sc.pathway.buffer(0.1) if sc.pathway is not None else None))

    # ---- 2. a catcher per building (largest building first), pond areas shared in proportion to the footprints
    styles = ("arc", "template") if str(sd(cfg, "catcher_style")) == "arc" else ("template",)
    areas = np.array([p.area for p in sc.group_polys])
    names = sc.group_names
    order = list(np.argsort(-areas))
    failed = {}

    def lims(tv):
        """per template: (rho, s_t_min, s_t_max, N, solidity of the pond)"""
        rho_ = min(1.0, 0.995 * tv["pair_ratio"] * 1.1 / gap)
        pp = Polygon(tv["pond_xy"]).buffer(0)
        return (rho_, lo_r / (float(tv["R"].min()) * rho_), hi_r / (float(tv["R"].max()) * rho_), len(tv["xy"]),
                pp.area / pp.convex_hull.area)

    def one_building(bi, lo_b, hi_b, n_target, ponds, region_cur, clip_max, style="template"):
        """Best catcher of building bi: dict or None.  ponds: pond polygons already placed.  style "template": the stored template; "arc": the
        arc catcher built for this building (pond beside it, tree arc round the pond's far side and downwind end)."""
        poly = sc.group_polys[bi]
        local = region_cur.intersection(poly.buffer(CATCH_REGION_M))
        if local.is_empty:
            return None
        shapely.prepare(local)
        pu = unary_union(ponds) if ponds else None
        if pu is not None:
            shapely.prepare(pu)
        shapely.prepare(poly)
        frame = T._building_frame(poly, f, n)
        if style == "arc":
            variants = arc_variants(tpl, 0.5 * (lo_b + hi_b), n_target, frame[1] - frame[0], gap, lo_r, hi_r)
            ks, rots, dus, dvs = ARC_K, ARC_ROT_DEG, ARC_SHIFT_U, ARC_SHIFT_V
        else:
            variants = [tpl]
            ks, rots, dus, dvs = T.COOL_K, T.COOL_ROT_DEG, T.COOL_SHIFT_M, T.COOL_SHIFT_M
        if not variants:
            return None
        if simple:                                                       # one pose: aligned with the wind, no shifts or rotations
            ks, rots, dus, dvs = ks[:1], (0.0,), (0.0,), (0.0,)

        def evaluate(tv, mirror, k, rot_deg, du, dv):
            rho_, s_t_min_, s_t_max_, N_, sol_ = lims(tv)
            s_area = math.sqrt(0.5 * (lo_b + hi_b) / tv["pond_area"])
            mp = T._pose(tv, f, n, frame, mirror, math.radians(rot_deg), du, dv)
            got = _pond_at(tv, mp, s_area, local, lo_b, hi_b, clip_max)
            if got is None:
                return None
            pond, s_p, loss = got
            if len(pond.interiors):                                      # a pond is one ring: no holes
                return None
            s_t = float(np.clip(s_p * k, s_t_min_, s_t_max_))
            P = mp(tv["xy"], s_t)
            Rt = np.clip(rho_ * s_t * tv["R"], lo_r, hi_r)
            ok = np.zeros(N_, bool)
            for rv in np.unique(np.round(Rt, 2)):
                sel = np.where(np.isclose(np.round(Rt, 2), rv))[0]
                ok[sel] = shapely.contains_xy(free(rv), P[sel, 0], P[sel, 1])
            shapely.prepare(pond)
            pts = shapely.points(P[:, 0], P[:, 1])
            ok &= shapely.distance(pts, pond) >= Rt + EPS
            if pu is not None:
                ok &= shapely.distance(pts, pu) >= Rt + EPS
            ok &= occ.free_of(P, Rt, gap)
            notch = max(0.0, sol_ - pond.area / pond.convex_hull.area)
            pen = 1.0 * loss + 1.5 * notch
            if style == "arc":                                           # the pond belongs beside its building
                pen += 0.8 * max(0.0, float(pond.distance(poly)) - ARC_DIST_FREE_M) / 40.0
                pen += 0.6 * max(0.0, float(pond.centroid.distance(poly)) - ARC_CENTRE_FREE_M) / 50.0
            return dict(tpl=tv, mirror=mirror, k=k, rot=rot_deg, du=du, dv=dv, pond=pond, s_p=s_p, s_t=s_t, loss=loss,
                        notch=notch, pen=pen, P=P, Rt=Rt, ok=ok, n_ok=int(ok.sum()), score=ok.mean() - pen)

        pool = []
        for tv in variants:
            for mirror in (1.0, -1.0):
                for k in ks:
                    for rot in rots:
                        for du in dus:
                            for dv in dvs:
                                e = evaluate(tv, mirror, k, rot, du, dv)
                                if e is not None:
                                    pool.append(e)
        if not pool:
            return None
        pool.sort(key=lambda e: -e["score"])
        fine = []
        for e0 in ([] if simple else pool[:(2 if style == "arc" else 3)]):
            for rot in (ARC_ROT_FINE if style == "arc" else T.COOL_ROT_FINE):
                for du in T.COOL_SHIFT_FINE:
                    for dv in T.COOL_SHIFT_FINE:
                        e = evaluate(e0["tpl"], e0["mirror"], e0["k"], e0["rot"] + rot, e0["du"] + du, e0["dv"] + dv)
                        if e is not None:
                            fine.append(e)
        pool = sorted(pool + fine, key=lambda e: -e["score"])
        best = None
        for e in pool[:(2 if simple else 5)]:
            pond = e["pond"]
            shapely.prepare(pond)
            ponds_u = pond if pu is None else unary_union([pond, pu])
            shapely.prepare(ponds_u)
            if style == "arc":                      # the arc is built on the pond as it was placed
                w = arc_on_pond(pond, poly, f, free, occ, ponds_u, gap, lo_r, hi_r, n_target, tpl["trees"][0], others=pu)
                if w is None:
                    continue
                P, Rt_, hn_, row_ = w
                tv2 = dict(e["tpl"], trees=[tpl["trees"][0]] * len(Rt_), hn=hn_, xy=P, R=Rt_,
                           arc=dict(e["tpl"]["arc"], rows=int(row_.max()) + 1, crown_radius_max_m=float(Rt_.max()), trees=int(len(Rt_))))
                sc_ = len(Rt_) / max(n_target, 1) - e["pen"]
                if best is None or sc_ > best["score"]:
                    best = dict(e, tpl=tv2, score=sc_, P=P, Rt=Rt_, take=np.arange(len(Rt_)), frame=frame, style=style)
                continue
            P, ok = _nudge(e["P"], e["Rt"], e["ok"], free, ponds_u, gap, occ)
            take = _select(np.where(ok)[0], e["tpl"]["R"], n_target)
            if len(take) < CATCH_MIN_TREES:
                continue
            sc_ = len(take) / max(n_target, 1) - e["pen"]
            if best is None or sc_ > best["score"]:
                best = dict(e, score=sc_, P=P, take=take, frame=frame, style=style)
        return best

    floor_a = 1.5 * pc["min_area_m2"]
    occ_start = (list(occ.x), list(occ.y), list(occ.R))

    def solve_all(tgt, skip, order_=None):
        """Sequential placement (largest building first) with pond targets `tgt` {building: m2}.  A building that cannot hold its pond tries smaller
        ones (RUNGS of the target, down to the minimum pond area).  Returns {building: result or None}."""
        occ.x, occ.y, occ.R = list(occ_start[0]), list(occ_start[1]), list(occ_start[2])
        out, ponds, region_cur, left = {}, [], region_all, struct_budget
        todo = [b for b in (order if order_ is None else order_) if b not in skip]
        for j, bi in enumerate(todo):
            n_b = max(CATCH_MIN_TREES, int(left // (len(todo) - j)))
            got = None
            for style in styles:                              # the arc catcher first; the stored template if the arc finds no place
                seen = set()
                for rung, clip_max in (CATCH_ATTEMPTS_SIMPLE if simple else CATCH_ATTEMPTS):
                    t = max(tgt[bi] * rung, floor_a)
                    if (round(t), str(clip_max)) in seen:
                        continue
                    seen.add((round(t), str(clip_max)))
                    got = one_building(bi, t * (1 - CATCH_SHARE_FLOOR), t * (1 + CATCH_SHARE_FLOOR), n_b, ponds, region_cur, clip_max, style)
                    if got is not None:
                        got.update(clip_max=clip_max, rung=rung, target=t)
                        break
                if got is not None:
                    break
            out[bi] = got
            if got is None:
                continue
            ponds.append(got["pond"])
            region_cur = region_cur.difference(got["pond"].buffer(ARC_POND_SEPARATION_M if got.get("style") == "arc" else 0.3))
            for i in got["take"]:
                occ.add(got["P"][i, 0], got["P"][i, 1], got["Rt"][i])
            if len(got["take"]):                              # the ponds of the buildings after this one keep off its trees
                region_cur = region_cur.difference(_closed_fill([_disc(got["P"][i, 0], got["P"][i, 1], got["Rt"][i] + (ARC_OTHER_POND_CLEAR_M if got.get("style") == "arc" else 0.3)) for i in got["take"]]))
            left -= len(got["take"])
        return out

    skip = set()
    w = areas / areas.sum()
    tgt = {b: float(max(mid * w[b], floor_a)) for b in range(nbld)}
    result, area_ok = None, False
    def solve_best(tgt, skip):
        """The buildings in two orders (largest first / smallest first: the one that goes first gets the best place); the order that gives the
        most catchers, then the better poses, wins."""
        best_, best_key = None, None
        for ord_ in ((order, order[::-1]) if len(order) <= ARC_BOTH_ORDERS_MAX else (order,)):
            r_ = solve_all(tgt, skip, ord_)
            key = (sum(v is not None for v in r_.values()), sum(v["score"] for v in r_.values() if v is not None))
            if best_key is None or key > best_key:
                best_, best_key, o_best = r_, key, ord_
            if len(order) < 2:
                break
        if o_best is not ord_:                               # solve_all leaves the trees of the LAST run in occ: redo the winner
            best_ = solve_all(tgt, skip, o_best)
        return best_

    for it in range(CATCH_ROUNDS):
        result = solve_best(tgt, skip)
        newly = [b for b, r in result.items() if r is None]
        for b in newly:
            failed[b] = "no place for its pond / trees"
            skip.add(b)
        done = {b: r for b, r in result.items() if r is not None}
        if not done:
            return None, ("no position for any pond (area %.0f-%.0f m2 in total) next to a building" % (lo_a, hi_a))
        if simple:                                           # no search for the pond total: what fits next to each building is what it gets
            area_ok = True
            break
        total = float(sum(r["pond"].area for r in done.values()))
        if lo_a * 1.003 <= total <= hi_a * 0.997 and len(done) <= pond_limit:
            area_ok = True
            break
        # the ponds must add up to the middle of the band: buildings that got their full target take the missing area (or give back the excess)
        full = [b for b, r in done.items() if r["rung"] == 1.0]
        for b, r in done.items():
            if r["rung"] != 1.0:
                tgt[b] = float(r["pond"].area)               # what fits there is what it gets
        room = mid - total
        if not full:
            break
        s_full = sum(tgt[b] for b in full)
        for b in full:
            tgt[b] = max(floor_a, tgt[b] + room * tgt[b] / s_full)
    if not area_ok:
        return None, ("the ponds placed (%.0f m2) cannot be brought into the allowed total %.0f-%.0f m2 - too little room next to the buildings"
                      % (sum(r["pond"].area for r in result.values() if r is not None), lo_a, hi_a))
    result = {b: r for b, r in result.items() if r is not None}
    active = [b for b in order if b in result]

    # ---- 3. ponds and lawns
    used = []
    pond_recs = []
    cut = []
    pond_polys = [result[b]["pond"] for b in active]
    pu = unary_union(pond_polys)
    for g in data["grassAreas"]:
        q = T._largest(L._poly(g["coordinates"]).buffer(0).difference(pu.buffer(0.3)).difference(static[1]))
        cut.append(q if (q is not None and q.area >= cfg["lawn"]["min_area_m2"]) else None)
    pond_new = L.discretize_patches(pond_polys, step, static[0])
    polys = pond_new + L.discretize_patches([q for q in cut if q is not None], step, unary_union([static[1]] + pond_new))
    for k, (b, poly) in enumerate(zip(active, polys[:len(pond_polys)]), 1):
        rec = copy.deepcopy(tpl["pond"])
        rec.setdefault("dimensions", {}).setdefault("depth", pc["default_depth_m"])
        rec["id"] = f"pond_{k}"
        L.patch_set_polygon(rec, poly, True, 0)
        pond_recs.append(rec)
    kept, it = [], iter(polys[len(pond_polys):])
    for g, q in zip(data["grassAreas"], cut):
        if q is not None:
            L.patch_set_polygon(g, next(it), False, 0)
            kept.append(g)
    data["ponds"] = pond_recs
    data["grassAreas"] = kept

    # ---- 4. trees: the pathway rows, then the catchers
    added = list(ptrees)
    for b in active:
        r = result[b]
        for i in r["take"]:
            added.append(dict(x=float(r["P"][i, 0]), y=float(r["P"][i, 1]), R=float(r["Rt"][i]), rec=r["tpl"]["trees"][i],
                              hn=float(r["tpl"]["hn"][i]), prio=1, kind="catcher"))
    notes = {}
    reserve = [_disc(a["x"], a["y"], a["R"] + EPS) for a in added]
    try:
        fitted = [] if simple else fit_patches(data, base, cfg, rng, reserve, static, step)
    except ValueError as e:
        return None, str(e)
    if fitted:
        notes["area_rescaled"] = fitted
    clear = None
    if sd(cfg, "air_corridor") and not simple:
        # the air that comes to a catcher must not meet trees that were not planted for it: upwind of every pond + building pair, as wide as the pair,
        # up to the site edge, no base tree stands (the arcs and the pathway rows are not touched)
        parts = []
        for b in active:
            hull = unary_union([result[b]["pond"], sc.group_polys[b]]).convex_hull
            far = affinity.translate(hull, -float(f[0]) * 1.2 * sc.site_diameter, -float(f[1]) * 1.2 * sc.site_diameter)
            parts.append(unary_union([hull, far]).convex_hull.buffer(AIR_CORRIDOR_MARGIN_M))
        clear = unary_union(parts).intersection(sc.site)
        shapely.prepare(clear)
    res, why = assemble(data, base_data, base, cfg, base_map, sc, added, cap, rng, notes, clear_zone=clear)
    if res is None:
        return None, why
    data, mine, removed_ids, counts = res
    n_cat = counts.get("catcher", 0)
    if n_cat < CATCH_MIN_TREES * len(active) * 0.5:
        return None, "too few catcher trees survive the rules"
    per_b = []
    for b in active:
        r = result[b]
        fu0, fu1, fv0, fv1 = r["frame"]
        per_b.append({"building": names[b], "size_m": [round(fu1 - fu0, 1), round(fv1 - fv0, 1)],
                      "style": r.get("style", "template"),
                      "arc": r["tpl"].get("arc"),
                      "side_of_wind": "left" if r["tpl"]["side"] * r["mirror"] > 0 else "right",
                      "mirrored_about_west_east_axis": r["mirror"] < 0,
                      "pose": {"rotate_deg": round(r["rot"], 1), "shift_along_wind_m": round(r["du"], 1),
                               "shift_across_wind_m": round(r["dv"], 1)},
                      "pond_area_m2": round(float(r["pond"].area), 1), "pond_shape": "adapted" if r.get("clip_max") == "reshape" else ("arc" if r.get("style") == "arc" else "template"),
                      "trees_planned": int(len(r["take"]))})
    pond_total = float(sum(L._poly(p["coordinates"]).area for p in data["ponds"]))
    info = {"op": "template", "kind": "cool_wind_catcher_all", "mode": "simple" if simple else "search", "template": Path(tpl["path"]).name, "season": "heat",
            "wind_from_deg_model": round(wind["from_deg"], 1),
            "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
            "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
            "buildings": nbld, "buildings_with_catcher": len(active),
            "buildings_without_catcher": {names[b]: why_ for b, why_ in failed.items()},
            "catchers": per_b,
            "scale": round(float(np.mean([result[b]["s_t"] for b in active])), 3),
            "pond_area_m2": round(pond_total, 1), "pond_area_allowed_m2": [round(lo_a, 1), round(hi_a, 1)],
            "ponds": len(data["ponds"]), "pond_patch_limit": pond_limit,
            "trees_in_template": int(np.mean([len(result[b]["tpl"]["xy"]) for b in active])), "trees_placed": len(mine), "trees_catcher": n_cat,
            "trees_pathway": counts.get("pathway", 0), "pathway_planting": pinfo,
            "tree_count_limit": int(cap), "tree_budget": bud,
            "base_trees_replaced": len(removed_ids), "air_corridor_area_m2": None if clear is None else round(float(clear.area), 1),
            "base_trees_removed_from_air_corridor": notes.get("base_trees_removed_from_air_corridor", 0),
            "template_tree_ids": sorted(mine, key=lambda i: int(i.split()[-1])),
            "rules_enforced_on_base": notes,
            "distance_from_upwind_site_edge_m": None, "close_to_inlet": None}
    return (data, [info]), None


# ============================================================================= wind blocker belt along the windward edge(s)
def windward_arcs(site, flow, min_w=BELT_WINDWARD_MIN):
    """Windward stretches of the site boundary as lists of vertices.  An edge is windward when the wind enters through it
    (-(outward normal . flow) >= min_w); consecutive windward edges form one stretch."""
    poly = orient(site, 1.0)                          # counter-clockwise: the site lies to the LEFT of the walk
    xy = np.array(poly.exterior.coords)[:-1]
    m = len(xy)
    f = np.asarray(flow, float)
    wind = []
    for i in range(m):
        p0, p1 = xy[i], xy[(i + 1) % m]
        t = p1 - p0
        ln = float(np.hypot(*t))
        if ln < 1e-9:
            wind.append(False)
            continue
        t /= ln
        outward = np.array([t[1], -t[0]])
        wind.append(float(-(outward @ f)) >= min_w)
    if not any(wind):
        return []
    if all(wind):
        return [list(map(tuple, xy)) + [tuple(xy[0])]]
    start = next(i for i in range(m) if wind[i] and not wind[i - 1])
    arcs, cur = [], None
    for k in range(m):
        i = (start + k) % m
        if wind[i]:
            if cur is None:
                cur = [tuple(xy[i])]
            cur.append(tuple(xy[(i + 1) % m]))
        elif cur is not None:
            arcs.append(cur)
            cur = None
    if cur is not None:
        arcs.append(cur)
    return arcs


def _belt_rows(tpl):
    """Row structure of the blocker template (flow towards +x): (R, tree spacing in a row, row pitch, [row records])."""
    a, b, R = tpl["a"], tpl["b"], tpl["R"]
    rows = {}
    for i in range(len(a)):
        rows.setdefault(round(float(a[i]), 1), []).append(i)
    keys = sorted(rows)
    Rm = float(np.mean(R))
    sp = []
    for k in keys:
        bb = np.sort(b[rows[k]])
        if len(bb) > 1:
            sp.append(float(np.median(np.diff(bb))))
    spacing = float(np.median(sp)) if sp else 2.4 * Rm
    pitch = float(np.median(np.diff(keys))) if len(keys) > 1 else spacing
    return Rm, spacing, pitch, [tpl["trees"][rows[k][0]] for k in keys]


def _row_points(line, spacing, phase):
    """Points every `spacing` m along a (multi)line, starting `phase` m from its start."""
    out = []
    parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
    for g in parts:
        if g.is_empty or g.length < 1e-6:
            continue
        s = phase
        while s <= g.length + 1e-9:
            p = g.interpolate(min(s, g.length))
            out.append((p.x, p.y))
            s += spacing
    return out


def _belt_adaptive(lines, Rmax, rows, pitch, spacing_ratio, occ0, free, gap, lo_r):
    """Staggered rows along the windward stretches.  Row r runs along the stretch at Rmax + gap + r x pitch from the site boundary; at every
    step the LARGEST crown (Rmax down to the minimum radius, steps of 0.5 m) that stands there without breaking a rule is planted, so that a narrow strip
    between the boundary and a building gets small trees instead of a hole.  Returns [(x, y, R, row, stretch)]."""
    loc = Occ()
    loc.x, loc.y, loc.R = list(occ0.x), list(occ0.y), list(occ0.R)
    opts = [float(r) for r in np.arange(Rmax, lo_r - 1e-9, -0.5)]
    if opts[-1] > lo_r + 1e-9:
        opts.append(float(lo_r))
    pts = []
    for r in range(rows):
        base = Rmax + BELT_FIRST_ROW_GAP + r * pitch
        for li, g in enumerate(lines):
            s = (0.5 * spacing_ratio * Rmax if r % 2 else 0.0) + 0.5 * Rmax
            while s < g.length - 0.25:
                p = g.interpolate(s)
                a, b = g.interpolate(max(s - 0.5, 0.0)), g.interpolate(min(s + 0.5, g.length))
                tx, ty = b.x - a.x, b.y - a.y
                ln = math.hypot(tx, ty)
                if ln < 1e-9:
                    s += 1.0
                    continue
                nx, ny = -ty / ln, tx / ln                         # into the site (the ring runs counter-clockwise)
                placed = None
                for R in opts:
                    for dd in (0.0, -1.0, 1.0, -2.0, 2.0):
                        d = max(base + dd, R + BELT_FIRST_ROW_GAP)
                        x, y = p.x + d * nx, p.y + d * ny
                        if (shapely.contains_xy(free(R), x, y) and loc.free_of(np.array([[x, y]]), np.array([R]), gap)[0]):
                            placed = (x, y, R)
                            break
                    if placed:
                        break
                if placed:
                    loc.add(*placed)
                    pts.append((*placed, r, li))
                    s += spacing_ratio * placed[2]
                else:
                    s += 1.0
    return pts


def _coverage(lines, pts):
    """Share of the windward edge length that crown shadows (projected on the edge) cover."""
    cover = 0.0
    for li, g in enumerate(lines):
        iv = []
        for x, y, R, r, l2 in pts:
            if l2 == li:
                s = g.project(Point(x, y))
                iv.append((max(0.0, s - R), min(g.length, s + R)))
        cur_end, c = -1.0, 0.0
        for a_, b_ in sorted(iv):
            a_ = max(a_, cur_end)
            if b_ > a_:
                c += b_ - a_
                cur_end = max(cur_end, b_)
        cover += c
    total = sum(g.length for g in lines)
    return min(1.0, cover / max(total, 1e-9))


def make_edge_blocker_individual(base_data, base, cfg, base_map, tpl, wind, seed=0):
    """(layout, ops) or (None, reason): a belt of staggered tree rows along the whole windward edge(s) of the site, dense trees on both
    sides of the pathways."""
    import layout_reflow as RF
    rng = np.random.default_rng(seed)
    tc = cfg["tree"]
    gap = tc["canopy_gap_factor"]
    lo_r, hi_r = tc["radius_range_m"]
    data = L.working_copy(base_data)
    todo = L.patches_out_of_band(data, base, cfg)
    sdata = dict(data, ponds=[]) if "pond" in todo else data
    sc = L.Scene(sdata, cfg, base.osm_polys)
    cap = L.max_tree_count(cfg, L.tree_count0(base, cfg), sc.pathway is not None)
    free = FreeCache(sc, cfg, sc.ponds)
    arcs = windward_arcs(sc.site, wind["flow"])
    if not arcs:
        return None, "the site has no windward edge"
    Rm, spacing_t, pitch_t, row_recs = _belt_rows(tpl)
    ratio = spacing_t / Rm                                  # centre-to-centre spacing in crown radii (template)
    lines = [LineString(a) for a in arcs]
    edge_len = float(sum(g.length for g in lines))

    def attempt(path_share):
        """Pathway trees with `path_share` of the tree-count limit, then the best belt the rest allows."""
        bud = budgets(cfg, cap, sc.pathway is not None, path_share)
        occ = Occ()
        ptrees, pinfo = plant_pathway_dense(sc, cfg, occ, bud["path"], free, base_data["trees"])
        struct_budget = bud["struct"] + max(0, bud["path"] - len(ptrees))
        tried, best = [], None
        for Rmax, rows in BELT_CONFIGS:
            Rmax = float(np.clip(Rmax, lo_r, hi_r))
            pts = _belt_adaptive(lines, Rmax, rows, pitch_t * Rmax / Rm, ratio, occ, free, gap, lo_r)
            cov = _coverage(lines, pts)
            tried.append((Rmax, rows, len(pts), round(cov, 3)))
            if pts and len(pts) <= struct_budget:
                key = cov - 0.0004 * len(pts)              # the most complete edge; among equals the fewer trees
                if best is None or key > best[0]:
                    best = (key, Rmax, rows, pts, cov)
        thinned = False
        if best is None:                                    # even the sparsest belt exceeds the budget: thin it evenly
            Rmax, rows = BELT_CONFIGS[-1]
            pts = _belt_adaptive(lines, float(Rmax), rows, pitch_t * Rmax / Rm, ratio, occ, free, gap, lo_r)
            if pts:
                pts = [pts[i] for i in T._thin(struct_budget, len(pts))]
                best, thinned = (0, float(Rmax), rows, pts, _coverage(lines, pts)), True
        return dict(share=path_share, bud=bud, occ=occ, ptrees=ptrees, pinfo=pinfo, best=best, tried=tried, thinned=thinned)

    # the pathways get PATH_SHARE of the limit; when that leaves the belt short of BELT_COVER_TARGET the belt takes more, down to PATH_SHARE_MIN
    ps0, ps_min = float(sd(cfg, "path_share")), float(sd(cfg, "path_share_min"))
    shares = [ps0] if (sc.pathway is None or ps_min >= ps0) else [float(x) for x in np.linspace(ps0, ps_min, 4)]
    chosen = None
    for ps in shares:
        a_ = attempt(ps)
        if a_["best"] is None:
            continue
        if chosen is None or a_["best"][4] > chosen["best"][4] + 1e-9:
            chosen = a_
        if a_["best"][4] >= float(sd(cfg, "belt_cover_target")):
            chosen = a_
            break
    if chosen is None:
        return None, "no belt tree fits along the windward edge"
    ptrees, pinfo, occ, bud, tried, thinned = (chosen[k] for k in ("ptrees", "pinfo", "occ", "bud", "tried", "thinned"))
    _, Rmax, rows, pts, cov = chosen["best"]

    added = list(ptrees)
    for x, y, R, r, li in pts:
        added.append(dict(x=float(x), y=float(y), R=float(R), rec=row_recs[min(r, len(row_recs) - 1)], hn=BELT_HN, prio=1, kind="belt"))
        occ.add(x, y, R)
    notes = {}
    reserve = [_disc(a["x"], a["y"], a["R"] + EPS) for a in added]
    try:
        fitted = fit_patches(data, base, cfg, rng, reserve, (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn")), cfg["patch"].get("discretization_m", 1.0))
    except ValueError as e:
        return None, str(e)
    if fitted:
        notes["area_rescaled"] = fitted
    res, why = assemble(data, base_data, base, cfg, base_map, sc, added, cap, rng, notes)
    if res is None:
        return None, why
    data, mine, removed_ids, counts = res
    n_belt = counts.get("belt", 0)
    if n_belt < 3:
        return None, "fewer than 3 belt trees survive the rules"
    Rs = [p[2] for p in pts]
    info = {"op": "template", "kind": "wind_blocker_edge", "template": Path(tpl["path"]).name, "season": "cold",
            "wind_from_deg_model": round(wind["from_deg"], 1),
            "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
            "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
            "scale": round(Rmax / Rm, 3), "crown_radius_max_m": Rmax, "crown_radius_min_m": float(min(Rs)), "rows": rows,
            "windward_stretches": len(lines), "windward_edge_length_m": round(edge_len, 1),
            "edge_covered_share": round(cov, 3),
            "belt_thinned_to_budget": thinned,
            "trees_in_template": len(tpl["trees"]), "trees_placed": len(mine), "trees_belt": n_belt,
            "trees_pathway": counts.get("pathway", 0), "pathway_planting": pinfo,
            "tree_count_limit": int(cap), "tree_budget": bud, "belt_configs_tried": [list(t) for t in tried],
            "base_trees_replaced": len(removed_ids),
            "template_tree_ids": sorted(mine, key=lambda i: int(i.split()[-1])),
            "rules_enforced_on_base": notes,
            "distance_from_upwind_site_edge_m": round(Rmax + BELT_FIRST_ROW_GAP, 1), "close_to_inlet": True}
    return (data, [info]), None


# ============================================================================= one-line description (for the logs)
def describe(kind, name, path, spec, o):
    pw = o.get("pathway_planting") or {}
    if o.get("mode") == "simple":
        bad = o.get("rule_violations_of_the_result") or {}
        sbad = ", ".join(f"{k}: {v}" for k, v in bad.items()) if bad else "none"
        miss = o.get("buildings_without_catcher") or {}
        return (f"{spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), simple mode (no ponds / lawns / trees of the base, "
                f"no search, no pathway planting): catcher on {o['buildings_with_catcher']} of {o['buildings']} buildings, {o['ponds']} ponds {o['pond_area_m2']:g} m2 "
                f"(allowed total {o['pond_area_allowed_m2'][0]:g}-{o['pond_area_allowed_m2'][1]:g}), {o['trees_catcher']} catcher trees; {o.get('side_flipped_buildings', 0)} catcher(s) on the other side of the wind, "
                f"{o.get('ponds_cut_to_free_ground', 0)} pond(s) cut to the free ground, {o.get('trees_moved', 0)} tree(s) moved, {o.get('trees_dropped', 0)} dropped; "
                f"no overlaps (site, buildings, pathways, ponds, trees); not enforced: {sbad}" + (f"  (no room at: {', '.join(miss)})" if miss else ""))
    if kind == "cool_wind_catcher_all":
        miss = o.get("buildings_without_catcher") or {}
        return (f"{spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), "
                f"catcher on {o['buildings_with_catcher']} of {o['buildings']} buildings, {o['ponds']} ponds {o['pond_area_m2']:g} m2, "
                f"{o['trees_catcher']} catcher trees{ptxt}" + (f"  (no room at: {', '.join(miss)})" if miss else ""))
    if kind == "wind_blocker_edge":
        return (f"{spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), "
                f"belt of {o['rows']} rows (crown radius {o['crown_radius_min_m']:g}-{o['crown_radius_max_m']:g} m, {o['trees_belt']} trees) along {o['windward_stretches']} windward edge(s) "
                f"({o['windward_edge_length_m']:g} m, {round(100 * o['edge_covered_share'])} % shaded){ptxt}"
                + ("  (belt thinned to the tree-count limit)" if o.get("belt_thinned_to_budget") else ""))
    return None






# #!/usr/bin/env python3
# """
# layout_seeds.py  --  seed individuals, version 3.

#     cool_wind_catcher_all   HEAT-case wind.  A COOL WIND CATCHER (pond + curved tree rows, jsons/cool_wind_catcher.json) is
#                             hung on EVERY building and aligned to the wind: pond and trees stand at the building's
#                             downwind face, on the side of the wind that has room.  The pond area of the layout (the
#                             allowed band 1.0 - 1.1 x A0) is shared between the buildings in proportion to their footprints.
#     wind_blocker_edge       COLD-case wind.  A BELT of staggered tree rows (built from jsons/tree_wind_breaker.json) along
#                             the COMPLETE windward edge or edges of the site, so that the wind is blocked where it enters.
#     both                    DENSE trees on BOTH sides of every pathway / road (crown radius 2.5 m, centre-to-centre
#                             >= 1.1 x the crown diameters, as close as the tree-count limit allows).

# Every individual obeys all the rules of layout_ops (inside the site, clear of buildings / ponds / pathways, canopy gap,
# crown radius, height band of the location, tree-count limit, pond / lawn totals, pond patches <= number of buildings).
# The tree-count limit is shared out: pathway trees first (seeds.path_share, default 45 % of the limit), then the catchers / the belt
# (seeds.struct_share, default 45 %, plus whatever the pathways leave unused); base trees fill what is left and give way where they are in
# the way.  A pattern that does not fit is thinned (smaller number, same shape) - it is never allowed to break a rule.
# """

# import copy
# import math
# from pathlib import Path

# import numpy as np
# import shapely
# from shapely import affinity
# from shapely.geometry import LineString, Point, Polygon
# from shapely.geometry.polygon import orient
# from shapely.ops import unary_union

# import layout_ops as L
# import layout_templates as T

# EPS = T.EPS

# # ---- tree-count budget (shares of the limit  floor(max_count_factor x n0) + max_new_trees)
# PATH_SHARE = 0.45
# STRUCT_SHARE = 0.45
# # ---- dense pathway planting
# PATH_R = 2.5                      # crown radius of a pathway tree (m)
# PATH_GAP_TO_EDGE = 0.5            # crown edge to pathway edge (m)  (pathway_planting.canopy_gap_to_path_m when present)
# PATH_SLIDE_M = (0.0, 1.0, -1.0, 2.0, -2.0)
# PATH_HN = 0.5                     # height class: 0 = lowest allowed height of the location, 1 = highest
# # ---- cool wind catcher on every building
# CATCH_SHARE_FLOOR = 0.012        # each pond's area is its target +- this share (so the total stays inside the band)
# CATCH_MIN_TREES = 3               # fewest trees a catcher may be left with
# # (share of the pond target, share of the template pond that buildings / roads / the site edge may cut off  -or-  "reshape": the pond keeps its place at the building,
# # but its shape adapts to the room there is)
# CATCH_ATTEMPTS = ((1.0, 0.2), (1.0, 0.45), (1.0, "reshape"), (0.7, "reshape"), (0.5, "reshape"), (0.35, "reshape"), (0.22, "reshape"), (0.12, "reshape"))
# RESHAPE_MIN_START = 0.15          # a reshaped pond starts from at least this share of the target area of free template pond
# CATCH_ATTEMPTS_SIMPLE = ((1.0, "reshape"), (0.5, "reshape"), (0.22, "reshape"))     # cool_mode "simple": the pond target of a building, then smaller ones
# CATCH_ROUNDS = 5                  # passes to bring the pond areas to the middle of the band
# CATCH_REGION_M = 90.0             # a catcher is searched within this distance of its building
# # ---- wind blocker belt
# BELT_WINDWARD_MIN = 0.38          # a site edge is windward when -(outward normal . flow) >= this (about 68 deg off head-on)
# BELT_CONFIGS = ((3.0, 3), (4.0, 3), (3.0, 2), (4.0, 2), (5.0, 2), (6.0, 2), (6.0, 1))     # (largest crown radius m, rows)
# BELT_FIRST_ROW_GAP = 0.5          # crown edge to the site boundary (m)
# BELT_COVER_TARGET = 0.80          # the belt should shade at least this share of the windward edge, else the pathways give up tree budget (to seeds.path_share_min)
# BELT_HN = 1.0                     # belt trees are as tall as their location allows


# # ============================================================================= helpers
# class Occ:
#     """Trees that already stand (x, y, crown radius)."""

#     def __init__(self):
#         self.x, self.y, self.R = [], [], []

#     def add(self, x, y, R):
#         self.x.append(float(x)); self.y.append(float(y)); self.R.append(float(R))

#     def arrays(self):
#         return np.array(self.x, float), np.array(self.y, float), np.array(self.R, float)

#     def free_of(self, P, R, gap):
#         """bool array: tree centres P (n,2) with crown radii R (n,) keep the canopy gap to every standing tree."""
#         P, R = np.atleast_2d(P), np.atleast_1d(R)
#         if not self.x:
#             return np.ones(len(P), bool)
#         x, y, r = self.arrays()
#         d = np.hypot(P[:, None, 0] - x[None, :], P[:, None, 1] - y[None, :])
#         return (d >= gap * (R[:, None] + r[None, :]) + EPS).all(axis=1)


# class FreeCache:
#     """Region where the centre of a tree of crown radius r may stand (layout_templates._free_region), per radius."""

#     def __init__(self, sc, cfg, ponds):
#         self.sc, self.cfg, self.ponds, self.c = sc, cfg, ponds, {}

#     def __call__(self, r):
#         k = round(float(r), 2)
#         if k not in self.c:
#             self.c[k] = T._free_region(self.sc, self.cfg, k, ponds=self.ponds)
#         return self.c[k]


# SEED_DEFAULTS = {"catcher_style": "arc", "air_corridor": True, "path_share": PATH_SHARE, "struct_share": STRUCT_SHARE, "path_share_min": 0.30, "path_crown_radius_m": PATH_R,
#                  "belt_cover_target": BELT_COVER_TARGET,
#                  # "simple" (default): individual_1 = the base layout WITHOUT its ponds, lawns and trees (buildings and pathways stay), one cool wind catcher
#                  # on every building aligned to the heat-case wind, one pose per building, no search and no pond-area fitting.  "search": the older
#                  # pose / pond-area search with dense pathway planting (slow on a site with many buildings).
#                  "cool_mode": "simple"}


# def sd(cfg, key):
#     """Setting of the seed individuals: constraints.json "seeds" block, else the default above."""
#     return (cfg.get("seeds") or {}).get(key, SEED_DEFAULTS[key])


# def budgets(cfg, cap, has_path, path_share=None):
#     """Tree budgets: pathway trees and the structure (catchers / belt).  The two add up to seeds.path_share + seeds.struct_share of the
#     limit (default 90 %); base trees get the rest."""
#     ps0, ss = float(sd(cfg, "path_share")), float(sd(cfg, "struct_share"))
#     ps = ps0 if path_share is None else path_share
#     if not has_path:
#         return dict(path=0, struct=int(math.floor((ps0 + ss) * cap)))
#     return dict(path=int(math.floor(ps * cap)), struct=int(math.floor((ps0 + ss - ps) * cap)))


# def _next_id(base_data, data):
#     ids = [int(str(t["id"]).split()[-1]) for t in list(base_data["trees"]) + list(data["trees"])
#            if str(t["id"]).split()[-1].isdigit()]
#     return max(ids + [0])


# def _closed_fill(discs, extra=None, k=1.5):
#     """Obstacle made of tree discs (and `extra` geometry): neighbouring discs are merged (closing by k m) and enclosed holes are filled.  A pond
#     outline cannot have holes (a patch is one ring), so ponds are kept out of the whole planted stretch, not just off each crown."""
#     parts = list(discs) + ([extra] if extra is not None else [])
#     if not parts:
#         return None
#     u = unary_union(parts).buffer(k, quad_segs=4).buffer(-k, quad_segs=4)
#     polys = [u] if u.geom_type == "Polygon" else [g for g in getattr(u, "geoms", []) if g.geom_type == "Polygon"]
#     return unary_union([Polygon(g.exterior) for g in polys]) if polys else None


# def _disc(x, y, r):
#     return Point(float(x), float(y)).buffer(float(r), 16)


# # ============================================================================= dense pathway trees
# def plant_pathway_dense(sc, cfg, occ, budget, free, base_trees):
#     """Trees on BOTH sides of every pathway.  Slots lie on the pathway outline (every edge of every road) at a spacing that starts at the
#     closest the canopy-gap rule allows and grows until the number of trees fits `budget`.  Trees are appended to `occ`.
#     Returns (list of dict(x, y, R, rec, prio, kind), info dict)."""
#     import layout_reflow as RF
#     tc = cfg["tree"]
#     gap = tc["canopy_gap_factor"]
#     if sc.pathway is None or budget <= 0 or not cfg["tree"]["pathway_planting"]["enabled"]:
#         return [], dict(enabled=False)
#     R = float(np.clip(sd(cfg, "path_crown_radius_m"), *tc["radius_range_m"]))
#     off = R + float(cfg["tree"]["pathway_planting"].get("canopy_gap_to_path_m", PATH_GAP_TO_EDGE))
#     rings = [r for poly in RF._polys(sc.pathway) for r in [poly.exterior, *poly.interiors] if r.length > 2.0]
#     total = float(sum(r.length for r in rings))
#     if total <= 0:
#         return [], dict(enabled=False)
#     s_min = gap * 2 * R + 0.1

#     def run(spacing):
#         loc = Occ()
#         loc.x, loc.y, loc.R = list(occ.x), list(occ.y), list(occ.R)
#         out = []
#         for ring in rings:
#             n_slot = max(1, int(round(ring.length / spacing)))
#             step = ring.length / n_slot
#             for k in range(n_slot):
#                 done = False
#                 for slide in PATH_SLIDE_M:
#                     c = RF._side_point(ring, k * step + slide, off, sc.pathway)
#                     if c is None:
#                         continue
#                     p = np.array([c])
#                     if (shapely.contains_xy(free(R), c[0], c[1]) and loc.free_of(p, np.array([R]), gap)[0]
#                             and sc.pathway.distance(Point(*c)) >= R + EPS):
#                         loc.add(c[0], c[1], R)
#                         out.append((float(c[0]), float(c[1])))
#                         done = True
#                         break
#         return out

#     spacing = s_min
#     pts = run(spacing)
#     for _ in range(12):
#         if len(pts) <= budget:
#             break
#         spacing *= max(1.05, 1.02 * len(pts) / budget)
#         pts = run(spacing)
#     if len(pts) > budget:                                   # last resort: even thinning
#         pts = [pts[i] for i in T._thin(budget, len(pts))]
#     out = []
#     for x, y in pts:
#         occ.add(x, y, R)
#         out.append(dict(x=x, y=y, R=R, rec=None, hn=PATH_HN, prio=0, kind="pathway"))
#     info = dict(enabled=True, trees=len(out), crown_radius_m=R, spacing_asked_m=round(spacing, 2),
#                 min_spacing_m=round(s_min, 2), pathway_edge_length_m=round(total, 1),
#                 mean_spacing_m=round(total / max(len(out), 1), 2), budget=int(budget))
#     return out, info


# # ============================================================================= assembling the layout
# def _tree_record(rec, cfg, base_trees, rng):
#     if rec is not None:
#         return copy.deepcopy(rec)
#     return L.new_tree(cfg, base_trees, rng)


# def assemble(data, base_data, base, cfg, base_map, sc, added, max_total, rng, notes, static=None, clear_zone=None):
#     """Put the `added` trees (dicts x, y, R, hn, rec, prio, kind) into `data`:  base trees that touch them give way (then, if
#     the tree-count limit needs it, the nearest base trees), heights follow the height band of each location, base trees that the
#     new ponds / lawns put in breach of a rule are moved, and the result must have no NEW rule violation (an added tree that causes one is
#     dropped, the lowest priority first).  Returns ((data, mine ids, removed base tree ids, kept added list), None) or (None, reason)."""
#     import layout_reflow as RF
#     tc = cfg["tree"]
#     gap = tc["canopy_gap_factor"]
#     nb = len(sc.tx)
#     added = sorted(added, key=lambda a: a["prio"])
#     if len(added) > max_total:
#         added = added[:max_total]
#     px = np.array([a["x"] for a in added]); py = np.array([a["y"] for a in added]); pr = np.array([a["R"] for a in added])
#     gone = np.zeros(nb, bool)
#     if nb and len(added):
#         d = np.hypot(px[:, None] - sc.tx[None, :], py[:, None] - sc.ty[None, :])
#         gone = (d < gap * (pr[:, None] + sc.tR[None, :]) + EPS).any(axis=0)
#         over = nb - int(gone.sum()) + len(added) - max_total
#         if over > 0:
#             for i in np.argsort(d.min(axis=0)):
#                 if over <= 0:
#                     break
#                 if not gone[i]:
#                     gone[i] = True
#                     over -= 1
#     if clear_zone is not None and nb:                       # base trees whose crown reaches the clear air corridor of a catcher give way
#         inz = shapely.distance(shapely.points(sc.tx, sc.ty), clear_zone) < sc.tR
#         notes["base_trees_removed_from_air_corridor"] = int((inz & ~gone).sum())
#         gone |= inz
#     removed_ids = {sc.tree_ids[i] for i in np.where(gone)[0]}
#     data["trees"] = [t for t in data["trees"] if t["id"] not in removed_ids]
#     ball = sc.ball_union
#     nid = _next_id(base_data, data)
#     base_trees = list(base_data["trees"])
#     new, prio_of, kind_of = [], {}, {}
#     for a in added:
#         t = _tree_record(a["rec"], cfg, base_trees, rng)
#         nid += 1
#         t["id"] = f"Tree {nid}"
#         t["position"] = {"x": a["x"], "y": a["y"], "z": t.get("position", {}).get("z", 0)}
#         h = T._band_height(cfg, a["hn"], a["x"], a["y"], ball, sc.pathway)
#         L.set_tree_geometry(t, a["R"], h, tc)
#         T._sync_vegetation(t, a["R"], h)
#         new.append(t)
#         prio_of[t["id"]] = a["prio"]
#         kind_of[t["id"]] = a["kind"]
#     data["trees"] = list(data["trees"]) + new
#     mine = {t["id"] for t in new}
#     moved = RF.fix_trees(data, base, cfg, rng)
#     if moved:
#         notes["trees_moved_off_rule_breaks"] = len([i for i in moved if i not in mine])
#     L.finalize_patches(data, cfg)
#     _separate_lawns_from_ponds(data, cfg, static if static is not None else (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn")), 1.0)
#     _fix_slits(data)
#     topped = _topup_totals(data, base, cfg, sc, static if static is not None else (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn")))
#     if topped:
#         notes["area_topped_up"] = topped
#     L.refresh_metadata(data)
#     for _ in range(8):
#         # road shade is mandatory for every individual: plant / enlarge / move trees (never the seed trees) until it holds
#         notes["pathway_shade"] = RF.ensure_pathway_shade(data, base, cfg, rng, protect=mine)
#         fsc = L.Scene(data, cfg, base.osm_polys)
#         nv = L.new_violations(fsc, base, base_map)
#         if not nv:
#             break
#         drop = set()
#         for typ, ids, _sev in nv:
#             mm = [i for i in ids if i in mine and i not in drop]
#             if typ == "tree_gap":
#                 mm = sorted(mm, key=lambda i: (prio_of.get(i, 9), int(i.split()[-1])))
#                 drop.update(mm[-1:])
#             else:
#                 drop.update(mm)
#         if not drop:
#             return None, "the layout has new rule violations that are not seed trees: " + ", ".join(sorted({t for t, _, _ in nv}))
#         data["trees"] = [t for t in data["trees"] if t["id"] not in drop]
#         mine -= drop
#         L.refresh_metadata(data)
#     else:
#         return None, "final rule check failed"
#     counts = {}
#     for i in mine:
#         counts[kind_of[i]] = counts.get(kind_of[i], 0) + 1
#     return (data, mine, removed_ids, counts), None


# FIT_FRACTIONS = (0.45, 0.7, 0.9)      # where in the allowed band the reflow aims (share of the band width above its lower limit)


# def fit_patches(data, base, cfg, rng, reserve, static, step):
#     """Bring the pond / lawn totals back into their bands when they are out (L.fit_patch_area) and clean up the outlines (overlap slivers,
#     short edges - re-discretizing shared pond / lawn boundaries pulls them back by up to 0.4 m, which can cost 100 m2 or more).  The reflow
#     aims at 45 % of the band; if the clean-up pushes a total out of the band again, the attempt is repeated aiming higher (70 %, 90 %).
#     `data` is edited in place.  Returns the list of categories that were rebuilt (empty: nothing was out of band)."""
#     todo = L.patches_out_of_band(data, base, cfg)
#     if not todo:
#         return []
#     last = None
#     for frac in FIT_FRACTIONS:
#         c2 = copy.deepcopy(cfg)
#         for cat in todo:
#             a0 = L.base_area0(base, cfg, cat)
#             lo, hi = L.area_band(cfg, cat, a0)
#             if hi - lo >= 0.03 * a0:
#                 c2[cat]["area_factor_range"][0] = (lo + (frac - 0.1) / 0.9 * (hi - lo)) / a0
#         d = L.working_copy(data)
#         L.fit_patch_area(d, base, c2, rng, pond_blockers=reserve)
#         _separate_lawns_from_ponds(d, cfg, static, step)
#         L.finalize_patches(d, cfg)
#         _fix_slits(d)
#         last = d
#         if not L.patches_out_of_band(d, base, cfg):
#             break
#     for k in ("ponds", "grassAreas"):
#         data[k] = last[k]
#     return todo


# def _topup_totals(data, base, cfg, sc, static, rounds=6):
#     """Narrow bands (a lawn that sits exactly at its irrigation cap: band = A0 +- 0.1 %) are missed by the clean-up of the outlines by a few
#     tens of m2.  Bring each total back to the middle of its band: the largest patch grows (buffer outwards, clipped to the free room) or shrinks.
#     Ponds keep off the crowns that stand, lawns off the ponds.  Returns the categories that were corrected."""
#     fixed = []
#     margin = cfg["patch"].get("edge_margin_m", 0.25)
#     for cat, key in (("pond", "ponds"), ("lawn", "grassAreas")):
#         a0 = L.base_area0(base, cfg, cat)
#         lo, hi = L.area_band(cfg, cat, a0)
#         slack = 1e-3 * a0
#         aim = 0.5 * (lo + hi)
#         for _ in range(rounds):
#             polys = [L._poly(q["coordinates"]).buffer(0) for q in data[key]]
#             if not polys:
#                 break
#             tot = sum(q.area for q in polys)
#             if lo - 0.5 * slack <= tot <= hi + 0.5 * slack:
#                 break
#             i = int(np.argmax([q.area for q in polys]))
#             own = [q for j, q in enumerate(polys) if j != i]
#             ko = [static[0] if cat == "pond" else static[1]]
#             if cat == "lawn":
#                 ko += [L._poly(q["coordinates"]).buffer(0) for q in data["ponds"]]
#             else:
#                 ko += [_disc(tr["position"]["x"], tr["position"]["y"], L._tree_R(tr) + EPS) for tr in data["trees"]]
#             ko += own
#             free = sc.site.buffer(-margin).difference(unary_union(ko))
#             want = aim - sum(q.area for q in own)
#             q0 = polys[i]
#             if tot < lo:
#                 a_, b_ = 0.0, 6.0 * math.sqrt(max(want, 1.0))
#                 sgn = 1.0
#             else:
#                 a_, b_ = 0.0, 0.5 * math.sqrt(q0.area)
#                 sgn = -1.0
#             best = None
#             for _k in range(26):
#                 d = 0.5 * (a_ + b_)
#                 q = T._largest((q0.buffer(sgn * d, quad_segs=6).intersection(free)) if sgn > 0 else q0.buffer(-d, quad_segs=6))
#                 ar = 0.0 if q is None else q.area
#                 best = q if q is not None else best
#                 if abs(ar - want) < 0.2 * slack:
#                     best = q
#                     break
#                 if (ar < want) == (sgn > 0):
#                     a_ = d
#                 else:
#                     b_ = d
#             if best is None or best.is_empty:
#                 break
#             L.patch_set_polygon(data[key][i], best, cat == "pond", 0)
#             L.finalize_patches(data, cfg)
#             _fix_slits(data)
#             if cat not in fixed:
#                 fixed.append(cat)
#             tot2 = sum(L._poly(q["coordinates"]).area for q in data[key])
#             aim += (0.5 * (lo + hi) - tot2)                  # the clean-up took / gave a little: aim a bit off the middle next time
#     return fixed


# SLIT_M = 0.2           # an outline whose own 0.1 m buffer closes a gap (gap < 0.2 m) "encloses something through a hairline slit"


# def _fix_slits(data):
#     """A pond / lawn outline that wraps round an obstacle through a channel that discretization pulled in to under 0.2 m (the audit calls this a
#     hairline slit; meshers cannot resolve it) is eroded by 0.11 m, which widens every such channel by 0.22 m.  Other outlines are left alone."""
#     n = 0
#     for key in ("ponds", "grassAreas"):
#         for p in data[key]:
#             poly = L._poly(p["coordinates"]).buffer(0)
#             c = poly.buffer(0.5 * SLIT_M, join_style=2)
#             if c.geom_type == "Polygon" and not c.interiors:
#                 continue
#             q = T._largest(poly.buffer(-0.55 * SLIT_M, join_style=2))
#             if q is not None and q.area > 0.5 * poly.area:
#                 L.patch_set_polygon(p, q, key == "ponds", 0)
#                 n += 1
#     return n


# def _separate_lawns_from_ponds(data, cfg, static, step):
#     """Discretization and the pond / lawn reflow leave slivers (a fraction of a m2) where two patches, or a patch and a building, touch.  Ponds
#     are cut back against the buildings / pathways and the ponds before them, lawns against all of these and the lawns before them.  Nothing is
#     re-discretized (that would shave metres off every shared boundary and could push a total under its band); finalize_patches() takes care of
#     vertices that end up closer than the discretization step."""
#     static_p, static_l = static if isinstance(static, tuple) else (static, static)       # (pond keep-out, lawn keep-out): buildings grown by 2 m / 1 m
#     taken = static_p                                            # ponds first (against buildings / pathways and each other), then the lawns
#     pond_polys = []
#     for p in data["ponds"]:
#         q = L._poly(p["coordinates"]).buffer(0)
#         if q.intersection(taken).area > 0:
#             q2 = T._largest(q.difference(taken))
#             if q2 is not None and q2.area >= 0.5 * q.area:
#                 L.patch_set_polygon(p, q2, True, 0)
#                 q = q2
#         taken = unary_union([taken, q])
#         pond_polys.append(q)
#     taken = unary_union([static_l] + pond_polys)                # lawns: the (smaller) lawn clearance from the buildings, the ponds, the lawns before
#     kept = []
#     for g in data["grassAreas"]:
#         q = L._poly(g["coordinates"]).buffer(0)
#         if q.intersection(taken).area > 0:
#             q = T._largest(q.difference(taken))
#             if q is None or q.area < cfg["lawn"]["min_area_m2"]:
#                 continue
#             L.patch_set_polygon(g, q, False, 0)
#         kept.append(g)
#         taken = unary_union([taken, L._poly(g["coordinates"]).buffer(0)])
#     data["grassAreas"] = kept


# # ============================================================================= cool wind catcher on every building
# def _nudge(P, Rt, ok, free, ponds, gap, occ, nudge_m=T.COOL_NUDGE_M, step=0.5):
#     """Tree.not ok -> nearest spot within nudge_m that is clear of the obstacles, the ponds, the trees that stand (occ) and the
#     pattern's own trees.  Largest crowns first."""
#     P, ok = P.copy(), ok.copy()
#     angs = np.linspace(0, 2 * math.pi, 16, endpoint=False)
#     rings = np.arange(step, nudge_m + 1e-9, step)
#     O = np.array([[r * math.cos(a), r * math.sin(a)] for r in rings for a in angs])
#     standing = list(np.where(ok)[0])
#     for i in np.argsort(-Rt):
#         if ok[i]:
#             continue
#         C = P[i] + O
#         m = shapely.contains_xy(free(Rt[i]), C[:, 0], C[:, 1])
#         if ponds is not None:
#             m &= shapely.distance(shapely.points(C[:, 0], C[:, 1]), ponds) >= Rt[i] + EPS
#         if m.any():
#             m &= occ.free_of(C, np.full(len(C), Rt[i]), gap)
#         if standing and m.any():
#             d = np.hypot(C[:, None, 0] - P[standing, 0][None, :], C[:, None, 1] - P[standing, 1][None, :])
#             m &= (d >= gap * (Rt[i] + Rt[standing])[None, :] + EPS).all(axis=1)
#         if m.any():
#             P[i] = C[int(np.argmax(m))]
#             ok[i] = True
#             standing.append(i)
#     return P, ok


# def _reshape(tpl, mp, s0, region, lo, hi):
#     """The pond's SHAPE adapts to the room there is: start from the part of the template pond (scale s0, pose mp) that is free, then grow it
#     (buffer outwards, clipped to the free region) or shrink it (buffer inwards) until its area is inside [lo, hi].  The pond keeps its place at the
#     building and as much of the template outline as the room allows.  Returns (polygon, s0, share of the pond that is not template shape) or None."""
#     target, tol = 0.5 * (lo + hi), 0.01 * (hi - lo) + 1e-9
#     g = Polygon(mp(tpl["pond_xy"], s0)).buffer(0)
#     q0 = T._largest(g.intersection(region))
#     if q0 is None or q0.area < RESHAPE_MIN_START * target:
#         return None

#     def shape(d):
#         q = q0.buffer(d, quad_segs=6) if abs(d) > 1e-9 else q0
#         return T._largest(q.intersection(region)) if not q.is_empty else None

#     def ok(q):
#         return q is not None and lo + tol <= q.area <= hi - tol

#     q = q0
#     if not ok(q0):
#         if q0.area < target:
#             a, b = 0.0, 4.0 * math.sqrt(target)
#         else:
#             a, b = -0.5 * math.sqrt(q0.area), 0.0
#         q = None
#         for _ in range(26):
#             d = 0.5 * (a + b)
#             qd = shape(d)
#             ar = 0.0 if qd is None else qd.area
#             if ok(qd):
#                 q = qd
#                 break
#             if ar < target:                                  # the area grows with d: too small -> go outwards
#                 a = d
#             else:
#                 b = d
#         if q is None:
#             return None
#     q = T._largest(q.buffer(0))
#     if q is None or q.area < 1.0:
#         return None
#     iou = q.intersection(g).area / max(q.union(g).area, 1e-9)
#     return q, s0, 0.6 * (1.0 - iou)


# def _pond_at(tpl, mp, s0, region, lo, hi, clip_max):
#     """layout_templates._pond_at with the allowed share of cut-off area as a parameter; clip_max == "reshape": the shape adapts (see _reshape)."""
#     if clip_max == "reshape":
#         return _reshape(tpl, mp, s0, region, lo, hi)
#     target, tol, s = 0.5 * (lo + hi), 0.01 * (hi - lo) + 1e-9, s0
#     for _ in range(6):
#         g = Polygon(mp(tpl["pond_xy"], s)).buffer(0)
#         q = T._largest(g.intersection(region))
#         if q is None or q.area < 1.0 or g.area <= 0:
#             return None
#         if lo + tol <= q.area <= hi - tol:
#             loss = 1.0 - q.area / g.area
#             return (q, s, loss) if loss <= clip_max else None
#         s *= math.sqrt(target / q.area)
#         if not 0.3 < s < 3.0:
#             return None
#     return None


# def _select(idx, Rtpl, n):
#     """n of the trees `idx`: the big-crown classes first, a class that does not fit completely is thinned evenly."""
#     idx = np.asarray(idx, int)
#     if len(idx) <= n:
#         return idx
#     cls = sorted(set(np.round(Rtpl[idx], 1)), reverse=True)
#     out = []
#     for c in cls:
#         room = n - len(out)
#         if room <= 0:
#             break
#         m = idx[np.isclose(np.round(Rtpl[idx], 1), c)]
#         out.extend(m if len(m) <= room else m[T._thin(room, len(m))])
#     return np.array(out, int)


# # ============================================================================= arc-shaped cool wind catcher (built for each building)
# # The catcher the stored template shows is one fixed picture.  The "arc" catcher is BUILT for every building from a few rules, so it adapts to the
# # building, the wind and the room there is (frame: u along the wind, d = distance from the building's lateral face, away from the building):
# #   * the POND is a long, leaf-shaped water body on the lateral side of the building, in the open area.  It starts upwind of the building (the
# #     wind crosses it before it reaches the building) and its downwind end leans in towards the building, so that the cooled air is guided to it;
# #   * the TREES stand in one or two rows on the pond's FAR side and round its DOWNWIND END, starting just upwind of the widest part of the pond and
# #     hooking back towards the building.  The wind enters between the pond and the building, crosses the water, and the tree arc stops it from
# #     leaving sideways / downwind, so the cooled air is held at the building.  Crowns are largest in the middle of the arc and smaller at its ends.
# ARC_VARIANTS = ((6.0, 0.5, 2.6), (12.0, 0.3, 2.8), (25.0, 0.5, 3.2), (25.0, 0.2, 3.2), (45.0, 0.5, 3.4), (60.0, 0.4, 3.6))   # (angle of the pond axis to the wind (deg), downwind end of the pond past the building's downwind face as a share of the building length, pond length / widest width)
# ARC_LADDER = ((2, 5.0), (2, 5.5), (2, 6.0))      # (rows, largest crown radius m): densest arc first; the first one inside the tree budget is used
# ARC_R_MIN = 2.5                   # smallest crown radius at the ends of the arc (m)
# ARC_ROW_SHRINK = 0.15             # an outer row's crowns are this much smaller than the row inside it
# ARC_POND_GAP = 1.6                # crown edge to the pond edge (m)
# ARC_POND_CLEAR = 2.6              # pond edge to the building's lateral face (m); a little more than the 2 m pond clearance
# ARC_NORMAL_BUILDING = 0.0         # the arc covers the pond's outline where the outward normal points at most this much (cosine) towards the building ...
# ARC_NORMAL_UPWIND = 0.0           # ... and where it points at most this much towards the wind (the upwind end of the pond stays open)
# ARC_END_TRIM = 0.06               # share of the arc's length left bare at each end (the openings: the wind enters / the cooled air leaves there)
# ARC_POND_SEPARATION_M = 4.0       # the pond of a later building keeps this far from the ponds before it (a passage for the wind)
# ARC_OTHER_POND_CLEAR_M = 4.0      # crown edge to a pond of another building (m)
# AIR_CORRIDOR_MARGIN_M = 18.0     # the clear air corridor is this much wider than the pond + building pair (m, each side)
# ARC_K = (1.0,)
# ARC_ROT_DEG = (0.0,)
# ARC_ROT_FINE = (0.0,)
# ARC_BOTH_ORDERS_MAX = 3          # up to this many buildings the placement is tried in two orders (largest first / smallest first)
# ARC_SHIFT_U = (-6.0, 0.0, 6.0)
# ARC_SHIFT_V = (0.0, 6.0)
# ARC_WIDTH_MAX_M = 40.0            # widest part of the pond (m), unless the length limit makes it wider
# ARC_LENGTH_MAX_M = 130.0          # longest pond (m)
# ARC_CENTRE_FREE_M = 25.0          # distance from the building to the pond's centre that costs nothing; beyond it the pose is penalised
# ARC_DIST_FREE_M = 12.0            # the pond may be this far from the building without penalty (distance between the outlines)


# def _arc_pond(area, phi_deg, shift_u, Lb, aspect=3.2):
#     """Leaf-shaped pond (template frame a, d) of the given area: its axis rises away from the building going upwind at phi_deg to the wind, its
#     downwind end is at a = shift_u * Lb, its lowest point is ARC_POND_CLEAR from the building's lateral face (d = 0)."""
#     phi = math.radians(phi_deg)
#     e = np.array([math.cos(phi), -math.sin(phi)])           # upwind end -> downwind end (the axis leans in towards the building)
#     nrm = np.array([math.sin(phi), math.cos(phi)])           # across the axis, away from the building
#     k = 0.70                                                 # area of the leaf / (length x width)
#     Wp = math.sqrt(area / (k * aspect))
#     if Wp > ARC_WIDTH_MAX_M:                                  # a big pond gets longer, not wider ...
#         Wp = ARC_WIDTH_MAX_M
#     Lp = area / (k * Wp)
#     if Lp > ARC_LENGTH_MAX_M:                                 # ... up to a length, then wider
#         Lp = ARC_LENGTH_MAX_M
#         Wp = area / (k * Lp)
#     t = np.linspace(0.0, 1.0, 41)
#     tw = t ** 0.756                                           # widest part at 40 % of the length from the upwind end
#     p = 2.4
#     g = np.clip(1.0 - np.abs(2 * tw - 1.0) ** p, 0.0, 1.0) ** (1.0 / p)
#     c = np.outer(t * Lp, e)
#     top = c + np.outer(0.5 * Wp * g, nrm)
#     bot = c - np.outer(0.5 * Wp * g, nrm)
#     xy = np.vstack([top, bot[::-1][1:-1]])
#     poly = Polygon(xy).buffer(0)
#     poly = orient(T._largest(poly), 1.0)
#     xy = np.array(poly.exterior.coords)[:-1]
#     xy = xy * math.sqrt(area / poly.area)                    # exact area (the leaf's area factor is only approximate)
#     c = c * math.sqrt(area / poly.area)
#     # the downwind end of the axis (t = 1) goes to a = shift_u * Lb;  the lowest point goes to d = ARC_POND_CLEAR
#     xy = xy + np.array([shift_u * Lb - c[-1, 0], 0.0])
#     xy[:, 1] += ARC_POND_CLEAR - xy[:, 1].min()
#     return xy


# def _smooth_periodic(a, w):
#     k = np.ones(2 * w + 1) / (2 * w + 1)
#     pad = np.concatenate([a[-w:], a, a[:w]], axis=0)
#     return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(a.shape[1])], axis=1)


# def _arc_trees(pond_xy, rows, Rmax, gap_eff, lo_r, hi_r, toward=(0.0, -1.0), upwind=(-1.0, 0.0), valid=None, quant=None, ds=0.25):
#     """Tree arc round the far side and the downwind end of a pond (see above).  Returns (xy (n,2), R (n,), row (n,)) in the order of the arc.
#     toward: direction from the pond to the building, or the building polygon (then the direction is taken point by point - the arc stays on the far
#     side of a pond that wraps round a building);  upwind: direction the wind comes from;  valid(q, R) -> bool: is a crown of radius R allowed at q
#     (None: always);  quant: crown radii are rounded to this step (m)."""
#     poly = orient(Polygon(pond_xy).buffer(0), 1.0)
#     ring = poly.exterior
#     Lr = ring.length
#     s_grid = np.arange(0.0, Lr, ds)
#     pts = np.array([ring.interpolate(s).coords[0] for s in s_grid])
#     tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
#     tang = _smooth_periodic(tang, max(2, int(2.0 / ds)))
#     tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
#     nor = np.stack([tang[:, 1], -tang[:, 0]], axis=1)         # outward (counter-clockwise ring)
#     if hasattr(toward, "geom_type"):
#         ln = shapely.get_coordinates(shapely.shortest_line(shapely.points(pts[:, 0], pts[:, 1]), toward)).reshape(-1, 2, 2)
#         tv = ln[:, 1, :] - ln[:, 0, :]
#         tv /= np.maximum(np.hypot(tv[:, 0], tv[:, 1]), 1e-9)[:, None]
#     else:
#         tv = np.tile(np.asarray(toward, float), (len(pts), 1))
#     up = np.asarray(upwind, float)
#     keep = ((nor * tv).sum(1) < ARC_NORMAL_BUILDING) & ((nor @ up) < ARC_NORMAL_UPWIND)
#     if keep.sum() < 8:
#         return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
#     # longest circular run of kept points
#     n = len(keep)
#     best, run0, run = (0, 0), None, 0
#     kk = np.concatenate([keep, keep])
#     for i, v in enumerate(kk):
#         if v:
#             if run == 0:
#                 run0 = i
#             run += 1
#             if run > best[0] and run <= n:
#                 best = (run, run0)
#         else:
#             run = 0
#     cnt, start = best
#     idx = (start + np.arange(cnt)) % n
#     cut_ = int(ARC_END_TRIM * len(idx))
#     if cut_ and len(idx) > 2 * cut_ + 8:
#         idx = idx[cut_:len(idx) - cut_]
#     P0, N0 = pts[idx], nor[idx]
#     lo = max(lo_r, 1.0)
#     out_xy, out_R, out_row = [], [], []
#     placed_xy, placed_R = np.zeros((0, 2)), np.zeros(0)
#     for row in range(rows):
#         shrink = (1.0 - ARC_ROW_SHRINK) ** row
#         i = 0
#         while i < len(P0):
#             t = i / max(len(P0) - 1, 1)
#             Rc = (ARC_R_MIN + (Rmax - ARC_R_MIN) * math.sin(math.pi * t) ** 0.8) * shrink
#             if quant:
#                 Rc = math.floor(Rc / quant) * quant
#             Rc = float(np.clip(Rc, lo, hi_r))
#             e0 = Rc + ARC_POND_GAP
#             ok_pos = None
#             for extra in ((0.0,) if row == 0 else np.arange(0.0, 3.0 * Rmax * (row + 1) + 1e-9, 0.5)):
#                 q = P0[i] + N0[i] * (e0 + extra)
#                 if valid is not None and not valid(q, Rc):
#                     continue
#                 if len(placed_R):
#                     d = np.hypot(placed_xy[:, 0] - q[0], placed_xy[:, 1] - q[1])
#                     if (d >= gap_eff * (placed_R + Rc)).all():
#                         ok_pos = q
#                         break
#                 else:
#                     ok_pos = q
#                     break
#             if ok_pos is None:
#                 i += 2
#                 continue
#             placed_xy = np.vstack([placed_xy, ok_pos])
#             placed_R = np.append(placed_R, Rc)
#             out_xy.append(ok_pos); out_R.append(Rc); out_row.append(row)
#             i += max(1, int(1.5 * Rc / ds))
#     return np.array(out_xy).reshape(-1, 2), np.array(out_R), np.array(out_row, int)


# def arc_on_pond(pond, bpoly, f, free, occ, ponds_u, gap, lo_r, hi_r, n_target, rec, others=None):
#     """The tree arc of a pond that has been placed (its real outline, so the arc follows whatever shape the pond got): the densest step of ARC_LADDER
#     that stays inside n_target trees, crowns only where the rules allow (free region, canopy gap to the trees that stand, clear of the ponds).
#     Returns (P, Rt, hn, row) or None."""
#     def valid(q, R):
#         if not shapely.contains_xy(free(R), q[0], q[1]):
#             return False
#         if ponds_u is not None and float(shapely.distance(shapely.points(q[0], q[1]), ponds_u)) < R + EPS:
#             return False
#         if others is not None and float(shapely.distance(shapely.points(q[0], q[1]), others)) < R + ARC_OTHER_POND_CLEAR_M:
#             return False                                             # the passage between two ponds stays open
#         return bool(occ.free_of(np.array([q]), np.array([R]), gap)[0])
#     pxy = np.array(pond.exterior.coords)[:-1]
#     got = None
#     for rows, Rmax in ARC_LADDER:
#         xy, R, row = _arc_trees(pxy, rows, min(Rmax, hi_r), gap + 0.03, lo_r, hi_r, toward=bpoly, upwind=(-f[0], -f[1]), valid=valid, quant=0.25)
#         got = (xy, R, row)
#         if 0 < len(R) <= n_target:
#             break
#     xy, R, row = got
#     if len(R) > n_target:
#         keep = _thin_to(row, n_target)
#         xy, R, row = xy[keep], R[keep], row[keep]
#     if len(R) < CATCH_MIN_TREES:
#         return None
#     hn = 0.5 + 0.5 * (R - R.min()) / max(R.max() - R.min(), 1e-9)
#     return xy, R, hn, row


# def _thin_to(row, n):
#     """Indices of the n trees that stay when an arc of trees (row number per tree, in the order of the arc) is over budget: the outer rows are
#     thinned first (evenly), then the inner one."""
#     idx = np.arange(len(row))
#     excess = len(row) - n
#     drop = np.zeros(len(row), bool)
#     for r in sorted(set(row.tolist()), reverse=True):
#         if excess <= 0:
#             break
#         m = idx[row == r]
#         k = min(excess, len(m))
#         if r == 0 and k < len(m):
#             drop[m[T._thin(k, len(m))]] = True
#         else:
#             drop[m[np.unique(np.round(np.linspace(0, len(m) - 1, k)).astype(int))] if k < len(m) else m] = True
#         excess -= k
#     return idx[~drop]


# def arc_variants(base_tpl, area, n_target, Lb, gap, lo_r, hi_r):
#     """Template-like dicts (see layout_templates.read_cool_template) of the arc catcher, one per shape variant (ARC_VARIANTS), for a pond of about
#     `area` m2 at a building of length Lb along the wind.  The tree arc is the densest one of ARC_LADDER that stays inside n_target trees."""
#     out = []
#     for phi, shift, asp in ARC_VARIANTS:
#         pxy = _arc_pond(area, phi, shift, Lb, asp)
#         got = None
#         for rows, Rmax in ARC_LADDER:
#             xy, R, row = _arc_trees(pxy, rows, min(Rmax, hi_r), gap + 0.03, lo_r, hi_r)
#             got = (xy, R, row, rows, Rmax)
#             if 0 < len(R) <= n_target:
#                 break
#         xy, R, row, rows, Rmax = got
#         if len(R) > n_target:                                  # even the sparsest ladder step is over budget: thin the outer row, then the arc
#             keep = _thin_to(row, n_target)
#             xy, R, row = xy[keep], R[keep], row[keep]
#         if len(R) < CATCH_MIN_TREES:
#             continue
#         D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
#         np.fill_diagonal(D, np.inf)
#         ratio = float((D / (1.1 * (R[:, None] + R[None, :]))).min())
#         hn = 0.5 + 0.5 * (R - R.min()) / max(R.max() - R.min(), 1e-9)
#         tv = dict(base_tpl)
#         tv.update(side=1.0, anchor=np.array([0.0, 0.0]), pond_xy=pxy, pond_area=float(Polygon(pxy).area),
#                   trees=[base_tpl["trees"][0]] * len(R), xy=xy, R=R, H=R, hn=hn, pair_ratio=ratio,
#                   arc=dict(angle_deg=phi, downwind_end_share=shift, aspect=asp, rows=int(rows), crown_radius_max_m=float(R.max()), trees=int(len(R))))
#         out.append(tv)
#     return out


# def make_simple_catchers_individual(base_data, base, cfg, base_map, tpl, wind):
#     """individual_1 in "simple" mode: take the base layout, remove every pond, lawn and tree (only buildings and pathways stay),
#     then hang ONE cool wind catcher (pond + tree arc) on EVERY building, turned to the wind of the heat case.  One pose per building (the template's
#     own: pond on the template side, no shift, no rotation), pond area shared in proportion to the footprints, no search, no clipping to free space, no
#     rule check on the result.  The ONE thing that is enforced is the site boundary: a catcher whose template side sticks out of the site is tried on the
#     other side of the wind (mirror image); what still lies outside is cut off (pond clipped to the site, trees whose crown is not inside dropped).
#     Overlaps (pond / trees on a neighbour building, a road, another catcher) are not repaired - they are counted in the
#     result ("overlaps") so they can be seen.  Returns ((data, [info]), None)."""
#     tc = cfg["tree"]
#     rng = np.random.default_rng(0)
#     data = L.working_copy(base_data)
#     # ponds, lawns and trees of the site are removed; one that lies OUTSIDE the site boundary (no part of a pond / lawn inside it, tree centre outside it)
#     # stays exactly as it is
#     site_poly = L._poly(data["site"]["coordinates"]).buffer(0)
#     kept = {"ponds": [], "grassAreas": [], "trees": []}
#     for k_ in ("ponds", "grassAreas"):
#         for it in data.get(k_, []):
#             try:
#                 outside = L._poly(it["coordinates"]).buffer(0).intersection(site_poly).area <= 1e-6
#             except Exception:
#                 outside = False
#             if outside:
#                 kept[k_].append(it)
#     for t in data.get("trees", []):
#         pos = t.get("position") or {}
#         if "x" in pos and not site_poly.contains(Point(float(pos["x"]), float(pos["y"]))):
#             kept["trees"].append(t)
#     for k_ in ("trees", "ponds", "grassAreas"):
#         data[k_] = list(kept[k_])
#     L.refresh_metadata(data)
#     sc = L.Scene(data, cfg, base.osm_polys)
#     nbld = len(sc.group_polys)
#     if not nbld:
#         return None, "the layout has no building"
#     f, n = T._frame(wind["flow"])
#     gap = tc["canopy_gap_factor"]
#     lo_r, hi_r = tc["radius_range_m"]
#     a0 = L.base_area0(base, cfg, "pond")
#     lo_a, hi_a = L.area_band(cfg, "pond", a0)
#     kept_pond_area = float(sum(L._poly(p["coordinates"]).area for p in kept["ponds"]))
#     mid = max(0.5 * (lo_a + hi_a) - kept_pond_area, 1.5 * cfg["pond"]["min_area_m2"])     # pond area of the catchers: the middle of the band, less the ponds that stay, shared by footprint
#     areas = np.array([p.area for p in sc.group_polys])
#     style = str(sd(cfg, "catcher_style"))
#     ponds, trees_added, per_b = [], [], []
#     margin = cfg["patch"].get("edge_margin_m", 0.25)
#     site_in = sc.site.buffer(-margin)
#     shapely.prepare(site_in)
#     crown_zone = {}

#     def inside_crowns(P, Rt):
#         """bool per tree: the whole crown is inside the site"""
#         ok = np.zeros(len(Rt), bool)
#         for r in np.unique(np.round(Rt, 2)):
#             if r not in crown_zone:
#                 crown_zone[r] = sc.site.buffer(-float(r))
#                 shapely.prepare(crown_zone[r])
#             sel = np.where(np.round(Rt, 2) == r)[0]
#             ok[sel] = shapely.contains_xy(crown_zone[r], P[sel, 0], P[sel, 1])
#         return ok

#     n_flipped = n_clipped = n_dropped = 0
#     for bi in np.argsort(-areas):
#         poly = sc.group_polys[bi]
#         frame = T._building_frame(poly, f, n)
#         a_b = float(max(mid * areas[bi] / areas.sum(), 1.5 * cfg["pond"]["min_area_m2"]))
#         tv = tpl
#         if style == "arc":
#             vs = arc_variants(tpl, a_b, 10 ** 6, frame[1] - frame[0], gap, lo_r, hi_r)
#             tv = vs[0] if vs else tpl
#         rho = min(1.0, 0.995 * tv["pair_ratio"] * 1.1 / gap)
#         s_p = math.sqrt(a_b / tv["pond_area"])
#         Rt = np.clip(rho * s_p * tv["R"], lo_r, hi_r)
#         # the template side of the wind first; the mirror image (other side of the wind) only if more of the catcher then lies inside the site
#         tried = []
#         for mirror in (1.0, -1.0):
#             mp = T._pose(tv, f, n, frame, mirror, 0.0, 0.0, 0.0)
#             pond_m = T._largest(Polygon(mp(tv["pond_xy"], s_p)).buffer(0))
#             if pond_m is None:
#                 continue
#             P_m = mp(tv["xy"], s_p)
#             ok_m = inside_crowns(P_m, Rt)
#             share = pond_m.intersection(sc.site).area / pond_m.area + float(ok_m.mean()) if len(Rt) else pond_m.intersection(sc.site).area / pond_m.area
#             tried.append((share, mirror, pond_m, P_m, ok_m))
#             if share >= (2.0 if len(Rt) else 1.0) - 1e-9:
#                 break                                          # fully inside on this side: no need to look at the other
#         if not tried:
#             continue
#         best_t = tried[0]
#         for t_ in tried[1:]:
#             if t_[0] > best_t[0] + 1e-9:
#                 best_t = t_
#         share, mirror, pond, P, ok = best_t
#         flipped = mirror < 0
#         n_flipped += int(flipped)
#         pond_c = T._largest(pond.intersection(site_in)) if pond.intersection(site_in).area < pond.area - 1e-6 else pond
#         if pond_c is None:
#             per_b.append({"building": sc.group_names[bi], "pond_area_m2": 0.0, "trees_planned": 0, "note": "its catcher lies outside the site on both sides"})
#             continue
#         clipped = pond_c is not pond
#         n_clipped += int(clipped)
#         dropped = int((~ok).sum())
#         n_dropped += dropped
#         ponds.append(pond_c)
#         for i in np.where(ok)[0]:
#             trees_added.append(dict(x=float(P[i, 0]), y=float(P[i, 1]), R=float(Rt[i]), rec=tv["trees"][i], hn=float(tv["hn"][i])))
#         per_b.append({"building": sc.group_names[bi], "size_m": [round(frame[1] - frame[0], 1), round(frame[3] - frame[2], 1)],
#                       "style": "arc" if tv is not tpl else "template", "side": "mirrored (other side of the wind)" if flipped else "template side",
#                       "pond_area_m2": round(float(pond_c.area), 1), "pond_clipped_to_site": clipped,
#                       "pond_share_inside_site_before_clip": round(float(pond.intersection(sc.site).area / pond.area), 3),
#                       "trees_planned": int(ok.sum()), "trees_dropped_outside_site": dropped})
#     if not ponds:
#         return None, "no pond could be drawn"
#     step = cfg["patch"].get("discretization_m", 1.0)
#     polys = L.discretize_patches(ponds, step, None)
#     recs = []
#     for k, poly in enumerate(polys, 1):
#         rec = copy.deepcopy(tpl["pond"])
#         rec.setdefault("dimensions", {}).setdefault("depth", cfg["pond"]["default_depth_m"])
#         rec["id"] = f"pond_{k}"
#         while rec["id"] in {str(p_.get("id")) for p_ in kept["ponds"]}:
#             rec["id"] += "_c"
#         L.patch_set_polygon(rec, poly, True, 0)
#         recs.append(rec)
#     data["ponds"] = list(kept["ponds"]) + recs
#     nid = _next_id(base_data, data)
#     base_trees = list(base_data["trees"])
#     ball = sc.ball_union
#     new = []
#     for a in trees_added:
#         t = _tree_record(a["rec"], cfg, base_trees, rng)
#         nid += 1
#         t["id"] = f"Tree {nid}"
#         t["position"] = {"x": a["x"], "y": a["y"], "z": t.get("position", {}).get("z", 0)}
#         h = T._band_height(cfg, a["hn"], a["x"], a["y"], ball, sc.pathway)
#         L.set_tree_geometry(t, a["R"], h, tc)
#         T._sync_vegetation(t, a["R"], h)
#         new.append(t)
#     data["trees"] = list(kept["trees"]) + new
#     L.refresh_metadata(data)
#     # what the unchecked layout breaks (reported, not repaired)
#     overlaps = {}
#     try:
#         fsc = L.Scene(data, cfg, base.osm_polys)
#         for typ, ids, _sev in L.check(fsc, base):
#             overlaps[typ] = overlaps.get(typ, 0) + 1
#     except Exception as e:
#         overlaps = {"check_failed": str(e)[:120]}
#     pond_total = float(sum(L._poly(p["coordinates"]).area for p in data["ponds"]))
#     info = {"op": "template", "kind": "cool_wind_catcher_all", "mode": "simple", "template": Path(tpl["path"]).name, "season": "heat",
#             "wind_from_deg_model": round(wind["from_deg"], 1),
#             "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
#             "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
#             "buildings": nbld, "buildings_with_catcher": len(ponds), "buildings_without_catcher": {p_["building"]: p_["note"] for p_ in per_b if p_.get("note")},
#             "catchers": per_b, "scale": 1.0,
#             "pond_area_m2": round(pond_total, 1), "pond_area_allowed_m2": [round(lo_a, 1), round(hi_a, 1)],
#             "ponds": len(data["ponds"]), "trees_placed": len(new), "trees_catcher": len(new), "trees_pathway": 0,
#             "side_flipped_buildings": n_flipped, "ponds_clipped_to_site": n_clipped, "trees_dropped_outside_site": n_dropped,
#             "kept_outside_the_site": {"ponds": sum(1 for _, gi in base.hosted_base["ponds"] if gi == -1), "grassAreas": sum(1 for _, gi in base.hosted_base["grassAreas"] if gi == -1),
#                                      "trees": len(kept["trees"])},
#             "pathway_planting": {"enabled": False, "skipped": "simple mode: only the catchers are planted"},
#             "base_trees_replaced": len(base_data.get("trees", [])) - len(kept["trees"]), "rules_not_applied": sorted(overlaps) if "check_failed" not in overlaps else overlaps,
#             "rule_violations_of_the_result": overlaps,
#             "template_tree_ids": sorted((t["id"] for t in new), key=lambda i: int(i.split()[-1])),
#             "distance_from_upwind_site_edge_m": None, "close_to_inlet": None}
#     return (data, [info]), None


# def make_cool_all_individual(base_data, base, cfg, base_map, tpl, wind, seed=0):
#     """(layout, ops) or (None, reason): a cool wind catcher on every building, dense trees on both sides of the pathways."""
#     import layout_reflow as RF
#     if str(sd(cfg, "cool_mode")) == "simple":
#         return make_simple_catchers_individual(base_data, base, cfg, base_map, tpl, wind)
#     rng = np.random.default_rng(seed)
#     tc, pc = cfg["tree"], cfg["pond"]
#     simple = False
#     data = L.working_copy(base_data)
#     if simple:                                             # only buildings and pathways stay: no ponds, lawns or trees (also none on roofs)
#         for k_ in ("trees", "ponds", "grassAreas"):
#             data[k_] = []
#         L.refresh_metadata(data)
#     sc = L.Scene(data, cfg, base.osm_polys)
#     nbld = len(sc.group_polys)
#     if not nbld:
#         return None, "the layout has no building"
#     f, n = T._frame(wind["flow"])
#     a0 = L.base_area0(base, cfg, "pond")
#     lo_a, hi_a = L.area_band(cfg, "pond", a0)
#     mid = 0.5 * (lo_a + hi_a)
#     gap = tc["canopy_gap_factor"]
#     lo_r, hi_r = tc["radius_range_m"]
#     cap = L.max_tree_count(cfg, L.tree_count0(base, cfg), sc.pathway is not None)
#     bud = budgets(cfg, cap, sc.pathway is not None)
#     pond_limit = L.patch_limit(cfg, "pond", nbld)
#     static = (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn"))     # buildings grown by 2 m (ponds) / 1 m (lawns), pathway
#     step = cfg["patch"].get("discretization_m", 1.0)
#     margin = cfg["patch"].get("edge_margin_m", 0.25)
#     free = FreeCache(sc, cfg, [])                              # the base ponds are replaced
#     occ = Occ()

#     # ---- 1. dense trees on both sides of the pathways
#     if simple:
#         ptrees, pinfo = [], {"skipped": "cool_mode simple: no pathway planting"}
#         struct_budget = int(cap)
#     else:
#         ptrees, pinfo = plant_pathway_dense(sc, cfg, occ, bud["path"], free, base_data["trees"])
#         struct_budget = bud["struct"] + max(0, bud["path"] - len(ptrees))
#     path_discs = [_disc(a["x"], a["y"], a["R"] + 0.3) for a in ptrees]
#     region_all = sc.site.buffer(-margin).difference(static[0].buffer(0.1))
#     if path_discs:
#         region_all = region_all.difference(_closed_fill(path_discs, sc.pathway.buffer(0.1) if sc.pathway is not None else None))

#     # ---- 2. a catcher per building (largest building first), pond areas shared in proportion to the footprints
#     styles = ("arc", "template") if str(sd(cfg, "catcher_style")) == "arc" else ("template",)
#     areas = np.array([p.area for p in sc.group_polys])
#     names = sc.group_names
#     order = list(np.argsort(-areas))
#     failed = {}

#     def lims(tv):
#         """per template: (rho, s_t_min, s_t_max, N, solidity of the pond)"""
#         rho_ = min(1.0, 0.995 * tv["pair_ratio"] * 1.1 / gap)
#         pp = Polygon(tv["pond_xy"]).buffer(0)
#         return (rho_, lo_r / (float(tv["R"].min()) * rho_), hi_r / (float(tv["R"].max()) * rho_), len(tv["xy"]),
#                 pp.area / pp.convex_hull.area)

#     def one_building(bi, lo_b, hi_b, n_target, ponds, region_cur, clip_max, style="template"):
#         """Best catcher of building bi: dict or None.  ponds: pond polygons already placed.  style "template": the stored template; "arc": the
#         arc catcher built for this building (pond beside it, tree arc round the pond's far side and downwind end)."""
#         poly = sc.group_polys[bi]
#         local = region_cur.intersection(poly.buffer(CATCH_REGION_M))
#         if local.is_empty:
#             return None
#         shapely.prepare(local)
#         pu = unary_union(ponds) if ponds else None
#         if pu is not None:
#             shapely.prepare(pu)
#         shapely.prepare(poly)
#         frame = T._building_frame(poly, f, n)
#         if style == "arc":
#             variants = arc_variants(tpl, 0.5 * (lo_b + hi_b), n_target, frame[1] - frame[0], gap, lo_r, hi_r)
#             ks, rots, dus, dvs = ARC_K, ARC_ROT_DEG, ARC_SHIFT_U, ARC_SHIFT_V
#         else:
#             variants = [tpl]
#             ks, rots, dus, dvs = T.COOL_K, T.COOL_ROT_DEG, T.COOL_SHIFT_M, T.COOL_SHIFT_M
#         if not variants:
#             return None
#         if simple:                                                       # one pose: aligned with the wind, no shifts or rotations
#             ks, rots, dus, dvs = ks[:1], (0.0,), (0.0,), (0.0,)

#         def evaluate(tv, mirror, k, rot_deg, du, dv):
#             rho_, s_t_min_, s_t_max_, N_, sol_ = lims(tv)
#             s_area = math.sqrt(0.5 * (lo_b + hi_b) / tv["pond_area"])
#             mp = T._pose(tv, f, n, frame, mirror, math.radians(rot_deg), du, dv)
#             got = _pond_at(tv, mp, s_area, local, lo_b, hi_b, clip_max)
#             if got is None:
#                 return None
#             pond, s_p, loss = got
#             if len(pond.interiors):                                      # a pond is one ring: no holes
#                 return None
#             s_t = float(np.clip(s_p * k, s_t_min_, s_t_max_))
#             P = mp(tv["xy"], s_t)
#             Rt = np.clip(rho_ * s_t * tv["R"], lo_r, hi_r)
#             ok = np.zeros(N_, bool)
#             for rv in np.unique(np.round(Rt, 2)):
#                 sel = np.where(np.isclose(np.round(Rt, 2), rv))[0]
#                 ok[sel] = shapely.contains_xy(free(rv), P[sel, 0], P[sel, 1])
#             shapely.prepare(pond)
#             pts = shapely.points(P[:, 0], P[:, 1])
#             ok &= shapely.distance(pts, pond) >= Rt + EPS
#             if pu is not None:
#                 ok &= shapely.distance(pts, pu) >= Rt + EPS
#             ok &= occ.free_of(P, Rt, gap)
#             notch = max(0.0, sol_ - pond.area / pond.convex_hull.area)
#             pen = 1.0 * loss + 1.5 * notch
#             if style == "arc":                                           # the pond belongs beside its building
#                 pen += 0.8 * max(0.0, float(pond.distance(poly)) - ARC_DIST_FREE_M) / 40.0
#                 pen += 0.6 * max(0.0, float(pond.centroid.distance(poly)) - ARC_CENTRE_FREE_M) / 50.0
#             return dict(tpl=tv, mirror=mirror, k=k, rot=rot_deg, du=du, dv=dv, pond=pond, s_p=s_p, s_t=s_t, loss=loss,
#                         notch=notch, pen=pen, P=P, Rt=Rt, ok=ok, n_ok=int(ok.sum()), score=ok.mean() - pen)

#         pool = []
#         for tv in variants:
#             for mirror in (1.0, -1.0):
#                 for k in ks:
#                     for rot in rots:
#                         for du in dus:
#                             for dv in dvs:
#                                 e = evaluate(tv, mirror, k, rot, du, dv)
#                                 if e is not None:
#                                     pool.append(e)
#         if not pool:
#             return None
#         pool.sort(key=lambda e: -e["score"])
#         fine = []
#         for e0 in ([] if simple else pool[:(2 if style == "arc" else 3)]):
#             for rot in (ARC_ROT_FINE if style == "arc" else T.COOL_ROT_FINE):
#                 for du in T.COOL_SHIFT_FINE:
#                     for dv in T.COOL_SHIFT_FINE:
#                         e = evaluate(e0["tpl"], e0["mirror"], e0["k"], e0["rot"] + rot, e0["du"] + du, e0["dv"] + dv)
#                         if e is not None:
#                             fine.append(e)
#         pool = sorted(pool + fine, key=lambda e: -e["score"])
#         best = None
#         for e in pool[:(2 if simple else 5)]:
#             pond = e["pond"]
#             shapely.prepare(pond)
#             ponds_u = pond if pu is None else unary_union([pond, pu])
#             shapely.prepare(ponds_u)
#             if style == "arc":                      # the arc is built on the pond as it was placed
#                 w = arc_on_pond(pond, poly, f, free, occ, ponds_u, gap, lo_r, hi_r, n_target, tpl["trees"][0], others=pu)
#                 if w is None:
#                     continue
#                 P, Rt_, hn_, row_ = w
#                 tv2 = dict(e["tpl"], trees=[tpl["trees"][0]] * len(Rt_), hn=hn_, xy=P, R=Rt_,
#                            arc=dict(e["tpl"]["arc"], rows=int(row_.max()) + 1, crown_radius_max_m=float(Rt_.max()), trees=int(len(Rt_))))
#                 sc_ = len(Rt_) / max(n_target, 1) - e["pen"]
#                 if best is None or sc_ > best["score"]:
#                     best = dict(e, tpl=tv2, score=sc_, P=P, Rt=Rt_, take=np.arange(len(Rt_)), frame=frame, style=style)
#                 continue
#             P, ok = _nudge(e["P"], e["Rt"], e["ok"], free, ponds_u, gap, occ)
#             take = _select(np.where(ok)[0], e["tpl"]["R"], n_target)
#             if len(take) < CATCH_MIN_TREES:
#                 continue
#             sc_ = len(take) / max(n_target, 1) - e["pen"]
#             if best is None or sc_ > best["score"]:
#                 best = dict(e, score=sc_, P=P, take=take, frame=frame, style=style)
#         return best

#     floor_a = 1.5 * pc["min_area_m2"]
#     occ_start = (list(occ.x), list(occ.y), list(occ.R))

#     def solve_all(tgt, skip, order_=None):
#         """Sequential placement (largest building first) with pond targets `tgt` {building: m2}.  A building that cannot hold its pond tries smaller
#         ones (RUNGS of the target, down to the minimum pond area).  Returns {building: result or None}."""
#         occ.x, occ.y, occ.R = list(occ_start[0]), list(occ_start[1]), list(occ_start[2])
#         out, ponds, region_cur, left = {}, [], region_all, struct_budget
#         todo = [b for b in (order if order_ is None else order_) if b not in skip]
#         for j, bi in enumerate(todo):
#             n_b = max(CATCH_MIN_TREES, int(left // (len(todo) - j)))
#             got = None
#             for style in styles:                              # the arc catcher first; the stored template if the arc finds no place
#                 seen = set()
#                 for rung, clip_max in (CATCH_ATTEMPTS_SIMPLE if simple else CATCH_ATTEMPTS):
#                     t = max(tgt[bi] * rung, floor_a)
#                     if (round(t), str(clip_max)) in seen:
#                         continue
#                     seen.add((round(t), str(clip_max)))
#                     got = one_building(bi, t * (1 - CATCH_SHARE_FLOOR), t * (1 + CATCH_SHARE_FLOOR), n_b, ponds, region_cur, clip_max, style)
#                     if got is not None:
#                         got.update(clip_max=clip_max, rung=rung, target=t)
#                         break
#                 if got is not None:
#                     break
#             out[bi] = got
#             if got is None:
#                 continue
#             ponds.append(got["pond"])
#             region_cur = region_cur.difference(got["pond"].buffer(ARC_POND_SEPARATION_M if got.get("style") == "arc" else 0.3))
#             for i in got["take"]:
#                 occ.add(got["P"][i, 0], got["P"][i, 1], got["Rt"][i])
#             if len(got["take"]):                              # the ponds of the buildings after this one keep off its trees
#                 region_cur = region_cur.difference(_closed_fill([_disc(got["P"][i, 0], got["P"][i, 1], got["Rt"][i] + (ARC_OTHER_POND_CLEAR_M if got.get("style") == "arc" else 0.3)) for i in got["take"]]))
#             left -= len(got["take"])
#         return out

#     skip = set()
#     w = areas / areas.sum()
#     tgt = {b: float(max(mid * w[b], floor_a)) for b in range(nbld)}
#     result, area_ok = None, False
#     def solve_best(tgt, skip):
#         """The buildings in two orders (largest first / smallest first: the one that goes first gets the best place); the order that gives the
#         most catchers, then the better poses, wins."""
#         best_, best_key = None, None
#         for ord_ in ((order, order[::-1]) if len(order) <= ARC_BOTH_ORDERS_MAX else (order,)):
#             r_ = solve_all(tgt, skip, ord_)
#             key = (sum(v is not None for v in r_.values()), sum(v["score"] for v in r_.values() if v is not None))
#             if best_key is None or key > best_key:
#                 best_, best_key, o_best = r_, key, ord_
#             if len(order) < 2:
#                 break
#         if o_best is not ord_:                               # solve_all leaves the trees of the LAST run in occ: redo the winner
#             best_ = solve_all(tgt, skip, o_best)
#         return best_

#     for it in range(CATCH_ROUNDS):
#         result = solve_best(tgt, skip)
#         newly = [b for b, r in result.items() if r is None]
#         for b in newly:
#             failed[b] = "no place for its pond / trees"
#             skip.add(b)
#         done = {b: r for b, r in result.items() if r is not None}
#         if not done:
#             return None, ("no position for any pond (area %.0f-%.0f m2 in total) next to a building" % (lo_a, hi_a))
#         if simple:                                           # no search for the pond total: what fits next to each building is what it gets
#             area_ok = True
#             break
#         total = float(sum(r["pond"].area for r in done.values()))
#         if lo_a * 1.003 <= total <= hi_a * 0.997 and len(done) <= pond_limit:
#             area_ok = True
#             break
#         # the ponds must add up to the middle of the band: buildings that got their full target take the missing area (or give back the excess)
#         full = [b for b, r in done.items() if r["rung"] == 1.0]
#         for b, r in done.items():
#             if r["rung"] != 1.0:
#                 tgt[b] = float(r["pond"].area)               # what fits there is what it gets
#         room = mid - total
#         if not full:
#             break
#         s_full = sum(tgt[b] for b in full)
#         for b in full:
#             tgt[b] = max(floor_a, tgt[b] + room * tgt[b] / s_full)
#     if not area_ok:
#         return None, ("the ponds placed (%.0f m2) cannot be brought into the allowed total %.0f-%.0f m2 - too little room next to the buildings"
#                       % (sum(r["pond"].area for r in result.values() if r is not None), lo_a, hi_a))
#     result = {b: r for b, r in result.items() if r is not None}
#     active = [b for b in order if b in result]

#     # ---- 3. ponds and lawns
#     used = []
#     pond_recs = []
#     cut = []
#     pond_polys = [result[b]["pond"] for b in active]
#     pu = unary_union(pond_polys)
#     for g in data["grassAreas"]:
#         q = T._largest(L._poly(g["coordinates"]).buffer(0).difference(pu.buffer(0.3)).difference(static[1]))
#         cut.append(q if (q is not None and q.area >= cfg["lawn"]["min_area_m2"]) else None)
#     pond_new = L.discretize_patches(pond_polys, step, static[0])
#     polys = pond_new + L.discretize_patches([q for q in cut if q is not None], step, unary_union([static[1]] + pond_new))
#     for k, (b, poly) in enumerate(zip(active, polys[:len(pond_polys)]), 1):
#         rec = copy.deepcopy(tpl["pond"])
#         rec.setdefault("dimensions", {}).setdefault("depth", pc["default_depth_m"])
#         rec["id"] = f"pond_{k}"
#         L.patch_set_polygon(rec, poly, True, 0)
#         pond_recs.append(rec)
#     kept, it = [], iter(polys[len(pond_polys):])
#     for g, q in zip(data["grassAreas"], cut):
#         if q is not None:
#             L.patch_set_polygon(g, next(it), False, 0)
#             kept.append(g)
#     data["ponds"] = pond_recs
#     data["grassAreas"] = kept

#     # ---- 4. trees: the pathway rows, then the catchers
#     added = list(ptrees)
#     for b in active:
#         r = result[b]
#         for i in r["take"]:
#             added.append(dict(x=float(r["P"][i, 0]), y=float(r["P"][i, 1]), R=float(r["Rt"][i]), rec=r["tpl"]["trees"][i],
#                               hn=float(r["tpl"]["hn"][i]), prio=1, kind="catcher"))
#     notes = {}
#     reserve = [_disc(a["x"], a["y"], a["R"] + EPS) for a in added]
#     try:
#         fitted = [] if simple else fit_patches(data, base, cfg, rng, reserve, static, step)
#     except ValueError as e:
#         return None, str(e)
#     if fitted:
#         notes["area_rescaled"] = fitted
#     clear = None
#     if sd(cfg, "air_corridor") and not simple:
#         # the air that comes to a catcher must not meet trees that were not planted for it: upwind of every pond + building pair, as wide as the pair,
#         # up to the site edge, no base tree stands (the arcs and the pathway rows are not touched)
#         parts = []
#         for b in active:
#             hull = unary_union([result[b]["pond"], sc.group_polys[b]]).convex_hull
#             far = affinity.translate(hull, -float(f[0]) * 1.2 * sc.site_diameter, -float(f[1]) * 1.2 * sc.site_diameter)
#             parts.append(unary_union([hull, far]).convex_hull.buffer(AIR_CORRIDOR_MARGIN_M))
#         clear = unary_union(parts).intersection(sc.site)
#         shapely.prepare(clear)
#     res, why = assemble(data, base_data, base, cfg, base_map, sc, added, cap, rng, notes, clear_zone=clear)
#     if res is None:
#         return None, why
#     data, mine, removed_ids, counts = res
#     n_cat = counts.get("catcher", 0)
#     if n_cat < CATCH_MIN_TREES * len(active) * 0.5:
#         return None, "too few catcher trees survive the rules"
#     per_b = []
#     for b in active:
#         r = result[b]
#         fu0, fu1, fv0, fv1 = r["frame"]
#         per_b.append({"building": names[b], "size_m": [round(fu1 - fu0, 1), round(fv1 - fv0, 1)],
#                       "style": r.get("style", "template"),
#                       "arc": r["tpl"].get("arc"),
#                       "side_of_wind": "left" if r["tpl"]["side"] * r["mirror"] > 0 else "right",
#                       "mirrored_about_west_east_axis": r["mirror"] < 0,
#                       "pose": {"rotate_deg": round(r["rot"], 1), "shift_along_wind_m": round(r["du"], 1),
#                                "shift_across_wind_m": round(r["dv"], 1)},
#                       "pond_area_m2": round(float(r["pond"].area), 1), "pond_shape": "adapted" if r.get("clip_max") == "reshape" else ("arc" if r.get("style") == "arc" else "template"),
#                       "trees_planned": int(len(r["take"]))})
#     pond_total = float(sum(L._poly(p["coordinates"]).area for p in data["ponds"]))
#     info = {"op": "template", "kind": "cool_wind_catcher_all", "mode": "simple" if simple else "search", "template": Path(tpl["path"]).name, "season": "heat",
#             "wind_from_deg_model": round(wind["from_deg"], 1),
#             "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
#             "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
#             "buildings": nbld, "buildings_with_catcher": len(active),
#             "buildings_without_catcher": {names[b]: why_ for b, why_ in failed.items()},
#             "catchers": per_b,
#             "scale": round(float(np.mean([result[b]["s_t"] for b in active])), 3),
#             "pond_area_m2": round(pond_total, 1), "pond_area_allowed_m2": [round(lo_a, 1), round(hi_a, 1)],
#             "ponds": len(data["ponds"]), "pond_patch_limit": pond_limit,
#             "trees_in_template": int(np.mean([len(result[b]["tpl"]["xy"]) for b in active])), "trees_placed": len(mine), "trees_catcher": n_cat,
#             "trees_pathway": counts.get("pathway", 0), "pathway_planting": pinfo,
#             "tree_count_limit": int(cap), "tree_budget": bud,
#             "base_trees_replaced": len(removed_ids), "air_corridor_area_m2": None if clear is None else round(float(clear.area), 1),
#             "base_trees_removed_from_air_corridor": notes.get("base_trees_removed_from_air_corridor", 0),
#             "template_tree_ids": sorted(mine, key=lambda i: int(i.split()[-1])),
#             "rules_enforced_on_base": notes,
#             "distance_from_upwind_site_edge_m": None, "close_to_inlet": None}
#     return (data, [info]), None


# # ============================================================================= wind blocker belt along the windward edge(s)
# def windward_arcs(site, flow, min_w=BELT_WINDWARD_MIN):
#     """Windward stretches of the site boundary as lists of vertices.  An edge is windward when the wind enters through it
#     (-(outward normal . flow) >= min_w); consecutive windward edges form one stretch."""
#     poly = orient(site, 1.0)                          # counter-clockwise: the site lies to the LEFT of the walk
#     xy = np.array(poly.exterior.coords)[:-1]
#     m = len(xy)
#     f = np.asarray(flow, float)
#     wind = []
#     for i in range(m):
#         p0, p1 = xy[i], xy[(i + 1) % m]
#         t = p1 - p0
#         ln = float(np.hypot(*t))
#         if ln < 1e-9:
#             wind.append(False)
#             continue
#         t /= ln
#         outward = np.array([t[1], -t[0]])
#         wind.append(float(-(outward @ f)) >= min_w)
#     if not any(wind):
#         return []
#     if all(wind):
#         return [list(map(tuple, xy)) + [tuple(xy[0])]]
#     start = next(i for i in range(m) if wind[i] and not wind[i - 1])
#     arcs, cur = [], None
#     for k in range(m):
#         i = (start + k) % m
#         if wind[i]:
#             if cur is None:
#                 cur = [tuple(xy[i])]
#             cur.append(tuple(xy[(i + 1) % m]))
#         elif cur is not None:
#             arcs.append(cur)
#             cur = None
#     if cur is not None:
#         arcs.append(cur)
#     return arcs


# def _belt_rows(tpl):
#     """Row structure of the blocker template (flow towards +x): (R, tree spacing in a row, row pitch, [row records])."""
#     a, b, R = tpl["a"], tpl["b"], tpl["R"]
#     rows = {}
#     for i in range(len(a)):
#         rows.setdefault(round(float(a[i]), 1), []).append(i)
#     keys = sorted(rows)
#     Rm = float(np.mean(R))
#     sp = []
#     for k in keys:
#         bb = np.sort(b[rows[k]])
#         if len(bb) > 1:
#             sp.append(float(np.median(np.diff(bb))))
#     spacing = float(np.median(sp)) if sp else 2.4 * Rm
#     pitch = float(np.median(np.diff(keys))) if len(keys) > 1 else spacing
#     return Rm, spacing, pitch, [tpl["trees"][rows[k][0]] for k in keys]


# def _row_points(line, spacing, phase):
#     """Points every `spacing` m along a (multi)line, starting `phase` m from its start."""
#     out = []
#     parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
#     for g in parts:
#         if g.is_empty or g.length < 1e-6:
#             continue
#         s = phase
#         while s <= g.length + 1e-9:
#             p = g.interpolate(min(s, g.length))
#             out.append((p.x, p.y))
#             s += spacing
#     return out


# def _belt_adaptive(lines, Rmax, rows, pitch, spacing_ratio, occ0, free, gap, lo_r):
#     """Staggered rows along the windward stretches.  Row r runs along the stretch at Rmax + gap + r x pitch from the site boundary; at every
#     step the LARGEST crown (Rmax down to the minimum radius, steps of 0.5 m) that stands there without breaking a rule is planted, so that a narrow strip
#     between the boundary and a building gets small trees instead of a hole.  Returns [(x, y, R, row, stretch)]."""
#     loc = Occ()
#     loc.x, loc.y, loc.R = list(occ0.x), list(occ0.y), list(occ0.R)
#     opts = [float(r) for r in np.arange(Rmax, lo_r - 1e-9, -0.5)]
#     if opts[-1] > lo_r + 1e-9:
#         opts.append(float(lo_r))
#     pts = []
#     for r in range(rows):
#         base = Rmax + BELT_FIRST_ROW_GAP + r * pitch
#         for li, g in enumerate(lines):
#             s = (0.5 * spacing_ratio * Rmax if r % 2 else 0.0) + 0.5 * Rmax
#             while s < g.length - 0.25:
#                 p = g.interpolate(s)
#                 a, b = g.interpolate(max(s - 0.5, 0.0)), g.interpolate(min(s + 0.5, g.length))
#                 tx, ty = b.x - a.x, b.y - a.y
#                 ln = math.hypot(tx, ty)
#                 if ln < 1e-9:
#                     s += 1.0
#                     continue
#                 nx, ny = -ty / ln, tx / ln                         # into the site (the ring runs counter-clockwise)
#                 placed = None
#                 for R in opts:
#                     for dd in (0.0, -1.0, 1.0, -2.0, 2.0):
#                         d = max(base + dd, R + BELT_FIRST_ROW_GAP)
#                         x, y = p.x + d * nx, p.y + d * ny
#                         if (shapely.contains_xy(free(R), x, y) and loc.free_of(np.array([[x, y]]), np.array([R]), gap)[0]):
#                             placed = (x, y, R)
#                             break
#                     if placed:
#                         break
#                 if placed:
#                     loc.add(*placed)
#                     pts.append((*placed, r, li))
#                     s += spacing_ratio * placed[2]
#                 else:
#                     s += 1.0
#     return pts


# def _coverage(lines, pts):
#     """Share of the windward edge length that crown shadows (projected on the edge) cover."""
#     cover = 0.0
#     for li, g in enumerate(lines):
#         iv = []
#         for x, y, R, r, l2 in pts:
#             if l2 == li:
#                 s = g.project(Point(x, y))
#                 iv.append((max(0.0, s - R), min(g.length, s + R)))
#         cur_end, c = -1.0, 0.0
#         for a_, b_ in sorted(iv):
#             a_ = max(a_, cur_end)
#             if b_ > a_:
#                 c += b_ - a_
#                 cur_end = max(cur_end, b_)
#         cover += c
#     total = sum(g.length for g in lines)
#     return min(1.0, cover / max(total, 1e-9))


# def make_edge_blocker_individual(base_data, base, cfg, base_map, tpl, wind, seed=0):
#     """(layout, ops) or (None, reason): a belt of staggered tree rows along the whole windward edge(s) of the site, dense trees on both
#     sides of the pathways."""
#     import layout_reflow as RF
#     rng = np.random.default_rng(seed)
#     tc = cfg["tree"]
#     gap = tc["canopy_gap_factor"]
#     lo_r, hi_r = tc["radius_range_m"]
#     data = L.working_copy(base_data)
#     todo = L.patches_out_of_band(data, base, cfg)
#     sdata = dict(data, ponds=[]) if "pond" in todo else data
#     sc = L.Scene(sdata, cfg, base.osm_polys)
#     cap = L.max_tree_count(cfg, L.tree_count0(base, cfg), sc.pathway is not None)
#     free = FreeCache(sc, cfg, sc.ponds)
#     arcs = windward_arcs(sc.site, wind["flow"])
#     if not arcs:
#         return None, "the site has no windward edge"
#     Rm, spacing_t, pitch_t, row_recs = _belt_rows(tpl)
#     ratio = spacing_t / Rm                                  # centre-to-centre spacing in crown radii (template)
#     lines = [LineString(a) for a in arcs]
#     edge_len = float(sum(g.length for g in lines))

#     def attempt(path_share):
#         """Pathway trees with `path_share` of the tree-count limit, then the best belt the rest allows."""
#         bud = budgets(cfg, cap, sc.pathway is not None, path_share)
#         occ = Occ()
#         ptrees, pinfo = plant_pathway_dense(sc, cfg, occ, bud["path"], free, base_data["trees"])
#         struct_budget = bud["struct"] + max(0, bud["path"] - len(ptrees))
#         tried, best = [], None
#         for Rmax, rows in BELT_CONFIGS:
#             Rmax = float(np.clip(Rmax, lo_r, hi_r))
#             pts = _belt_adaptive(lines, Rmax, rows, pitch_t * Rmax / Rm, ratio, occ, free, gap, lo_r)
#             cov = _coverage(lines, pts)
#             tried.append((Rmax, rows, len(pts), round(cov, 3)))
#             if pts and len(pts) <= struct_budget:
#                 key = cov - 0.0004 * len(pts)              # the most complete edge; among equals the fewer trees
#                 if best is None or key > best[0]:
#                     best = (key, Rmax, rows, pts, cov)
#         thinned = False
#         if best is None:                                    # even the sparsest belt exceeds the budget: thin it evenly
#             Rmax, rows = BELT_CONFIGS[-1]
#             pts = _belt_adaptive(lines, float(Rmax), rows, pitch_t * Rmax / Rm, ratio, occ, free, gap, lo_r)
#             if pts:
#                 pts = [pts[i] for i in T._thin(struct_budget, len(pts))]
#                 best, thinned = (0, float(Rmax), rows, pts, _coverage(lines, pts)), True
#         return dict(share=path_share, bud=bud, occ=occ, ptrees=ptrees, pinfo=pinfo, best=best, tried=tried, thinned=thinned)

#     # the pathways get PATH_SHARE of the limit; when that leaves the belt short of BELT_COVER_TARGET the belt takes more, down to PATH_SHARE_MIN
#     ps0, ps_min = float(sd(cfg, "path_share")), float(sd(cfg, "path_share_min"))
#     shares = [ps0] if (sc.pathway is None or ps_min >= ps0) else [float(x) for x in np.linspace(ps0, ps_min, 4)]
#     chosen = None
#     for ps in shares:
#         a_ = attempt(ps)
#         if a_["best"] is None:
#             continue
#         if chosen is None or a_["best"][4] > chosen["best"][4] + 1e-9:
#             chosen = a_
#         if a_["best"][4] >= float(sd(cfg, "belt_cover_target")):
#             chosen = a_
#             break
#     if chosen is None:
#         return None, "no belt tree fits along the windward edge"
#     ptrees, pinfo, occ, bud, tried, thinned = (chosen[k] for k in ("ptrees", "pinfo", "occ", "bud", "tried", "thinned"))
#     _, Rmax, rows, pts, cov = chosen["best"]

#     added = list(ptrees)
#     for x, y, R, r, li in pts:
#         added.append(dict(x=float(x), y=float(y), R=float(R), rec=row_recs[min(r, len(row_recs) - 1)], hn=BELT_HN, prio=1, kind="belt"))
#         occ.add(x, y, R)
#     notes = {}
#     reserve = [_disc(a["x"], a["y"], a["R"] + EPS) for a in added]
#     try:
#         fitted = fit_patches(data, base, cfg, rng, reserve, (RF.static_blockers(sc, "pond"), RF.static_blockers(sc, "lawn")), cfg["patch"].get("discretization_m", 1.0))
#     except ValueError as e:
#         return None, str(e)
#     if fitted:
#         notes["area_rescaled"] = fitted
#     res, why = assemble(data, base_data, base, cfg, base_map, sc, added, cap, rng, notes)
#     if res is None:
#         return None, why
#     data, mine, removed_ids, counts = res
#     n_belt = counts.get("belt", 0)
#     if n_belt < 3:
#         return None, "fewer than 3 belt trees survive the rules"
#     Rs = [p[2] for p in pts]
#     info = {"op": "template", "kind": "wind_blocker_edge", "template": Path(tpl["path"]).name, "season": "cold",
#             "wind_from_deg_model": round(wind["from_deg"], 1),
#             "flow_unit_vector": [round(wind["flow"][0], 4), round(wind["flow"][1], 4)],
#             "rotation_deg": round(math.degrees(math.atan2(wind["flow"][1], wind["flow"][0])), 1),
#             "scale": round(Rmax / Rm, 3), "crown_radius_max_m": Rmax, "crown_radius_min_m": float(min(Rs)), "rows": rows,
#             "windward_stretches": len(lines), "windward_edge_length_m": round(edge_len, 1),
#             "edge_covered_share": round(cov, 3),
#             "belt_thinned_to_budget": thinned,
#             "trees_in_template": len(tpl["trees"]), "trees_placed": len(mine), "trees_belt": n_belt,
#             "trees_pathway": counts.get("pathway", 0), "pathway_planting": pinfo,
#             "tree_count_limit": int(cap), "tree_budget": bud, "belt_configs_tried": [list(t) for t in tried],
#             "base_trees_replaced": len(removed_ids),
#             "template_tree_ids": sorted(mine, key=lambda i: int(i.split()[-1])),
#             "rules_enforced_on_base": notes,
#             "distance_from_upwind_site_edge_m": round(Rmax + BELT_FIRST_ROW_GAP, 1), "close_to_inlet": True}
#     return (data, [info]), None


# # ============================================================================= one-line description (for the logs)
# def describe(kind, name, path, spec, o):
#     pw = o.get("pathway_planting") or {}
#     if o.get("mode") == "simple":
#         bad = o.get("rule_violations_of_the_result") or {}
#         sbad = ", ".join(f"{k} x{v}" for k, v in bad.items()) if bad else "none"
#         return (f"{spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), simple mode (no ponds / lawns / trees of the base, "
#                 f"no pathway planting, no rule check except the site boundary): catcher on {o['buildings_with_catcher']} of {o['buildings']} buildings, {o['ponds']} ponds {o['pond_area_m2']:g} m2, "
#                 f"{o['trees_catcher']} catcher trees; site boundary: {o.get('side_flipped_buildings', 0)} catcher(s) put on the other side of the wind, {o.get('ponds_clipped_to_site', 0)} pond(s) clipped, "
#                 f"{o.get('trees_dropped_outside_site', 0)} tree(s) dropped; other rule breaks left as drawn: {sbad}")
#     ptxt = (f"; {o.get('trees_pathway', 0)} pathway trees on both sides (one per {pw.get('mean_spacing_m', '?')} m of road edge, closest allowed "
#             f"{pw.get('min_spacing_m', '?')} m, crown radius {pw.get('crown_radius_m', '?'):g} m)" if pw.get("enabled") else "; no pathways")
#     if kind == "cool_wind_catcher_all":
#         miss = o.get("buildings_without_catcher") or {}
#         return (f"{spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), "
#                 f"catcher on {o['buildings_with_catcher']} of {o['buildings']} buildings, {o['ponds']} ponds {o['pond_area_m2']:g} m2, "
#                 f"{o['trees_catcher']} catcher trees{ptxt}" + (f"  (no room at: {', '.join(miss)})" if miss else ""))
#     if kind == "wind_blocker_edge":
#         return (f"{spec['label']}: {name}  <- {path.name}, {spec['season']}-case wind from {o['wind_from_deg_model']:g} deg (model), "
#                 f"belt of {o['rows']} rows (crown radius {o['crown_radius_min_m']:g}-{o['crown_radius_max_m']:g} m, {o['trees_belt']} trees) along {o['windward_stretches']} windward edge(s) "
#                 f"({o['windward_edge_length_m']:g} m, {round(100 * o['edge_covered_share'])} % shaded){ptxt}"
#                 + ("  (belt thinned to the tree-count limit)" if o.get("belt_thinned_to_budget") else ""))
#     return None