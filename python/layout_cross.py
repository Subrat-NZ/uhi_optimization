"""
layout_cross.py  --  crossover and mutation of whole layouts for the NSGA-II driver.

Every child is built so that all Stage 1 rules hold (it is validated with layout_ops.new_violations and
rejected otherwise):

    buildings   each physical building (a whole stack of parts) takes the translation of parent A or of parent B (or a 75 / 50 / 25 %
                blend of it when that spot is taken); stacked parts move together as always.  Trees / lawns / ponds that
                stand on a building are not crossed over: they follow their building (layout_ops.attach_hosted)
    lawns/ponds each category comes from A or B, is cut back around the new building positions and refilled to
                the allowed total area (layout_reflow.reflow_patches)
    trees       one side of a random line through the site comes from A, the other from B; trees that break a
                rule are moved or dropped (fix_trees), the count is brought to the mean of the parents,
                heights follow the location zones
"""
import copy
import math

import numpy as np
from shapely import affinity
from shapely.geometry import Point

import layout_ops as L
import layout_reflow as R


def _renumber(trees):
    for i, t in enumerate(trees, 1):
        t["id"] = f"Tree {i}"
    return trees


def crossover(pa, pb, base, cfg, rng, base_map):
    """Child of two layouts (dicts) that share the same base.  Returns (child, info) or None."""
    n = len(base.groups)
    pa, pb = L.working_copy(pa), L.working_copy(pb)          # without what stands on a building (it follows its building, see attach_hosted)
    sa, sb = L.Scene(pa, cfg, base.osm_polys), L.Scene(pb, cfg, base.osm_polys)
    if sa.group_ids != base.group_ids or sb.group_ids != base.group_ids:
        return None
    child = L.working_copy(base.d)
    sc = L.Scene(child, cfg, base.osm_polys)
    offs = [(sa.centroid(g) - base.centroid(g), sb.centroid(g) - base.centroid(g)) for g in range(n)]
    cur = [np.zeros(2) for _ in range(n)]
    from_a = 0
    # ---- buildings
    for gi in rng.permutation(n):
        gi = int(gi)
        order = [0, 1] if rng.random() < 0.5 else [1, 0]
        done = False
        for src in order:
            for f in (1.0, 0.75, 0.5, 0.25):
                off = offs[gi][src] * f
                if float(np.hypot(*off)) < 1e-9:
                    continue
                poly = affinity.translate(base.group_polys[gi], float(off[0]), float(off[1]))
                if L._building_move_ok(sc, base, cfg, gi, gi, poly):
                    d = off - cur[gi]
                    L.translate_group(child, sc, gi, float(d[0]), float(d[1]))
                    sc.group_polys[gi], cur[gi] = poly, off
                    from_a += int(src == 0)
                    done = True
                    break
            if done:
                break
        if not done and not L._building_move_ok(sc, base, cfg, gi, gi, sc.group_polys[gi]):
            return None
    # ---- lawns and ponds
    srcs = [pa, pb] if rng.random() < 0.5 else [pb, pa]
    rep = None
    for attempt in range(2):
        child["grassAreas"] = copy.deepcopy((srcs[0] if attempt == 0 else srcs[1])["grassAreas"])
        child["ponds"] = copy.deepcopy((srcs[1] if attempt == 0 else srcs[0])["ponds"])
        rep = R.reflow_patches(child, base, cfg, rng)
        if rep is not None:
            break
    if rep is None:
        return None
    # ---- trees: split along a random line
    minx, miny, maxx, maxy = base.site.bounds
    cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
    th = float(rng.uniform(0, math.pi))
    nx, ny = math.cos(th), math.sin(th)
    c0 = float(rng.normal(0, 0.15 * base.site_diameter))
    side = lambda t: (t["position"]["x"] - cx) * nx + (t["position"]["y"] - cy) * ny - c0
    trees = [copy.deepcopy(t) for t in pa["trees"] if side(t) >= 0] + [copy.deepcopy(t) for t in pb["trees"] if side(t) < 0]
    cap = L.max_tree_count(cfg, L.tree_count0(base, cfg), sc.pathway is not None)
    target = min(cap, int(round(0.5 * (len(pa["trees"]) + len(pb["trees"])))))
    if len(trees) > target:
        keep = sorted(rng.permutation(len(trees))[:target])
        trees = [trees[i] for i in keep]
    child["trees"] = _renumber(trees)
    sc2 = L.Scene(child, cfg, base.osm_polys)
    R.fix_trees(child, base, cfg, rng, pl=R.TreePlacer(sc2, cfg, child), sc=sc2)
    sc2 = L.Scene(child, cfg, base.osm_polys)
    pl = R.TreePlacer(sc2, cfg, child)
    T, tc, added = child["trees"], cfg["tree"], 0
    while len(T) < target and added < 3 * target + 10:
        added += 1
        R0 = float(np.clip(L._rt(rng.uniform(*tc["new_tree_radius_range_m"]), tc["size_step_m"]), *tc["radius_range_m"]))
        spot = pl.place(rng, R0)
        if spot is None:
            break
        t = L.new_tree(cfg, T, rng)
        t["id"] = f"Tree {len(T) + 1000 + added}"
        R._set_pos(t, spot[0], spot[1])
        L.set_tree_geometry(t, spot[2], float(t["height"]), tc)
        T.append(t)
        pl.xs.append(spot[0]); pl.ys.append(spot[1]); pl.Rs.append(spot[2])
    _renumber(T)
    sc3 = L.Scene(child, cfg, base.osm_polys)
    R.rezone_heights(child, sc3, cfg, rng)
    shade = R.ensure_pathway_shade(child, base, cfg, rng)      # road shade is mandatory for every individual
    L.refresh_metadata(child)
    # ---- validate (rules inherited from the base are tolerated, nothing new is)
    sc3 = L.Scene(child, cfg, base.osm_polys)
    viol = L.new_violations(sc3, base, base_map)
    if viol:
        bad = {i for t, ids, _ in viol if t.startswith("tree_") for i in ids}
        rest = [v for v in viol if not v[0].startswith("tree_")]
        if rest or not bad or len(bad) > 8:
            return None
        child["trees"] = _renumber([t for t in child["trees"] if t["id"] not in bad])
        L.refresh_metadata(child)
        if L.new_violations(L.Scene(child, cfg, base.osm_polys), base, base_map):
            return None
    return child, {"op": "crossover", "buildings_from_a": from_a, "buildings": n, "trees": len(child["trees"]),
                   "split_angle_deg": round(math.degrees(th), 1), "pathway_shade": shade}


def mutate(data, base, cfg, rng, base_map, n_ops, strength):
    """n_ops small random operations (layout_ops.try_op).  Returns (data, ops) - ops may be empty."""
    cur, ops = L.working_copy(data), []
    for _ in range(n_ops):
        r = L.try_op(cur, base, cfg, rng, strength, base_map)
        if r is None:
            continue
        cur, info = r
        ops.append(info)
    return cur, ops
