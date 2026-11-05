#!/usr/bin/env python3
"""
layout_ops.py  --  edit and check layout JSONs (buildings, trees, ponds, lawns)

Used by generate_candidates.py.  Everything here works on the layout JSON that comes from the
scene editor (keys: site, buildings, trees, footprints, ponds, grassAreas, osmBuildings, ...).

What may change (Stage 1 rules, all values editable in input/constraints.json):
  * buildings : translation only (no rotation, no shape / size / height change).  A building
                stacked on another (set-back masses, internal floors: "Building 3",
                "building_3_mass_1", "building_3_internal_floor_2" ...) is ONE building: all its
                parts move together and its footprint for every clearance / overlap rule is the
                union of the parts.  Max shift = 10 % of the site diameter from the ORIGINAL
                (base) position; shifts snap to the scene grid (1 m).
  * on a      : trees, lawns and ponds that stand on a building (centre / most of the outline inside
    building    the footprint of its stack in the BASE layout) are part of that building: they are
                taken out while the optimiser works (they are not counted in the tree / lawn / pond
                budgets and are never edited) and put back, rigidly moved with their building, when a
                layout is written (attach_hosted).
  * trees     : move, resize (crown radius 1-6 m, height by location), add; may be dropped only
                when a building move breaks a rule.  Crown inside site; centre-to-building
                distance >= crown radius + 5 % of the smallest editable building width; canopy-
                edge gap 10 % (centre distance >= 1.10 (Ri + Rj)); no tree-pond overlap (tree-
                lawn overlap is allowed); height 12-15 m in the open, 8-10 m near buildings.
  * ponds /   : shift, rotate, stretch, scale, reshape, add or remove patches.  Inside the site,
    lawns       no overlap with buildings or with each other, patch area >= minimum, and the
                TOTAL area of each category stays within A0 .. 1.10 A0 (A0 = base total).
                Pond depth never changes.
Rule violations that already exist in the BASE layout are 'inherited': they are tolerated as long
as they do not get worse (the base layout has to be repaired separately, see the work plan).
"""

import copy
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# ----------------------------------------------------------------------------- configuration
DEFAULTS = {
    "building": {"max_shift_frac_of_site_diameter": 0.10, "snap_step_m": None,   # None -> scene gridSize
                 "overlap_tol_m2": 1e-3,
                 # every building move is at least this fraction of the building's LONGEST PROJECTED SPAN
                 # (max of east-west and north-south extent) and at most the 10 % limit above
                 "min_shift_frac_of_length": 0.5},
    "tree": {"radius_range_m": [1.0, 6.0], "canopy_gap_factor": 1.10,
             # centre-to-building-edge distance must be >= factor * crown radius (+ optional extra)
             "building_clearance_factor": 1.10, "building_clearance_extra_frac_of_width": 0.0,
             "clearance_applies_to_osm": True,
             "height_open_m": [12.0, 15.0], "height_near_building_m": [8.0, 10.0],
             "height_pathway_m": [8.0, 12.0],      # pathways (= roads = paths; see 'zones' below)
             "tree_on_pathway_allowed": False,     # crowns may NOT overlap pathways (roads = paths); True = allowed
             "near_building_distance_m": 10.0,      # ASSUMPTION: what counts as 'near buildings'
             "max_count_factor": 1.5,               # ASSUMPTION: tree budget = 1.5 x base count
             # a tree whose centre is within this distance of a pathway edge is a 'street / path' tree
             # (height band height_pathway_m).  Needed so that trees planted along the sides get that band.
             "pathway_zone_distance_m": 6.0,
             # A base WITHOUT trees gets trees (otherwise nothing could be planted): canopy cover =
             # canopy_cover_frac_if_missing x the open space (site minus buildings, OSM, pathways, lawns),
             # mean crown = the middle of new_tree_radius_range_m.  This count replaces 'base count' in
             # the +10 pct rule and in the max_count_factor cap.  Look of a new tree: template.
             "create_if_missing": True, "canopy_cover_frac_if_missing": 0.10,
             "template": {"height": 7, "crownDiameter": 4, "dimensions": {"trunkHeight": 2.8000000000000003, "trunkRadius": 0.3, "crownHeight": 4.2, "crownRadius": 2, "totalHeight": 7, "crownDiameter": 4}, "properties": {"species": "Generic Tree", "maturity": "Mature", "shadingArea": 12.566370614359172, "volume": 34.30200298699575}, "materials": {"trunk": {"color": 9127187, "opacity": 1, "transparent": False, "roughness": 0.8, "metalness": 0, "emissive": 0}, "crown": {"color": 7048739, "opacity": 1, "transparent": False, "roughness": 0.9, "metalness": 0, "emissive": 0}, "vegetation": {"leaf_area_index": 4, "height": 8, "crown_diameter": 6, "transpiration_rate": 0.6, "canopy_density": 0.8}}},
             # Generations plant trees along BOTH sides of every pathway wherever the earlier rules leave room
             # (crown radius 1-6 m, centre >= 1.1 R from buildings, crown gaps >= 1.1 (Ri+Rj), crown inside the
             # site, off ponds); slots without room are skipped.  These trees are on top of the +10 % and of
             # max_count_factor (the tree-count cap grows by max_new_trees).
             "pathway_planting": {"enabled": True, "spacing_m": 10.0,         # centre-to-centre along a pathway side
                                  "radius_range_m": [2.0, 4.0],
                                  "canopy_gap_to_path_m": 0.5,               # crown edge >= this far from the path edge
                                  "fill_fraction_range": [0.6, 1.0],         # share of the slots that is used (random)
                                  "max_new_trees": 60,                       # hard budget per candidate
                                  "slide_tries_m": [0.0, 1.5, -1.5, 3.0, -3.0],    # shift along the path when a slot is blocked
                                  # ---- ROAD SHADE: a rule for EVERY individual (random, seeds, crossover children, mutations).
                                  # Samples every sample_step_m along both edges of every pathway; a sample is shaded when a crown
                                  # edge is within reach_m of it.  Samples where not even a crown of min_radius_m fits (buildings,
                                  # ponds, site edge) do not count.  >= min_coverage of the others must be shaded, otherwise the
                                  # layout is rejected; a repair (layout_reflow.ensure_pathway_shade) plants trees up to
                                  # target_coverage first, picks the crown radius that fits (shade_radius_range_m), enlarges nearby
                                  # trees (up to grow_max_radius_m) and, when the tree-count limit is reached, moves trees that stand far
                                  # from every road (never the protected seed trees).
                                  "shade_required": True, "min_coverage": 0.80, "target_coverage": 0.90,
                                  "reach_m": 3.0, "sample_step_m": 2.0, "min_radius_m": 1.5,
                                  "shade_radius_range_m": [2.5, 5.0], "grow_max_radius_m": 6.0,
                                  "max_relocated_frac": 0.5},
             "max_dropped_trees": 3, "trunk_height_frac": 0.4, "trunk_radius_m": 0.3,
             "move_sigma_m": 15.0, "relocate_probability": 0.25, "resize_sigma_m": 2.0,
             "new_tree_radius_range_m": [1.5, 4.0], "size_step_m": 0.5},
    # ponds: max_patches is raised to the number of buildings (one pond per building is possible) unless
    # max_patches_follows_buildings is false; the limit never goes below max_patches (see patch_limit)
    "pond": {"max_patches": 4, "max_patches_follows_buildings": True, "min_area_m2": 25.0, "area_factor_range": [1.0, 1.10],
             # distance every pond keeps from every building (generated footprints and OSM buildings); overlap is never allowed
             "building_clearance_m": 2.0,
             # a base without ponds gets ponds: total area = open_space_frac_if_missing x the open space
             # (site minus buildings, pathways and lawns).  Depth is fixed, never a design variable.
             "create_if_missing": True, "open_space_frac_if_missing": 0.10, "default_depth_m": 2.0,
             "template": {"material": {"thermal_conductivity": 0.6, "density": 1000, "specific_heat": 4186,
                                       "surface_roughness": 0.0001, "albedo": 0.06}}},
    "lawn": {"max_patches": 6, "min_area_m2": 25.0, "area_factor_range": [1.0, 1.10],
             # distance every lawn keeps from every building (generated footprints and OSM buildings); overlap is never allowed
             "building_clearance_m": 1.0,
             # a base without lawns gets lawns: A0 = open_space_frac_if_missing x the open space (like ponds)
             "create_if_missing": True, "open_space_frac_if_missing": 0.10,
             # Irrigation budget in litres/day (NOT pond water).  The irrigation depth is FIXED at 5 mm/day,
             # so lawn area <= water / 5  (1 L on 1 m2 = 1 mm).  2000 L/day -> 400 m2.
             # null -> the budget is what the BASE lawn needs:  A0 * 0.005 m * 1000 L/m3 = 5 * A0 L/day,
             # i.e. the total lawn area cannot grow (position, shape and count may still change).
             "water_available_l_per_day": None,
             "template": {"material": {"leaf_area_index": 3, "height": 0.3, "transpiration_rate": 0.5,
                                       "surface_roughness": 0.05, "albedo": 0.25}}},
    # Seed individuals (layout_seeds.py): how the tree-count limit  floor(max_count_factor x n0) + max_new_trees  is shared out.  Pathway trees get
    # path_share of it, the cool wind catchers / the wind-blocker belt struct_share, base trees the rest.  The belt may take budget from the pathways
    # (down to path_share_min) until it shades belt_cover_target of the windward edge.  Raise max_count_factor / max_new_trees for denser seeds.
    "seeds": {"catcher_style": "arc", "air_corridor": True, "path_share": 0.45, "struct_share": 0.45, "path_share_min": 0.30, "path_crown_radius_m": 2.5, "belt_cover_target": 0.80},
    # Pathways (roads and paths are the same thing).  Read from the scene's 'pathways' list (and
    # 'roads' / 'osmRoads' if present).  You may also give explicit polygons:
    # {"pathway": [[[x,y],...], ...]}; default width (m) is used when a pathway is a centre line.
    # allow_unreadable_pathways: true = go on (with a warning) when the roads of the scene cannot be read; trees may then land on roads.
    "zones": {"pathway": [], "default_pathway_width_m": 6.0, "allow_unreadable_pathways": False},
    # NEW ponds / lawns are seeded in the largest open areas: the point with the most free room around it
    # (pole of inaccessibility of the free space) is rank 1, the next free spot rank 2, ...  A patch is grown
    # as the part of a disc that is VISIBLE from that point (never encloses a building, no holes).
    # largest_open_rank_probs: chance of using rank 1, 2, 3 ...  ([1.0] = always the largest open area)
    "patch": {"largest_open_rank_probs": [0.8, 0.2], "edge_margin_m": 0.25, "rays": 360, "min_width_m": 4.0,
              "shift_sigma_m": 20.0, "rotate_sigma_deg": 60.0, "stretch_sigma": 0.5,
              "scale_sigma": 0.3, "jitter_sigma_frac": 0.15, "new_patch_share_range": [0.15, 0.45],
              "new_patch_aspect_range": [1.0, 3.0], "new_patch_vertices": 12,
              "discretization_m": 1.0,
              # a clearance counts as kept when the distance is >= clearance - this (the outlines are resampled every 1 m)
              "clearance_tol_m": 0.05},
    "ops": {"category_weights": {"building": 0.25, "tree": 0.40, "pond": 0.15, "lawn": 0.20},
            "tree_weights": {"move": 0.40, "resize": 0.30, "add": 0.25, "remove": 0.0},
            "patch_weights": {"shift": 0.30, "rotate": 0.10, "stretch": 0.15, "scale": 0.15,
                              "jitter": 0.10, "add": 0.12, "remove": 0.08}},
    # a child that is practically a copy of an existing individual is not simulated (a CFD run costs hours and the
    # run-to-run noise of the simulation is larger than the effect of a few centimetres)
    "novelty": {"enabled": True,
                "min_building_shift_m": 1.0,        # a child is "the same" as an individual only if ALL THREE hold:
                "min_tree_changes": 3,              #   no building moved more than this, fewer than this many trees
                "min_patch_change_frac": 0.02},     #   differ (position / size), and each lawn / pond total changed < 2 %
    # ranges applied to every GENERATION candidate (mutations stay small)
    "generation": {"all_buildings_move": True,          # every physical building moves (min .. max shift)
                   "tree_count_increase_frac": 0.10,     # trees = base count + 10 %
                   "tree_relocate_fraction": 0.25, "tree_resize_fraction": 0.50,
                   "normalize_tree_heights": True,       # all heights follow the location zones
                   "min_shift_relax_steps": [1.0, 0.5, 0.25],   # factor on the 0.5 L minimum if it is impossible
                   "building_attempts": 6,                      # random building orders tried per candidate
                   "min_extra_ops": {"pond": 1, "lawn": 1},
                   "attempts": 6},
    "tol": 1e-6,
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_constraints(path=None):
    p = Path(path) if path else None
    return _merge(DEFAULTS, json.loads(p.read_text()) if p and p.exists() else {})


# ----------------------------------------------------------------------------- JSON <-> geometry
def _num(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def _xy(vs, key=("x", "y")):
    return [(v[key[0]], v[key[1]]) for v in vs]


def _poly(vs, key=("x", "y")):
    return Polygon(_xy(vs, key))


STACK_Z_TOL = 0.30        # m: two parts are stacked when their heights touch / overlap within this gap ...
STACK_OVERLAP = 0.50      # ... and their footprints overlap by this share of the smaller one (or they are the same building family)
_FAMILY = re.compile(r"^\s*(building)[\s_\-]*(\d+)", re.I)
_GROUP_CACHE = {}


def _bld_poly(b):
    p = Polygon(_xy(b["geometry"]["footprint"]))
    return p if p.is_valid else p.buffer(0)


def _bld_z(b):
    z0 = b.get("position", {}).get("z")
    if z0 is None:
        bv = b.get("dimensions", {}).get("baseVertices") or [{}]
        z0 = bv[0].get("z", 0.0)
    return float(z0 or 0.0), float(z0 or 0.0) + float(b.get("height", 0.0) or 0.0)


def _building_family(b):
    """('building', N) for ids like 'Building 3', 'building_3', 'building_3_mass_1', 'building_3_internal_floor_2', else None."""
    m = _FAMILY.match(str(b.get("id", "")))
    return (m.group(1).lower(), int(m.group(2))) if m else None


def building_groups(data):
    """Index lists of the parts of ONE physical building.  A part belongs to a stack when
        * it has the same footprint as another part (floors), or
        * it belongs to the same building family ('Building 3' / 'building_3_mass_1' / 'building_3_internal_floor_2') and its footprint
          overlaps the other part, or
        * its footprint overlaps >= 50 % of the smaller footprint AND the heights touch (a mass sitting on a slab).
    The result depends on shapes, heights and ids only, so it is the same after any translation of the stack."""
    B = data["buildings"]
    n = len(B)
    key = tuple((b["id"], len(b["geometry"]["footprint"]), round(_bld_z(b)[0], 3), round(_bld_z(b)[1], 3),
                 round(_bld_poly(b).area, 3)) for b in B)
    hit = _GROUP_CACHE.get(key)
    if hit is not None:
        return [list(g) for g in hit]
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            par[max(a, b)] = min(a, b)

    polys = [_bld_poly(b) for b in B]
    zs = [_bld_z(b) for b in B]
    fam = [_building_family(b) for b in B]
    sig = {}
    for i, b in enumerate(B):
        k = tuple(sorted((round(v["x"], 6), round(v["y"], 6)) for v in b["geometry"]["footprint"]))
        if k in sig:
            union(i, sig[k])
        else:
            sig[k] = i
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            pi, pj = polys[i], polys[j]
            if pi.is_empty or pj.is_empty or not pi.intersects(pj):
                continue
            a = pi.intersection(pj).area
            if a <= 1e-6:
                continue
            m = min(pi.area, pj.area)
            gap = max(zs[i][0] - zs[j][1], zs[j][0] - zs[i][1])
            if (fam[i] is not None and fam[i] == fam[j] and a > 0.01 * m) or (gap <= STACK_Z_TOL and a >= STACK_OVERLAP * m):
                union(i, j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    groups = sorted(out.values(), key=lambda g: g[0])
    if len(_GROUP_CACHE) > 256:
        _GROUP_CACHE.clear()
    _GROUP_CACHE[key] = [tuple(g) for g in groups]
    return groups


def group_polygon(data, g):
    """Footprint of one physical building: the footprint of its parts if they all have the same one, else the union of the parts."""
    ps = [_bld_poly(data["buildings"][i]) for i in g]
    sigs = {tuple(sorted((round(x, 6), round(y, 6)) for x, y in p.exterior.coords)) for p in ps if p.geom_type == "Polygon"}
    if len(ps) == 1 or (len(sigs) == 1 and all(p.geom_type == "Polygon" for p in ps)):
        return _poly(data["buildings"][g[0]]["geometry"]["footprint"])
    u = unary_union(ps)
    if u.geom_type != "Polygon":
        u = u.buffer(0.01, join_style=2).buffer(-0.01, join_style=2)
        if u.geom_type != "Polygon":
            u = u.convex_hull
    return u


def _patch_poly(p):
    q = _poly(p["coordinates"])
    return q if q.is_valid else q.buffer(0)


def hosted_indices(data, groups, gpolys, ponds=None, lawns=None, only_ids=None):
    """Trees, ponds and lawns that the optimiser must not touch: {'trees': {i: group}, 'ponds': {...}, 'grassAreas': {...}}.
    - standing ON a building (group index >= 0): a tree when its centre is inside the footprint of the stack; a pond / lawn when more than half of
      its outline is.  They follow their building.
    - OUTSIDE THE SITE (group index -1): a pond / lawn with no part inside the site boundary.  It stays exactly as it is in the base layout, in
      every individual and offspring (never moved, resized, removed, crossed over or counted in the pond / lawn area budget).
    only_ids ({'trees': set, ...}): count only items with these ids (the ones that are hosted / outside in the BASE layout)."""
    out = {"trees": {}, "ponds": {}, "grassAreas": {}}
    T = data.get("trees", [])
    if T and gpolys:
        xs = np.array([t["position"]["x"] for t in T], float)
        ys = np.array([t["position"]["y"] for t in T], float)
        for gi, gp in enumerate(gpolys):
            m = shapely.contains_xy(gp, xs, ys) | shapely.contains_xy(gp.buffer(1e-6), xs, ys)
            for i in np.where(m)[0]:
                if only_ids is None or T[int(i)]["id"] in only_ids["trees"]:
                    out["trees"].setdefault(int(i), gi)
    site = None
    try:
        site = _poly(data["site"]["coordinates"])
        site = site if site.is_valid else site.buffer(0)
    except Exception:
        site = None
    for key, polys in (("ponds", ponds), ("grassAreas", lawns)):
        P = data.get(key, [])
        if polys is None:
            polys = [_patch_poly(p) for p in P] if P else []
        for i, q in enumerate(polys):
            if q.is_empty or q.area <= 0 or (only_ids is not None and P[i]["id"] not in only_ids[key]):
                continue
            qq = q if q.is_valid else q.buffer(0)
            for gi, gp in enumerate(gpolys or []):
                if q.bounds[0] > gp.bounds[2] or q.bounds[2] < gp.bounds[0] or q.bounds[1] > gp.bounds[3] or q.bounds[3] < gp.bounds[1]:
                    continue
                if qq.intersection(gp).area > 0.5 * qq.area:
                    out[key][i] = gi
                    break
            if i not in out[key] and site is not None:
                sb = site.bounds
                apart = q.bounds[0] > sb[2] or q.bounds[2] < sb[0] or q.bounds[1] > sb[3] or q.bounds[3] < sb[1]
                if apart or qq.intersection(site).area <= 1e-6:
                    out[key][i] = -1                              # outside the site boundary: frozen
    return out


def ground_view(data, only_ids=None):
    """`data` without the trees / lawns / ponds that stand on a building (the same object if there are none).  Use it on a layout that
    comes from outside (the base, a file, a parent) - NOT on data in the middle of an operation, where a misplaced tree must stay visible."""
    groups = building_groups(data)
    gp = [group_polygon(data, g) for g in groups]
    h = hosted_indices(data, groups, gp, only_ids=only_ids)
    if not any(h.values()):
        return data
    out = dict(data)
    for key in ("trees", "ponds", "grassAreas"):
        out[key] = [x for i, x in enumerate(data.get(key, [])) if i not in h[key]]
    return out


def _shift_xy(obj, dx, dy):
    """Add (dx, dy) to every {'x', 'y'} point found anywhere inside obj (z and everything else untouched)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("x"), (int, float)) and isinstance(obj.get("y"), (int, float)):
            obj["x"] += dx
            obj["y"] += dy
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _shift_xy(v, dx, dy)
    elif isinstance(obj, list):
        for v in obj:
            _shift_xy(v, dx, dy)


def hosted_items(base_data):
    """Items standing on a building of the BASE layout: {'trees': [(item, group index)], 'ponds': [...], 'grassAreas': [...]} (deep copies),
    plus 'centroids' (group centroids of the base)."""
    groups = building_groups(base_data)
    gp = [group_polygon(base_data, g) for g in groups]
    h = hosted_indices(base_data, groups, gp)
    out = {k: [(copy.deepcopy(base_data[k][i]), gi) for i, gi in sorted(h[k].items())] for k in h}
    out["ids"] = [tuple(base_data["buildings"][i]["id"] for i in g) for g in groups]
    out["centroids"] = [(p.centroid.x, p.centroid.y) for p in gp]
    return out


def attach_hosted(data, base):
    """A COPY of layout `data` (which holds only the free trees / lawns / ponds) with the items of the base that stand on a building put
    back, each moved by exactly the translation of its building.  `base`: a Scene or the base layout dict.  Idempotent: items that stand
    on a building in `data` are replaced by the base's.  Ground trees whose id collides with a hosted tree get a fresh id."""
    hb = base.hosted_base if hasattr(base, "hosted_base") else hosted_items(base)
    if not any(hb[k] for k in ("trees", "ponds", "grassAreas")):
        return data
    data = ground_view(data, {k: {it["id"] for it, _ in hb[k]} for k in ("trees", "ponds", "grassAreas")})
    groups = building_groups(data)
    ids = [tuple(data["buildings"][i]["id"] for i in g) for g in groups]
    cen = [group_polygon(data, g).centroid for g in groups]
    off = {}
    for gi, bid in enumerate(hb["ids"]):
        if bid in ids:
            c = cen[ids.index(bid)]
            off[gi] = (c.x - hb["centroids"][gi][0], c.y - hb["centroids"][gi][1])
    out = dict(data)
    for key in ("groups", "metadata"):
        if key in data:
            out[key] = copy.deepcopy(data[key])
    for key in ("trees", "ponds", "grassAreas"):
        add = []
        for item, gi in hb[key]:
            it = copy.deepcopy(item)
            dx, dy = off.get(gi, (0.0, 0.0))
            if dx or dy:
                _shift_xy(it, dx, dy)
            add.append(it)
        out[key] = list(data.get(key, [])) + add
    taken = {str(t["id"]) for t in out["trees"][len(data.get("trees", [])):]}
    nxt = max([int(str(t["id"]).split()[-1]) for t in out["trees"] if str(t["id"]).split()[-1].isdigit()] + [0])
    ground = []
    for t in out["trees"][:len(data.get("trees", []))]:
        if str(t["id"]) in taken:
            nxt += 1
            t = dict(t, id=f"Tree {nxt}")
        ground.append(t)
    out["trees"] = ground + out["trees"][len(data.get("trees", [])):]
    refresh_metadata(out)
    return out


def patch_limit(cfg, cat, n_buildings):
    """Largest allowed number of ponds / lawns.  Ponds: max(max_patches, number of physical buildings) while
    pond.max_patches_follows_buildings is true (default); lawns: max_patches."""
    pc = cfg[cat]
    n = int(pc["max_patches"])
    if cat == "pond" and pc.get("max_patches_follows_buildings", True):
        n = max(n, int(n_buildings))
    return n


def _tree_R(t):
    return float(t.get("dimensions", {}).get("crownRadius", t["crownDiameter"] / 2.0))


_GEOM_KEYS = ("coordinates", "points", "vertices", "path", "centerline", "centreline", "polyline", "nodes",
              "baseVertices", "outline", "footprint", "ring", "polygon", "line", "geometry")


def _pt3(v):
    """(x, y, z or None) of one point: {'x','y'[,'z']} or [x, y[, z]]"""
    if isinstance(v, dict):
        if "x" in v and ("y" in v or "z" in v):
            return (float(v["x"]), float(v["y"]) if "y" in v else float(v["z"]), float(v["z"]) if "z" in v else None)
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(t, (int, float)) for t in v[:2]):
        return (float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 and isinstance(v[2], (int, float)) else None)
    return None


def _points_of(raw, depth=0):
    """all (x, y, z) points found in a geometry value: list of points, single point, GeoJSON-like nested rings, or a dict that holds one of _GEOM_KEYS"""
    if raw is None or depth > 5:
        return []
    if isinstance(raw, dict):
        p = _pt3(raw)
        if p is not None:
            return [p]
        for k in _GEOM_KEYS:
            if raw.get(k):
                out = _points_of(raw[k], depth + 1)
                if out:
                    return out
        return []
    if isinstance(raw, (list, tuple)):
        if not raw:
            return []
        if all(isinstance(t, (int, float)) for t in raw):                     # flat numbers: one point, or x1, y1, x2, y2, ...
            if len(raw) in (2, 3):
                return [_pt3(raw)]
            return [(float(raw[i]), float(raw[i + 1]), None) for i in range(0, len(raw) - 1, 2)] if len(raw) % 2 == 0 else []
        if all(isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple, dict)) for v in raw):   # rings: take the first
            return _points_of(raw[0], depth + 1)
        out = []
        for v in raw:
            p = _pt3(v)
            if p is not None:
                out.append(p)
            else:
                out += _points_of(v, depth + 1)
        return out
    return []


def _plan_xy(pts):
    """plan coordinates of the points: (x, y); when y is the same everywhere and z varies, the scene stores plan y as z (OSM convention)"""
    arr = [(p[0], p[1], p[2]) for p in pts]
    ys = [a[1] for a in arr]
    zs = [a[2] for a in arr if a[2] is not None]
    if len(zs) == len(arr) and len(arr) >= 2 and (max(ys) - min(ys)) < 1e-9 and (max(zs) - min(zs)) > 1e-9:
        return [(a[0], a[2]) for a in arr]
    return [(a[0], a[1]) for a in arr]


def _first_num(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _pathway_geometry(r, w0):
    """Geometry of one pathway entry, or None.  Understood: an outline polygon; a centre line (buffered by its width); one centre point with
    length / width / rotation in 'dimensions' (a rectangle).  Geometry may sit in any of _GEOM_KEYS at the top level or inside 'dimensions' / 'geometry'."""
    pts3 = []
    for src in (r, r.get("dimensions") or {}, r.get("geometry") or {}):
        if isinstance(src, dict):
            for k in _GEOM_KEYS:
                if src.get(k):
                    pts3 = _points_of(src[k])
                    if len(pts3) >= 2 or (len(pts3) == 1 and src is not r):
                        break
                    pts3 = pts3 if len(pts3) == 1 else []
            if len(pts3) >= 2:
                break
    dims = r.get("dimensions") if isinstance(r.get("dimensions"), dict) else {}
    if len(pts3) == 1 or (not pts3 and isinstance(r.get("coordinates"), dict)):
        # a centre point + size  ->  rectangle
        c = _points_of(r.get("coordinates")) or pts3
        if not c:
            return None
        cx, cy = _plan_xy(c)[0]
        L = _first_num(dims, ("length", "len", "depth", "height")) or _first_num(r, ("length",))
        W = _first_num(dims, ("width", "w", "roadWidth", "pathWidth")) or _first_num(r, ("width", "pathWidth"))
        if L is None or W is None:
            return None
        ang = None
        for src in (dims, r):
            for k in ("rotation", "angle", "heading", "yaw", "orientation", "rotationY", "azimuth"):
                if isinstance(src.get(k), (int, float)):
                    ang = float(src[k]); break
            if ang is not None:
                break
        ang = ang or 0.0
        ang = math.radians(ang) if abs(ang) > 6.4 else ang                 # degrees if clearly larger than 2 pi
        ca, sa = math.cos(ang), math.sin(ang)
        return Polygon([(cx + ca * u - sa * v, cy + sa * u + ca * v) for u, v in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2))])
    if len(pts3) < 2:
        return None
    pts = _plan_xy(pts3)
    w = r.get("width") or r.get("pathWidth") or dims.get("width") or w0
    kind = str(r.get("type", r.get("shape", ""))).lower()
    polyish = len(pts) >= 3 and (r.get("closed") or r.get("isPolygon") or r.get("area") is True
                                 or kind in ("polygon", "area", "region", "surface"))
    ar = dims.get("area")
    if not polyish and len(pts) >= 3 and ar:
        # the scene editor stores a road as its outline polygon; recognise it by the stored area
        try:
            polyish = Polygon(pts).area >= 0.5 * float(ar)     # an outline encloses ~its stored area; a centre line encloses ~0
        except Exception:
            pass
    if not polyish and len(pts) >= 4 and pts[0] == pts[-1]:
        polyish = True
    g = Polygon(pts) if polyish else shapely.LineString(pts).buffer(float(w) / 2.0, cap_style="flat")
    return g if (g.is_valid and not g.is_empty) else g.buffer(0)


def _pathway_zone(data, cfg):
    """Union of all pathways (roads = paths), or None when the scene has none.  Raises if a
    pathway list exists but nothing could be read from it (never ignore it silently) - unless zones.allow_unreadable_pathways is true:
    then a loud warning is printed and the run goes on WITHOUT the road zone (trees may then be placed on roads)."""
    zc = cfg.get("zones", {})
    w0 = float(zc.get("default_pathway_width_m", 6.0))
    geoms = [Polygon([tuple(v) for v in ring]) for ring in zc.get("pathway", []) if len(ring) >= 3]
    for key in ("pathways", "roads", "osmRoads"):
        items = data.get(key) or []
        got = []
        for r in items:
            try:
                g = _pathway_geometry(r, w0) if isinstance(r, dict) else None
            except Exception:
                g = None
            if g is not None and not g.is_empty:
                got.append(g)
        if items and not got:
            first = items[0] if isinstance(items[0], dict) else {"entry": items[0]}
            sample = json.dumps(first, default=str)
            sample = sample if len(sample) <= 900 else sample[:900] + " ..."
            msg = (f"'{key}' has {len(items)} entries but no geometry could be read; entry keys: {sorted(first)[:12]}; first entry: {sample} "
                   f"- send this sample so the reader can be extended, or set zones.allow_unreadable_pathways to true in constraints.json to run without the road zone")
            if zc.get("allow_unreadable_pathways"):
                print("WARNING: " + msg, file=sys.stderr)
                continue
            raise ValueError(msg)
        geoms += got
    return shapely.union_all(geoms) if geoms else None


def on_pathway(cfg, pathway, x, y):
    """True when a tree centre counts as a street / path tree: inside a pathway or within
    tree.pathway_zone_distance_m of its edge."""
    return bool(pathway is not None and pathway.distance(Point(x, y)) <= cfg["tree"]["pathway_zone_distance_m"] + 1e-9)


def _shade_unions(scene):
    """(buildings, osm, ponds) unions, cached on the scene."""
    u = getattr(scene, "_shade_u", None)
    if u is None:
        u = (unary_union(scene.group_polys) if scene.group_polys else None,
             unary_union(scene.osm_polys) if scene.osm_polys else None,
             unary_union(scene.ponds) if scene.ponds else None)
        scene._shade_u = u
    return u


def shade_samples(scene):
    """Sample points on both edges of every pathway (cached on the scene).
    Returns dict P (n,2) points, N (n,2) unit normal pointing away from the pathway, Tg (n,2) unit tangent along the edge,
    plantable (n,) bool: a crown of min_radius_m fits near the sample (centre slid by up to 4 m along the edge) ignoring trees."""
    c = getattr(scene, "_shade", None)
    if c is not None:
        return c
    cfg, pp = scene.cfg, scene.cfg["tree"]["pathway_planting"]
    tc = cfg["tree"]
    empty = {"P": np.zeros((0, 2)), "N": np.zeros((0, 2)), "Tg": np.zeros((0, 2)), "plantable": np.zeros(0, bool)}
    pw = scene.pathway
    if pw is None or pw.is_empty:
        scene._shade = empty
        return empty
    r = float(pp["min_radius_m"])
    off = r + float(pp["canopy_gap_to_path_m"])
    # everything that does not depend on the layout (pathway, site edge, OSM buildings) is computed once and kept
    key = (shapely.to_wkb(pw), shapely.to_wkb(scene.site), id(scene.osm_polys), float(pp["sample_step_m"]), r, off,
           bool(tc["clearance_applies_to_osm"]), float(tc["building_clearance_factor"]))
    st = _SHADE_STATIC.get(key)
    if st is None or st["osm"] is not scene.osm_polys:
        st = _shade_static(scene, pw, pp, tc, r, off)
        if len(_SHADE_STATIC) > 6:
            _SHADE_STATIC.clear()
        _SHADE_STATIC[key] = st
    P, N, Tg = st["P"], st["N"], st["Tg"]
    if len(P) == 0:
        scene._shade = empty
        return empty
    bg, _bo, pd_ = _shade_unions(scene)
    kf = tc["building_clearance_factor"]
    extra = tc["building_clearance_extra_frac_of_width"] * scene.min_width
    plant = np.zeros(len(P), bool)
    for pts, ok0 in st["cands"]:
        ok = ok0.copy()
        idx = np.where(ok)[0]
        if len(idx):
            if bg is not None:
                ok[idx] &= shapely.distance(pts[idx], bg) >= kf * r + extra
            idx = np.where(ok)[0]
            if pd_ is not None and len(idx):
                ok[idx] &= shapely.distance(pts[idx], pd_) >= r
        plant |= ok
    out = {"P": P, "N": N, "Tg": Tg, "plantable": plant}
    scene._shade = out
    return out


_SHADE_STATIC = {}


def _shade_static(scene, pw, pp, tc, r, off):
    """Road-edge samples and the part of the 'a crown fits here' test that never changes during a run."""
    polys = list(pw.geoms) if hasattr(pw, "geoms") else [pw]
    step = float(pp["sample_step_m"])
    Ps, Ns, Ts = [], [], []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            ln = ring.length
            if ln < 2.0:
                continue
            line = shapely.LineString(ring.coords)
            n = max(2, int(round(ln / step)))
            s = np.arange(n) * ln / n
            pts = shapely.get_coordinates(shapely.line_interpolate_point(line, s))
            a = shapely.get_coordinates(shapely.line_interpolate_point(line, (s - 0.5) % ln))
            b = shapely.get_coordinates(shapely.line_interpolate_point(line, (s + 0.5) % ln))
            t = b - a
            nt = np.hypot(t[:, 0], t[:, 1])
            ok = nt > 1e-9
            t = t[ok] / nt[ok, None]
            pts = pts[ok]
            nrm = np.column_stack([-t[:, 1], t[:, 0]])
            in1 = shapely.contains_xy(pw, pts[:, 0] + 0.3 * nrm[:, 0], pts[:, 1] + 0.3 * nrm[:, 1])
            nrm = np.where(in1[:, None], -nrm, nrm)
            in2 = shapely.contains_xy(pw, pts[:, 0] + 0.3 * nrm[:, 0], pts[:, 1] + 0.3 * nrm[:, 1])
            keep = ~in2
            Ps.append(pts[keep]); Ns.append(nrm[keep]); Ts.append(t[keep])
    st = {"osm": scene.osm_polys, "P": np.zeros((0, 2)), "N": np.zeros((0, 2)), "Tg": np.zeros((0, 2)), "cands": []}
    if not Ps:
        return st
    P, N, Tg = np.vstack(Ps), np.vstack(Ns), np.vstack(Ts)
    kf = tc["building_clearance_factor"]
    bo = unary_union(scene.osm_polys) if (scene.osm_polys and tc["clearance_applies_to_osm"]) else None
    site_edge = scene.site.exterior
    for dl in (0.0, 2.0, -2.0, 4.0, -4.0):
        C = P + off * N + dl * Tg
        pts = shapely.points(C[:, 0], C[:, 1])
        ok = shapely.contains(scene.site, pts) & (shapely.distance(pts, site_edge) >= r)
        if bo is not None:
            ok &= shapely.distance(pts, bo) >= kf * r
        ok &= shapely.distance(pts, pw) >= r + 0.5 * float(pp["canopy_gap_to_path_m"])
        st["cands"].append((pts, ok))
    st["P"], st["N"], st["Tg"] = P, N, Tg
    return st


def shade_dmin(P, tx, ty, tR):
    """Distance from every sample to the nearest crown EDGE (inf without trees)."""
    if len(tx) == 0 or len(P) == 0:
        return np.full(len(P), np.inf)
    return (np.hypot(P[:, 0, None] - np.asarray(tx)[None, :], P[:, 1, None] - np.asarray(ty)[None, :]) - np.asarray(tR)[None, :]).min(axis=1)


def shade_coverage(scene):
    """(coverage, shaded mask, samples dict): share of the plantable road-edge samples that have a crown edge within reach_m.
    1.0 when the layout has no pathway or nothing is plantable."""
    S = shade_samples(scene)
    reach = float(scene.cfg["tree"]["pathway_planting"]["reach_m"])
    dm = shade_dmin(S["P"], scene.tx, scene.ty, scene.tR)
    sh = dm <= reach + 1e-9
    npl = int(S["plantable"].sum())
    cov = float(sh[S["plantable"]].mean()) if npl else 1.0
    return cov, sh, S


def tree_count0(base, cfg):
    """Tree count the +10 pct rule and the count cap refer to: the base count, or - for a base without trees -
    the number that gives canopy_cover_frac_if_missing of the open space."""
    cache = base.__dict__.setdefault("_a0", {})
    if "trees" not in cache:
        n = len(base.tx)
        tc = cfg["tree"]
        if n == 0 and tc.get("create_if_missing"):
            r = 0.5 * sum(tc["new_tree_radius_range_m"])
            n = int(round(tc["canopy_cover_frac_if_missing"] * open_space_area(base) / (math.pi * r * r)))
        cache["trees"] = n
    return cache["trees"]


def new_tree(cfg, T, rng):
    """A tree record to copy: a random existing tree, or the configured template when there is none."""
    if T:
        return copy.deepcopy(T[int(rng.integers(len(T)))])
    t = copy.deepcopy(cfg["tree"]["template"])
    t["position"] = {"x": 0.0, "y": 0.0, "z": 0}
    t["id"] = "Tree 0"
    return t


def max_tree_count(cfg, n_base, has_pathway):
    tc = cfg["tree"]
    pp = tc["pathway_planting"]
    return math.floor(tc["max_count_factor"] * n_base) + (int(pp["max_new_trees"]) if pp["enabled"] and has_pathway else 0)


def height_band(cfg, dist_to_building, on_pathway):
    """Allowed total tree height [lo, hi] for a location.  Priority: near buildings (strictest),
    then pathway, then open ground."""
    tc = cfg["tree"]
    if dist_to_building <= tc["near_building_distance_m"]:
        return tc["height_near_building_m"]
    if on_pathway:
        return tc["height_pathway_m"]
    return tc["height_open_m"]


class Scene:
    """Read-only geometric view of a layout dict (rebuild after every edit)."""

    def __init__(self, data, cfg, osm_polys=None, split_hosted=False):
        """split_hosted=True for the BASE (or any layout from outside): trees / lawns / ponds that stand on a building are left out of the
        scene arrays and kept in hosted_base.  Data in the middle of an optimisation is already without them (working_copy)."""
        self.d, self.cfg = data, cfg
        self.site = _poly(data["site"]["coordinates"])
        sv = np.array(_xy(data["site"]["coordinates"]))
        self.site_diameter = float(max(np.hypot(*(a - b)) for a in sv for b in sv))
        self.groups = building_groups(data)
        self.group_ids = [tuple(data["buildings"][i]["id"] for i in g) for g in self.groups]
        self.group_names = [data["buildings"][g[0]]["id"] for g in self.groups]
        self.group_polys = [group_polygon(data, g) for g in self.groups]      # a stack of parts = ONE building (union footprint)
        self.osm_polys = osm_polys if osm_polys is not None else [
            _poly(o["dimensions"]["baseVertices"], ("x", "z")) for o in data.get("osmBuildings", [])]
        self.osm_ids = [o["id"] for o in data.get("osmBuildings", [])]
        spans = [max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in self.group_polys]
        self.min_width = min(spans) if spans else 0.0
        self.group_len = spans                       # longest projected span (max of E-W, N-S) per building
        # trees / lawns / ponds that stand on a building belong to it: the scene (checks, operators) sees only the free ones
        all_ponds = [_patch_poly(p) for p in data["ponds"]]
        all_lawns = [_patch_poly(p) for p in data["grassAreas"]]
        self.hosted = (hosted_indices(data, self.groups, self.group_polys, all_ponds, all_lawns) if split_hosted
                       else {"trees": {}, "ponds": {}, "grassAreas": {}})
        hs = self.hosted
        T = [t for i, t in enumerate(data["trees"]) if i not in hs["trees"]]
        self.tree_ids = [t["id"] for t in T]
        self.tx = np.array([t["position"]["x"] for t in T], float)
        self.ty = np.array([t["position"]["y"] for t in T], float)
        self.tR = np.array([_tree_R(t) for t in T], float)
        self.th = np.array([float(t["height"]) for t in T], float)
        self.ponds = [_poly(p["coordinates"]) for i, p in enumerate(data["ponds"]) if i not in hs["ponds"]]
        self.lawns = [_poly(p["coordinates"]) for i, p in enumerate(data["grassAreas"]) if i not in hs["grassAreas"]]
        self.pond_ids = [p["id"] for i, p in enumerate(data["ponds"]) if i not in hs["ponds"]]
        self.lawn_ids = [p["id"] for i, p in enumerate(data["grassAreas"]) if i not in hs["grassAreas"]]
        self.pathway = _pathway_zone(data, cfg)

    @property
    def ball_union(self):
        if not hasattr(self, "_bu"):
            parts = list(self.group_polys) + list(self.osm_polys)
            self._bu = unary_union(parts) if parts else None
        return self._bu

    def centroid(self, gi):
        c = self.group_polys[gi].centroid
        return np.array([c.x, c.y])

    @property
    def hosted_base(self):
        """Items that stand on a building of this (base) scene, see hosted_items()."""
        if not hasattr(self, "_hb"):
            self._hb = hosted_items(self.d)
        return self._hb

    @property
    def hosted_ids(self):
        hb = self.hosted_base
        return {k: {it["id"] for it, _ in hb[k]} for k in ("trees", "ponds", "grassAreas")}


# ----------------------------------------------------------------------------- the checks
def _snap_step(cfg, data):
    s = cfg["building"]["snap_step_m"]
    return float(s) if s else float(data.get("settings", {}).get("units", {}).get("gridSize", 1.0) or 1.0)


def open_space_area(base):
    """Site area not taken by buildings (own + OSM), pathways or lawns."""
    parts = list(base.group_polys) + list(base.osm_polys) + list(base.lawns)
    if base.pathway is not None:
        parts.append(base.pathway)
    return float(base.site.difference(unary_union(parts)).area) if parts else float(base.site.area)


def base_area0(base, cfg, cat):
    """A0 of a category = total area of the base's ponds / lawns INSIDE the site (ponds / lawns outside the site boundary are frozen, see
    hosted_indices: they are not part of any area, count or clearance rule).  A base without any inside gets A0 = open_space_frac_if_missing
    (10 % by default) of the open space (create_if_missing)."""
    cache = base.__dict__.setdefault("_a0", {})
    if cat not in cache:
        a = sum(p.area for p in (base.ponds if cat == "pond" else base.lawns))
        if a <= 0 and cfg[cat].get("create_if_missing"):
            a = cfg[cat]["open_space_frac_if_missing"] * open_space_area(base)
        cache[cat] = a
    return cache[cat]


IRRIGATION_DEPTH_MM = 5.0          # fixed design value, not configurable


def lawn_water_l_per_day(cfg, a0):
    """Irrigation water available to the lawns (L/day): the configured value, or - when none is
    given - what the base lawn needs at 5 mm/day:  A0[m2] * 0.005[m] * 1000[L/m3]."""
    q = cfg["lawn"].get("water_available_l_per_day")
    return float(q) if q else a0 * IRRIGATION_DEPTH_MM / 1000.0 * 1000.0


def area_band(cfg, cat, a0):
    """Allowed TOTAL area [lo, hi] (m2) of ponds / lawns.  Lawns are also capped by the irrigation
    budget:  hi = min(1.10 A0, water[L/day] / 5 mm)."""
    pc = cfg[cat]
    lo, hi = pc["area_factor_range"][0] * a0, pc["area_factor_range"][1] * a0
    if cat == "lawn":
        hi = min(hi, lawn_water_l_per_day(cfg, a0) / IRRIGATION_DEPTH_MM)
        lo = min(lo, hi)
    return lo, hi


def clearance_m(cfg, cat):
    """Required distance (m) between a pond / lawn and every building (0 = only 'no overlap')."""
    return float(cfg[cat].get("building_clearance_m", 0.0))


_OSM_ZONES = {}          # id(osm_polys list) -> (list, union, {clearance: union grown by clearance})


def _osm_zone(osm_polys, clearance):
    """Union of the (never changing) OSM footprints grown by `clearance`, computed once per list of OSM polygons."""
    key = id(osm_polys)
    hit = _OSM_ZONES.get(key)
    if hit is None or hit[0] is not osm_polys:
        parts = [p if p.is_valid else p.buffer(0) for p in osm_polys if p is not None and not p.is_empty]
        if len(_OSM_ZONES) > 8:
            _OSM_ZONES.clear()
        hit = _OSM_ZONES[key] = (osm_polys, unary_union(parts) if parts else None, {})
    u, grown = hit[1], hit[2]
    if u is None:
        return None
    c = round(float(clearance), 6)
    if c not in grown:
        grown[c] = u.buffer(c, quad_segs=16) if c > 1e-9 else u
    return grown[c]


def building_zone(group_polys, osm_polys, clearance):
    """Union of all building footprints grown by `clearance` m (the area a pond / lawn must stay out of).
    The OSM part does not change during a run, so it is grown once and kept (_osm_zone)."""
    own = [p if p.is_valid else p.buffer(0) for p in group_polys if p is not None and not p.is_empty]
    zo = _osm_zone(osm_polys, clearance) if len(osm_polys) else None
    if not own and zo is None:
        return None
    if own:
        u = unary_union(own)
        zg = u.buffer(clearance, quad_segs=16) if clearance > 1e-9 else u
        return zg if zo is None else unary_union([zg, zo])
    return zo


def scene_zone(scene, cat):
    """Keep-out zone of the buildings of `scene` for a patch of category `cat` ('pond' or 'lawn')."""
    return building_zone(scene.group_polys, scene.osm_polys, clearance_m(scene.cfg, cat))


SLIT_M = 0.2             # an outline whose own 0.1 m buffer closes a gap (gap < 0.2 m) encloses something through a hairline slit


def has_slit(poly):
    """True if a pond / lawn outline wraps round an obstacle through a channel under SLIT_M wide, or touches itself in a point (the constrained
    triangulation of the ground fails on such outlines and snappyHexMesh cannot resolve them)."""
    c = poly.buffer(0.5 * SLIT_M, join_style=2)
    return not (c.geom_type == "Polygon" and not c.interiors)


def unslit(poly):
    """poly eroded by 0.11 m if it has a slit (every channel gets 0.22 m wider), else unchanged."""
    if not has_slit(poly):
        return poly
    q = poly.buffer(-0.55 * SLIT_M, join_style=2)
    if q.geom_type != "Polygon":
        q = max((x for x in getattr(q, "geoms", []) if x.geom_type == "Polygon"), key=lambda x: x.area, default=None)
    return q if (q is not None and not q.is_empty and q.area > 0.5 * poly.area) else poly


def check(scene, base):
    """All Stage 1 violations of `scene` as a list of (type, ids, severity>0)."""
    cfg, tc = scene.cfg, scene.cfg["tree"]
    tol, V = cfg["tol"], []
    bt = cfg["building"]["overlap_tol_m2"]
    ctol = cfg["patch"].get("clearance_tol_m", 0.05)
    base_by_ids = {ids: i for i, ids in enumerate(base.group_ids)}

    # ---- buildings
    max_shift = cfg["building"]["max_shift_frac_of_site_diameter"] * base.site_diameter
    for gi, ids in enumerate(scene.group_ids):
        name, poly = scene.group_names[gi], scene.group_polys[gi]
        bi = base_by_ids.get(ids)
        if bi is None:
            V.append(("building_group_changed", (name,), 1.0))
            continue
        off = float(np.linalg.norm(scene.centroid(gi) - base.centroid(bi)))
        if off > max_shift + 1e-6:
            V.append(("building_shift", (name,), off - max_shift))
        # translation only: every coordinate of every part moved by the same vector
        d_vec = scene.centroid(gi) - base.centroid(bi)
        for i in scene.groups[gi]:
            if _rigid_error(scene.d["buildings"][i], _base_building(base, scene.d["buildings"][i]["id"]), d_vec) > 1e-6:
                V.append(("building_not_rigid", (scene.d["buildings"][i]["id"],), 1.0))
        out = poly.difference(scene.site).area
        if out > bt:
            V.append(("building_outside_site", (name,), out))
        for gj in range(gi + 1, len(scene.group_polys)):
            a = poly.intersection(scene.group_polys[gj]).area
            if a > bt:
                V.append(("building_overlap", tuple(sorted((name, scene.group_names[gj]))), a))
        if scene.pathway is not None and poly.intersects(scene.pathway):
            a = poly.intersection(scene.pathway).area
            if a > bt:
                V.append(("building_pathway_overlap", (name,), a))
        for j, op in enumerate(scene.osm_polys):
            if poly.intersects(op):
                a = poly.intersection(op).area
                if a > bt:
                    V.append(("building_overlap", tuple(sorted((name, scene.osm_ids[j]))), a))

    # ---- trees
    n = len(scene.tx)
    pts = shapely.points(scene.tx, scene.ty)
    lo_r, hi_r = tc["radius_range_m"]
    extra = tc["building_clearance_extra_frac_of_width"] * scene.min_width   # min_width = max(E-W, N-S span)
    kf = tc["building_clearance_factor"]
    near_d = np.full(n, np.inf)
    allp = scene.group_polys + scene.osm_polys
    if n and allp:
        D = shapely.distance(pts[:, None], np.array(allp, dtype=object)[None, :])
        ne = len(scene.group_polys)
        near_d = D.min(axis=1)
        k_osm = kf if tc["clearance_applies_to_osm"] else 0.0
        kvec = np.concatenate([np.full(ne, kf), np.full(len(scene.osm_polys), k_osm)])
        evec = np.concatenate([np.full(ne, extra), np.zeros(len(scene.osm_polys))])
        bname = scene.group_names + scene.osm_ids
        deficit = (kvec[None, :] * scene.tR[:, None] + evec[None, :]) - D
        for i, j in zip(*np.where(deficit > tol)):
            V.append(("tree_building_clearance", (scene.tree_ids[i], bname[j]), float(deficit[i, j])))
    pw_d = shapely.distance(pts, scene.pathway) if (n and scene.pathway is not None) else None
    for i in range(n):
        R, tid = scene.tR[i], scene.tree_ids[i]
        if R < lo_r - tol or R > hi_r + tol:
            V.append(("tree_radius", (tid,), float(max(lo_r - R, R - hi_r))))
        inside = scene.site.contains(pts[i])
        bd = scene.site.exterior.distance(pts[i])
        sev = (R + bd) if not inside else max(0.0, R - bd)
        if sev > tol:
            V.append(("tree_outside_site", (tid,), float(sev)))
        on_pw = bool(pw_d is not None and pw_d[i] <= tc["pathway_zone_distance_m"] + 1e-9)
        lo_h, hi_h = height_band(cfg, near_d[i], on_pw)
        if scene.pathway is not None and not tc["tree_on_pathway_allowed"]:
            dpw = scene.pathway.distance(pts[i])
            if dpw < R - tol:
                V.append(("tree_pathway_overlap", (tid,), float(R - dpw)))
        if scene.th[i] < lo_h - tol or scene.th[i] > hi_h + tol:
            V.append(("tree_height", (tid,), float(max(lo_h - scene.th[i], scene.th[i] - hi_h))))
        for pi, pp in enumerate(scene.ponds):
            dd = pts[i].distance(pp)
            if dd < R - tol:
                V.append(("tree_pond_overlap", (tid, scene.pond_ids[pi]), float(R - dd)))
    if n > 1:
        dx = scene.tx[:, None] - scene.tx[None, :]
        dy = scene.ty[:, None] - scene.ty[None, :]
        dist = np.hypot(dx, dy)
        need = tc["canopy_gap_factor"] * (scene.tR[:, None] + scene.tR[None, :])
        for i, j in zip(*np.where(np.triu(need - dist, 1) > tol)):
            V.append(("tree_gap", tuple(sorted((scene.tree_ids[i], scene.tree_ids[j]))), float(need[i, j] - dist[i, j])))
    max_trees = max_tree_count(cfg, tree_count0(base, cfg), scene.pathway is not None)
    if n > max_trees:
        V.append(("tree_count", ("trees",), float(n - max_trees)))
    sp = tc["pathway_planting"]
    if scene.pathway is not None and sp["enabled"] and sp.get("shade_required", True) and scene is not base:
        cov = shade_coverage(scene)[0]
        if cov < sp["min_coverage"] - 1e-9:
            V.append(("pathway_shade", ("pathway",), float(sp["min_coverage"] - cov)))

    # ---- ponds and lawns
    for cat, polys, ids, base_polys in (("pond", scene.ponds, scene.pond_ids, base.ponds),
                                        ("lawn", scene.lawns, scene.lawn_ids, base.lawns)):
        pc = cfg[cat]
        mp_ = patch_limit(cfg, cat, len(scene.group_polys))
        if len(polys) > mp_:
            V.append((f"{cat}_count", (cat,), float(len(polys) - mp_)))
        a0 = base_area0(base, cfg, cat)
        tot = sum(p.area for p in polys)
        lo_a, hi_a = area_band(cfg, cat, a0)
        slack = 1e-3 * max(a0, 1.0)
        if tot < lo_a - slack or tot > hi_a + slack:
            V.append((f"{cat}_total_area", (cat,), float(max(lo_a - tot, tot - hi_a))))
        for p, pid in zip(polys, ids):
            if not p.is_valid:
                V.append((f"{cat}_invalid_shape", (pid,), 1.0))
                continue
            if p.area < pc["min_area_m2"] - tol:
                V.append((f"{cat}_too_small", (pid,), float(pc["min_area_m2"] - p.area)))
            if has_slit(p):
                V.append((f"{cat}_slit", (pid,), 1.0))
            out = p.difference(scene.site).area
            if out > bt:
                V.append((f"{cat}_outside_site", (pid,), float(out)))
            if scene.pathway is not None and p.intersects(scene.pathway):
                a = p.intersection(scene.pathway).area
                if a > bt:
                    V.append((f"{cat}_pathway_overlap", (pid,), float(a)))
            cl = clearance_m(cfg, cat)
            for bp, bn in zip(allp, scene.group_names + scene.osm_ids):
                if p.intersects(bp):
                    a = p.intersection(bp).area
                    if a > bt:
                        V.append((f"{cat}_building_overlap", (pid, bn), float(a)))
                if cl > 0:
                    dd = p.distance(bp)
                    if dd < cl - ctol:
                        V.append((f"{cat}_building_clearance", (pid, bn), float(cl - dd)))
    pat = [(p, i) for p, i in zip(scene.ponds + scene.lawns, scene.pond_ids + scene.lawn_ids) if p.is_valid]
    for a in range(len(pat)):
        for b in range(a + 1, len(pat)):
            if pat[a][0].intersects(pat[b][0]):
                ar = pat[a][0].intersection(pat[b][0]).area
                if ar > bt:
                    V.append(("patch_overlap", tuple(sorted((pat[a][1], pat[b][1]))), float(ar)))
    return V


def _base_building(base, bid):
    for b in base.d["buildings"]:
        if b["id"] == bid:
            return b
    return None


def _rigid_error(b, b0, d_vec):
    """Largest deviation of any coordinate of building b from 'base building moved by d_vec'."""
    if b0 is None:
        return 1.0
    dx, dy = float(d_vec[0]), float(d_vec[1])
    err = 0.0

    def cmp(p, p0, with_z=True):
        nonlocal err
        err = max(err, abs(p["x"] - p0["x"] - dx), abs(p["y"] - p0["y"] - dy))
        if with_z and "z" in p and "z" in p0:
            err = max(err, abs(p["z"] - p0["z"]))

    cmp(b["position"], b0["position"])
    cmp(b["geometry"]["centroid"], b0["geometry"]["centroid"])
    for a, a0 in zip(b["geometry"]["footprint"], b0["geometry"]["footprint"]):
        cmp(a, a0, False)
    for a, a0 in zip(b["dimensions"]["baseVertices"], b0["dimensions"]["baseVertices"]):
        cmp(a, a0, False)
    if len(b["faces"]) != len(b0["faces"]):
        return 1.0
    for f, f0 in zip(b["faces"], b0["faces"]):
        cmp(f["center"], f0["center"])
        for a, a0 in zip(f["vertices"], f0["vertices"]):
            cmp(a, a0)
    if b["height"] != b0["height"] or b["dimensions"]["height"] != b0["dimensions"]["height"]:
        err = max(err, 1.0)
    return err


def new_violations(scene, base, base_map):
    """Violations that are NOT inherited from the base layout (or got worse than in the base)."""
    out = []
    for t, ids, sev in check(scene, base):
        b = base_map.get((t, ids))
        if b is None or sev > b + 1e-4:
            out.append((t, ids, sev))
    return out


HARD_RULES = ("lawn_total_area", "pond_total_area",    # never tolerated, even if the base breaks them
              "lawn_building_overlap", "pond_building_overlap", "patch_overlap",
              "lawn_building_clearance", "pond_building_clearance", "lawn_slit", "pond_slit",
              "tree_pathway_overlap", "pathway_shade")   # (e.g. a water budget below what the base lawn needs;
                                                      #  lawns/ponds on buildings, lawn-pond overlap)


def violation_map(viols):
    m = {}
    for t, ids, sev in viols:
        if t in HARD_RULES:
            continue
        m[(t, ids)] = max(sev, m.get((t, ids), 0.0))
    return m


# ----------------------------------------------------------------------------- JSON edits
def _shift_list(vs, dx, dy):
    for v in vs:
        v["x"] += dx
        v["y"] += dy


def translate_group(data, scene, gi, dx, dy):
    """Move one physical building (all stacked parts, faces, footprint entry) by (dx, dy)."""
    old_fps = [sorted((round(v["x"], 6), round(v["y"], 6)) for v in data["buildings"][i]["geometry"]["footprint"]) for i in scene.groups[gi]]
    names = set()
    for i in scene.groups[gi]:
        b = data["buildings"][i]
        names.add(b["id"])
        b["position"]["x"] += dx
        b["position"]["y"] += dy
        b["geometry"]["centroid"]["x"] += dx
        b["geometry"]["centroid"]["y"] += dy
        _shift_list(b["geometry"]["footprint"], dx, dy)
        _shift_list(b["dimensions"]["baseVertices"], dx, dy)
        for f in b["faces"]:
            _shift_list(f["vertices"], dx, dy)
            f["center"]["x"] += dx
            f["center"]["y"] += dy
            for co in f.get("cutouts", []):
                if "vertices" in co:
                    _shift_list(co["vertices"], dx, dy)
                if "center" in co:
                    co["center"]["x"] += dx
                    co["center"]["y"] += dy
    old_mirs = [sorted((x, round(-y, 6) + 0.0) for x, y in fp) for fp in old_fps]
    for fp in data.get("footprints", []):
        same_id = any(fp.get("id") == f"shape_fp_{n}" for n in names)
        fxy = sorted((round(v["x"], 6), round(v["y"], 6) + 0.0) for v in fp["vertices"])
        direct, mirrored = fxy in old_fps, fxy in old_mirs
        if same_id or direct or mirrored:
            # footprints[] is stored in the scene's mirrored frame (y_footprint = -y_building) in
            # the app's files; keep whichever convention THIS file uses (default: mirrored).
            sgn = 1.0 if (direct and not mirrored) else -1.0
            _shift_list(fp["vertices"], dx, sgn * dy)


def set_tree_geometry(t, R, h, tc):
    th = tc["trunk_height_frac"] * h
    d = t.setdefault("dimensions", {})
    d.update(trunkHeight=th, crownHeight=h - th, crownRadius=_num(R), totalHeight=_num(h),
             crownDiameter=_num(2 * R))
    d.setdefault("trunkRadius", tc["trunk_radius_m"])
    t["height"], t["crownDiameter"] = _num(h), _num(2 * R)
    p = t.setdefault("properties", {})
    p["shadingArea"] = math.pi * R * R
    p["volume"] = 4.0 / 3.0 * math.pi * R ** 3 + math.pi * d["trunkRadius"] ** 2 * th


def discretize_ring(poly, step=1.0, corner_deg=35.0):
    """Re-discretize the outer ring of a pond / lawn polygon so that NO two neighbouring vertices are closer
    than `step` metres (default 1 m).  Smoothed outlines (buffer(), splines) carry thousands of vertices a few
    millimetres apart, which makes the surface mesh / snappyHexMesh ground triangulation needlessly hard.

    Genuine corners (heading change > corner_deg, measured over +-step/2 of arc length) are kept exactly; the
    stretch between two corners is resampled at equal arc-length spacing ds = L / floor(L / step) >= step.
    Returns the new Polygon, or the input if the result would be invalid or change the area by > 1 %."""
    xy = np.asarray(poly.exterior.coords)[:-1]
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9]
    xy = xy[keep]
    if len(xy) < 4 or step <= 0:
        return poly
    ring = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = cum[-1]
    if total < 3 * step:
        return poly

    def at(sv):                                   # point at arc length sv (wraps around the ring)
        sv = np.mod(sv, total)
        return np.stack([np.interp(sv, cum, ring[:, 0]), np.interp(sv, cum, ring[:, 1])], axis=-1)

    h = 0.5 * step
    s_v = cum[:-1]
    d_in, d_out = at(s_v) - at(s_v - h), at(s_v + h) - at(s_v)
    ang = np.degrees(np.abs(np.arctan2(d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0],
                                       (d_in * d_out).sum(axis=1))))
    cand = np.flatnonzero(ang > corner_deg)
    corners = []                                   # one corner per cluster of candidates within `step` of arc length
    for i in cand:
        if corners and (s_v[i] - s_v[corners[-1]]) < step:
            if ang[i] > ang[corners[-1]]:
                corners[-1] = i
        else:
            corners.append(i)
    if len(corners) > 1 and (total - s_v[corners[-1]] + s_v[corners[0]]) < step:
        corners.pop() if ang[corners[-1]] <= ang[corners[0]] else corners.pop(0)
    if not corners:
        corners = [0]
    out = []
    for j, ci in enumerate(corners):
        a = s_v[ci]
        b = s_v[corners[(j + 1) % len(corners)]]
        b = b + total if b <= a else b
        L = b - a
        n = max(1, int(math.floor(L / step)))
        pts = at(a + np.arange(n) * (L / n))       # n points, the end corner starts the next section
        while n > 1 and np.linalg.norm(np.diff(np.vstack([pts, at(b)]), axis=0), axis=1).min() < step * 0.999:
            n -= 1
            pts = at(a + np.arange(n) * (L / n))
        out.append(pts)
    ring_pts = np.vstack(out)
    lim = step * 0.999
    for _ in range(5):                              # a pair of neighbours that ended up closer than `step` (arcs): merge
        d = np.linalg.norm(np.diff(np.vstack([ring_pts, ring_pts[:1]]), axis=0), axis=1)
        bad = np.flatnonzero(d < lim)
        if not len(bad) or len(ring_pts) <= 4:
            break
        drop = set()
        for i in bad:
            if i not in drop and (i - 1) % len(ring_pts) not in drop:
                drop.add((i + 1) % len(ring_pts))   # drop the later vertex of the short edge
        ring_pts = ring_pts[[k for k in range(len(ring_pts)) if k not in drop]]
    new = Polygon(ring_pts)
    if not new.is_valid or new.is_empty:
        new = new.buffer(0)
        if new.geom_type != "Polygon":
            return poly
    if abs(new.area - poly.area) > 0.01 * poly.area:
        return poly
    return new


def discretize_patches(polys, step=1.0, keep_out=None):
    """discretize_ring() for a list of patches that must not overlap each other (or `keep_out`, e.g. building
    footprints).  Independent resampling of two touching outlines (pond next to lawn) would cut their shared
    boundary differently and leave slivers of overlap, so each patch is resampled in turn and, where it would
    overlap an earlier patch / keep_out more than the original did, is pulled back by a small gap (up to 0.4 m)."""
    if not step:
        return list(polys)
    out, orig = [], []
    ko = [keep_out] if keep_out is not None and not keep_out.is_empty else []
    def _u(parts):
        if not parts:
            return None
        if len(parts) == 1 and parts[0].is_valid:
            return parts[0]
        g = unary_union([x if x.is_valid else x.buffer(0) for x in parts])
        return g if g.is_valid else g.buffer(0)

    for poly in polys:
        if not poly.is_valid:
            poly = poly.buffer(0)
        # only what lies within 1 m of the patch matters (the largest gap is 0.4 m): clip the (large) keep-out and the other patches to it
        minx, miny, maxx, maxy = poly.bounds
        win = shapely.box(minx - 1.0, miny - 1.0, maxx + 1.0, maxy + 1.0)
        near = lambda g: g.intersection(win) if g.intersects(win) else None
        kl = [x for x in (near(k) for k in ko) if x is not None and not x.is_empty]
        blocker = _u([q for q in (near(o) for o in out) if q is not None and not q.is_empty] + kl)
        orig_blocker = _u([q for q in (near(o) for o in orig) if q is not None and not q.is_empty] + kl)
        orig.append(poly)
        if blocker is None or blocker.is_empty:
            out.append(discretize_ring(poly, step))
            continue
        allowed = poly.intersection(orig_blocker).area + 1e-4          # overlap the ORIGINAL outlines already had
        best = None
        for gap in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
            g = poly.difference(blocker.buffer(gap)) if gap else poly
            if g.geom_type != "Polygon":
                g = max((x for x in getattr(g, "geoms", []) if x.geom_type == "Polygon"), key=lambda x: x.area, default=None)
            if g is None or g.is_empty:
                break
            cand = discretize_ring(g, step)
            best = cand
            if cand.intersection(blocker).area <= allowed:
                break
        out.append(best if best is not None else poly)
    return out


def finalize_patches(data, cfg):
    """Safety net applied to every candidate before it is written: any pond / lawn outline that still has
    neighbouring vertices closer than patch.discretization_m (e.g. a smoothed parent whose lawns were not touched
    by this candidate's operations) is re-discretized.  Outlines that are already fine are left alone."""
    step = cfg.get("patch", {}).get("discretization_m", 1.0)
    if not step:
        return 0
    items = [(key, p) for key in ("ponds", "grassAreas") for p in data.get(key, [])]
    def short(p):
        a = np.array([[q["x"], q["y"]] for q in p["coordinates"]])
        return np.linalg.norm(np.diff(np.vstack([a, a[:1]]), axis=0), axis=1).min() < step * 0.999
    polys0 = [_poly(p["coordinates"]) for _, p in items]
    slit = [q.is_valid and has_slit(q) for q in polys0]
    if not any(short(p) for _, p in items) and not any(slit):
        return 0
    sc = Scene(data, cfg)
    npd = len(data.get("ponds", []))
    polys = [x if x.is_valid else x.buffer(0) for x in polys0]
    # ponds first (kept out of the buildings by their clearance), then the lawns (kept out of buildings by theirs and of the ponds)
    new = discretize_patches(polys[:npd], step, scene_zone(sc, "pond"))
    zl = scene_zone(sc, "lawn")
    new += discretize_patches(polys[npd:], step, unary_union([g for g in [zl] + new if g is not None and not g.is_empty]))
    n = 0
    for (key, p), poly, sl in zip(items, new, slit):
        if short(p) or sl:
            patch_set_polygon(p, unslit(poly) if sl else poly, key == "ponds", 0)
            n += 1
    return n


def _other_patches(scene, cat):
    """What a pond (lawn) outline must not cut into when it is re-discretized: the lawns (ponds), the buildings
    grown by the clearance of the category, the pathway."""
    parts = list(scene.lawns if cat == "pond" else scene.ponds)
    z = scene_zone(scene, cat)
    if z is not None:
        parts.append(z)
    if getattr(scene, "pathway", None) is not None:
        parts.append(scene.pathway)
    parts = [x for x in parts if x is not None and not x.is_empty]
    return unary_union(parts) if parts else None


def patch_set_polygon(patch, poly, is_pond, step=1.0):
    if step:
        poly = discretize_ring(poly, step)
    xy = list(poly.exterior.coords)[:-1]
    patch["coordinates"] = [{"x": float(x), "y": float(y)} for x, y in xy]
    patch["dimensions"]["area"] = poly.area
    if is_pond:
        patch["dimensions"]["volume"] = poly.area * patch["dimensions"]["depth"]    # depth never changes


def refresh_metadata(data):
    si = data.get("metadata", {}).get("sceneInfo")
    if si is not None:
        for k, key in (("totalBuildings", "buildings"), ("totalTrees", "trees"), ("totalFootprints", "footprints"),
                       ("totalPonds", "ponds"), ("totalGrassAreas", "grassAreas")):
            if k in si:
                si[k] = len(data[key])
    for g in data.get("groups", {}).get("treeGroups", [])[:1]:       # one group that lists all trees
        g["treeIds"] = [t["id"] for t in data["trees"]]
        if data["trees"]:
            xs = [t["position"]["x"] for t in data["trees"]]
            ys = [t["position"]["y"] for t in data["trees"]]
            g["boundingBox"] = {"min": {"x": min(xs), "y": 0, "z": -max(ys)},
                                "max": {"x": max(xs), "y": 0, "z": -min(ys)},
                                "center": {"x": (min(xs) + max(xs)) / 2, "y": 0, "z": -(min(ys) + max(ys)) / 2},
                                "size": {"x": max(xs) - min(xs), "y": 0, "z": max(ys) - min(ys)}}
    if "metadata" in data and "timestamp" in data["metadata"]:
        data["metadata"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def working_copy(data):
    """Copy only the parts that edits touch; the large fixed OSM context is shared."""
    out = dict(data)
    for k in ("buildings", "trees", "footprints", "ponds", "grassAreas", "groups", "metadata"):
        if k in data:
            out[k] = pickle.loads(pickle.dumps(data[k], protocol=pickle.HIGHEST_PROTOCOL))     # a deep copy of plain JSON data, ~4x faster than copy.deepcopy
    return ground_view(out)         # what stands on a building stays out of the optimisation (attach_hosted puts it back)


# ----------------------------------------------------------------------------- random operations
def _rt(x, step):
    return round(x / step) * step


def _in_site_point(scene, rng, tries=300):
    minx, miny, maxx, maxy = scene.site.bounds
    for _ in range(tries):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if scene.site.contains(p):
            return p.x, p.y
    return None


def _height_for(scene, x, y, cfg, rng, current=None):
    tc = cfg["tree"]
    allp = scene.group_polys + scene.osm_polys
    d = min(Point(x, y).distance(p) for p in allp) if allp else np.inf
    lo, hi = height_band(cfg, d, on_pathway(cfg, scene.pathway, x, y))
    if current is not None and lo <= current <= hi:
        return current
    return float(np.clip(_rt(rng.uniform(lo, hi), tc["size_step_m"]), lo, hi))


def _building_move_ok(scene, base, cfg, gi, bi, poly):
    """Geometry of a moved building: inside the site, no overlap with other buildings, OSM buildings,
    and no more pathway overlap than the base had.  (Lawns, ponds and trees make room afterwards.)"""
    bt = cfg["building"]["overlap_tol_m2"]
    if poly.difference(scene.site).area > bt:
        return False
    for gj, p in enumerate(scene.group_polys):
        if gj != gi and poly.intersects(p) and poly.intersection(p).area > bt:
            return False
    for op in scene.osm_polys:
        if poly.intersects(op) and poly.intersection(op).area > bt:
            return False
    if scene.pathway is not None and poly.intersects(scene.pathway):
        cache = base.__dict__.setdefault("_pw_overlap", {})
        if bi not in cache:
            cache[bi] = base.group_polys[bi].intersection(base.pathway).area if base.pathway is not None else 0.0
        if poly.intersection(scene.pathway).area > cache[bi] + bt:
            return False
    return True


def op_shift_building(data, base, cfg, rng, strength, scene, gi=None, fresh=False, min_scale=1.0):
    """Translate one physical building.  The offset from its ORIGINAL position always lies in the ring
    [min_shift_frac_of_length x longest projected span, max_shift_frac x site diameter].  fresh=True draws
    a new offset from the whole ring (generation); otherwise the current offset is perturbed (mutation).
    Offsets that would hit another building, an OSM building, the site boundary or a pathway are rejected."""
    gi = int(rng.integers(len(scene.groups))) if gi is None else gi
    bi = base.group_ids.index(scene.group_ids[gi])
    bc = cfg["building"]
    max_shift = bc["max_shift_frac_of_site_diameter"] * base.site_diameter
    dmin = min(bc["min_shift_frac_of_length"] * base.group_len[bi] * min_scale, 0.98 * max_shift)
    step = _snap_step(cfg, data)
    cur = scene.centroid(gi) - base.centroid(bi)
    if not fresh and np.linalg.norm(cur) > 1e-9:       # a mutation never forces a building further out than it is
        dmin = min(dmin, 0.7 * float(np.linalg.norm(cur)))
    sigma = max(strength, 0.05) * max_shift * 0.5
    for _ in range(600 if fresh else 150):
        if fresh or np.linalg.norm(cur) < 1e-9:
            mag, ang = rng.uniform(dmin, 0.999 * max_shift), rng.uniform(0, 2 * math.pi)
            new = np.array([mag * math.cos(ang), mag * math.sin(ang)])
        else:
            new = cur + rng.normal(0.0, sigma, 2)
        n = float(np.linalg.norm(new))
        if n > 0.999 * max_shift:
            new *= 0.999 * max_shift / n
        elif n < dmin * 1.02:
            if n < 1e-9:
                continue
            new *= dmin * 1.02 / n
        new = np.array([_rt(new[0], step), _rt(new[1], step)])
        n = float(np.linalg.norm(new))
        if n < dmin or n > max_shift:
            continue
        if np.linalg.norm(new - cur) < 1e-9:
            continue
        poly = affinity.translate(base.group_polys[bi], float(new[0]), float(new[1]))
        if not _building_move_ok(scene, base, cfg, gi, bi, poly):
            continue
        d = new - cur
        translate_group(data, scene, gi, float(d[0]), float(d[1]))
        return {"op": "shift_building", "building": scene.group_names[gi],
                "offset_from_base_m": [float(new[0]), float(new[1])], "shift_m": round(n, 2),
                "min_shift_m": round(dmin, 2)}
    return None


def op_tree(kind, data, base, cfg, rng, strength, scene):
    tc, T = cfg["tree"], data["trees"]
    if kind == "add":
        if len(T) + 1 > max_tree_count(cfg, tree_count0(base, cfg), scene.pathway is not None):
            return None
        pos = _in_site_point(scene, rng)
        if pos is None:
            return None
        t = new_tree(cfg, T, rng)
        n = max([int(str(x["id"]).split()[-1]) for x in T if str(x["id"]).split()[-1].isdigit()] + [0])
        t["id"] = f"Tree {n + 1}"
        t["position"].update(x=pos[0], y=pos[1])
        R = float(np.clip(_rt(rng.uniform(*tc["new_tree_radius_range_m"]), tc["size_step_m"]), *tc["radius_range_m"]))
        set_tree_geometry(t, R, _height_for(scene, pos[0], pos[1], cfg, rng), tc)
        T.append(t)
        return {"op": "add_tree", "tree": t["id"], "position": pos, "radius_m": R}
    if not T:
        return None
    i = int(rng.integers(len(T)))
    t = T[i]
    if kind == "remove":
        T.pop(i)
        return {"op": "remove_tree", "tree": t["id"]}
    R = _tree_R(t)
    x, y = t["position"]["x"], t["position"]["y"]
    h = float(t["height"])
    if kind == "move":
        if rng.random() < tc["relocate_probability"]:
            pos = _in_site_point(scene, rng)
            if pos is None:
                return None
            x, y = pos
        else:
            x, y = x + rng.normal(0, max(strength, 0.05) * tc["move_sigma_m"]), y + rng.normal(0, max(strength, 0.05) * tc["move_sigma_m"])
        t["position"].update(x=float(x), y=float(y))
        h = _height_for(scene, x, y, cfg, rng, current=h)
        set_tree_geometry(t, R, h, tc)
        return {"op": "move_tree", "tree": t["id"], "position": [float(x), float(y)]}
    if kind == "resize":
        R = float(np.clip(_rt(R + rng.normal(0, max(strength, 0.05) * tc["resize_sigma_m"]), tc["size_step_m"]), *tc["radius_range_m"]))
        h = _height_for(scene, x, y, cfg, rng, current=h if rng.random() < 0.5 else None)
        set_tree_geometry(t, R, h, tc)
        return {"op": "resize_tree", "tree": t["id"], "radius_m": R, "height_m": h}
    return None


def _ellipse(cx, cy, area, aspect, theta, nv, rng, jitter=0.05):
    b = math.sqrt(area / (math.pi * aspect))
    a = aspect * b
    ang = np.sort(rng.uniform(0, 2 * math.pi, nv)) if False else np.linspace(0, 2 * math.pi, nv, endpoint=False)
    rad = 1.0 + rng.uniform(-jitter, jitter, nv)
    pts = [(a * r * math.cos(t), b * r * math.sin(t)) for t, r in zip(ang, rad)]
    poly = affinity.rotate(Polygon(pts), math.degrees(theta), origin=(0, 0))
    return affinity.translate(poly, cx, cy)


def _normalize_area(polys, lo, hi):
    """Rescale all patches about their centroids so the total lies in [lo, hi] (m2)."""
    tot = sum(p.area for p in polys)
    if hi - lo < 0.004 * max(hi, 1.0):                     # (almost) fixed total, e.g. water-limited lawn
        target = 0.5 * (lo + hi)
    else:
        target = min(max(tot, lo * 1.0005), hi * 0.9995) if (tot < lo or tot > hi) else tot
    if target == tot or tot <= 0:
        return polys
    s = math.sqrt(target / tot)
    return [affinity.scale(p, s, s, origin="centroid") for p in polys]


def _template_patch(cat, cfg):
    from layout_reflow import _default_patch
    return _default_patch(cat, cfg)


def op_patch(kind, cat, data, base, cfg, rng, strength, scene):
    key = "ponds" if cat == "pond" else "grassAreas"
    P, pc, sc = data[key], cfg[cat], cfg["patch"]
    polys = [_poly(p["coordinates"]) for p in P]
    a0 = base_area0(base, cfg, cat)
    s = max(strength, 0.05)
    info = {"op": f"{kind}_{cat}"}
    if kind == "add":
        if len(P) + 1 > patch_limit(cfg, cat, len(building_groups(data))):
            return None
        pos = _in_site_point(scene, rng)
        if pos is None:
            return None
        area = max(pc["min_area_m2"], a0 * rng.uniform(*sc["new_patch_share_range"]))
        poly = _ellipse(pos[0], pos[1], area, rng.uniform(*sc["new_patch_aspect_range"]),
                        rng.uniform(0, math.pi), sc["new_patch_vertices"], rng)
        newp = copy.deepcopy(P[int(rng.integers(len(P)))]) if P else _template_patch(cat, cfg)
        newp["id"] = f"{cat if cat == 'pond' else 'grass'}_{int(time.time() * 1000) + int(rng.integers(1_000_000))}"
        P.append(newp)
        polys.append(poly)
        info["patch"] = newp["id"]
        idx = len(P) - 1
    else:
        if not P:
            return None
        idx = int(rng.integers(len(P)))
        poly = polys[idx]
        if kind == "remove":
            if len(P) <= 1:
                return None
            info["patch"] = P[idx]["id"]
            P.pop(idx)
            polys.pop(idx)
            idx = None
        else:
            info["patch"] = P[idx]["id"]
            c = poly.centroid
            if kind == "shift":
                poly = affinity.translate(poly, rng.normal(0, s * sc["shift_sigma_m"]), rng.normal(0, s * sc["shift_sigma_m"]))
            elif kind == "rotate":
                poly = affinity.rotate(poly, rng.normal(0, s * sc["rotate_sigma_deg"]), origin="centroid")
            elif kind == "stretch":
                phi = rng.uniform(0, 180)
                lam = math.exp(rng.normal(0, s * sc["stretch_sigma"]))
                poly = affinity.rotate(poly, -phi, origin="centroid")
                poly = affinity.scale(poly, lam, 1 / lam, origin="centroid")
                poly = affinity.rotate(poly, phi, origin="centroid")
            elif kind == "scale":
                f = math.exp(rng.normal(0, s * sc["scale_sigma"]))
                poly = affinity.scale(poly, f, f, origin="centroid")
            elif kind == "jitter":
                pts = []
                for (x, y) in list(poly.exterior.coords)[:-1]:
                    k = 1.0 + rng.normal(0, s * sc["jitter_sigma_frac"])
                    pts.append((c.x + (x - c.x) * k, c.y + (y - c.y) * k))
                poly = Polygon(pts)
            else:
                return None
            polys[idx] = poly
    polys = _normalize_area(polys, *area_band(cfg, cat, a0))
    stp = cfg["patch"].get("discretization_m", 1.0)
    polys = [unslit(q) if q.is_valid else q for q in discretize_patches(polys, stp, _other_patches(scene, cat))]
    for p, poly in zip(P, polys):
        patch_set_polygon(p, poly, cat == "pond", 0)
    return info


# ----------------------------------------------------------------------------- novelty (is a child just a copy?)
def novelty_features(data):
    """Compact description of a layout used to compare layouts: building centroids, trees, lawn / pond unions."""
    from shapely.ops import unary_union
    data = ground_view(data)
    b = {}
    for x in data.get("buildings", []):
        c = (x.get("geometry") or {}).get("centroid")
        if c:
            b[x["id"]] = (float(c["x"]), float(c["y"]))
    t = np.array([[tr["position"]["x"], tr["position"]["y"], tr["dimensions"]["crownRadius"], tr["dimensions"]["totalHeight"]]
                  for tr in data.get("trees", [])], float).reshape(-1, 4)
    pat = {}
    for key in ("grassAreas", "ponds"):
        polys = [_poly(p["coordinates"]).buffer(0) for p in data.get(key, []) if len(p.get("coordinates", [])) >= 3]
        pat[key] = unary_union(polys) if polys else None
    return {"b": b, "t": t, "p": pat}


def novelty_distance(fa, fb):
    """(largest building shift [m], number of trees that differ, largest relative lawn / pond change) between two layouts."""
    from scipy.spatial import cKDTree
    shifts = [math.hypot(fa["b"][k][0] - fb["b"][k][0], fa["b"][k][1] - fb["b"][k][1]) for k in fa["b"] if k in fb["b"]]
    shift = max(shifts) if shifts else 0.0
    if set(fa["b"]) != set(fb["b"]):
        shift = float("inf")
    A, B = fa["t"], fb["t"]
    if len(A) == 0 or len(B) == 0:
        dt = max(len(A), len(B))
    else:
        d, j = cKDTree(B[:, :2]).query(A[:, :2])
        same = (d < 0.05) & (np.abs(A[:, 2] - B[j, 2]) < 1e-9) & (np.abs(A[:, 3] - B[j, 3]) < 1e-9)
        d2, j2 = cKDTree(A[:, :2]).query(B[:, :2])
        same2 = (d2 < 0.05) & (np.abs(B[:, 2] - A[j2, 2]) < 1e-9) & (np.abs(B[:, 3] - A[j2, 3]) < 1e-9)
        dt = int(max((~same).sum(), (~same2).sum()))
    frac = 0.0
    for key in ("grassAreas", "ponds"):
        pa, pb = fa["p"][key], fb["p"][key]
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            frac = max(frac, 1.0)
            continue
        frac = max(frac, pa.symmetric_difference(pb).area / max(pa.area, pb.area, 1e-9))
    return float(shift), int(dt), float(frac)


def too_similar(fa, fb, cfg):
    """True if two layouts are practically the same (see DEFAULTS['novelty'])."""
    n = cfg.get("novelty", {})
    if not n.get("enabled", True):
        return False
    shift, dt, frac = novelty_distance(fa, fb)
    return shift < n["min_building_shift_m"] and dt < n["min_tree_changes"] and frac < n["min_patch_change_frac"]


def pick_category(cfg, rng):
    w = cfg["ops"]["category_weights"]
    cats, p = list(w), np.array(list(w.values()), float)
    return str(rng.choice(cats, p=p / p.sum()))


def apply_random_op(data, base, cfg, rng, strength, category=None, scene=None):
    """Pick one operation by the configured weights and apply it to `data` in place.
    Returns an info dict, or None if the operation could not be applied."""
    scene = scene or Scene(data, cfg, base.osm_polys)
    cat = category or pick_category(cfg, rng)
    if cat == "building":
        return op_shift_building(data, base, cfg, rng, strength, scene)
    if cat == "tree":
        tw = cfg["ops"]["tree_weights"]
        kinds, p = list(tw), np.array(list(tw.values()), float)
        return op_tree(str(rng.choice(kinds, p=p / p.sum())), data, base, cfg, rng, strength, scene)
    pw = cfg["ops"]["patch_weights"]
    kinds, p = list(pw), np.array(list(pw.values()), float)
    return op_patch(str(rng.choice(kinds, p=p / p.sum())), cat, data, base, cfg, rng, strength, scene)


def try_op(data, base, cfg, rng, strength, base_map, category=None, tries=40):
    """Apply one random op to a COPY; accept only if no new violation remains (trees that a
    building move pushes into a rule break are dropped, up to max_dropped_trees)."""
    category = category or pick_category(cfg, rng)     # keep the category across retries, otherwise easy
    for k in range(tries):                              # categories (trees) would always win over hard ones
        # tight layouts: a big move usually hits something, so the step is halved every 8 failed tries
        strength_k = strength * 0.5 ** min(4, k // 8)
        trial = working_copy(data)
        scene = Scene(trial, cfg, base.osm_polys)
        info = apply_random_op(trial, base, cfg, rng, strength_k, category, scene)
        if info is None:
            continue
        if info["op"] == "shift_building":               # lawns / ponds make room, trees move out of the way
            from layout_reflow import fix_trees, reflow_patches
            rep = reflow_patches(trial, base, cfg, rng)
            if rep is None:
                continue
            info["moved_trees"] = fix_trees(trial, base, cfg, rng)
        from layout_reflow import ensure_pathway_shade        # road shade is mandatory: restore it if this operation lost it
        shade = ensure_pathway_shade(trial, base, cfg, rng)
        if shade["planted"] or shade["enlarged"] or shade["relocated"]:
            info["pathway_shade"] = shade
        viol = new_violations(Scene(trial, cfg, base.osm_polys), base, base_map)
        if viol and info["op"] == "shift_building":
            bad = {i for t, ids, _ in viol if t.startswith("tree_") for i in ids if i in {x["id"] for x in trial["trees"]}}
            if bad and len(bad) <= cfg["tree"]["max_dropped_trees"]:
                trial["trees"] = [t for t in trial["trees"] if t["id"] not in bad]
                info["dropped_trees"] = sorted(bad)
                viol = new_violations(Scene(trial, cfg, base.osm_polys), base, base_map)
        if not viol:
            refresh_metadata(trial)
            return trial, info
    return None


def patches_break_clearance(data, base, cfg):
    """Categories ('pond', 'lawn') with a patch that overlaps a building, is closer to one than its clearance
    (building_clearance_m of the category) or has a hairline slit - such a layout must be reflowed (cut back and refilled)."""
    sc = Scene(data, cfg, base.osm_polys)
    bld = building_zone(sc.group_polys, sc.osm_polys, 0.0)
    out = []
    if bld is None:
        return out
    ctol = cfg["patch"].get("clearance_tol_m", 0.05)
    for cat, polys in (("pond", sc.ponds), ("lawn", sc.lawns)):
        cl = clearance_m(cfg, cat)
        if any(p.is_valid and not p.is_empty and (p.distance(bld) < max(cl - ctol, 0.0) + 1e-9 or has_slit(p)) for p in polys):
            out.append(cat)
    return out


def patches_out_of_band(data, base, cfg):
    """Categories ('pond', 'lawn') that must be rebuilt by fit_patch_area: the total area is outside the allowed band
    (water budget below the base lawn, a base without ponds ...) or a patch is too close to a building."""
    todo = []
    for cat, key in (("pond", "ponds"), ("lawn", "grassAreas")):
        a0 = base_area0(base, cfg, cat)
        lo, hi = area_band(cfg, cat, a0)
        tot = sum(_poly(p["coordinates"]).area for p in data[key])
        if not (lo - 1e-3 * a0 <= tot <= hi + 1e-3 * a0):
            todo.append(cat)
    for cat in patches_break_clearance(data, base, cfg):
        if cat not in todo:
            todo.append(cat)
    return todo


def fit_patch_area(data, base, cfg, rng=None, pond_blockers=None):
    """If the total lawn / pond area is outside its allowed band (water budget below the base lawn, or a base
    without ponds) or a pond / lawn overlaps a building or is closer to one than its clearance (ponds 2 m, lawns 1 m),
    reshape the patches (see layout_reflow.reflow_patches).  Returns the list of categories that had to be rebuilt."""
    from layout_reflow import reflow_patches
    rng = rng if rng is not None else np.random.default_rng(0)
    todo = patches_out_of_band(data, base, cfg)
    if todo and reflow_patches(data, base, cfg, rng, pond_blockers) is None:
        raise ValueError("could not fit lawn / pond areas to their allowed band (and clearance from the buildings)")
    return todo


def validate(data, base, cfg, stripped=False):
    """(new_violations, inherited_violations) of a layout against the base layout.  stripped=False: a layout file (what stands on a
    building of the base is left out); True: data in the middle of an operation."""
    if not stripped:
        data = ground_view(data, base.hosted_ids)
    sc = Scene(data, cfg, base.osm_polys)
    bm = violation_map(check(base, base))
    allv = check(sc, base)
    new = [v for v in allv if bm.get((v[0], v[1])) is None or v[2] > bm[(v[0], v[1])] + 1e-4]
    inh = [v for v in allv if v not in new]
    return new, inh