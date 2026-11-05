#!/usr/bin/env python3
"""check_invariants.py  --  independent audit of candidate files against the base layout.

Does NOT use layout_ops.check; it re-derives the key rules with plain geometry, so a bug in the
generator cannot hide itself.

    python check_invariants.py input/individual_*.json            (base = input/base.json)
    python check_invariants.py --base other_base.json input/individual_*.json

Rules that the BASE already breaks (overlaps, tree heights, ...) are tolerated when a candidate does
not make them worse; the water budget, the area bands, lawn / pond overlaps with buildings or each other and the
clearance from the buildings (ponds >= 2 m, lawns >= 1 m) are always hard.

Stacked buildings (a mass on a slab, internal floors: "Building 3", "building_3_mass_1", "building_3_internal_floor_2") are ONE building:
they must move together and every overlap / clearance rule uses the union of their footprints.  Trees, lawns and ponds that stand on a
building in the BASE must still be there, glued to it (moved by exactly the translation of their building, nothing else changed);
they are not part of the tree / lawn / pond budgets.
"""
import argparse, copy, json, math, re, sys
from pathlib import Path
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(ROOT / "input" / "base.json"))
ap.add_argument("--constraints", default=str(ROOT / "input" / "constraints.json"))
ap.add_argument("files", nargs="+")
A = ap.parse_args()
base = json.loads(Path(A.base).read_text())
cp = Path(A.constraints)
CON = json.loads(cp.read_text()) if cp.exists() else {}
KF = CON.get("tree", {}).get("building_clearance_factor", 1.10)
SHIFT = CON.get("building", {}).get("max_shift_frac_of_site_diameter", 0.10)
Q = CON.get("lawn", {}).get("water_available_l_per_day")
CL_POND = CON.get("pond", {}).get("building_clearance_m", 2.0)     # minimum distance pond - building (generated and OSM)
CL_LAWN = CON.get("lawn", {}).get("building_clearance_m", 1.0)     # minimum distance lawn - building
CTOL = CON.get("patch", {}).get("clearance_tol_m", 0.05)
DEPTH = 5.0   # mm/day, fixed
P = lambda cs: Polygon([(v["x"], v["y"]) for v in cs])


def diff_paths(a, b, p=""):
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if num(a) and num(b):                      # 3 and 3.0 are the same number (a moved copy of an integer coordinate is a float)
        if abs(a - b) > 1e-9: yield p
        return
    if type(a) != type(b):
        yield p; return
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b: yield f"{p}/{k}"
            else: yield from diff_paths(a[k], b[k], f"{p}/{k}")
    elif isinstance(a, list):
        if len(a) != len(b): yield p; return
        for i, (x, y) in enumerate(zip(a, b)): yield from diff_paths(x, y, f"{p}[{i}]")
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and abs(a - b) < 1e-9):
        yield p


def fp_key(b):
    return tuple((round(v["x"], 6), round(v["y"], 6)) for v in b["geometry"]["footprint"])


def roads_union(d):
    gs = []
    for r in d.get("roads", []) + d.get("pathways", []):
        pts = [(v["x"], v["y"]) for v in r["coordinates"]]
        if len(pts) >= 3:
            gs.append(Polygon(pts).buffer(0))
    return unary_union(gs) if gs else None


POSITIONAL = ("position", "geometry", "faces", "baseVertices", "vertices", "center", "centroid", "boundingBox", "bounds")
site = P(base["site"]["coordinates"])
diam = max(math.dist(p, q) for p in site.exterior.coords for q in site.exterior.coords)
base_pw = roads_union(base)


def bpoly(b):
    return P(b["geometry"]["footprint"]).buffer(0)


def stack_groups(d):
    """Independent re-implementation: connected components of 'is part of the same stack' (same footprint | same Building N family and
    overlapping | footprints overlap by >= half of the smaller one and the heights touch within 0.3 m).  -> {first index: [indices]}"""
    B = d["buildings"]; n = len(B)
    polys = [bpoly(b) for b in B]
    z = []
    for b in B:
        z0 = float((b.get("position") or {}).get("z", 0) or 0)
        z.append((z0, z0 + float(b.get("height", 0) or 0)))
    fam = [(m.group(1).lower(), int(m.group(2))) if (m := re.match(r"^\s*(building)[\s_\-]*(\d+)", str(b["id"]), re.I)) else None for b in B]
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if fp_key(B[i]) == fp_key(B[j]):
                adj[i].add(j); adj[j].add(i); continue
            if not polys[i].intersects(polys[j]):
                continue
            a = polys[i].intersection(polys[j]).area
            if a <= 1e-6:
                continue
            m_ = min(polys[i].area, polys[j].area)
            gap = max(z[i][0] - z[j][1], z[j][0] - z[i][1])
            if (fam[i] and fam[i] == fam[j] and a > 0.01 * m_) or (gap <= 0.3 and a >= 0.5 * m_):
                adj[i].add(j); adj[j].add(i)
    seen, out = set(), {}
    for i in range(n):
        if i in seen:
            continue
        comp, todo = [], [i]
        seen.add(i)
        while todo:
            k = todo.pop(); comp.append(k)
            for q in adj[k]:
                if q not in seen:
                    seen.add(q); todo.append(q)
        out[min(comp)] = sorted(comp)
    return out


base_groups = stack_groups(base)
gpoly = lambda d, idx: unary_union([bpoly(d["buildings"][i]) for i in idx]).buffer(0)


def shift_xy(o, dx, dy):
    if isinstance(o, dict):
        if isinstance(o.get("x"), (int, float)) and isinstance(o.get("y"), (int, float)):
            o["x"] += dx; o["y"] += dy
        for v in o.values():
            if isinstance(v, (dict, list)): shift_xy(v, dx, dy)
    elif isinstance(o, list):
        for v in o: shift_xy(v, dx, dy)


SITE0 = P(base["site"]["coordinates"]).buffer(0)


def hosted_of(d):
    """{(key, id): group first index} of the trees / ponds / lawns standing on a stack of d (centre inside, or > half the outline inside)."""
    out = {}
    gp = {k: gpoly(d, idx) for k, idx in base_groups.items()}
    for t in d["trees"]:
        pt = Point(t["position"]["x"], t["position"]["y"])
        for k, q in gp.items():
            if q.buffer(1e-6).contains(pt): out[("trees", t["id"])] = k; break
    for key in ("ponds", "grassAreas"):
        for p_ in d[key]:
            q = P(p_["coordinates"]).buffer(0)
            for k, g in gp.items():
                if q.area > 0 and q.intersection(g).area > 0.5 * q.area: out[(key, p_["id"])] = k; break
            else:
                if q.area > 0 and q.intersection(SITE0).area <= 1e-6: out[(key, p_["id"])] = None       # outside the site: must stay exactly as in the base
    return out


BASE_HOST = hosted_of(base)
BASE_FULL = base
base = dict(base)
for _k in ("trees", "ponds", "grassAreas"):
    base[_k] = [x for x in BASE_FULL[_k] if (_k, x["id"]) not in BASE_HOST]      # the budgets / rules below are for the free items


def ground(d):
    d = dict(d)
    for _k in ("trees", "ponds", "grassAreas"):
        d[_k] = [x for x in d[_k] if (_k, x["id"]) not in BASE_HOST]
    return d


def open_space(d):
    parts = [gpoly(d, idx) for idx in base_groups.values()]
    parts += [Polygon([(v["x"], v["z"]) for v in o["dimensions"]["baseVertices"]]) for o in d["osmBuildings"]]
    parts += [P(g["coordinates"]) for g in d["grassAreas"]]
    pw = roads_union(d)
    if pw is not None:
        parts.append(pw)
    return site.difference(unary_union(parts)).area


def mirror(poly):
    return Polygon([(x, -y) for x, y in poly.exterior.coords])


def iou(a, b):
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def fp_relations(d):
    """footprints[] entry -> ('mirror' | 'direct' | None) relative to the building footprint of the same id."""
    out = {}
    byid = {b["id"]: b for b in d["buildings"]}
    for fp in d.get("footprints", []):
        bid = fp["id"].replace("shape_fp_", "", 1)
        b = byid.get(bid)
        if b is None:
            continue
        pf, pb = Polygon([(v["x"], v["y"]) for v in fp["vertices"]]).buffer(0), P(b["geometry"]["footprint"]).buffer(0)
        out[fp["id"]] = frozenset(r for r, q in (("direct", pb), ("mirror", mirror(pb))) if iou(pf, q) > 0.99)   # both when symmetric
    return out


def components(d):
    """Every placed object as {key: polygon in the BUILDING frame}.  Groups of stacked parts count once."""
    comp = {}
    for k, idx in base_groups.items():
        comp[("building", d["buildings"][idx[0]]["id"])] = gpoly(d, idx)
    for o in d["osmBuildings"]:
        comp[("osm", o["id"])] = Polygon([(v["x"], v["z"]) for v in o["dimensions"]["baseVertices"]]).buffer(0)
    for r in d.get("roads", []) + d.get("pathways", []):
        if len(r["coordinates"]) >= 3:
            key = ("road", "all roads")
            q = P(r["coordinates"]).buffer(0)
            comp[key] = q if key not in comp else comp[key].union(q)
    for p_ in d["ponds"]:
        comp[("pond", p_["id"])] = P(p_["coordinates"])
    for g in d["grassAreas"]:
        comp[("lawn", g["id"])] = P(g["coordinates"])
    return comp


def overlap_pairs(d, tol=0.01):
    """Pairwise overlap areas between all objects, except tree-lawn (trees are points here) and static-static pairs."""
    comp = components(d); keys = list(comp); out = {}
    static = ("osm", "road")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if a[0] in static and b[0] in static:
                continue
            ar = comp[a].intersection(comp[b]).area
            if ar > tol:
                out[(a, b)] = ar
    return out


def base_pwd(t):
    return base_pw.distance(Point(t['position']['x'], t['position']['y'])) if base_pw is not None else 1e9


BASE_OVL = None
BASE_REL = None

OPEN0 = open_space(base)
GEN = CON.get("generation", {})
TC = CON.get("tree", {})


def main():
    global BASE_OVL, BASE_REL
    BASE_OVL, BASE_REL = overlap_pairs(base), fp_relations(base)
    fails = 0
    for f in A.files:
        d = json.loads(Path(f).read_text()); err = []
        if set(d) != set(BASE_FULL): err.append("top-level keys differ")
        for k in ("osmBuildings", "osm_buildings", "osmRoads", "roads", "pathways", "dxfEntities", "site"):
            if d.get(k) != BASE_FULL.get(k): err.append(f"{k} changed")
        maxshift = 0.0
        if len(d["buildings"]) != len(base["buildings"]): err.append("building count")
        else:
            shifts = []
            for b0, b1 in zip(base["buildings"], d["buildings"]):
                if b0["id"] != b1["id"]: err.append("building order/id")
                bad = [p for p in diff_paths(b0, b1) if not any(s in p for s in POSITIONAL)]
                if bad: err.append(f"{b0['id']} non-positional change {bad[:2]}")
                shifts.append((b1["position"]["x"] - b0["position"]["x"], b1["position"]["y"] - b0["position"]["y"]))
                dm = lambda b: {k: v for k, v in b["dimensions"].items() if k != "baseVertices"}
                if b0["height"] != b1["height"] or dm(b0) != dm(b1): err.append("building size changed")
                for u, w in zip(b0["dimensions"].get("baseVertices", []), b1["dimensions"].get("baseVertices", [])):
                    if abs((w["x"] - u["x"]) - shifts[-1][0]) > 1e-6 or abs((w["y"] - u["y"]) - shifts[-1][1]) > 1e-6:
                        err.append(f"{b0['id']} baseVertices shift != position shift"); break
                for u, w in zip(b0["geometry"]["footprint"], b1["geometry"]["footprint"]):
                    if abs((w["x"] - u["x"]) - shifts[-1][0]) > 1e-6 or abs((w["y"] - u["y"]) - shifts[-1][1]) > 1e-6:
                        err.append(f"{b0['id']} footprint shift != position shift"); break
            s0 = np.array(shifts)
            if {tuple(v) for v in stack_groups(d).values()} != {tuple(v) for v in base_groups.values()}: err.append("stack structure changed")
            for k, idx in base_groups.items():           # parts of one physical building move together
                if np.ptp(s0[idx], axis=0).max() > 1e-6: err.append(f"stacked parts moved differently {np.ptp(s0[idx], axis=0)}")
                maxshift = max(maxshift, float(np.hypot(*s0[idx[0]])))
            if maxshift > SHIFT * diam + 1e-6: err.append(f"shift {maxshift:.2f} > {SHIFT * diam:.2f}")
        # trees / lawns / ponds on a building: still there, moved by exactly the translation of their building, nothing else changed
        ids = [t["id"] for t in d["trees"]]
        if len(set(ids)) != len(ids): err.append("duplicate tree ids")
        if len(d["buildings"]) == len(base["buildings"]):
            for (key, iid), k in BASE_HOST.items():
                it0 = next(x for x in BASE_FULL[key] if x["id"] == iid)
                found = [x for x in d[key] if x["id"] == iid]
                if len(found) != 1: err.append(f"GLUE {key[:-1]} {iid} (" + ("outside the site" if k is None else f"on {base['buildings'][k]['id']}") + ") is missing"); continue
                if k is None:
                    bad = [p_ for p_ in diff_paths(it0, found[0]) if True]
                    if bad: err.append(f"OUTSIDE {key[:-1]} {iid} (outside the site) was changed ({bad[:2]})")
                    continue
                sh = (d["buildings"][k]["position"]["x"] - base["buildings"][k]["position"]["x"], d["buildings"][k]["position"]["y"] - base["buildings"][k]["position"]["y"])
                exp = copy.deepcopy(it0); shift_xy(exp, *sh)
                bad = [p_ for p_ in diff_paths(exp, found[0]) if True]
                if bad: err.append(f"GLUE {key[:-1]} {iid} is not rigidly on {base['buildings'][k]['id']} ({bad[:2]})")
        d = ground(d)
        ids = [t["id"] for t in d["trees"]]
        for t in d["trees"]:
            R = t["dimensions"]["crownRadius"]
            if abs(t["crownDiameter"] - 2 * R) > 1e-6 or abs(t["height"] - t["dimensions"]["totalHeight"]) > 1e-6: err.append(f"{t['id']} inconsistent fields")
            if not 1 - 1e-9 <= R <= 6 + 1e-9: err.append(f"{t['id']} radius {R}")
        # tree-building clearance >= KF*R (editable moved buildings + OSM); inherited if the tree is untouched
        bpolys = [gpoly(d, idx) for idx in base_groups.values()]
        bpolys += [Polygon([(v["x"], v["z"]) for v in o["dimensions"]["baseVertices"]]) for o in d["osmBuildings"]]
        base_bpolys = [gpoly(base, idx) for idx in base_groups.values()] + \
                      [Polygon([(v["x"], v["z"]) for v in o["dimensions"]["baseVertices"]]) for o in base["osmBuildings"]]
        base_t = {t["id"]: (t["position"]["x"], t["position"]["y"], t["dimensions"]["crownRadius"]) for t in base["trees"]}
        for t in d["trees"]:
            R = t["dimensions"]["crownRadius"]; pt = Point(t["position"]["x"], t["position"]["y"])
            dmin = min(pt.distance(bp) for bp in bpolys)
            worse = True
            if t["id"] in base_t:                         # tolerated only if not worse than in the base
                bx, by, bR = base_t[t["id"]]
                bd = min(Point(bx, by).distance(bp) for bp in base_bpolys)
                worse = (dmin - KF * R) < (bd - KF * bR) - 1e-6
            if dmin < KF * R - 1e-6 and worse:
                err.append(f"{t['id']} clearance {dmin:.2f} < {KF * R:.2f}")
        # ponds / lawns: total area band, ids, depth, validity
        for key, name in (("ponds", "pond"), ("grassAreas", "lawn")):
            a0 = sum(P(p["coordinates"]).area for p in base[key]); a1 = sum(P(p["coordinates"]).area for p in d[key])      # inside the site only (the ones outside are frozen and not counted)
            if a0 == 0:                                             # none in the base: 10 % of the open space (ponds and lawns)
                a0 = CON.get("pond" if key == "ponds" else "lawn", {}).get("open_space_frac_if_missing", 0.10) * OPEN0
            hi = 1.1 * a0 if key != "grassAreas" else min(1.1 * a0, (Q if Q else a0 * 0.005 * 1000) / DEPTH)
            lo = min(a0, hi)
            if not lo - 1e-3 * a0 <= a1 <= hi + 1e-3 * a0: err.append(f"{name} area {a1 / a0:.3f} x A0 (allowed {lo / a0:.3f}-{hi / a0:.3f})")
            pid = [p["id"] for p in d[key]]
            if len(set(pid)) != len(pid): err.append(f"duplicate {name} ids")
            # patch count: lawns <= lawn.max_patches; ponds <= max(pond.max_patches, number of buildings) unless pond.max_patches_follows_buildings is false
            lim = CON.get(name, {}).get("max_patches", 4 if key == "ponds" else 6)
            if key == "ponds" and CON.get("pond", {}).get("max_patches_follows_buildings", True): lim = max(lim, len(base_groups))
            if len(d[key]) > lim: err.append(f"{len(d[key])} {name}s > limit {lim}")
            if key == "ponds" and base[key] and {p["dimensions"]["depth"] for p in d[key]} != {p["dimensions"]["depth"] for p in base[key]}: err.append("pond depth changed")
            if key == "ponds" and not base[key] and {p["dimensions"]["depth"] for p in d[key]} - {CON.get("pond", {}).get("default_depth_m", 2.0)}: err.append("created pond depth != default")
            for p in d[key]:
                if not P(p["coordinates"]).is_valid: err.append(f"{p['id']} invalid polygon")
                closed = P(p["coordinates"]).buffer(0.1, join_style=2)       # a reader that only sees the outer ring
                if closed.geom_type != "Polygon" or len(closed.interiors) > 0:
                    err.append(f"{p['id']} encloses something through a hairline slit")
        # overlaps with pathways must not be worse than in the base
        pw = roads_union(d)
        if pw is not None:
            for key, label in (("ponds", "pond"), ("grassAreas", "lawn")):
                b0 = sum(P(p["coordinates"]).intersection(base_pw).area for p in base[key])
                b1 = sum(P(p["coordinates"]).intersection(pw).area for p in d[key])
                if b1 > b0 + 1e-2: err.append(f"{label}-pathway overlap {b1:.1f} > base {b0:.1f}")
            for (k, idx) in base_groups.items():
                a0 = gpoly(base, idx).intersection(base_pw).area
                a1 = gpoly(d, idx).intersection(pw).area
                if a1 > a0 + 1e-2: err.append(f"building {base['buildings'][idx[0]]['id']} pathway overlap {a1:.1f} > base {a0:.1f}")
        # trees must not overlap pathways (generations: never; others: not worse than in the base)
        pwt = roads_union(d)
        if pwt is not None and not TC.get("tree_on_pathway_allowed", False):
            btp = {t["id"]: base_pwd(t) for t in base["trees"]}
            bad = []
            for t in d["trees"]:
                R = t["dimensions"]["crownRadius"]; dd = pwt.distance(Point(t["position"]["x"], t["position"]["y"]))
                if dd < R - 1e-6 and (Path(f).name.startswith(("individual_", "generation_")) or t["id"] not in btp or dd < btp[t["id"]] - 1e-6):
                    bad.append(t["id"])
            if bad: err.append(f"{len(bad)} tree crowns overlap a pathway (e.g. {bad[0]})")
        # ---- road shade (every individual / offspring except the untouched base copy): both edges of every road have a crown edge
        # within reach_m for >= min_coverage of the samples where a crown still fits.  Independent re-implementation.
        SP = TC.get("pathway_planting", {})
        if pwt is not None and SP.get("enabled", True) and SP.get("shade_required", True) and Path(f).name not in ("individual_0.json", "generation_0.json"):
            reach, step_s, rmin = SP.get("reach_m", 3.0), SP.get("sample_step_m", 2.0), SP.get("min_radius_m", 1.5)
            off = rmin + SP.get("canopy_gap_to_path_m", 0.5)
            ball = unary_union([gpoly(d, idx) for idx in base_groups.values()] +
                               [Polygon([(v["x"], v["z"]) for v in o["dimensions"]["baseVertices"]]) for o in d["osmBuildings"]])
            pnd = unary_union([P(q["coordinates"]) for q in d["ponds"]]) if d["ponds"] else None
            TX = np.array([[t["position"]["x"], t["position"]["y"], t["dimensions"]["crownRadius"]] for t in d["trees"]]).reshape(-1, 3)
            tot = shaded = 0
            for pg in (list(pwt.geoms) if hasattr(pwt, "geoms") else [pwt]):
                for ring in [pg.exterior, *pg.interiors]:
                    n_ = max(2, int(round(ring.length / step_s)))
                    for k_ in range(n_):
                        sp = ring.length * k_ / n_
                        q0, qa, qb = ring.interpolate(sp), ring.interpolate((sp - 0.5) % ring.length), ring.interpolate((sp + 0.5) % ring.length)
                        tx_, ty_ = qb.x - qa.x, qb.y - qa.y; nn = math.hypot(tx_, ty_)
                        if nn < 1e-9: continue
                        tx_, ty_ = tx_ / nn, ty_ / nn
                        nx_, ny_ = -ty_, tx_
                        if pwt.contains(Point(q0.x + .3 * nx_, q0.y + .3 * ny_)): nx_, ny_ = -nx_, -ny_
                        if pwt.contains(Point(q0.x + .3 * nx_, q0.y + .3 * ny_)): continue
                        fits = False
                        for dl in (0, 2, -2, 4, -4):
                            c = Point(q0.x + off * nx_ + dl * tx_, q0.y + off * ny_ + dl * ty_)
                            if site.contains(c) and site.exterior.distance(c) >= rmin and ball.distance(c) >= KF * rmin and pwt.distance(c) >= rmin \
                               and (pnd is None or pnd.distance(c) >= rmin):
                                fits = True; break
                        if not fits: continue
                        tot += 1
                        if len(TX) and (np.hypot(TX[:, 0] - q0.x, TX[:, 1] - q0.y) - TX[:, 2]).min() <= reach + 1e-9: shaded += 1
            if tot and shaded / tot < SP.get("min_coverage", 0.8) - 0.03:
                err.append(f"road shade {shaded / tot:.2f} < {SP.get('min_coverage', 0.8):.2f}")
        # ---- independent geometry audit: NO prohibited overlap between any two objects (trees may overlap lawns)
        # lawn / pond with a building (generated or OSM) or with each other: NEVER allowed, whatever the base does
        HARD = {"pond", "lawn", "building", "osm"}
        for (a, b), ar in overlap_pairs(d).items():
            hard = {a[0], b[0]} <= HARD and bool({a[0], b[0]} & {"pond", "lawn"})
            b0 = 0.0 if hard else BASE_OVL.get((a, b), 0.0)
            if ar > b0 + 1e-2:
                err.append(f"OVERLAP {a[0]} {a[1][-14:]} x {b[0]} {b[1][-14:]} = {ar:.2f} m2 (base {b0:.2f})")
        # clearance: ponds >= pond.building_clearance_m (2 m), lawns >= lawn.building_clearance_m (1 m) from every building
        blds = [("building " + str(d["buildings"][idx[0]]["id"]), gpoly(d, idx)) for idx in base_groups.values()] + \
               [("osm " + str(o["id"]), Polygon([(v["x"], v["z"]) for v in o["dimensions"]["baseVertices"]]).buffer(0)) for o in d["osmBuildings"]]
        for kind, key, cl in (("pond", "ponds", CL_POND), ("lawn", "grassAreas", CL_LAWN)):
            for g in d[key]:
                q = P(g["coordinates"]).buffer(0)
                for nm, bp in blds:
                    dist = q.distance(bp)
                    if dist < cl - CTOL:
                        err.append(f"CLEARANCE {kind} {str(g['id'])[-14:]} is {dist:.2f} m from {nm} (needs >= {cl:g} m)")
        # footprints[] (2-D footprint list) must follow its building in the SAME frame convention as in the base
        for fid, rel in fp_relations(d).items():
            b0 = BASE_REL.get(fid)
            if b0 and not (rel & b0 if rel else False):
                err.append(f"footprints[] {fid} no longer matches its building ({sorted(b0)} -> {sorted(rel) or 'NONE'})")
        # ---- ranges that apply to GENERATION candidates only
        notes = []
        name = Path(f).name
        if name.startswith(("individual_", "generation_")) and name not in ("individual_0.json", "generation_0.json"):
            n0 = len(base["trees"]); need = n0 + math.ceil(GEN.get("tree_count_increase_frac", 0.10) * n0)
            if n0 == 0:      # no trees in the base: canopy cover x open space / mean crown area
                rr = 0.5 * sum(TC.get("new_tree_radius_range_m", [1.5, 4.0]))
                need = int(round(TC.get("canopy_cover_frac_if_missing", 0.10) * OPEN0 / (math.pi * rr * rr)))
            if len(d["trees"]) < need: err.append(f"trees {len(d['trees'])} < {need} (+10 %)")
            # tree heights follow the location zones
            bu = unary_union(bpolys); pwz = roads_union(d)
            nh = TC.get("near_building_distance_m", 10.0)
            bandN, bandP, bandO = TC.get("height_near_building_m", [8, 10]), TC.get("height_pathway_m", [8, 12]), TC.get("height_open_m", [12, 15])
            badh = 0
            for t in d["trees"]:
                pt = Point(t["position"]["x"], t["position"]["y"])
                lo, hi = bandN if bu.distance(pt) <= nh else (bandP if (pwz is not None and pwz.distance(pt) <= TC.get("pathway_zone_distance_m", 6.0) + 1e-9) else bandO)
                if not lo - 1e-6 <= t["height"] <= hi + 1e-6: badh += 1
            if badh and GEN.get("normalize_tree_heights", True): err.append(f"{badh} trees outside their height zone")
            # minimum building shift 0.5 x longest projected span (relaxed where physically impossible)
            relaxed = []
            for k, idx in base_groups.items():
                poly = gpoly(base, idx); L_ = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
                sh = float(np.hypot(*s0[idx[0]]))
                need_ = min(0.5 * L_, 0.98 * SHIFT * diam)
                if sh < need_ - 1e-6: relaxed.append(f"{base['buildings'][idx[0]]['id']} {sh:.0f}<{need_:.0f}")
            if relaxed: notes.append("min-shift relaxed: " + ", ".join(relaxed))
            if any(np.hypot(*s0[idx[0]]) < 1e-9 for idx in base_groups.values()): notes.append("a building did not move")
        print(f"{Path(f).name:<20} {'OK  ' if not err else 'FAIL'} shift={maxshift:5.1f} m  trees={len(d['trees']):3d} ponds={len(d['ponds'])} lawns={len(d['grassAreas'])}  {'; '.join(sorted(set(err)))[:300]}  {' | '.join(notes)}")
        fails += bool(err)
    sys.exit(1 if fails else 0)


main()