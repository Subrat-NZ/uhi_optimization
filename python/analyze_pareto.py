#!/usr/bin/env python3
"""
analyze_pareto.py  --  why are the best (Pareto) layouts better than the base case?   (post-processing, run it after the optimisation)

Reads what the optimisation already wrote (nothing is simulated again):

    output/top5.json  (or the Pareto front of output/objectives.csv)      which layouts to analyse
    run/<name>/<heat|cold>/temperature_humidity/postProcessing/utci_points.csv + utci.json      pedestrian-height results of every layout
    run/<name>/<season>/temperature_humidity/weatherData/target.json                            wind / sun / air of the hour
    input/<name>.json                                                                           the layouts (buildings, trees, ponds, lawns)

and compares every selected layout with the BASE case (individual_0 = base.json) for the heat hour and for the cold hour:

  1. HOW MUCH, WHERE      UTCI maps of base / layout / difference on a common grid, category-area table, hot spots and cold spots.
  2. WHY  (which driver)  The UTCI is a function of four things at the pedestrian: mean radiant temperature Tmrt, wind speed, air temperature Ta,
                          relative humidity RH.  The change of UTCI is split EXACTLY into the four (Shapley / order-averaged substitution
                          with the UTCI function itself, so the four add up to the change):  Tmrt (shade, sky view, hot / cool surfaces),
                          wind (ventilation), Ta (evaporative / shade cooling of the air), RH (humidity).  Tmrt is split further, by a regression on the
                          sunlit share, the sky view and Ta, into shade, sky view, air and "surfaces / long-wave".
  3. WAKES, VENTILATION   Calm (wake) regions = wind below `--wake-frac` x the reference wind: area, length, width, which building they trail, what
                          changed.  Mean wind ratio, calm share, fast channels.
  4. HOT / COLD SPOTS     Regions of the base case where the stress is worst (heat: UTCI >= the `--hot-pct` percentile and >= 26 C; cold:
                          <= the (100-`--cold-pct`) percentile and <= 9 C): were they resolved, reduced, still there?  New ones?  For every region the
                          drivers and the layout changes (trees, ponds, lawns, building shifts) next to / upwind of it.
  5. MITIGATION           Rule-based reading of the numbers: what worked, what is left, what to try next, and the trade-off between the hot and the cold hour.
                          The rules are listed in the report, so you can see why each sentence is there.

Output (default output/analysis/):  analysis_report.md  +  analysis_report.html (one file, pictures inside),  figures/*.png,  analysis_summary.json,
                                    tables/*.csv

    python3 python/analyze_pareto.py                                  # top5.json of ./output, base = individual_0, all seasons
    python3 python/analyze_pareto.py --top 3 --season heat            # the first 3, heat hour only
    python3 python/analyze_pareto.py --names offspring_10 individual_2 --base individual_0
    python3 python/analyze_pareto.py --root /data/UHI_optimization_template_271026_scotch --outdir /tmp/analysis
    python3 python/analyze_pareto.py --synthetic-demo ...             (see make_synthetic_demo.py: a made-up project to try the analyzer)

Limits (also printed in the report): one hour per season, a steady frozen-flow run, results at ONE height (1.5 m): wakes and ventilation are those of the
pedestrian level; a statement like "tree shade lowered UTCI" is an association between the layout change and the field change, not a controlled experiment.
"""
import argparse
import base64
import csv
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
from matplotlib.patches import Circle, Polygon as MPoly            # noqa: E402

from scipy import ndimage as ndi                                  # noqa: E402
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator   # noqa: E402
from scipy.spatial import cKDTree                                 # noqa: E402

import shapely                                                    # noqa: E402
from shapely.geometry import Point, Polygon                       # noqa: E402
from shapely.ops import unary_union                               # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

SEASONS = ("heat", "cold")
SEASON_TXT = {"heat": "heat-stress hour", "cold": "cold-stress hour"}
CATEGORIES = [(46.0, "extreme heat stress"), (38.0, "very strong heat stress"), (32.0, "strong heat stress"),
              (26.0, "moderate heat stress"), (9.0, "no thermal stress"), (0.0, "slight cold stress"),
              (-13.0, "moderate cold stress"), (-27.0, "strong cold stress"), (-40.0, "very strong cold stress")]
DRIVERS = ("Tmrt", "wind", "Ta", "RH")
DRIVER_COL = {"Tmrt": "Tmrt_C", "wind": "wind_ms", "Ta": "Tair_C", "RH": "RH_pct"}
DRIVER_LONG = {"Tmrt": "mean radiant temperature (shade / sky view / hot surfaces)", "wind": "wind speed (ventilation)",
               "Ta": "air temperature (evaporation, shade, mixing)", "RH": "relative humidity"}
COL = {"Tmrt": "#d9822b", "wind": "#2b7bba", "Ta": "#c0392b", "RH": "#4f9d69", "total": "#1c2830"}
INK, MUTED = "#1c2830", "#6d7780"
FIELDS = ("Tair_C", "RH_pct", "wind_ms", "Tmrt_C", "UTCI_C", "sunlit", "sky_open")

RAINBOW = LinearSegmentedColormap.from_list("rainbow_desaturated", [
    (0.000, (0.278431, 0.278431, 0.858824)), (0.143, (0.000000, 0.000000, 0.360784)), (0.285, (0.000000, 1.000000, 1.000000)),
    (0.429, (0.000000, 0.501961, 0.000000)), (0.571, (1.000000, 1.000000, 0.000000)), (0.714, (1.000000, 0.380392, 0.000000)),
    (0.857, (0.419608, 0.000000, 0.000000)), (1.000, (0.878431, 0.301961, 0.301961))])


def category(v):
    for lo, name in CATEGORIES:
        if v > lo:
            return name
    return "extreme cold stress"


def log(msg):
    print(f"[analyze] {msg}", flush=True)


# ============================================================================= reading the project
def _xy(seq):
    out = []
    for p in seq or []:
        if isinstance(p, dict):
            out.append((float(p["x"]), float(p["y"] if "y" in p else p["z"])))      # osm buildings store the plan y as z
        else:
            out.append((float(p[0]), float(p[1])))
    return out


def _poly(seq):
    pts = _xy(seq)
    if len(pts) < 3:
        return None
    g = Polygon(pts)
    if not g.is_valid:
        g = g.buffer(0)
    if g.is_empty:
        return None
    if g.geom_type != "Polygon":
        g = max(getattr(g, "geoms", []), key=lambda q: q.area)
    return g


class Layout:
    """site, buildings (id -> polygon, height), trees, ponds, lawns of one layout file"""

    def __init__(self, path):
        J = json.loads(Path(path).read_text())
        J = J.get("siteOnly_json", J)
        self.path = str(path)
        self.site = _poly((J.get("site") or {}).get("coordinates"))
        self.buildings, self.height = {}, {}
        for b in J.get("buildings") or []:
            vs = (b.get("dimensions") or {}).get("baseVertices") or (b.get("geometry") or {}).get("footprint")
            g = _poly(vs)
            if g is not None:
                self.buildings[b["id"]] = g
                self.height[b["id"]] = float(b.get("height") or (b.get("dimensions") or {}).get("height") or 0.0)
        osm = J.get("osmBuildings") or J.get("osm_buildings") or []
        for k, b in enumerate(osm):
            g = _poly((b.get("dimensions") or {}).get("baseVertices") or (b.get("geometry") or {}).get("footprint") or b.get("coordinates"))
            if g is not None:
                self.buildings[f"osm_{k}"] = g
                self.height[f"osm_{k}"] = float(b.get("height") or 10.0)
        self.trees = []
        for t in J.get("trees") or []:
            r = (t.get("dimensions") or {}).get("crownRadius") or 0.5 * float(t.get("crownDiameter") or 4.0)
            self.trees.append((str(t.get("id")), float(t["position"]["x"]), float(t["position"]["y"]), float(r), float(t.get("height") or 8.0)))
        self.ponds = [g for g in (_poly(p.get("coordinates")) for p in J.get("ponds") or []) if g is not None]
        self.lawns = [g for g in (_poly(p.get("coordinates")) for p in J.get("grassAreas") or []) if g is not None]
        self.pond_ids = [p.get("id") for p in J.get("ponds") or []]

    def building_union(self):
        return unary_union(list(self.buildings.values())) if self.buildings else None


def find_pp(run_dir, name, season):
    """(postProcessing folder, job folder) of one layout / season, or None"""
    job = Path(run_dir) / name / season
    for pp in (job / "temperature_humidity" / "postProcessing", job / "postProcessing"):
        if (pp / "utci_points.csv").is_file():
            return pp, job
    hits = sorted(job.glob("*/postProcessing/utci_points.csv"))
    if hits:
        return hits[0].parent, job
    return None


def read_points(csv_path, height=None):
    """utci_points.csv -> dict of arrays; one row per (x, y).  Duplicated (x, y) rows (several heights, or the same point written twice) are collapsed:
    the row whose z is closest to `height` is kept, else the rows are averaged.  Returns (points, note)"""
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    names = d.dtype.names
    out = {k: np.atleast_1d(np.asarray(d[k], float)) for k in ("x", "y", "z") + FIELDS if k in names}
    for k in FIELDS:
        out.setdefault(k, np.full(len(out["x"]), np.nan))
    n0 = len(out["x"])
    xy = np.round(np.column_stack([out["x"], out["y"]]), 3)
    _, first, inv, cnt = np.unique(xy, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    note = f"{n0} rows, {len(cnt)} distinct (x, y)"
    if len(cnt) < n0:
        z = out.get("z")
        zl = np.unique(np.round(z, 3)) if z is not None else np.array([])
        if z is not None and len(zl) > 1:
            h = float(height) if height is not None else float(zl[np.argmin(np.abs(zl - 1.5))])
            order = np.lexsort((np.abs(z - h), inv))                  # per (x, y): the row closest to the wanted height first
            pick = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
            out = {k: v[pick] for k, v in out.items()}
            note += f"; z levels {', '.join(f'{v:g}' for v in zl[:6])} m -> kept the rows nearest z = {h:g} m"
        else:
            avg = {}
            for k, v in out.items():
                sm = np.zeros(len(cnt))
                np.add.at(sm, inv, np.where(np.isfinite(v), v, 0.0))
                ct = np.zeros(len(cnt))
                np.add.at(ct, inv, np.isfinite(v).astype(float))
                avg[k] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
            out = avg
            note += "; duplicated points averaged"
    return out, note


def read_boundary(job, pp):
    """boundary of the hour (wind speed / direction, air, sun): weatherData/target.json of the job"""
    for p in (pp.parent / "weatherData" / "target.json", job / "weatherData" / "target.json", job / "temperature_humidity" / "weatherData" / "target.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text()).get("boundary", {})
            except Exception:
                pass
    return {}


def merge_near_points(pts, area):
    """Sample points that lie almost on top of each other (several points per CFD cell, or the same cell written with tiny offsets) make the
    spacing estimate collapse to a few cm; then every map keeps only the ground within ~2 x that distance of a point (a few % of the site) and
    all areas / means / spots are computed on that sliver.  When most points have a neighbour much closer than sqrt(area / N) (the smallest
    spacing the points can really have), every group of such points is replaced by one point: mean position, mean of every field.
    Returns (pts, note)"""
    n = len(pts["x"])
    if n < 6 or not area or area <= 0:
        return pts, ""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    xy = np.column_stack([pts["x"], pts["y"]])
    tree = cKDTree(xy)
    s0 = float(np.sqrt(area / n))
    if np.median(tree.query(xy, k=2)[0][:, 1]) >= 0.3 * s0:
        return pts, ""
    pairs = tree.query_pairs(0.3 * s0, output_type="ndarray")
    if len(pairs) == 0:
        return pts, ""
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    out = {}
    for key, v in pts.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        sm = np.bincount(lab, weights=np.where(ok, v, 0.0), minlength=k)
        ct = np.bincount(lab, weights=ok.astype(float), minlength=k)
        out[key] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
    return out, f"; {n} points lie in clusters (several within {0.3 * s0:.2f} m), merged to {k} locations"


class Case:
    """one layout in one season"""

    def __init__(self, name, season, run_dir, in_dir):
        self.name, self.season = name, season
        hit = find_pp(run_dir, name, season)
        if hit is None:
            raise FileNotFoundError(f"no utci_points.csv for {name} / {season} under {run_dir}")
        self.pp, self.job = hit
        self.summary = json.loads((self.pp / "utci.json").read_text()) if (self.pp / "utci.json").is_file() else {}
        self.pts, self.point_note = read_points(self.pp / "utci_points.csv", self.summary.get("height_m"))
        self.boundary = read_boundary(self.job, self.pp)
        lj = Path(in_dir) / f"{name}.json"
        if not lj.is_file():
            c = sorted((self.job / "momentum").glob("*.json"))
            lj = c[0] if c else None
        if lj is None:
            raise FileNotFoundError(f"no layout json for {name}")
        self.layout = Layout(lj)
        self.pts, merged_note = merge_near_points(self.pts, float(self.layout.site.area))
        if merged_note:
            self.point_note += merged_note
            print(f"[analyze] {name}/{season}: {self.point_note}", file=sys.stderr)
        xy = np.column_stack([self.pts["x"], self.pts["y"]])
        d = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.array([0.0])
        self.spacing = float(1.8 * np.median(d))
        if not self.spacing > 0:
            raise ValueError(f"{name}/{season}: the sample points of utci_points.csv have no spacing ({self.point_note}); check the file")

    def flow(self):
        """unit vector of the wind (towards where it blows), model frame"""
        f = self.boundary.get("wind_flow_unit_vector_xy_model")
        if f and len(f) == 2:
            n = math.hypot(f[0], f[1])
            if n > 1e-9:
                return np.array([f[0] / n, f[1] / n])
        wd = self.boundary.get("wind_from_deg_model")
        if wd is None:
            return None
        return np.array([-math.sin(math.radians(wd)), -math.cos(math.radians(wd))])


# ============================================================================= grid
class Grid:
    def __init__(self, site, cell):
        x0, y0, x1, y1 = site.bounds
        self.cell = float(cell)
        self.x = np.arange(x0 + 0.5 * cell, x1, cell)
        self.y = np.arange(y0 + 0.5 * cell, y1, cell)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.shape = self.X.shape
        self.extent = (x0, x0 + self.cell * len(self.x), y0, y0 + self.cell * len(self.y))
        self.area = self.cell ** 2
        self.site_mask = shapely.contains_xy(site, self.X, self.Y)


def rasterise(case, grid, gap_factor=2.2):
    """fields of one case on the grid (linear interpolation of the sample points; NaN outside the site, in buildings and farther than
    gap_factor x the point spacing from every sample)"""
    p = case.pts
    xy = np.column_stack([p["x"], p["y"]])
    ok = np.isfinite(p["UTCI_C"])
    xy = xy[ok]
    tree = cKDTree(xy)
    near = tree.query(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))[0].reshape(grid.shape)
    bu = case.layout.building_union()
    inb = shapely.contains_xy(bu, grid.X, grid.Y) if bu is not None else np.zeros(grid.shape, bool)
    valid = grid.site_mask & ~inb & (near <= gap_factor * case.spacing)
    F = {}
    nn_idx = tree.query(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))[1].reshape(grid.shape)
    for k in FIELDS:
        v = p[k][ok]
        if not np.isfinite(v).any():
            F[k] = np.full(grid.shape, np.nan)
            continue
        good = np.isfinite(v)
        try:
            lin = LinearNDInterpolator(xy[good], v[good])(grid.X, grid.Y)
        except Exception:
            lin = np.full(grid.shape, np.nan)
        nn = v[nn_idx]                                   # nearest sample of the same field (outside the convex hull of the samples)
        a = np.where(np.isfinite(lin), lin, nn)
        F[k] = np.where(valid, a, np.nan)
    F["valid"] = valid
    open_cells = int((grid.site_mask & ~inb).sum())
    if open_cells and valid.sum() < 0.5 * open_cells:
        print(f"[analyze] WARNING {case.name}/{case.season}: only {100 * valid.sum() / open_cells:.0f} % of the open ground is within {gap_factor} x "
              f"{case.spacing:.2f} m of a sample point - the sample spacing looks wrong ({case.point_note}); maps and areas will be incomplete",
              file=sys.stderr)
    return F


# ============================================================================= UTCI and its exact split
WIND_FACTOR = 1.0                  # same as compute_utci.py --wind-factor (multiplies the CFD wind before UTCI)


def utci_fn(Ta, tmrt, v, rh):
    from pythermalcomfort.models import utci
    Ta, tmrt, v, rh = (np.asarray(a, float) for a in (Ta, tmrt, v, rh))
    v = np.clip(v * WIND_FACTOR, 0.5, 17.0)
    tr = np.clip(tmrt, Ta - 30.0, Ta + 70.0)
    rh = np.clip(rh, 1.0, 100.0)
    res = utci(tdb=Ta, tr=tr, v=v, rh=rh, units="SI", limit_inputs=False)
    return np.asarray(getattr(res, "utci", res), float)


def shapley_split(B, C, mask):
    """B, C: field dicts of base / candidate on the grid; mask: cells valid in both.  Returns dict(contrib={driver: array (cells,)},
    u_base, u_cand (recomputed UTCI), total) with  sum(contrib) = u_cand - u_base  exactly (order-averaged substitution)."""
    idx = np.where(mask)
    cols = [DRIVER_COL[d] for d in DRIVERS]
    b = [B[c][idx] for c in cols]
    c = [C[c][idx] for c in cols]
    n = len(DRIVERS)
    cache = {}

    def val(sub):
        if sub not in cache:
            args = [c[i] if (sub >> i) & 1 else b[i] for i in range(n)]
            cache[sub] = utci_fn(Ta=args[2], tmrt=args[0], v=args[1], rh=args[3])
        return cache[sub]

    fact = [math.factorial(k) for k in range(n + 1)]
    contrib = {}
    for i, d in enumerate(DRIVERS):
        acc = np.zeros(len(idx[0]))
        for sub in range(1 << n):
            if (sub >> i) & 1:
                continue
            k = bin(sub).count("1")
            w = fact[k] * fact[n - k - 1] / fact[n]
            acc += w * (val(sub | (1 << i)) - val(sub))
        full = np.full(B["valid"].shape, np.nan)
        full[idx] = acc
        contrib[d] = full
    u0, u1 = np.full(B["valid"].shape, np.nan), np.full(B["valid"].shape, np.nan)
    u0[idx], u1[idx] = val(0), val((1 << n) - 1)
    return dict(contrib=contrib, u_base=u0, u_cand=u1)


def excess_split(F, region_mask, valid):
    """why is this region hotter / colder than the REST of the site in the same layout?  Region means vs the means of the rest of the open ground,
    split over Tmrt, wind, Ta, RH (same exact method).  Returns dict(contrib={driver: dC}, total, u_region, u_rest)"""
    rest = valid & ~region_mask
    if not region_mask.any() or not rest.any():
        return None
    mk = lambda m: {c: np.array([[float(np.nanmean(F[c][m]))]]) for c in DRIVER_COL.values()}      # noqa: E731
    Bm, Cm = mk(rest), mk(region_mask)
    ones = np.ones((1, 1), bool)
    Bm["valid"] = Cm["valid"] = ones
    sp = shapley_split(Bm, Cm, ones)
    return dict(contrib={d: float(sp["contrib"][d][0, 0]) for d in DRIVERS}, total=float(sp["u_cand"][0, 0] - sp["u_base"][0, 0]),
                u_region=float(sp["u_cand"][0, 0]), u_rest=float(sp["u_base"][0, 0]))


def tmrt_attribution(B, C, mask):
    """split the change of Tmrt into shade (sunlit share), sky view, air temperature and the rest (surfaces / long-wave): linear regression
    Tmrt ~ 1 + sunlit + sky_open + Ta on the cells of both cases.  Returns dict(coef, r2, parts={name: array})"""
    idx = np.where(mask)
    A, y = [], []
    for F in (B, C):
        A.append(np.column_stack([np.ones(len(idx[0])), F["sunlit"][idx], F["sky_open"][idx], F["Tair_C"][idx]]))
        y.append(F["Tmrt_C"][idx])
    A, y = np.vstack(A), np.concatenate(y)
    okk = np.isfinite(A).all(1) & np.isfinite(y)
    if okk.sum() < 20 or np.ptp(A[okk, 1]) < 1e-6 and np.ptp(A[okk, 2]) < 1e-6:
        return None
    coef, *_ = np.linalg.lstsq(A[okk], y[okk], rcond=None)
    res = y[okk] - A[okk] @ coef
    r2 = 1.0 - float(res.var() / max(y[okk].var(), 1e-9))
    parts = {}
    for name, j, key in (("shade (sunlit share)", 1, "sunlit"), ("sky view", 2, "sky_open"), ("air temperature", 3, "Tair_C")):
        full = np.full(B["valid"].shape, np.nan)
        full[idx] = coef[j] * (C[key][idx] - B[key][idx])
        parts[name] = full
    dt = np.full(B["valid"].shape, np.nan)
    dt[idx] = C["Tmrt_C"][idx] - B["Tmrt_C"][idx]
    rest = dt.copy()
    for v in parts.values():
        rest = rest - v
    parts["surfaces / long-wave (rest)"] = rest
    return dict(coef=[float(c) for c in coef], r2=r2, parts=parts, dTmrt=dt)


# ============================================================================= regions
def label_mask(mask, grid, min_area):
    """connected regions (8-neighbour, one closing step so a 1-cell gap does not split a region) larger than min_area m2:
    list of dict(mask, area, centroid, ...) sorted by area"""
    if not mask.any():
        return []
    m = ndi.binary_closing(mask, structure=np.ones((3, 3), bool)) & (mask | ndi.binary_dilation(mask, iterations=1))
    lab, n = ndi.label(m, structure=np.ones((3, 3), int))
    out = []
    for i in range(1, n + 1):
        mm = (lab == i) & mask
        a = float(mm.sum() * grid.area)
        if a < min_area:
            continue
        out.append(dict(mask=mm, area=a, centroid=(float(grid.X[mm].mean()), float(grid.Y[mm].mean()))))
    out.sort(key=lambda r: -r["area"])
    return out


def compass(dx, dy):
    """name of the direction of (dx, dy) in the model frame (+y north, +x east)"""
    if math.hypot(dx, dy) < 1e-9:
        return "centre"
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return names[int(round(math.degrees(math.atan2(dy, dx)) / 45.0)) % 8]


def region_extent(mask, grid, flow):
    """length along the flow and width across it (m) of a region, from its cells"""
    xs, ys = grid.X[mask], grid.Y[mask]
    if flow is None:
        return float(np.ptp(xs)), float(np.ptp(ys))
    u = xs * flow[0] + ys * flow[1]
    v = -xs * flow[1] + ys * flow[0]
    return float(np.ptp(u) + grid.cell), float(np.ptp(v) + grid.cell)


def nearest_building(layout, pt, flow=None):
    """(id, distance) of the building nearest to pt; with flow: the nearest building UPWIND of pt (the one whose wake it may be), else the nearest"""
    best = None
    P = Point(pt)
    for bid, g in layout.buildings.items():
        d = g.distance(P)
        if flow is not None:
            c = g.centroid
            if (pt[0] - c.x) * flow[0] + (pt[1] - c.y) * flow[1] <= 0:
                continue
        if best is None or d < best[1]:
            best = (bid, d)
    if best is None and flow is not None:
        return nearest_building(layout, pt, None)
    return best


# ============================================================================= layout differences
def layout_diff(Lb, Lc):
    """what changed between two layouts: building shifts, trees added / removed, canopy, ponds, lawns"""
    out = {"buildings": []}
    for bid, g in Lc.buildings.items():
        if bid in Lb.buildings and not bid.startswith("osm_"):
            d = (g.centroid.x - Lb.buildings[bid].centroid.x, g.centroid.y - Lb.buildings[bid].centroid.y)
            out["buildings"].append(dict(id=bid, dx=round(d[0], 2), dy=round(d[1], 2), shift=round(math.hypot(*d), 2)))
    sh = [b["shift"] for b in out["buildings"]]
    out["building_shift_mean"] = float(np.mean(sh)) if sh else 0.0
    out["building_shift_max"] = float(np.max(sh)) if sh else 0.0

    def key(t):
        return (round(t[1], 2), round(t[2], 2), round(t[3], 2))
    kb, kc = {key(t) for t in Lb.trees}, {key(t) for t in Lc.trees}
    out["trees_base"], out["trees_cand"] = len(Lb.trees), len(Lc.trees)
    out["trees_added"] = [t for t in Lc.trees if key(t) not in kb]
    out["trees_removed"] = [t for t in Lb.trees if key(t) not in kc]
    out["canopy_base"] = float(sum(math.pi * t[3] ** 2 for t in Lb.trees))
    out["canopy_cand"] = float(sum(math.pi * t[3] ** 2 for t in Lc.trees))
    out["mean_height_base"] = float(np.mean([t[4] for t in Lb.trees])) if Lb.trees else 0.0
    out["mean_height_cand"] = float(np.mean([t[4] for t in Lc.trees])) if Lc.trees else 0.0
    out["pond_area_base"] = float(sum(p.area for p in Lb.ponds))
    out["pond_area_cand"] = float(sum(p.area for p in Lc.ponds))
    out["lawn_area_base"] = float(sum(p.area for p in Lb.lawns))
    out["lawn_area_cand"] = float(sum(p.area for p in Lc.lawns))
    out["ponds_cand"], out["ponds_base"] = len(Lc.ponds), len(Lb.ponds)
    return out


def region_layout_context(region_mask, grid, flow, Lb, Lc, diff, near_m=25.0, upwind_len=70.0):
    """layout changes around one region: new / removed trees within near_m of it, trees added in a corridor upwind of it, ponds / lawns in the
    candidate near it (and in the base), buildings that moved near it"""
    from shapely.geometry import MultiPoint
    pts = np.column_stack([grid.X[region_mask], grid.Y[region_mask]])
    reg = MultiPoint(pts[::max(1, len(pts) // 400)]).buffer(0.75 * grid.cell)
    ctx = {}
    near = reg.buffer(near_m)
    shapely.prepare(near)
    added = [t for t in diff["trees_added"] if near.contains(Point(t[1], t[2]))]
    removed = [t for t in diff["trees_removed"] if near.contains(Point(t[1], t[2]))]
    ctx["trees_added_near"], ctx["trees_removed_near"] = len(added), len(removed)
    ctx["canopy_added_near_m2"] = float(sum(math.pi * t[3] ** 2 for t in added))
    ctx["canopy_removed_near_m2"] = float(sum(math.pi * t[3] ** 2 for t in removed))
    if flow is not None:
        c = reg.centroid
        u = lambda x, y: (x - c.x) * flow[0] + (y - c.y) * flow[1]            # noqa: E731
        v = lambda x, y: -(x - c.x) * flow[1] + (y - c.y) * flow[0]           # noqa: E731
        half = 0.5 * region_extent(region_mask, grid, flow)[1] + 10.0
        up_added = [t for t in diff["trees_added"] if -upwind_len <= u(t[1], t[2]) <= 0 and abs(v(t[1], t[2])) <= half]
        up_removed = [t for t in diff["trees_removed"] if -upwind_len <= u(t[1], t[2]) <= 0 and abs(v(t[1], t[2])) <= half]
        ctx["trees_added_upwind"], ctx["trees_removed_upwind"] = len(up_added), len(up_removed)

        def pond_upwind(L):
            n = 0.0
            for p in L.ponds:
                q = p.centroid
                if -upwind_len * 1.5 <= u(q.x, q.y) <= 0 and abs(v(q.x, q.y)) <= half + 15.0:
                    n += p.area
            return n
        ctx["pond_area_upwind_base"], ctx["pond_area_upwind_cand"] = pond_upwind(Lb), pond_upwind(Lc)
    ctx["pond_area_near_base"] = float(sum(p.intersection(near).area for p in Lb.ponds))
    ctx["pond_area_near_cand"] = float(sum(p.intersection(near).area for p in Lc.ponds))
    ctx["lawn_area_near_base"] = float(sum(p.intersection(near).area for p in Lb.lawns))
    ctx["lawn_area_near_cand"] = float(sum(p.intersection(near).area for p in Lc.lawns))
    moved = [b for b in diff["buildings"] if b["shift"] > 0.5 and Lc.buildings[b["id"]].distance(reg) <= near_m + 15.0]
    ctx["buildings_moved_near"] = [(b["id"], b["shift"]) for b in moved]
    return ctx


# ============================================================================= the analysis of one candidate in one season
def stress_areas(U, valid, grid):
    """area (m2) per UTCI category"""
    out = {}
    u = U[valid]
    edges = sorted(c[0] for c in CATEGORIES)
    names = {c[0]: c[1] for c in CATEGORIES}
    lo = -1e9
    for e in edges + [1e9]:
        nm = "extreme cold stress" if lo == -1e9 else names[lo]
        out[nm] = float(((u > lo) & (u <= e)).sum() * grid.area)
        lo = e
    return out


def field_stats(F, valid, grid):
    s = {}
    for k in FIELDS:
        v = F[k][valid]
        s[k] = float(np.nanmean(v)) if v.size else float("nan")
    s["utci_p10"], s["utci_p90"] = (float(np.nanpercentile(F["UTCI_C"][valid], q)) for q in (10, 90))
    s["area_m2"] = float(valid.sum() * grid.area)
    return s


def wake_regions(F, valid, grid, flow, u_ref, frac, min_area, layout):
    """calm regions: wind < frac * u_ref"""
    calm = valid & (F["wind_ms"] < frac * u_ref)
    regs = label_mask(calm, grid, min_area)
    out = []
    for k, r in enumerate(regs, 1):
        m = r["mask"]
        ln, wd = region_extent(m, grid, flow)
        nb = nearest_building(layout, r["centroid"], flow)
        out.append(dict(id=f"W{k}", area=r["area"], centroid=r["centroid"], length_m=ln, width_m=wd, mask=m,
                        wind_mean=float(np.nanmean(F["wind_ms"][m])), utci_mean=float(np.nanmean(F["UTCI_C"][m])),
                        building=None if nb is None else nb[0], building_dist=None if nb is None else float(nb[1])))
    return calm, out


def match_regions(regs_a, regs_b, min_overlap=0.2):
    """for every region of A: (index of the best overlapping region of B or None, overlap share of the smaller of the two)"""
    out = []
    for r in regs_a:
        best, bo = None, 0.0
        for j, s in enumerate(regs_b):
            ov = float((r["mask"] & s["mask"]).sum()) / max(min(float(r["mask"].sum()), float(s["mask"].sum())), 1.0)
            if ov > bo:
                best, bo = j, ov
        out.append((best if bo >= min_overlap else None, bo))
    return out


def match_wakes(wa, wb, max_shift_m=45.0):
    """base wake -> layout wake: first the wake that trails the same building (nearest centroid within max_shift_m, since a moved building drags its wake
    along), else the one it overlaps.  Returns (list of index or None per wake of wa, set of indices of wb that were matched)"""
    res, used = [], set()
    ov = match_regions(wa, wb)
    for k, w in enumerate(wa):
        cand = [j for j, v in enumerate(wb) if j not in used and v["building"] is not None and v["building"] == w["building"]
                and math.hypot(v["centroid"][0] - w["centroid"][0], v["centroid"][1] - w["centroid"][1]) <= max_shift_m]
        if cand:
            j = min(cand, key=lambda j: math.hypot(wb[j]["centroid"][0] - w["centroid"][0], wb[j]["centroid"][1] - w["centroid"][1]))
        else:
            j = ov[k][0]
            if j in used:
                j = None
        res.append(j)
        if j is not None:
            used.add(j)
    return res, used


UNIT = {"Tmrt": "°C", "wind": "m/s", "Ta": "°C", "RH": "%"}
NDEC = {"Tmrt": 1, "wind": 2, "Ta": 1, "RH": 1}
ORDER = ("Ta", "Tmrt", "RH", "wind")                    # order of the columns of the value tables


def zone_vals(mask, B, C, both, sp, grid):
    """area-averaged inputs of a zone in the base and in the layout, their change, and the part of the UTCI change each one caused (exact split):
    dict(area, n, UTCI=dict(b, c, d), Ta=dict(b, c, d, eff), Tmrt=..., RH=..., wind=...)"""
    m = mask & both
    if not m.any():
        return None
    v = dict(area=float(mask.sum() * grid.area), n=int(m.sum()))
    ub, uc = float(np.nanmean(B["UTCI_C"][m])), float(np.nanmean(C["UTCI_C"][m]))
    v["UTCI"] = dict(b=ub, c=uc, d=uc - ub)
    for dr in DRIVERS:
        col = DRIVER_COL[dr]
        b, c = float(np.nanmean(B[col][m])), float(np.nanmean(C[col][m]))
        v[dr] = dict(b=b, c=c, d=c - b, eff=float(np.nanmean(sp["contrib"][dr][m])))
    return v


def vtxt(v, dr, eff=True, nl=False):
    """'36.1 → 35.9 (−0.2) → −0.27' : base → layout (change) → effect on UTCI (°C)"""
    n = NDEC[dr]
    x = v[dr]
    s = f"{x['b']:.{n}f} → {x['c']:.{n}f} ({x['d']:+.{n}f})"
    if eff:
        s += ("\n" if nl else " → ") + f"{x['eff']:+.2f}"
    return s


def ctxt(v, dr):
    """compact label text for drawing on a map"""
    n = NDEC[dr] if dr != "RH" else 0
    return f"{dr if dr != 'wind' else 'U'} {v[dr]['b']:.{n}f}→{v[dr]['c']:.{n}f}"


def inputs_sentence(v, sgn=None):
    """'Ta 36.1 → 35.9 °C (−0.2) lowers UTCI by 0.27 °C; Tmrt ...' for the four inputs in order of their effect"""
    items = sorted(DRIVERS, key=lambda d: -abs(v[d]["eff"]))
    parts = []
    for d in items:
        x = v[d]
        n = NDEC[d]
        verb = "raises" if x["eff"] > 0.005 else ("lowers" if x["eff"] < -0.005 else "no effect on")
        tail = f" {verb} UTCI by {abs(x['eff']):.2f} °C" if abs(x["eff"]) >= 0.005 else " no effect on UTCI"
        parts.append(f"{d} {x['b']:.{n}f} → {x['c']:.{n}f} {UNIT[d]} ({x['d']:+.{n}f}){tail}")
    return "; ".join(parts)


class SeasonResult:
    pass


def analyse_pair(base, cand, grid, a):
    """everything for one candidate / one season"""
    R = SeasonResult()
    R.season, R.name = cand.season, cand.name
    heat = cand.season == "heat"
    sgn = -1.0 if heat else 1.0                       # benefit = sgn * dUTCI
    B, C = rasterise(base, grid), rasterise(cand, grid)
    both = B["valid"] & C["valid"]
    if not B["valid"].any() or not C["valid"].any() or not both.any():
        raise ValueError(f"{cand.name}/{cand.season}: no open ground on the analysis grid (valid cells base {int(B['valid'].sum())}, layout {int(C['valid'].sum())}, both {int(both.sum())}); "
                         f"grid {grid.shape[1]} x {grid.shape[0]} cells of {grid.cell:g} m, site bounds {tuple(round(v, 1) for v in cand.layout.site.bounds) if cand.layout.site is not None else None}, "
                         f"sample points x {np.nanmin(cand.pts['x']):.1f}..{np.nanmax(cand.pts['x']):.1f}, y {np.nanmin(cand.pts['y']):.1f}..{np.nanmax(cand.pts['y']):.1f}")
    R.B, R.C, R.both, R.sgn = B, C, both, sgn
    R.flow = cand.flow() if cand.flow() is not None else base.flow()
    u10 = cand.boundary.get("wind_speed_10m_ms") or base.boundary.get("wind_speed_10m_ms")
    R.u10 = float(u10) if u10 else float("nan")
    # reference wind = free pedestrian-level wind of the BASE case (90th percentile of its 1.5 m field), same value for both layouts
    R.u_ref = float(a.u_ref) if a.u_ref else float(np.nanpercentile(B["wind_ms"][B["valid"]], 90))
    R.stats_b, R.stats_c = field_stats(B, B["valid"], grid), field_stats(C, C["valid"], grid)
    R.stats_b_both, R.stats_c_both = field_stats(B, both, grid), field_stats(C, both, grid)
    R.cat_b, R.cat_c = stress_areas(B["UTCI_C"], B["valid"], grid), stress_areas(C["UTCI_C"], C["valid"], grid)
    R.obj_b = base.summary.get("utci_C")
    R.obj_c = cand.summary.get("utci_C")
    R.diff = layout_diff(base.layout, cand.layout)

    # ---- exact split of the UTCI change
    sp = shapley_split(B, C, both)
    R.split = sp
    tot = sp["u_cand"] - sp["u_base"]
    R.dU = tot
    R.contrib_mean = {d: float(np.nanmean(sp["contrib"][d][both])) for d in DRIVERS}
    R.dU_mean = float(np.nanmean(tot[both]))
    R.split_err = float(np.nanmax(np.abs(sum(sp["contrib"][d] for d in DRIVERS) - tot)[both])) if both.any() else 0.0
    R.csv_err = float(np.nanmean(np.abs(sp["u_base"][both] - B["UTCI_C"][both])))
    R.tmrt = tmrt_attribution(B, C, both)
    if R.tmrt is not None:
        R.tmrt_parts_mean = {k: float(np.nanmean(v[both])) for k, v in R.tmrt["parts"].items()}
        R.dTmrt_mean = float(np.nanmean(R.tmrt["dTmrt"][both]))

    R.site_vals = zone_vals(both, B, C, both, sp, grid)

    # ---- hot / cold spots of the base case
    ub = B["UTCI_C"]
    if heat:
        thr = max(float(np.nanpercentile(ub[B["valid"]], a.hot_pct)), 26.0)
        spot_b = B["valid"] & (ub >= thr)
        spot_c = C["valid"] & (C["UTCI_C"] >= thr)
    else:
        thr = min(float(np.nanpercentile(ub[B["valid"]], 100 - a.cold_pct)), 9.0)
        spot_b = B["valid"] & (ub <= thr)
        spot_c = C["valid"] & (C["UTCI_C"] <= thr)
    R.thr = thr
    R.spot_b, R.spot_c = spot_b, spot_c
    R.spot_area_b, R.spot_area_c = float(spot_b.sum() * grid.area), float(spot_c.sum() * grid.area)
    regs = label_mask(spot_b, grid, a.min_region_m2)
    R.regions = []
    for k, r in enumerate(regs, 1):
        m = r["mask"] & both
        if m.sum() < 3:
            continue
        info = dict(id=("H" if heat else "C") + str(k), area=r["area"], centroid=r["centroid"], mask=r["mask"])
        info["u_base"] = float(np.nanmean(B["UTCI_C"][m]))
        info["u_cand"] = float(np.nanmean(C["UTCI_C"][m]))
        info["d_u"] = info["u_cand"] - info["u_base"]
        info["benefit"] = sgn * info["d_u"]
        if heat:
            still = (C["UTCI_C"][m] >= thr)
        else:
            still = (C["UTCI_C"][m] <= thr)
        info["resolved_share"] = float(1.0 - still.mean())
        info["contrib"] = {d: float(np.nanmean(sp["contrib"][d][m])) for d in DRIVERS}
        for key, lab in (("Tair_C", "Ta"), ("RH_pct", "RH"), ("wind_ms", "wind"), ("Tmrt_C", "Tmrt"), ("sunlit", "sunlit"), ("sky_open", "sky_open")):
            info[lab + "_b"], info[lab + "_c"] = float(np.nanmean(B[key][m])), float(np.nanmean(C[key][m]))
        if R.tmrt is not None:
            info["tmrt_parts"] = {k2: float(np.nanmean(v[m])) for k2, v in R.tmrt["parts"].items()}
        dom = max(DRIVERS, key=lambda d: sgn * info["contrib"][d])
        info["dominant"] = dom
        if info["benefit"] > 0.3 and info["resolved_share"] >= 0.7:
            info["status"] = "resolved"
        elif info["benefit"] > 0.3 and info["resolved_share"] >= 0.3:
            info["status"] = "partly resolved"
        elif info["benefit"] > 0.3:
            info["status"] = "reduced"
        elif info["benefit"] < -0.3:
            info["status"] = "worse"
        else:
            info["status"] = "unchanged"
        info["context"] = region_layout_context(r["mask"], grid, R.flow, base.layout, cand.layout, R.diff)
        nb = nearest_building(cand.layout, r["centroid"], R.flow)
        info["near_building"] = None if nb is None else nb[0]
        info["near_building_dist"] = None if nb is None else float(nb[1])
        info["cause_b"], info["cause_c"] = excess_split(B, r["mask"] & B["valid"], B["valid"]), excess_split(C, r["mask"] & C["valid"], C["valid"])
        info["wind_ratio_b"], info["wind_ratio_c"] = info["wind_b"] / R.u_ref, info["wind_c"] / R.u_ref
        info["vals"] = zone_vals(r["mask"], B, C, both, sp, grid)
        R.regions.append(info)
    new_mask = spot_c & ~spot_b & both & (sgn * tot < -0.3)          # stressed in the layout, not in the base, and clearly worse than the base
    new_regs = label_mask(new_mask, grid, a.min_region_m2)
    R.new_regions = []
    for k, r in enumerate(new_regs, 1):
        m = r["mask"]
        info = dict(id=("nH" if heat else "nC") + str(k), area=r["area"], centroid=r["centroid"], mask=m)
        info["u_cand"] = float(np.nanmean(C["UTCI_C"][m]))
        for key, lab in (("Tair_C", "Ta"), ("wind_ms", "wind"), ("Tmrt_C", "Tmrt"), ("sunlit", "sunlit")):
            info[lab + "_c"] = float(np.nanmean(C[key][m]))
        b2 = (m & both)
        if b2.any():
            info["u_base"] = float(np.nanmean(B["UTCI_C"][b2]))
            info["contrib"] = {d: float(np.nanmean(sp["contrib"][d][b2])) for d in DRIVERS}
            info["dominant"] = max(DRIVERS, key=lambda d: -sgn * info["contrib"][d])        # the driver that made it worse
        info["context"] = region_layout_context(m, grid, R.flow, base.layout, cand.layout, R.diff)
        nb = nearest_building(cand.layout, r["centroid"], R.flow)
        info["near_building"] = None if nb is None else nb[0]
        info["cause_c"] = excess_split(C, m & C["valid"], C["valid"])
        info["sunlit_c"], info["sky_open_c"] = float(np.nanmean(C["sunlit"][m])), float(np.nanmean(C["sky_open"][m]))
        info["RH_c"] = float(np.nanmean(C["RH_pct"][m]))
        info["vals"] = zone_vals(m, B, C, both, sp, grid)
        R.new_regions.append(info)

    # ---- wakes / ventilation
    calm_b, wb = wake_regions(B, B["valid"], grid, R.flow, R.u_ref, a.wake_frac, a.min_region_m2, base.layout)
    calm_c, wc = wake_regions(C, C["valid"], grid, R.flow, R.u_ref, a.wake_frac, a.min_region_m2, cand.layout)
    R.calm_b, R.calm_c, R.wakes_b, R.wakes_c = calm_b, calm_c, wb, wc
    R.vent = dict(
        wind_ratio_b=R.stats_b["wind_ms"] / R.u_ref, wind_ratio_c=R.stats_c["wind_ms"] / R.u_ref,
        calm_area_b=float(calm_b.sum() * grid.area), calm_area_c=float(calm_c.sum() * grid.area),
        calm_share_b=float(calm_b.sum() / max(B["valid"].sum(), 1)), calm_share_c=float(calm_c.sum() / max(C["valid"].sum(), 1)),
        fast_share_b=float(((B["wind_ms"] > 1.3 * R.u_ref) & B["valid"]).sum() / max(B["valid"].sum(), 1)),
        fast_share_c=float(((C["wind_ms"] > 1.3 * R.u_ref) & C["valid"]).sum() / max(C["valid"].sum(), 1)),
        longest_wake_b=max([w["length_m"] for w in wb], default=0.0), longest_wake_c=max([w["length_m"] for w in wc], default=0.0),
        wind_ratio_in_spots_b=float(np.nanmean(B["wind_ms"][spot_b]) / R.u_ref) if spot_b.any() else float("nan"),
        wind_ratio_in_spots_c=float(np.nanmean(C["wind_ms"][spot_c]) / R.u_ref) if spot_c.any() else float("nan"))
    mt, used = match_wakes(wb, wc)
    for w, j in zip(wb, mt):
        w["fate"] = "gone" if j is None else ("smaller" if wc[j]["area"] < 0.8 * w["area"] else ("larger" if wc[j]["area"] > 1.25 * w["area"] else "similar"))
        w["cand_match"] = None if j is None else wc[j]["id"]
        w["cand_area"] = 0.0 if j is None else wc[j]["area"]
        m = w["mask"] & both
        if m.any():
            w["d_wind"] = float(np.nanmean(C["wind_ms"][m] - B["wind_ms"][m]))
            w["d_u"] = float(np.nanmean(C["UTCI_C"][m] - B["UTCI_C"][m]))
            w["contrib"] = {d: float(np.nanmean(sp["contrib"][d][m])) for d in DRIVERS}
        w["context"] = region_layout_context(w["mask"], grid, R.flow, base.layout, cand.layout, R.diff)
        w["vals"] = zone_vals(w["mask"], B, C, both, sp, grid)
    for j, w in enumerate(wc):
        w["is_new"] = j not in used
        w["vals"] = zone_vals(w["mask"], B, C, both, sp, grid)
    # ---- fast channels
    return R


# ============================================================================= drawing
def _footprint_patches(ax, layout, tree_alpha=1.0, faces=True, tree_lw=0.5):
    if layout.site is not None:
        ax.plot(*layout.site.exterior.xy, color=INK, lw=1.0)
    for g in layout.lawns:
        ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#d8ecd0" if faces else "none", ec="#8fb98a", lw=0.5, alpha=0.8, zorder=3))
    for g in layout.ponds:
        ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#9fcbff" if faces else "none", ec="#2b6cb0", lw=0.9, alpha=0.85 if faces else 1.0, zorder=4))
    for bid, g in layout.buildings.items():
        ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#8a8f94", ec=INK, lw=0.7, zorder=6))
    for t in layout.trees:
        ax.add_patch(Circle((t[1], t[2]), t[3], fc="none", ec="#2f7d32", lw=tree_lw, alpha=tree_alpha, zorder=5))


def _setup(ax, grid, title, layout=None, small=False):
    ax.set_aspect("equal")
    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_title(title, fontsize=9 if small else 10, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#c8ced3")
    if layout is not None:
        _footprint_patches(ax, layout)


def _flow_arrow(ax, grid, flow, label=True):
    if flow is None:
        return
    x0, x1, y0, y1 = grid.extent
    L = 0.12 * (x1 - x0)
    cx, cy = x0 + 0.1 * (x1 - x0), y1 - 0.07 * (y1 - y0)
    ax.annotate("", xy=(cx + L * flow[0], cy + L * flow[1]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#222"), zorder=20)
    if label:
        ax.text(cx, cy - 0.025 * (y1 - y0), "wind", fontsize=7, color="#222", ha="left", va="top", zorder=20)


def _nice_levels(vmin, vmax, n=12):
    span = max(vmax - vmin, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    lo, hi = math.floor(vmin / step + 1e-9) * step, math.ceil(vmax / step - 1e-9) * step
    return np.arange(lo, hi + 0.5 * step, step)


def _field(ax, grid, Z, vmin, vmax, cmap, levels=None, lines=None, fill_levels=12):
    Zm = np.ma.masked_invalid(Z)
    lv = _nice_levels(vmin, vmax, fill_levels)
    cf = ax.contourf(grid.X, grid.Y, np.clip(Zm, lv[0], lv[-1]), levels=lv, cmap=cmap, extend="neither", zorder=1)
    if lines is not None and len(lines):
        try:
            cs = ax.contour(grid.X, grid.Y, Zm, levels=lines, colors="#222", linewidths=0.5, alpha=0.6, zorder=2)
            ax.clabel(cs, fontsize=6, fmt="%g")
        except Exception:
            pass
    return cf


def _outline(ax, grid, mask, **kw):
    if mask is None or not mask.any():
        return
    try:
        ax.contour(grid.X, grid.Y, mask.astype(float), levels=[0.5], **kw)
    except Exception:
        pass


def _cbar(fig, cf, ax, label):
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return cb


def _label_regions(ax, regs, color="#111", prefix=None, fs=7):
    for r in regs:
        ax.text(r["centroid"][0], r["centroid"][1], r["id"], fontsize=fs, ha="center", va="center", color=color, weight="bold", zorder=25,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def _label_zones(ax, zones, fs=5.3, color="#111"):
    """write the area-averaged UTCI, Ta, Tmrt, RH and wind (base → layout) of every zone on the map; boxes that would overlap are pushed apart
    and joined to their zone by a thin line"""
    anns = []
    for z in zones:
        v = z.get("vals")
        if v is None:
            continue
        txt = f"{z['id']}\nUTCI {v['UTCI']['b']:.1f}→{v['UTCI']['c']:.1f}\n" + "\n".join(ctxt(v, d) for d in ORDER)
        an = ax.annotate(txt, xy=z["centroid"], xytext=z["centroid"], textcoords="data", fontsize=fs, ha="center", va="center", color=color, zorder=25, linespacing=1.05,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", lw=0.4, alpha=0.9), arrowprops=dict(arrowstyle="-", lw=0.5, color="#444", shrinkA=0, shrinkB=0))
        anns.append((an, z["centroid"]))
    if len(anns) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    axbb = ax.get_window_extent(rend)
    pos = [np.array(ax.transData.transform(c), float) for _, c in anns]            # display coordinates of the label centres
    size = []
    for an, _ in anns:
        bb = matplotlib.text.Text.get_window_extent(an, rend)
        size.append((bb.width + 6.0, bb.height + 6.0))
    for _ in range(80):
        moved = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
                ox = 0.5 * (size[i][0] + size[j][0]) - abs(dx)
                oy = 0.5 * (size[i][1] + size[j][1]) - abs(dy)
                if ox > 0 and oy > 0:
                    moved = True
                    if oy <= ox:
                        sgn_ = 1.0 if dy >= 0 else -1.0
                        pos[i][1] -= sgn_ * 0.5 * (oy + 1.0)
                        pos[j][1] += sgn_ * 0.5 * (oy + 1.0)
                    else:
                        sgn_ = 1.0 if dx >= 0 else -1.0
                        pos[i][0] -= sgn_ * 0.5 * (ox + 1.0)
                        pos[j][0] += sgn_ * 0.5 * (ox + 1.0)
        for i in range(len(anns)):                                               # stay inside the axes
            pos[i][0] = float(np.clip(pos[i][0], axbb.x0 + 0.5 * size[i][0], axbb.x1 - 0.5 * size[i][0]))
            pos[i][1] = float(np.clip(pos[i][1], axbb.y0 + 0.5 * size[i][1], axbb.y1 - 0.5 * size[i][1]))
        if not moved:
            break
    for (an, c), pp in zip(anns, pos):
        an.xyann = tuple(inv.transform(pp))


def fig_maps(R, base, cand, grid, path):
    """2 x 4 panels: UTCI base / layout / change / spots, and the four drivers"""
    heat = R.season == "heat"
    ub, uc = R.B["UTCI_C"], R.C["UTCI_C"]
    allv = np.concatenate([ub[R.B["valid"]], uc[R.C["valid"]]])
    vmin, vmax = float(np.floor(np.nanpercentile(allv, 1) - 0.5)), float(np.ceil(np.nanpercentile(allv, 99) + 0.5))
    if vmax - vmin < 4:
        vmax = vmin + 4
    cat_lines = [c[0] for c in CATEGORIES if vmin < c[0] < vmax]
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.6))
    # row 1
    cf = _field(axs[0, 0], grid, ub, vmin, vmax, RAINBOW, lines=cat_lines)
    _setup(axs[0, 0], grid, f"UTCI base  ({base.name})  mean {R.stats_b['UTCI_C']:.1f} °C", base.layout)
    _cbar(fig, cf, axs[0, 0], "UTCI °C")
    cf = _field(axs[0, 1], grid, uc, vmin, vmax, RAINBOW, lines=cat_lines)
    _setup(axs[0, 1], grid, f"UTCI  {cand.name}  mean {R.stats_c['UTCI_C']:.1f} °C", cand.layout)
    _cbar(fig, cf, axs[0, 1], "UTCI °C")
    d = np.where(R.both, R.dU, np.nan)
    lim = max(0.5, float(np.nanpercentile(np.abs(d[R.both]), 98)))
    cold = R.season != "heat"                      # heat: a rise of UTCI is red; cold: a rise of UTCI is blue (a fall is red)
    DIFF_CMAP = "RdBu" if cold else "RdBu_r"
    cf = _field(axs[0, 2], grid, d, -lim, lim, DIFF_CMAP, lines=None)
    _setup(axs[0, 2], grid, "change of UTCI (layout − base)\n" + ("blue = warmer, red = cooler" if cold else "blue = cooler, red = warmer"), cand.layout)
    _outline(axs[0, 2], grid, R.spot_b, colors="k", linewidths=1.2)
    _outline(axs[0, 2], grid, R.spot_c, colors="k", linewidths=1.2, linestyles="--")
    _cbar(fig, cf, axs[0, 2], "ΔUTCI °C")
    # spots panel
    ax = axs[0, 3]
    _setup(ax, grid, f"{'hot' if heat else 'cold'} spots of the base (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C)\nand what became of them\n(zone means, base → layout: UTCI, Ta °C, Tmrt °C, RH %, U m/s)", cand.layout, small=True)
    colors = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c"}
    for r in R.regions:
        ax.contourf(grid.X, grid.Y, np.where(r["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[colors[r["status"]]], alpha=0.75, zorder=2)
    for r in R.new_regions:
        ax.contourf(grid.X, grid.Y, np.where(r["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#8e44ad"], alpha=0.8, zorder=2)
    _label_zones(ax, R.regions)
    _label_zones(ax, R.new_regions, color="#4a235a")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=c, label=k) for k, c in colors.items()] + [mp.Patch(color="#8e44ad", label="new in layout")],
              fontsize=6, loc="lower left", framealpha=0.85)
    # row 2: drivers
    def diffpanel(ax, key, title, unit, scale=1.0):
        dd = np.where(R.both, (R.C[key] - R.B[key]) * scale, np.nan)
        lm = max(0.01, float(np.nanpercentile(np.abs(dd[R.both]), 98)))
        cf_ = _field(ax, grid, dd, -lm, lm, DIFF_CMAP)
        _setup(ax, grid, title, cand.layout)
        _cbar(fig, cf_, ax, unit)
        return cf_
    diffpanel(axs[1, 0], "Tmrt_C", "Δ Tmrt (layout − base)", "°C")
    ax = axs[1, 1]
    diffpanel(ax, "wind_ms", f"Δ wind speed (reference wind {R.u_ref:.1f} m/s)\nwake: solid = base, dashed = layout", "m/s")
    _outline(ax, grid, R.calm_b, colors="k", linewidths=1.1)
    _outline(ax, grid, R.calm_c, colors="k", linewidths=1.1, linestyles="--")
    _flow_arrow(ax, grid, R.flow)
    diffpanel(axs[1, 2], "Tair_C", "Δ air temperature", "°C")
    diffpanel(axs[1, 3], "RH_pct", "Δ relative humidity", "%")
    fig.suptitle(f"{cand.name} vs base — {SEASON_TXT[R.season]}   (site mean UTCI {R.stats_b['UTCI_C']:.2f} → {R.stats_c['UTCI_C']:.2f} °C)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fig_drivers(R, grid, path):
    """why: Shapley split of the UTCI change (site mean and the regions), Tmrt split"""
    sgn = R.sgn
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1.6, 1.1]))
    ax = axs[0]
    names = list(DRIVERS)
    vals = [sgn * R.contrib_mean[d] for d in names]
    ax.barh(range(len(names)), vals, color=[COL[d] for d in names])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{d}\nΔ {R.site_vals[d]['d']:+.{NDEC[d]}f} {UNIT[d]}" for d in names], fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(f"site mean: benefit {sgn * R.dU_mean:+.2f} °C\n(benefit = {'cooler' if sgn < 0 else 'warmer'})", fontsize=10)
    ax.margins(x=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("°C of UTCI benefit  (label: actual change of the input)")
    ax = axs[1]
    labels, per = [], []
    for r in R.regions:
        vv = r["vals"]
        labels.append(f"{r['id']}\n{r['area']:.0f} m²\n" + "\n".join(f"Δ{'U' if d == 'wind' else d} {vv[d]['d']:+.{NDEC[d]}f}" for d in ORDER))
        per.append([sgn * r["contrib"][d] for d in DRIVERS])
    if per:
        per = np.array(per)
        x = np.arange(len(labels))
        pos, neg = np.zeros(len(labels)), np.zeros(len(labels))
        for j, d in enumerate(DRIVERS):
            v = per[:, j]
            bot = np.where(v >= 0, pos, neg)
            ax.bar(x, v, bottom=bot, color=COL[d], label=d, width=0.6)
            pos += np.where(v >= 0, v, 0)
            neg += np.where(v < 0, v, 0)
        ax.plot(x, [r["benefit"] for r in R.regions], "kD", ms=5, label="total")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.axhline(0, color=INK, lw=0.8)
        ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.set_title(f"base {'hot' if R.season == 'heat' else 'cold'}-spot regions: benefit by driver", fontsize=10)
    ax.set_ylabel("°C of UTCI benefit")
    ax = axs[2]
    if R.tmrt is not None:
        parts = R.tmrt_parts_mean
        names2 = list(parts.keys())
        v2 = [-p if R.season == "heat" else p for p in parts.values()]
        ax.barh(range(len(names2)), v2, color="#d9822b", alpha=0.85)
        ax.set_yticks(range(len(names2)))
        ax.set_yticklabels(names2, fontsize=8)
        for i, v in enumerate(v2):
            ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.invert_yaxis()
        ax.margins(x=0.25)
        ax.set_title(f"inside Tmrt: Δ Tmrt = {R.dTmrt_mean:+.2f} °C  (regression R² {R.tmrt['r2']:.2f})\nbenefit in Tmrt °C", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fig_wakes(R, base, cand, grid, path):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    ratio_b, ratio_c = R.B["wind_ms"] / R.u_ref, R.C["wind_ms"] / R.u_ref
    vmax = float(np.ceil(10 * max(1.6, np.nanpercentile(np.concatenate([ratio_b[R.B["valid"]], ratio_c[R.C["valid"]]]), 99))) / 10)
    for ax, ratio, F, name, lay, wk in ((axs[0], ratio_b, R.B, base.name, base.layout, R.wakes_b), (axs[1], ratio_c, R.C, cand.name, cand.layout, R.wakes_c)):
        cf = _field(ax, grid, ratio, 0.0, vmax, "viridis", lines=[0.5, 1.0, 1.3])
        _setup(ax, grid, f"wind speed / reference wind — {name}\ncalm (< {R.wake_frac:g}) regions outlined", lay)
        _outline(ax, grid, (F["wind_ms"] < R.wake_frac * R.u_ref) & F["valid"], colors="w", linewidths=1.4)
        _label_regions(ax, wk, color="#7a1f1f")
        _flow_arrow(ax, grid, R.flow)
        _cbar(fig, cf, ax, "U / U_ref")
    ax = axs[2]
    _setup(ax, grid, "wake regions: what became of the base wakes\n(zone means, base → layout: UTCI, Ta, Tmrt, RH, U)", cand.layout)
    fate_col = {"gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b"}
    for w in R.wakes_b:
        ax.contourf(grid.X, grid.Y, np.where(w["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[fate_col[w["fate"]]], alpha=0.75, zorder=2)
    for w in R.wakes_c:
        if w.get("is_new"):
            ax.contourf(grid.X, grid.Y, np.where(w["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#8e44ad"], alpha=0.8, zorder=2)
    _label_zones(ax, R.wakes_b)
    _label_zones(ax, [w for w in R.wakes_c if w.get("is_new")], color="#4a235a")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=c, label=f"base wake: {k}") for k, c in fate_col.items()] + [mp.Patch(color="#8e44ad", label="new wake in layout")],
              fontsize=6, loc="lower left", framealpha=0.85)
    _flow_arrow(ax, grid, R.flow)
    fig.suptitle(f"ventilation and wakes — {cand.name} vs base — {SEASON_TXT[R.season]}", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def zone_table_rows(zones, with_status=True):
    rows = []
    for z in zones:
        v = z.get("vals")
        if v is None:
            continue
        tag = z.get("status") or z.get("fate") or ("new" if z.get("is_new") else "")
        if z["id"].startswith("n"):
            tag = "new"
        rows.append([f"{z['id']}\n{tag}", f"{z['area']:.0f}", f"{v['UTCI']['b']:.1f} → {v['UTCI']['c']:.1f}\n({v['UTCI']['d']:+.2f})"] + [vtxt(v, d, True, True) for d in ORDER])
    return rows


def fig_zones(R, cand, grid, path, kind):
    """map of the zones with their area-mean values written on them + a table: for Ta, Tmrt, RH and wind the base value → layout value (change)
    and the effect of that change on UTCI.  kind = 'spots' (hot / cold spots of the base and new ones) or 'wakes' (calm regions)"""
    heat = R.season == "heat"
    if kind == "spots":
        zones = R.regions + R.new_regions
        ttl = f"{'hot' if heat else 'cold'} spots of the base (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C) and new ones"
        colr = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c", "new": "#8e44ad"}
        colof = lambda z: colr["new"] if z["id"].startswith("n") else colr[z["status"]]            # noqa: E731
    else:
        zones = R.wakes_b + [w for w in R.wakes_c if w.get("is_new")]
        ttl = f"calm regions (wind < {R.wake_frac:g} x {R.u_ref:.2f} m/s): base wakes and new ones"
        colr = {"gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b", "new": "#8e44ad"}
        colof = lambda z: colr["new"] if z.get("is_new") else colr[z["fate"]]                     # noqa: E731
    zones = [z for z in zones if z.get("vals") is not None]
    if not zones:
        return False
    rows = zone_table_rows(zones)
    sv = R.site_vals
    rows.append(["whole site\n(open ground)", f"{sv['area']:.0f}", f"{sv['UTCI']['b']:.1f} → {sv['UTCI']['c']:.1f}\n({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d, True, True) for d in ORDER])
    nrow = len(rows)
    fig = plt.figure(figsize=(21, max(8.0, 1.2 + 0.62 * nrow)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
    axm, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _setup(axm, grid, ttl, cand.layout)
    for z in zones:
        axm.contourf(grid.X, grid.Y, np.where(z["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[colof(z)], alpha=0.7, zorder=2)
    _label_zones(axm, zones, fs=5.6)
    _flow_arrow(axm, grid, R.flow)
    import matplotlib.patches as mp
    axm.legend(handles=[mp.Patch(color=c, label=k) for k, c in colr.items()], fontsize=7, loc="lower left", framealpha=0.9)
    axt.axis("off")
    cols = ["zone", "area m²", "UTCI °C\n(change)", "Ta °C\nbase → layout (change)\neffect on UTCI °C", "Tmrt °C\nbase → layout (change)\neffect on UTCI °C",
            "RH %\nbase → layout (change)\neffect on UTCI °C", "wind m/s\nbase → layout (change)\neffect on UTCI °C"]
    tb = axt.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colWidths=[0.1, 0.07, 0.12, 0.18, 0.18, 0.18, 0.18])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.6)
    tb.scale(1, 2.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c8ced3")
        if r == 0:
            cell.set_facecolor("#e8eef2")
            cell.set_text_props(weight="bold", fontsize=7)
            cell.set_height(cell.get_height() * 1.25)
        elif r == nrow:
            cell.set_facecolor("#f2f4f6")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_facecolor(colof(zones[r - 1]))
            cell.set_text_props(color="white", weight="bold")
        elif c >= 3:                                         # shade the cell by the effect on UTCI (heat: blue cooler, red warmer; cold: blue warmer, red cooler)
            eff = (sv if r == nrow else zones[r - 1]["vals"])[ORDER[c - 3]]["eff"]
            a_ = min(abs(eff) / 1.5, 1.0) * 0.55
            cell.set_facecolor((0.84, 0.19, 0.15, a_) if (eff > 0) != (not heat) else (0.17, 0.48, 0.73, a_))
    fig.suptitle(f"{cand.name} vs base — {SEASON_TXT[R.season]}: zone-averaged inputs and what they did to UTCI   (effect: − = cooler, + = warmer)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return True


# ============================================================================= plain two-picture view: heat (or cold) spots, ventilation
# Each picture: base map | layout map | key, then a table of the spots / calm zones with their Ta, Tmrt, RH and wind, then a few plain sentences.
# Everything is placed in inches on a fixed canvas, so boxes cannot overlap.
import textwrap                                                       # noqa: E402

BAND_EDGES = [-80, -40, -27, -13, 0, 9, 26, 32, 38, 46, 80]
BAND_NAMES = ["extreme cold stress", "very strong cold stress", "strong cold stress", "moderate cold stress", "slight cold stress", "no thermal stress",
              "moderate heat stress", "strong heat stress", "very strong heat stress", "extreme heat stress"]
BAND_RANGE = ["below −40", "−40 to −27", "−27 to −13", "−13 to 0", "0 to 9", "9 to 26", "26 to 32", "32 to 38", "38 to 46", "above 46"]
BAND_COL = ["#08306b", "#2b5fa8", "#4a8bc9", "#86b6e0", "#c5dcf0", "#e6f2dc", "#fde0a1", "#fbb04e", "#e5612a", "#9c2412"]
STATUS_COL = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c", "new": "#8e44ad",
              "gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b"}
STATUS_TXT = {"resolved": "gone", "partly resolved": "partly gone", "reduced": "reduced", "unchanged": "unchanged", "worse": "worse", "new": "NEW",
              "gone": "gone", "smaller": "smaller", "similar": "similar", "larger": "larger"}
SIMPLE_W = 16.0
SIMPLE_DPI = 200                                                      # --dpi
MAX_ROWS = 8
BLUE_TXT, RED_TXT, GREY_TXT = "#1f5f99", "#b3261e", "#5b666e"


def _inch_axes(fig, W, H, x, y, w, h):
    """axes covering the box (x, y, w, h) given in inches from the top-left of the canvas; its data units are inches too (y downwards)"""
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return ax


def _short_where(centroid, layout, site_centre, flow=None, near=45.0):
    d = compass(centroid[0] - site_centre[0], centroid[1] - site_centre[1])
    s = f"{d} part" if d != "centre" else "centre"
    nb = nearest_building(layout, centroid, None)
    if nb is not None and nb[1] <= near:
        s += f", next to {_bn(nb[0])}" if nb[1] < 12 else f", {nb[1]:.0f} m from {_bn(nb[0])}"
    return s


def _bn(bid):
    """short, readable building name: 'building_3_internal_floor_0' -> 'B3'"""
    t = str(bid)
    t = re.sub(r"[_\- ]?(internal|external|inner|outer)?[_\- ]?floor[_\- ]?\d+$", "", t, flags=re.I)
    t = re.sub(r"[_\- ](internal|external)$", "", t, flags=re.I)
    t = re.sub(r"^building[_\- ]?", "B", t, flags=re.I) if re.match(r"^building[_\-]", t, flags=re.I) else t
    return t or str(bid)


def _building_groups(layout):
    """buildings that share a footprint (e.g. the floors of one building) -> list of (label, polygon of the largest member)"""
    ids = list(layout.buildings)
    polys = [layout.buildings[i] for i in ids]
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    try:
        from shapely.strtree import STRtree
        tree = STRtree(polys)
        pairs = ((i, int(j)) for i in range(len(polys)) for j in tree.query(polys[i]) if int(j) > i)
    except Exception:
        pairs = ((i, j) for i in range(len(polys)) for j in range(i + 1, len(polys)))
    for i, j in pairs:
        try:
            inter = polys[i].intersection(polys[j]).area
        except Exception:
            continue
        if inter > 0.5 * min(polys[i].area, polys[j].area):
            parent[find(j)] = find(i)
    groups = {}
    for k in range(len(ids)):
        groups.setdefault(find(k), []).append(k)
    out = []
    for mem in groups.values():
        big = max(mem, key=lambda k: polys[k].area)
        names = []
        for k in sorted(mem, key=lambda k: -polys[k].area):
            n = _bn(ids[k])
            if n not in names:
                names.append(n)
        out.append(("\n".join(names[:2]), polys[big]))
    return out


def _building_labels(ax, layout, fs=8.0):
    for label, g in _building_groups(layout):
        if g.area < 120:
            continue
        p = g.representative_point()
        ax.text(p.x, p.y, label, fontsize=fs, color="white", ha="center", va="center", weight="bold", zorder=8, clip_on=True, linespacing=1.0)


def _scale_bar(ax, grid, length=50.0):
    x0, x1, y0, y1 = grid.extent
    xs, ys = x1 - 0.06 * (x1 - x0) - length, y0 + 0.05 * (y1 - y0)
    ax.plot([xs, xs + length], [ys, ys], color=INK, lw=2.2, solid_capstyle="butt", zorder=30)
    ax.text(xs + length / 2, ys + 0.012 * (y1 - y0), f"{length:g} m", fontsize=8, ha="center", va="bottom", color=INK, zorder=30,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


def _map_panel(ax, grid, layout, title, subtitle):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#aab2b8")
    ax.set_title(title, fontsize=12.5, color=INK, weight="bold", loc="left", pad=22)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5, color=GREY_TXT, ha="left", va="bottom")
    _footprint_patches(ax, layout, tree_alpha=0.55, faces=False, tree_lw=0.4)
    _building_labels(ax, layout)
    _scale_bar(ax, grid)


def _wind_arrow_simple(ax, grid, flow):
    if flow is None:
        return
    x0, x1, y0, y1 = grid.extent
    L = 0.14 * (x1 - x0)
    cx, cy = x0 + 0.12 * (x1 - x0), y1 - 0.09 * (y1 - y0)
    ax.annotate("", xy=(cx + L * flow[0], cy + L * flow[1]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=INK), zorder=30)
    ax.text(cx - 0.01 * (x1 - x0), cy + 0.02 * (y1 - y0), "wind", fontsize=9, color=INK, ha="center", va="bottom", zorder=30, weight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


def _place_badges(fig, ax, items, r_px=12.0):
    """numbered circles on a map: one per zone at its centre; circles that would touch are pushed apart and joined to their zone by a thin line"""
    if not items:
        return
    fig.canvas.draw()
    inv = ax.transData.inverted()
    true = [np.array(ax.transData.transform(it["xy"]), float) for it in items]
    pos = [p.copy() for p in true]
    axbb = ax.get_window_extent()
    d_min = 2 * r_px + 4
    for _ in range(200):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                v = pos[j] - pos[i]
                dist = float(np.hypot(*v))
                if dist < d_min:
                    moved = True
                    if dist < 1e-6:
                        v, dist = np.array([1.0, 0.3]), 1.04
                    push = 0.5 * (d_min - dist + 1.0) * v / dist
                    pos[i] -= push
                    pos[j] += push
        for i in range(len(pos)):
            pos[i][0] = float(np.clip(pos[i][0], axbb.x0 + r_px + 2, axbb.x1 - r_px - 2))
            pos[i][1] = float(np.clip(pos[i][1], axbb.y0 + r_px + 2, axbb.y1 - r_px - 2))
        if not moved:
            break
    for it, t, p in zip(items, true, pos):
        xy = it["xy"]
        xyt = tuple(inv.transform(p))
        if np.hypot(*(p - t)) > 0.6 * r_px:
            ax.plot([xy[0], xyt[0]], [xy[1], xyt[1]], color="#111", lw=0.9, zorder=26)
        ax.plot([xy[0]], [xy[1]], "o", ms=3.2, color="#111", zorder=27)
        ax.text(xyt[0], xyt[1], it["label"], fontsize=10, color="white", ha="center", va="center", weight="bold", zorder=28,
                bbox=dict(boxstyle="circle,pad=0.28", fc=it["color"], ec="#111", lw=1.0))


def _rows_of_zones(zones, cap=MAX_ROWS):
    """largest zones first; returns (shown, hidden)"""
    z = sorted([x for x in zones if x.get("vals") is not None], key=lambda g: -g["area"])
    return z[:cap], z[cap:]


def _fmt_eff(x, dr, cold=False):
    """text and colour of an effect on UTCI.  Heat: a rise of UTCI is red, a fall blue.  Cold: a rise of UTCI is blue, a fall red."""
    if abs(x) < 0.05:
        return "UTCI ±0", GREY_TXT
    return f"UTCI {x:+.1f}".replace("-", "−"), (RED_TXT if (x > 0) != cold else BLUE_TXT)


def _num(x, nd):
    return f"{x:.{nd}f}".replace("-", "−") if round(x, nd) != 0 else f"{0:.{nd}f}"


def _delta(x, nd):
    if round(x, nd) == 0:
        return f"{0:.{nd}f}"
    return f"{x:+.{nd}f}".replace("-", "−")


def _draw_zone_table(ax, W, rows, site, header_first="Spot", with_site=True, cell_h=0.56, cold=False):
    """rows: list of dict(num, color, status, where, area, vals);  columns: spot | where | UTCI | Ta | Tmrt | RH | wind.  Returns the height used (inches)"""
    colw = [1.5, 3.05, 1.8, 2.2, 2.2, 2.2, 2.2]
    x = [0.0]
    for w in colw:
        x.append(x[-1] + w)
    hdr = [(header_first, ""), ("Where, how big", ""), ("UTCI °C", "feels-like temp."), ("Air temp. Ta °C", "base → layout"), ("Radiant temp. Tmrt °C", "base → layout"),
           ("Humidity RH %", "base → layout"), ("Wind m/s", "base → layout")]
    hh = 0.62
    ax.add_patch(plt.Rectangle((0, 0), x[-1], hh, fc="#e8eef2", ec="#c8ced3", lw=0.8))
    for k, (t, s) in enumerate(hdr):
        cx = 0.5 * (x[k] + x[k + 1])
        ax.text(cx, hh * (0.40 if s else 0.5), t, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
        if s:
            ax.text(cx, hh * 0.78, s, ha="center", va="center", fontsize=8.5, color=GREY_TXT)
    y = hh
    allrows = list(rows) + ([dict(num=None, status="", where="Whole site (open ground)", area=site["area"], vals=site, site=True)] if with_site and site else [])
    for rw in allrows:
        v = rw["vals"]
        is_site = rw.get("site", False)
        ax.add_patch(plt.Rectangle((0, y), x[-1], cell_h, fc="#f1f3f5" if is_site else "white", ec="#c8ced3", lw=0.8))
        # first column: numbered badge + status
        if not is_site:
            ax.text(x[0] + 0.32, y + cell_h / 2, str(rw["num"]), ha="center", va="center", fontsize=10.5, color="white", weight="bold",
                    bbox=dict(boxstyle="circle,pad=0.28", fc=rw["color"], ec="#111", lw=0.9))
            ax.text(x[0] + 0.62, y + cell_h / 2, rw["status"], ha="left", va="center", fontsize=8.3, color=INK)
        # where / how big
        ax.text(x[1] + 0.08, y + cell_h * 0.34, rw["where"], ha="left", va="center", fontsize=9.5, color=INK, weight="bold" if is_site else "normal")
        ax.text(x[1] + 0.08, y + cell_h * 0.72, f"{rw['area']:,.0f} m²", ha="left", va="center", fontsize=9, color=GREY_TXT)
        # UTCI
        u = v["UTCI"]
        ax.text(0.5 * (x[2] + x[3]), y + cell_h * 0.34, f"{_num(u['b'], 1)} → {_num(u['c'], 1)}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
        ax.text(0.5 * (x[2] + x[3]), y + cell_h * 0.73, f"{_delta(u['d'], 1)} °C", ha="center", va="center", fontsize=9, color=GREY_TXT if abs(u["d"]) <= 0.05 else (RED_TXT if (u["d"] > 0) != cold else BLUE_TXT))
        for k, d in enumerate(ORDER):
            c = 3 + k
            vv = v[d]
            nd = NDEC[d] if d != "RH" else 1
            eff = vv["eff"]
            a_ = min(abs(eff) / 1.5, 1.0) * 0.5
            if a_ > 0.02:
                ax.add_patch(plt.Rectangle((x[c], y), colw[c], cell_h, fc=(0.84, 0.19, 0.15, a_) if (eff > 0) != cold else (0.17, 0.48, 0.73, a_), ec="none"))
            cx = 0.5 * (x[c] + x[c + 1])
            ax.text(cx, y + cell_h * 0.30, f"{_num(vv['b'], nd)} → {_num(vv['c'], nd)}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
            t2, c2 = _fmt_eff(eff, d, cold)
            ax.text(cx, y + cell_h * 0.72, f"({_delta(vv['d'], nd)})   {t2}", ha="center", va="center", fontsize=8.8, color=c2)
        y += cell_h
    return y


GLOSSARY = ("UTCI = the 'feels-like' temperature a person experiences (it combines the four columns on its right).   Ta = air temperature.   "
            "Tmrt = radiant temperature: the heat a person receives from the sun and from hot surfaces (shade lowers it).   RH = relative humidity.   "
            "Each cell: base → layout, (change), and what that change alone did to UTCI (blue = lowers UTCI, red = raises it).")
GLOSSARY_COLD = GLOSSARY.replace("(blue = lowers UTCI, red = raises it)", "(blue = raises UTCI, red = lowers it)")


def _glossary(heat):
    return GLOSSARY if heat else GLOSSARY_COLD


def _band_legend(ax, y, present, w):
    """colour chips of the UTCI categories that occur on the maps"""
    ax.text(0.0, y, "Colour on the maps = UTCI category", fontsize=10.5, weight="bold", color=INK, va="top")
    y += 0.34
    for k in range(len(BAND_NAMES) - 1, -1, -1):
        if k not in present:
            continue
        ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc=BAND_COL[k], ec="#777", lw=0.5))
        ax.text(0.45, y + 0.1, f"{BAND_NAMES[k]}", fontsize=8.8, va="center", color=INK)
        ax.text(w, y + 0.1, BAND_RANGE[k] + " °C", fontsize=8.3, va="center", ha="right", color=GREY_TXT)
        y += 0.27
    return y


def _symbol_legend(ax, y, items):
    """items: [(kind, colour, text)] kind in patch / circle / line"""
    for kind, col, txt in items:
        if kind == "patch":
            ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc=col, ec="#555", lw=0.6))
        elif kind == "outline":
            ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc="none", ec=col, lw=1.4))
        elif kind == "dash":
            ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc="none", ec=col, lw=1.4, ls="--"))
        elif kind == "circle":
            ax.plot([0.17], [y + 0.1], "o", ms=9, mfc=col, mec="#111", mew=0.9)
        ax.text(0.45, y + 0.1, txt, fontsize=8.8, va="center", color=INK)
        y += 0.28
    return y


def _site_centre(grid):
    m = grid.site_mask
    return (float(grid.X[m].mean()), float(grid.Y[m].mean()))


# ---------------------------------------------------------------------------------------- the words (2-3 sentences each)
def _drv_plain(d):
    return {"Tmrt": "radiant heat from sun and hot surfaces (shade)", "wind": "wind", "Ta": "air temperature", "RH": "humidity"}[d]


def _area_weighted(zones, fn):
    tot = sum(z["area"] for z in zones)
    return sum(z["area"] * fn(z) for z in zones) / tot if tot > 0 else 0.0


def _vtxt_short(d, b, c):
    nd = NDEC[d] if d != "RH" else 1
    lab = {"Ta": "Ta", "Tmrt": "Tmrt", "RH": "RH", "wind": "wind"}[d]
    return f"{lab} {_num(b, nd)} → {_num(c, nd)} {UNIT[d]}"


def short_spot_inference(R, base, cand, shown_index):
    """three sentences about the hot / cold spots (list of str)"""
    heat = R.season == "heat"
    word = "hot" if heat else "cold"
    sb, sc = R.spot_area_b, R.spot_area_c
    n_all = len(R.regions)
    n_res = sum(1 for g in R.regions if g["status"] == "resolved")
    pct = 100 * (sb - sc) / sb if sb > 0 else 0.0
    out = []
    verb = "shrinks" if sc < sb else ("grows" if sc > sb else "stays")
    s1 = (f"{word.capitalize()} ground (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C) {verb} from {sb:,.0f} to {sc:,.0f} m² ({'−' if sc <= sb else '+'}{abs(pct):.0f} %)")
    if n_all:
        s1 += f": {n_res} of {n_all} base {word} spots are gone"
    if R.regions:
        zs = R.regions
        ub, uc = _area_weighted(zs, lambda z: z["u_base"]), _area_weighted(zs, lambda z: z["u_cand"])
        s1 += f", and UTCI inside them goes {ub:.1f} → {uc:.1f} °C ({_delta(uc - ub, 1)})"
    out.append(s1 + ".")
    if R.regions:
        contrib = {d: _area_weighted(zs, lambda z, d=d: z["contrib"][d]) for d in DRIVERS}
        vb = {d: _area_weighted(zs, lambda z, d=d: z["vals"][d]["b"]) for d in DRIVERS}
        vc = {d: _area_weighted(zs, lambda z, d=d: z["vals"][d]["c"]) for d in DRIVERS}
        order = [d for d in sorted(DRIVERS, key=lambda d: -abs(contrib[d])) if abs(contrib[d]) >= 0.1][:2]
        if order:
            bits = [f"{_drv_plain(d)} ({_vtxt_short(d, vb[d], vc[d])}, worth {_delta(contrib[d], 1)} °C of UTCI)" for d in order]
            s2 = "Main cause: " + " and ".join(bits)
            dd = R.diff
            if dd["trees_cand"] != dd["trees_base"]:
                s2 += f"; the layout has {dd['trees_cand']} trees instead of {dd['trees_base']} (canopy {dd['canopy_base']:.0f} → {dd['canopy_cand']:.0f} m²)"
            out.append(s2 + ".")
    left = sorted([g for g in R.regions if g["status"] in ("unchanged", "worse", "reduced", "partly resolved")], key=lambda g: -g["area"])
    s3 = ""
    if left:
        g = left[0]
        cause, _act = mitigation_for_region(g, R, R.season)
        s3 = f"Still {word}: spot {shown_index.get(id(g), '?')} ({g['area']:,.0f} m², UTCI {g['u_cand']:.1f} °C), cause: {cause}"
    if R.new_regions:
        g = max(R.new_regions, key=lambda x: x["area"])
        nb = g.get("near_building")
        s3 += ("; " if s3 else "") + f"new {word} ground of {sum(x['area'] for x in R.new_regions):,.0f} m²" + (f" appears next to {_bn(nb)}" if nb else " appears")
    if not s3:
        s3 = f"No {word} spot of at least 80 m² remains and no new one appears" if n_all else f"The base case has no {word} spot of at least 80 m², and the layout adds none"
    out.append(s3.rstrip(".") + ".")
    return out


def short_vent_inference(R, base, cand, shown_index):
    heat = R.season == "heat"
    v = R.vent
    wb = R.wakes_b
    newc = [w for w in R.wakes_c if w.get("is_new")]
    thr = R.wake_frac * R.u_ref
    n_gone = sum(1 for w in wb if w["fate"] == "gone")
    n_small = sum(1 for w in wb if w["fate"] == "smaller")
    out = [f"Calm ground (wind < {thr:.2f} m/s) goes from {v['calm_area_b']:,.0f} to {v['calm_area_c']:,.0f} m² ({100 * v['calm_share_b']:.0f} → {100 * v['calm_share_c']:.0f} % of the open ground) "
           f"and the mean wind from {R.stats_b['wind_ms']:.2f} to {R.stats_c['wind_ms']:.2f} m/s: of {len(wb)} base calm zones {n_gone} are gone and {n_small} smaller"
           + (f", {len(newc)} new one(s) appear" if newc else "") + "."]
    cw = R.sgn * R.contrib_mean["wind"]
    cm = R.calm_b & R.B["valid"]
    if heat:
        s2 = ""
        if cm.any():
            pen = float(np.nanmean(R.B["UTCI_C"][cm])) - R.stats_b["UTCI_C"]
            s2 = f"In the hot hour still air costs heat: the base calm zones were {_delta(pen, 1)} °C in UTCI against the site mean. "
        if cw > 0.15:
            s2 += f"The wind change alone lowers the site-mean UTCI by {cw:.1f} °C, so ventilation is part of the benefit."
        elif cw < -0.15:
            s2 += f"The wind change alone raises the site-mean UTCI by {abs(cw):.1f} °C; the benefit comes from shade and air temperature instead."
        else:
            s2 += "The wind change alone hardly moves the site-mean UTCI, so the benefit does not come from ventilation."
    else:
        s2 = ("In the cold hour calm air is shelter from the wind chill. " + (f"The wind change alone {'raises' if cw > 0 else 'lowers'} the site-mean UTCI by {abs(cw):.1f} °C "
              f"({'less' if cw > 0 else 'more'} wind chill)."))
    out.append(s2)
    shown, _ = _rows_of_zones(wb + newc, cap=1)
    s3 = ""
    if shown:
        w = shown[0]
        vv = w["vals"]
        fate = "new in the layout" if w.get("is_new") else {"gone": "gone in the layout", "smaller": "smaller in the layout", "similar": "about the same", "larger": "larger in the layout"}[w["fate"]]
        s3 = (f"Largest zone (zone {shown_index.get(id(w), '?')}{', behind ' + _bn(w['building']) if w.get('building') else ''}, {w['area']:,.0f} m², {w['length_m']:.0f} m long): {fate}; "
              f"wind {vv['wind']['b']:.2f} → {vv['wind']['c']:.2f} m/s, UTCI {vv['UTCI']['b']:.1f} → {vv['UTCI']['c']:.1f} °C")
        if heat and not w.get("is_new") and w["fate"] in ("similar", "larger") and w.get("building"):
            s3 += f"; to fix it shift {_bn(w['building'])} off the wind line or leave a gap upwind"
    if s3:
        out.append(s3 + ".")
    return out


def _write_inference(path, title, sentences):
    path.write_text(title + "\n\n" + "\n".join(sentences) + "\n")


def _inference_box(ax, wbox, hbox, sentences, fs=11.0):
    ax.add_patch(plt.Rectangle((0, 0), wbox, hbox, fc="#f6f8f9", ec="#d5dbdf", lw=0.8))
    ax.text(0.15, 0.12, "What it means", fontsize=12.5, weight="bold", color=INK, va="top")
    nch = int((wbox - 0.6) / (fs * 0.0070))
    y = 0.5
    for s in sentences:
        ls = textwrap.wrap(s, width=nch)
        ax.text(0.15, y, "\n".join(ls), fontsize=fs, color="#222", va="top", linespacing=1.3)
        y += len(ls) * 0.215 + 0.1


def _inference_height(wbox, sentences, fs=11.0):
    nch = int((wbox - 0.6) / (fs * 0.0070))
    return 0.5 + sum(len(textwrap.wrap(s, width=nch)) * 0.215 + 0.1 for s in sentences) + 0.15


def _index_map(zones):
    z, _ = _rows_of_zones(zones, cap=10 ** 6)
    return {id(x): k for k, x in enumerate(z, 1)}


# ---------------------------------------------------------------------------------------- the two figures
def fig_simple_spots(R, base, cand, grid, path):
    heat = R.season == "heat"
    word = "Hot" if heat else "Cold"
    zones_all = R.regions + R.new_regions
    shown, hidden = _rows_of_zones(zones_all)
    centre = _site_centre(grid)
    rows = []
    for k, z in enumerate(shown, 1):
        is_new = z["id"].startswith("n")
        st = "new" if is_new else z["status"]
        rows.append(dict(num=k, color=STATUS_COL[st], status=STATUS_TXT[st], where=_short_where(z["centroid"], cand.layout, centre), area=z["area"], vals=z["vals"], zone=z, is_new=is_new))
    paras = short_spot_inference(R, base, cand, _index_map(zones_all))
    W = SIMPLE_W
    M = 0.4
    ms = 5.55
    n_rows = len(rows) + 1
    tab_h = 0.62 + 0.56 * n_rows + (0.32 if not rows else 0.0)
    hid_txt = f"{len(hidden)} smaller {word.lower()} spot(s) ({sum(h['area'] for h in hidden):,.0f} m²) are not shown; see tables/regions.csv." if hidden else ""
    para_h = _inference_height(W - 2 * M, paras)
    H = 0.95 + 0.55 + ms + 0.3 + tab_h + 0.75 + para_h + 0.25
    fig = plt.figure(figsize=(W, H), facecolor="white")
    fig.text(M / W, 1 - 0.38 / H, f"{word} spots: {cand.name} compared with the base case — {SEASON_TXT[R.season]}", fontsize=17, weight="bold", color=INK, va="center")
    fig.text(M / W, 1 - 0.74 / H, f"Pedestrian height; {word.lower()} spot = UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C in the base case.  Numbers on the maps are the spots listed in the table below.",
             fontsize=10.5, color=GREY_TXT, va="center")
    y0 = 0.95 + 0.55
    xs = [M, M + ms + 0.3]
    allu = np.concatenate([R.B["UTCI_C"][R.B["valid"]], R.C["UTCI_C"][R.C["valid"]]])
    uv0, uv1 = float(np.nanpercentile(allu, 1)), float(np.nanpercentile(allu, 99))
    lvu = _nice_levels(uv0, max(uv1, uv0 + 3.0), 12)
    cmu = "YlOrRd" if heat else "Blues_r"
    axes = []
    for k, (Fld, lay, name, stats, spot) in enumerate(((R.B, base.layout, "BASE CASE", R.stats_b, R.spot_b), (R.C, cand.layout, f"{cand.name.upper()}  (best layout)", R.stats_c, R.spot_c))):
        ax = fig.add_axes([xs[k] / W, 1 - (y0 + ms) / H, ms / W, ms / H])
        Z = np.ma.masked_invalid(np.where(Fld["valid"], Fld["UTCI_C"], np.nan))
        cfu = ax.contourf(grid.X, grid.Y, np.clip(Z, lvu[0], lvu[-1]), levels=lvu, cmap=cmu, zorder=1)
        sub = f"mean UTCI {stats['UTCI_C']:.1f} °C   ·   {word.lower()} ground {(R.spot_area_b if k == 0 else R.spot_area_c):,.0f} m²"
        _map_panel(ax, grid, lay, name, sub)
        _outline(ax, grid, R.spot_b, colors="k", linewidths=1.6, linestyles="-" if k == 0 else "--")
        if k == 1 and R.new_regions:
            for z in R.new_regions:
                _outline(ax, grid, z["mask"], colors="#8e44ad", linewidths=1.6, linestyles="--")
        axes.append(ax)
    # key
    kx = M + 2 * ms + 0.3 + 0.45
    kw = W - M - kx
    axk = _inch_axes(fig, W, H, kx, y0 - 0.05, kw, ms + 0.1)
    axk.text(0.0, 0.05, "Colour on the maps = UTCI, °C", fontsize=10.5, weight="bold", color=INK, va="top")
    cax = fig.add_axes([kx / W, 1 - (y0 - 0.05 + 0.78) / H, kw / W, 0.2 / H])
    cb = fig.colorbar(cfu, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8.5)
    present = [k for k in range(len(BAND_NAMES)) if BAND_EDGES[k + 1] > lvu[0] and BAND_EDGES[k] < lvu[-1]]
    yk = 1.2
    axk.text(0.0, yk, "UTCI categories (stress level)", fontsize=10.0, weight="bold", color=INK, va="top")
    yk += 0.3
    for k in reversed(present):
        axk.text(0.0, yk + 0.095, BAND_RANGE[k] + " °C", fontsize=8.5, va="center", color=GREY_TXT)
        axk.text(1.15, yk + 0.095, BAND_NAMES[k], fontsize=8.5, va="center", color=INK)
        yk += 0.25
    yk += 0.12
    axk.text(0.0, yk, "On the maps", fontsize=10.0, weight="bold", color=INK, va="top")
    yk += 0.3
    axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="#8a8f94", ec="#555", lw=0.6))
    axk.text(0.45, yk + 0.095, "building (name inside)", fontsize=8.5, va="center", color=INK)
    yk += 0.25
    axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="none", ec="#2b6cb0", lw=1.2))
    axk.text(0.45, yk + 0.095, "ponds, lawns, tree crowns (thin lines)", fontsize=8.5, va="center", color=INK)
    yk += 0.25
    axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="none", ec="k", lw=1.5))
    axk.text(0.45, yk + 0.095, "black outline = the spot (dashed in the layout)", fontsize=8.5, va="center", color=INK)
    yk += 0.37
    axk.text(0.0, yk, f"Circle colour = what became of the spot", fontsize=10.0, weight="bold", color=INK, va="top")
    yk += 0.3
    for col, txt in [(STATUS_COL["resolved"], "gone (≥ 70 % no longer " + ("hot)" if heat else "cold)")), (STATUS_COL["partly resolved"], "partly gone"),
                     (STATUS_COL["reduced"], "reduced, still " + word.lower()), (STATUS_COL["unchanged"], "unchanged"), (STATUS_COL["worse"], "worse"),
                     (STATUS_COL["new"], "new in the layout (purple outline)")]:
        axk.plot([0.17], [yk + 0.095], "o", ms=8.5, mfc=col, mec="#111", mew=0.9)
        axk.text(0.45, yk + 0.095, txt, fontsize=8.5, va="center", color=INK)
        yk += 0.25
    # badges on both maps
    badge_items = [dict(label=str(r["num"]), xy=r["zone"]["centroid"], color=STATUS_COL["new" if r["is_new"] else r["zone"]["status"]]) for r in rows]
    for ax in axes:
        _place_badges(fig, ax, badge_items)
    # table
    ty = y0 + ms + 0.3
    axt = _inch_axes(fig, W, H, M, ty, W - 2 * M, tab_h + 0.75)
    used = _draw_zone_table(axt, W, rows, R.site_vals, cold=not heat)
    gy = used + 0.1
    if not rows:
        axt.text(0.0, used + 0.08, f"No {word.lower()} spot of at least {80:.0f} m² in the base case.", fontsize=11, color=INK, va="top")
        gy = used + 0.42
    axt.text(0.0, gy, "\n".join(textwrap.wrap(_glossary(heat), 205)) + ("\n" + hid_txt if hid_txt else ""), fontsize=8.6, color=GREY_TXT, va="top", linespacing=1.3)
    # sentences
    py = ty + tab_h + 0.75
    axp = _inch_axes(fig, W, H, M, py, W - 2 * M, para_h)
    _inference_box(axp, W - 2 * M, para_h, paras)
    fig.savefig(path, dpi=SIMPLE_DPI, facecolor="white")
    plt.close(fig)
    _write_inference(Path(path).with_suffix(".txt"), f"{word} spots - {cand.name} vs base - {SEASON_TXT[R.season]}", paras)
    return True


def fig_simple_vent(R, base, cand, grid, path):
    heat = R.season == "heat"
    wb = R.wakes_b
    newc = [w for w in R.wakes_c if w.get("is_new")]
    zones = wb + newc
    shown, hidden = _rows_of_zones(zones)
    centre = _site_centre(grid)
    rows = []
    for k, z in enumerate(shown, 1):
        is_new = bool(z.get("is_new"))
        st = "new" if is_new else z["fate"]
        w = z
        where = _short_where(z["centroid"], cand.layout if is_new else base.layout, centre)
        if z.get("building"):
            where = f"behind {_bn(z['building'])}, {compass(z['centroid'][0] - centre[0], centre[1] * 0 + z['centroid'][1] - centre[1])} part"
        rows.append(dict(num=k, color=STATUS_COL[st], status=STATUS_TXT[st], where=where, area=z["area"], vals=z["vals"], zone=z, is_new=is_new))
    paras = short_vent_inference(R, base, cand, _index_map(zones))
    W = SIMPLE_W
    M = 0.4
    ms = 5.55
    n_rows = len(rows) + 1
    tab_h = 0.62 + 0.56 * n_rows + (0.32 if not rows else 0.0)
    hid_txt = f"{len(hidden)} smaller calm zone(s) ({sum(h['area'] for h in hidden):,.0f} m²) are not shown; see tables/wakes.csv." if hidden else ""
    para_h = _inference_height(W - 2 * M, paras)
    H = 0.95 + 0.55 + ms + 0.3 + tab_h + 0.75 + para_h + 0.25
    fig = plt.figure(figsize=(W, H), facecolor="white")
    thr = R.wake_frac * R.u_ref
    fig.text(M / W, 1 - 0.38 / H, f"Ventilation: {cand.name} compared with the base case — {SEASON_TXT[R.season]}", fontsize=17, weight="bold", color=INK, va="center")
    fig.text(M / W, 1 - 0.74 / H, f"Wind speed at pedestrian height.  Calm zone = wind below {thr:.2f} m/s ({R.wake_frac:g} × the free wind of the base case, {R.u_ref:.2f} m/s); "
                                  f"{'still air traps heat, so calm zones are a problem in this hour.' if heat else 'calm air is shelter from the wind chill in this hour.'}",
             fontsize=10.5, color=GREY_TXT, va="center")
    y0 = 0.95 + 0.55
    xs = [M, M + ms + 0.3]
    vmax = float(np.ceil(10 * max(1.0, np.nanpercentile(np.concatenate([R.B["wind_ms"][R.B["valid"]], R.C["wind_ms"][R.C["valid"]]]), 99))) / 10)
    lv = _nice_levels(0.0, vmax, 10)
    axes, cf = [], None
    for k, (Fld, lay, name, stats, calm, wk) in enumerate(((R.B, base.layout, "BASE CASE", R.stats_b, R.calm_b, R.wakes_b), (R.C, cand.layout, f"{cand.name.upper()}  (best layout)", R.stats_c, R.calm_c, R.wakes_c))):
        ax = fig.add_axes([xs[k] / W, 1 - (y0 + ms) / H, ms / W, ms / H])
        Z = np.ma.masked_invalid(np.where(Fld["valid"], Fld["wind_ms"], np.nan))
        cf = ax.contourf(grid.X, grid.Y, np.clip(Z, lv[0], lv[-1]), levels=lv, cmap="Blues", zorder=1)
        sub = f"mean wind {stats['wind_ms']:.2f} m/s   ·   calm ground {(R.vent['calm_area_b'] if k == 0 else R.vent['calm_area_c']):,.0f} m²"
        _map_panel(ax, grid, lay, name, sub)
        ax.contourf(grid.X, grid.Y, np.where(calm, 1.0, np.nan), levels=[0.5, 1.5], colors=["#c0392b"], alpha=0.35, zorder=2)
        _outline(ax, grid, calm, colors="#9c1c10", linewidths=1.4)
        _wind_arrow_simple(ax, grid, R.flow)
        axes.append(ax)
    kx = M + 2 * ms + 0.3 + 0.45
    kw = W - M - kx
    axk = _inch_axes(fig, W, H, kx, y0 - 0.05, kw, ms + 0.1)
    axk.text(0.0, 0.05, "Colour on the maps = wind speed", fontsize=10.5, weight="bold", color=INK, va="top")
    cax = fig.add_axes([(kx) / W, 1 - (y0 - 0.05 + 0.75) / H, kw / W, 0.2 / H])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("m/s", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    yk = 1.35
    axk.text(0.0, yk, "Symbols", fontsize=10.5, weight="bold", color=INK, va="top")
    yk += 0.34
    yk = _symbol_legend(axk, yk, [("patch", "#8a8f94", "building (name inside)"), ("outline", "#2b6cb0", "pond outline"), ("outline", "#8fb98a", "lawn outline"), ("outline", "#2f7d32", "tree crown")])
    yk += 0.12
    axk.text(0.0, yk, "Calm zones", fontsize=10.5, weight="bold", color=INK, va="top")
    yk += 0.34
    axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.2, fc="#c0392b", alpha=0.35, ec="#9c1c10", lw=1.4))
    axk.text(0.45, yk + 0.1, "calm zone: wind < %.2f m/s" % thr, fontsize=8.6, va="center", color=INK)
    yk += 0.28
    for kind, col, txt in [("circle", STATUS_COL["gone"], "base calm zone: gone in the layout"), ("circle", STATUS_COL["smaller"], "smaller"), ("circle", STATUS_COL["similar"], "about the same"),
                           ("circle", STATUS_COL["larger"], "larger"), ("circle", STATUS_COL["new"], "new in the layout")]:
        axk.plot([0.17], [yk + 0.1], "o", ms=9, mfc=col, mec="#111", mew=0.9)
        axk.text(0.45, yk + 0.1, txt, fontsize=8.6, va="center", color=INK)
        yk += 0.28
    badge_items = [dict(label=str(r["num"]), xy=r["zone"]["centroid"], color=r["color"]) for r in rows]
    for ax in axes:
        _place_badges(fig, ax, badge_items)
    ty = y0 + ms + 0.3
    axt = _inch_axes(fig, W, H, M, ty, W - 2 * M, tab_h + 0.75)
    used = _draw_zone_table(axt, W, rows, R.site_vals, header_first="Calm zone", cold=not heat)
    gy = used + 0.1
    if not rows:
        axt.text(0.0, used + 0.08, "No calm zone of at least 80 m² in either layout.", fontsize=11, color=INK, va="top")
        gy = used + 0.42
    axt.text(0.0, gy, "\n".join(textwrap.wrap(_glossary(heat), 205)) + ("\n" + hid_txt if hid_txt else ""), fontsize=8.6, color=GREY_TXT, va="top", linespacing=1.3)
    py = ty + tab_h + 0.75
    axp = _inch_axes(fig, W, H, M, py, W - 2 * M, para_h)
    _inference_box(axp, W - 2 * M, para_h, paras)
    fig.savefig(path, dpi=SIMPLE_DPI, facecolor="white")
    plt.close(fig)
    _write_inference(Path(path).with_suffix(".txt"), f"ventilation - {cand.name} vs base - {SEASON_TXT[R.season]}", paras)
    return True


def fig_categories(R, path):
    names = [c[1] for c in CATEGORIES][::-1]
    names = ["extreme cold stress"] + [c[1] for c in CATEGORIES][::-1][:0] + [c[1] for c in CATEGORIES][::-1]
    cats = ["extreme cold stress", "very strong cold stress", "strong cold stress", "moderate cold stress", "slight cold stress", "no thermal stress",
            "moderate heat stress", "strong heat stress", "very strong heat stress", "extreme heat stress"]
    cats = [c for c in cats if R.cat_b.get(c, 0) > 0 or R.cat_c.get(c, 0) > 0]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(cats))
    ax.bar(x - 0.2, [100 * R.cat_b[c] / R.stats_b["area_m2"] for c in cats], 0.4, color="#8a9aa6", label="base")
    ax.bar(x + 0.2, [100 * R.cat_c[c] / R.stats_c["area_m2"] for c in cats], 0.4, color="#2b7bba", label=R.name)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" stress", "\nstress").replace("no thermal\nstress", "no thermal\nstress") for c in cats], fontsize=7)
    ax.set_ylabel("% of the walkable area")
    ax.legend(fontsize=8)
    ax.set_title(f"share of the area per UTCI category — {SEASON_TXT[R.season]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fig_overview(results, path):
    """all selected layouts: benefit by driver, per season"""
    names = sorted({r.name for r in results})
    seasons = [s for s in SEASONS if any(r.season == s for r in results)]
    fig, axs = plt.subplots(1, len(seasons), figsize=(6.5 * len(seasons) + 1, 4.8), squeeze=False)
    for ax, s in zip(axs[0], seasons):
        rs = {r.name: r for r in results if r.season == s}
        x = np.arange(len(rs))
        pos, neg = np.zeros(len(rs)), np.zeros(len(rs))
        for d in DRIVERS:
            v = np.array([rs[n].sgn * rs[n].contrib_mean[d] for n in rs])
            ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), color=COL[d], label=d, width=0.6)
            pos += np.where(v >= 0, v, 0)
            neg += np.where(v < 0, v, 0)
        ax.plot(x, [rs[n].sgn * rs[n].dU_mean for n in rs], "kD", ms=6, label="total")
        ax.set_xticks(x)
        ax.set_xticklabels(list(rs), rotation=20, fontsize=8)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_ylabel("°C of UTCI benefit (site mean)")
        ax.set_title(f"{SEASON_TXT[s]}: what produced the benefit", fontsize=10)
        ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ============================================================================= text: findings and mitigation
def fmt_area(a):
    return f"{a:,.0f} m²"


def where_text(centroid, layout, site_centre, flow=None):
    d = compass(centroid[0] - site_centre[0], centroid[1] - site_centre[1])
    nb = nearest_building(layout, centroid, flow)
    s = f"{d} part of the site"
    if nb is not None:
        s += f", {nb[1]:.0f} m from {nb[0]}"
    return s


def driver_sentence(contrib, sgn, total=None, drivers=DRIVERS):
    """'Tmrt -3.1 °C (82 %), wind ...' in order of importance, signed as UTCI change; the share is shown only for terms that go the same way as the total"""
    tot = sum(contrib[d] for d in drivers) if total is None else total
    items = sorted(drivers, key=lambda d: -abs(contrib[d]))
    parts = []
    for d in items:
        if abs(tot) > 1e-6 and contrib[d] * tot > 0:
            parts.append(f"{d} {contrib[d]:+.2f} °C ({100 * contrib[d] / tot:.0f} %)")
        elif abs(contrib[d]) >= 0.005:
            parts.append(f"{d} {contrib[d]:+.2f} °C (opposes)")
        else:
            parts.append(f"{d} {contrib[d]:+.2f} °C")
    return ", ".join(parts)


def explain_region(r, R, season):
    """one paragraph for a hot / cold region"""
    heat = season == "heat"
    ctx = r["context"]
    s = (f"{r['status'].upper()}: UTCI {r['u_base']:.1f} → {r['u_cand']:.1f} °C ({r['d_u']:+.1f} °C); {100 * r['resolved_share']:.0f} % of its area is no longer "
         f"{'above' if heat else 'below'} {R.thr:.1f} °C" + (" (cells just beyond the threshold flip with a small change)" if r["status"] in ("unchanged", "reduced") and r["resolved_share"] >= 0.5 else "") + ". Zone means, base → layout, and what each change did to UTCI: " + inputs_sentence(r["vals"]) + ".")
    cb, _vb = cause_of(r.get("cause_b"), heat)
    if cb:
        s += f" In the base it was {'hotter' if heat else 'colder'} than the rest of the site mainly because of {cb} ({(_vb if heat else -_vb):+.1f} °C)."
    det = []
    if abs(r["sunlit_c"] - r["sunlit_b"]) >= 0.1:
        det.append(f"sunlit share {r['sunlit_b']:.2f} → {r['sunlit_c']:.2f}")
    if abs(r["sky_open_c"] - r["sky_open_b"]) >= 0.05:
        det.append(f"sky view {r['sky_open_b']:.2f} → {r['sky_open_c']:.2f}")
    det.append(f"wind {r['wind_ratio_b']:.2f} → {r['wind_ratio_c']:.2f} of the reference wind")
    s += " (" + "; ".join(det) + ")."
    ch = []
    if ctx["trees_added_near"]:
        ch.append(f"{ctx['trees_added_near']} new trees within 25 m (+{ctx['canopy_added_near_m2']:.0f} m² canopy)")
    if ctx["trees_removed_near"]:
        ch.append(f"{ctx['trees_removed_near']} trees removed within 25 m")
    if ctx.get("trees_added_upwind"):
        ch.append(f"{ctx['trees_added_upwind']} new trees in the 70 m upwind corridor")
    if ctx.get("trees_removed_upwind"):
        ch.append(f"{ctx['trees_removed_upwind']} trees removed from the upwind corridor")
    dp = ctx["pond_area_near_cand"] - ctx["pond_area_near_base"]
    if abs(dp) > 30:
        ch.append(f"pond area within 25 m {ctx['pond_area_near_base']:.0f} → {ctx['pond_area_near_cand']:.0f} m²")
    dpu = ctx.get("pond_area_upwind_cand", 0) - ctx.get("pond_area_upwind_base", 0)
    if abs(dpu) > 30:
        ch.append(f"pond area upwind {ctx['pond_area_upwind_base']:.0f} → {ctx['pond_area_upwind_cand']:.0f} m²")
    dl = ctx["lawn_area_near_cand"] - ctx["lawn_area_near_base"]
    if abs(dl) > 30:
        ch.append(f"lawn within 25 m {ctx['lawn_area_near_base']:.0f} → {ctx['lawn_area_near_cand']:.0f} m²")
    for bid, sh in ctx["buildings_moved_near"]:
        ch.append(f"{bid} moved {sh:.1f} m")
    s += (" Layout change near it: " + "; ".join(ch) + ".") if ch else " No layout change within 25 m (the change comes from elsewhere in the flow or from the whole site)."
    return s


def cause_of(cc, heat, min_c=0.3):
    """dominant driver of the excess of a region over the rest of the site: (driver or None, its contribution in the stress direction)"""
    if not cc:
        return None, 0.0
    sc = {d: (cc["contrib"][d] if heat else -cc["contrib"][d]) for d in DRIVERS}
    d = max(DRIVERS, key=lambda k: sc[k])
    return (d if sc[d] >= min_c else None), sc[d]


def mitigation_for_region(r, R, season):
    """(cause sentence, action sentence) for a region that is still stressed in the layout"""
    heat = season == "heat"
    d, val = cause_of(r.get("cause_c"), heat)
    wr = r.get("wind_c", float("nan")) / R.u_ref if "wind_c" in r else float("nan")
    sun, sky, tm, ta = r.get("sunlit_c", float("nan")), r.get("sky_open_c", float("nan")), r.get("Tmrt_c", float("nan")), r.get("Ta_c", float("nan"))
    nb = r.get("near_building")
    if d is None:
        return ("no single driver explains it (every driver within 0.3 °C of the rest of the site)",
                "mixed: the whole site is stressed here rather than a pocket; use the site-wide levers (shade and ventilation for heat, shelter and sun for cold).")
    if heat:
        if d == "Tmrt" and sun > 0.5:
            return (f"radiant load, +{val:.1f} °C of UTCI over the rest of the site (sunlit {sun:.2f}, Tmrt {tm:.0f} °C vs air {ta:.0f} °C)",
                    "shade it: wide, tall crowns (radius 4-6 m, height 10-15 m) on the sun side of the spot, or put a building / canopy so that its shadow falls on it; lawn or water instead of dark paving.")
        if d == "Tmrt":
            return (f"radiation although shaded, +{val:.1f} °C (sunlit {sun:.2f}, sky view {sky:.2f}, Tmrt {tm:.0f} °C)",
                    "hot surfaces / trapped long-wave: open the sky a little (thin or lift the canopy), cool the surfaces (lawn, water, light paving).")
        if d == "wind":
            return (f"calm air, +{val:.1f} °C (wind {wr:.2f} of the reference)",
                    "open a ventilation path: no tree row or wall directly upwind, line a gap between buildings up with the wind"
                    + (f", shift the building upwind of it ({_bn(nb)}) sideways" if nb else "") + ".")
        if d == "Ta":
            return (f"warm air, +{val:.1f} °C (Ta {ta:.1f} °C)",
                    "water or lawn UPWIND of the spot with a clear path to it (cooled air travels with the wind) rather than at the spot itself.")
        return (f"humid air, +{val:.1f} °C", "keep ponds downwind of where people stay, or make them smaller / further away.")
    if d == "wind":
        return (f"wind chill, {val:.1f} °C colder than the rest of the site (wind {wr:.2f} of the reference)",
                "shelter: 2-3 staggered tree rows (crown radius 3-5 m) across the wind 10-25 m upwind of it; close channels between buildings that point along the wind.")
    if d == "Tmrt":
        return (f"radiation, {val:.1f} °C colder (sunlit {sun:.2f}, Tmrt {tm:.1f} °C vs air {ta:.1f} °C)",
                "keep winter sun access: no dense canopy on the sun side, a warm / low-albedo surface where the sun reaches it.")
    if d == "Ta":
        return (f"cold air, {val:.1f} °C below the rest (Ta {ta:.1f} °C)", "exposed to the approach air: shelter it from the approach direction so the air mixes less.")
    return (f"humidity, {val:.1f} °C", "small effect in the cold hour; check the numbers before acting.")


def build_findings(R, base, cand, grid):
    """list of strings (markdown) that explain one candidate / season"""
    sgn, heat = R.sgn, R.season == "heat"
    L = []
    obj = ""
    if R.obj_b is not None and R.obj_c is not None:
        obj = f" (objective, area-mean of the sample points: {R.obj_b:.2f} → {R.obj_c:.2f} °C, {R.obj_c - R.obj_b:+.2f})"
    L.append(f"**{SEASON_TXT[R.season].capitalize()}.** Site-mean UTCI {R.stats_b['UTCI_C']:.2f} → {R.stats_c['UTCI_C']:.2f} °C on the common grid{obj}; "
             f"category {category(R.stats_b['UTCI_C'])} → {category(R.stats_c['UTCI_C'])}. Where both layouts have open ground the mean change is "
             f"{R.dU_mean:+.2f} °C.")
    c = R.contrib_mean
    ranking = sorted(DRIVERS, key=lambda d: -sgn * c[d])
    top = ranking[0]
    pos = [d for d in DRIVERS if sgn * c[d] > 0.05]
    neg = [d for d in DRIVERS if sgn * c[d] < -0.05]
    L.append("*Site mean, inputs and their effect on UTCI* (base → layout; the effect is the part of the UTCI change that this input alone caused): " + inputs_sentence(R.site_vals) + ".")
    L.append(f"*Why:* the change of UTCI splits into  {driver_sentence(c, sgn, R.dU_mean)}.")
    if abs(R.dU_mean) < 0.15:
        L.append("The site mean hardly changes; the benefit, if any, is local (see the regions below).")
    else:
        word = "lowered" if R.dU_mean < 0 else "raised"
        s = f"The mean UTCI was {word} mainly by **{DRIVER_LONG[top]}**"
        if sgn * c[top] > 0:
            s += f" (benefit {sgn * c[top]:+.2f} °C of the net {sgn * R.dU_mean:+.2f} °C)."
        else:
            s += "."
        if neg:
            s += " Working against the benefit: " + ", ".join(f"{d} ({sgn * c[d]:+.2f} °C)" for d in neg) + "."
        L.append(s)
    if R.tmrt is not None and abs(R.dTmrt_mean) > 0.1:
        p = R.tmrt_parts_mean
        items = sorted(p.items(), key=lambda kv: -abs(kv[1]))
        L.append(f"*Inside Tmrt* (Δ Tmrt {R.dTmrt_mean:+.2f} °C): " + ", ".join(f"{k} {v:+.2f}" for k, v in items) +
                 f" °C. Regression of Tmrt on sunlit share, sky view and Ta explains R² = {R.tmrt['r2']:.2f}. Sunlit share {R.stats_b['sunlit']:.2f} → {R.stats_c['sunlit']:.2f}, "
                 f"sky view {R.stats_b['sky_open']:.2f} → {R.stats_c['sky_open']:.2f}.")
    v = R.vent
    L.append(f"*Ventilation* at 1.5 m: mean wind {R.stats_b['wind_ms']:.2f} → {R.stats_c['wind_ms']:.2f} m/s ({v['wind_ratio_b']:.2f} → {v['wind_ratio_c']:.2f} of the reference wind "
             f"{R.u_ref:.1f} m/s); calm area (< {R.wake_frac:g} of reference) {fmt_area(v['calm_area_b'])} → {fmt_area(v['calm_area_c'])} "
             f"({100 * v['calm_share_b']:.0f} → {100 * v['calm_share_c']:.0f} % of the open ground); longest wake {v['longest_wake_b']:.0f} → {v['longest_wake_c']:.0f} m.")
    if heat:
        wind_help = sgn * c["wind"]
        if wind_help > 0.15:
            L.append(f"→ Ventilation **improved** the heat case: it accounts for {wind_help:+.2f} °C of the benefit.")
        elif wind_help < -0.15:
            L.append(f"→ Ventilation got **worse** in the heat case ({wind_help:+.2f} °C); the benefit came from the other drivers.")
        else:
            L.append("→ Ventilation changed little in the heat case; the benefit does not come from wind.")
    else:
        wind_help = sgn * c["wind"]
        if wind_help > 0.15:
            L.append(f"→ Less wind at the pedestrian reduced the cold stress: wind accounts for {wind_help:+.2f} °C of the benefit (wind chill reduced).")
        elif wind_help < -0.15:
            L.append(f"→ More wind increased the cold stress ({wind_help:+.2f} °C); the other drivers compensated.")
        else:
            L.append("→ Wind chill hardly changed in the cold case; the benefit is from radiation / air temperature.")
    # regions
    L.append(f"*{'Hot' if heat else 'Cold'} spots of the base* (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C): {fmt_area(R.spot_area_b)} in the base, {fmt_area(R.spot_area_c)} in the layout "
             f"({R.spot_area_c - R.spot_area_b:+,.0f} m²); {len(R.regions)} region(s), {len(R.new_regions)} new region(s) appear.")
    return L


def mitigation_text(results, base, grids, a):
    """consolidated inference over all analysed layouts / seasons (list of markdown lines)"""
    out = []
    by = {s: [r for r in results if r.season == s] for s in SEASONS}
    for s in SEASONS:
        rs = by[s]
        if not rs:
            continue
        heat = s == "heat"
        sgn = -1.0 if heat else 1.0
        out += [f"### {SEASON_TXT[s].capitalize()}", ""]
        m = {d: float(np.mean([sgn * r.contrib_mean[d] for r in rs])) for d in DRIVERS}
        tot = float(np.mean([sgn * r.dU_mean for r in rs]))
        out += [f"**What worked.** Average over the {len(rs)} layouts: benefit {tot:+.2f} °C of site-mean UTCI = " +
                ", ".join(f"{d} {m[d]:+.2f}" for d in sorted(DRIVERS, key=lambda d: -abs(m[d]))) + " °C.", ""]
        lead = max(DRIVERS, key=lambda d: m[d])
        if heat:
            tips = {"Tmrt": "The heat stress is dissolved mainly by cutting the radiant load: keep shade (tree crowns, building shadow) on the walkable ground and do not lose canopy to the other objective.",
                    "wind": "The heat stress is dissolved mainly by ventilation: protect the open corridors along the wind and keep tree rows and walls out of the approach path of the hot spots.",
                    "Ta": "The heat stress is dissolved mainly by cooler air: evaporation (ponds, lawns) and canopy cooling upwind of the spots is working; keep a clear wind path from the water to the places that need it.",
                    "RH": "Humidity is the leading term: check that ponds do not raise RH where people stay."}
        else:
            tips = {"Tmrt": "The cold stress is reduced mainly by radiation: sun reaches the walkable ground; keep sun access when adding trees.",
                    "wind": "The cold stress is reduced mainly by wind shelter: keep the shelter, and keep it from closing the heat-hour ventilation (see the trade-off below).",
                    "Ta": "The cold stress is reduced mainly by warmer air (less mixing with the cold approach air, sheltered pockets).",
                    "RH": "Humidity is the leading term; its effect on cold stress is small, check the numbers before drawing conclusions."}
        out += [f"**Leading driver: {lead}.** " + tips[lead], ""]
        # what is left: area-weighted causes over all leftover regions
        causes, tot_area = {}, 0.0
        for r in rs:
            for reg in r.regions:
                if reg["status"] in ("unchanged", "worse", "reduced", "partly resolved"):
                    d, _ = cause_of(reg.get("cause_c"), heat)
                    causes[d or "mixed"] = causes.get(d or "mixed", 0.0) + reg["area"]
                    tot_area += reg["area"]
        if tot_area > 0:
            out += ["**What is left.** Cause of the remaining " + ("hot" if heat else "cold") + " spots, weighted by area over all layouts: " +
                    ", ".join(f"{k} {100 * v / tot_area:.0f} %" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) + ".", ""]
        out += ["**Per layout: what to try next.**", ""]
        for r in rs:
            left = [g for g in r.regions if g["status"] in ("unchanged", "worse", "reduced", "partly resolved")]
            left.sort(key=lambda g: -g["area"])
            big = [g for g in left if g["area"] >= 150.0 or g["status"] == "worse"][:4]
            n_res = sum(1 for g in r.regions if g["status"] == "resolved")
            head = f"- **{r.name}**: {n_res} of {len(r.regions)} base spots resolved, {len(left)} left ({sum(g['area'] for g in left):,.0f} m²)"
            if r.new_regions:
                head += f", {len(r.new_regions)} new ({sum(g['area'] for g in r.new_regions):,.0f} m²)"
            out.append(head + ".")
            seen = {}
            for g in big:
                cause, act = mitigation_for_region(g, r, s)
                if act in seen:
                    act = f"same action as {seen[act]}."
                else:
                    seen[act] = g["id"]
                out.append(f"  - {g['id']} ({g['area']:,.0f} m², {g['status']}, UTCI {g['u_cand']:.1f} °C): cause — {cause}. Try: {act}")
            if len(left) > len(big):
                rest = [g for g in left if g not in big]
                out.append(f"  - {len(rest)} smaller region(s) ({sum(g['area'] for g in rest):,.0f} m²): same levers as above.")
            if r.new_regions:
                dom = {}
                for g in r.new_regions:
                    dom[g.get("dominant", "?")] = dom.get(g.get("dominant", "?"), 0.0) + g["area"]
                nbs = sorted({g.get("near_building") for g in r.new_regions if g.get("near_building")})
                up = sum(g["context"].get("trees_added_upwind", 0) for g in r.new_regions)
                out.append(f"  - new stressed ground, worsened mainly by {', '.join(k for k, v in sorted(dom.items(), key=lambda kv: -kv[1]))}, next to {', '.join(map(str, nbs[:4]))}: "
                           f"check what was added upwind of it ({up} trees added in its upwind corridors) and what moved; it is the price of the change that helped elsewhere.")
            if heat:
                hot_w = [w for w in r.wakes_c if w["utci_mean"] > r.stats_c["UTCI_C"] + 0.3 and w["area"] >= 150.0]
                hot_w.sort(key=lambda w: -w["area"])
                for w in hot_w[:2]:
                    out.append(f"  - wake {w['id']} ({w['area']:,.0f} m², {w['length_m']:.0f} m long, UTCI {w['utci_mean']:.1f} °C vs site {r.stats_c['UTCI_C']:.1f}), trails {w['building']}"
                               f"{' (new in this layout)' if w.get('is_new') else ''}: shift {w['building']} off the wind line, narrow it across the wind, or leave a gap upwind.")
        out.append("")
    # trade-off
    if by["heat"] and by["cold"]:
        out += ["### Trade-off between the hot and the cold hour", ""]
        for h in by["heat"]:
            c_ = next((x for x in by["cold"] if x.name == h.name), None)
            if c_ is None:
                continue
            hw, cw = -h.contrib_mean["wind"], c_.contrib_mean["wind"]
            ht, ct = -h.contrib_mean["Tmrt"], c_.contrib_mean["Tmrt"]
            msg = (f"- **{h.name}**: heat benefit {-h.dU_mean:+.2f} °C (wind {hw:+.2f}, Tmrt {ht:+.2f}), cold benefit {c_.dU_mean:+.2f} °C (wind {cw:+.2f}, Tmrt {ct:+.2f}).")
            if hw < -0.1 and cw > 0.1:
                msg += " The windbreak helps the cold hour but costs ventilation in the heat hour: a belt with gaps towards the heat-wind direction would keep both."
            elif hw > 0.1 and cw < -0.1:
                msg += " The ventilation gained for the heat hour is paid for by more wind chill in the cold hour: shelter the cold-wind side separately."
            elif ht > 0.1 and ct < -0.1:
                msg += " Shade that cools the heat hour also lowers Tmrt in the cold hour: keep the shade where the heat spots are, and keep the sun side open in winter."
            elif ht > 0.1 and cw > 0.1:
                msg += " Shade for the heat hour and shelter for the cold hour pull in the same direction: trees do both here."
            else:
                msg += " No direct conflict between the two hours in the drivers."
            out.append(msg)
        out.append("")
    return out


RULES_TABLE = [
    ("Heat: Tmrt dominates the benefit, sunlit share fell", "Shade (tree crowns, building shadow) did the work", "More / larger crowns at the remaining sunlit spots; protect existing canopy"),
    ("Heat: wind dominates, calm area shrank", "Ventilation did the work", "Keep corridors along the heat wind open; no tree rows / walls directly upwind of spots"),
    ("Heat: Ta dominates (air cooler)", "Evaporation / canopy cooling reached the spots", "Water and lawn upwind of the spots, open path from water to spot"),
    ("Heat: wake region behind a building with UTCI above the site mean", "Wake traps heat", "Shift the building off the wind line, split / narrow it across the wind, leave a gap upwind"),
    ("Heat: RH term negative", "Humid air from ponds is carried onto people", "Ponds downwind of seating, or smaller ponds, or more evaporation distance"),
    ("Cold: wind dominates, wind ratio fell", "Windbreak / building shelter did the work", "Keep the belt; gaps only where heat ventilation needs them"),
    ("Cold: spot with wind ratio > 1.15", "Wind channel / exposure", "Staggered rows 10-25 m upwind across the wind; close channels aligned with the wind"),
    ("Cold: spot shaded and Tmrt < Ta", "Missing sun / radiant heat", "Sun access (no dense canopy on the sun side), warm surfaces"),
    ("A region got worse", "Something placed upwind or beside it", "Check the layout changes listed for the region and undo that one"),
]


# ============================================================================= report
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def md_zone_table(zones, site_vals):
    rows = []
    for z in zones:
        v = z.get("vals")
        if v is None:
            continue
        tag = "new" if (z["id"].startswith("n") or z.get("is_new")) else (z.get("status") or z.get("fate") or "")
        rows.append([f"{z['id']}" + ("" if z["id"][0] in "HCn" else (" (layout)" if z.get("is_new") else " (base)")), tag, f"{z['area']:.0f}", f"{v['UTCI']['b']:.1f} → {v['UTCI']['c']:.1f} ({v['UTCI']['d']:+.2f})"] +
                    [vtxt(v, d) for d in ORDER])
    sv = site_vals
    rows.append(["whole site", "", f"{sv['area']:.0f}", f"{sv['UTCI']['b']:.1f} → {sv['UTCI']['c']:.1f} ({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d) for d in ORDER])
    return md_table(["zone", "status", "area m²", "UTCI °C"] + [f"{d} {UNIT[d]}" for d in ORDER], rows)


def md_to_html(md, img_dir=None, embed=True):
    """small converter for the subset used here (headings, paragraphs, bullets, tables, images, **bold**, *italic*, `code`)"""
    try:
        import markdown as mdl
        html = mdl.markdown(md, extensions=["tables"])
    except Exception:
        html = _mini_md(md)
    if embed and img_dir is not None:
        tot = sum(f.stat().st_size for f in Path(img_dir).glob("figures/*.png"))
        if tot > 30e6:                                    # HD pictures: do not inline them (the html would be huge); link the files instead
            embed = False
    if embed and img_dir is not None:
        def rep(m):
            p = Path(img_dir) / m.group(1)
            if p.is_file():
                return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
            return m.group(0)
        html = re.sub(r'src="([^"]+\.png)"', rep, html)
    css = ("body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#1c2830;line-height:1.5}"
           "h1,h2,h3{color:#14323f}table{border-collapse:collapse;margin:10px 0;font-size:13px}td,th{border:1px solid #cfd6dc;padding:4px 8px;text-align:left}"
           "th{background:#eef2f5}img{max-width:100%;border:1px solid #dde3e8;margin:8px 0}code{background:#eef2f5;padding:1px 4px;border-radius:3px}"
           "li{margin:3px 0}@media(prefers-color-scheme:dark){body{background:#12181c;color:#dfe6ea}h1,h2,h3{color:#bcd6e2}th{background:#1d272d}td,th{border-color:#34424a}code{background:#1d272d}}")
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>UHI Pareto analysis</title><style>{css}</style></head><body>{html}</body></html>"


def _inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def _mini_md(md):
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            n = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{n}>{_inline(ln[n:].strip())}</h{n}>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            i -= 1
            t = "<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr>"
            for r in rows[2:]:
                t += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</table>")
        elif ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            i -= 1
            out.append("</ul>")
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


# ============================================================================= selection
def read_objectives(out_dir):
    p = Path(out_dir) / "objectives.csv"
    if not p.is_file():
        return {}
    rows = {}
    for r in csv.DictReader(p.open()):
        try:
            rows[r["name"]] = dict(f1=float(r["f1"]), f2=float(r["f2"]), utci_heat=float(r["utci_heat"]), utci_cold=float(r["utci_cold"]), status=r.get("status", "ok"))
        except (KeyError, ValueError):
            continue
    return rows


def pareto_front(obj):
    names = [n for n, r in obj.items() if r.get("status", "ok") == "ok"]
    front = []
    for n in names:
        a = obj[n]
        if not any((obj[m]["f1"] <= a["f1"] and obj[m]["f2"] <= a["f2"]) and (obj[m]["f1"] < a["f1"] or obj[m]["f2"] < a["f2"]) for m in names if m != n):
            front.append(n)
    return sorted(front, key=lambda n: obj[n]["f1"] + obj[n]["f2"])


def select_names(a, obj):
    if a.names:
        return list(a.names)
    t5 = Path(a.output_dir) / "top5.json"
    if t5.is_file():
        try:
            names = [r["name"] for r in json.loads(t5.read_text())]
            return names[:a.top]
        except Exception:
            pass
    if obj:
        return pareto_front(obj)[:a.top]
    raise SystemExit("no top5.json / objectives.csv found: give --names")


# ============================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder (contains input/, run/, output/)")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--input-dir", type=Path, help="folder with the layout jsons (default <root>/input)")
    ap.add_argument("--output-dir", type=Path, help="folder with top5.json / objectives.csv (default <root>/output)")
    ap.add_argument("--outdir", type=Path, help="where the analysis is written (default <output-dir>/analysis)")
    ap.add_argument("--base", default="individual_0", help="name of the base case (default individual_0 = base.json)")
    ap.add_argument("--names", nargs="+", help="layouts to analyse (default: top5.json, else the Pareto front)")
    ap.add_argument("--top", type=int, default=5, help="how many of top5.json / of the Pareto front")
    ap.add_argument("--season", choices=("heat", "cold", "both"), default="both")
    ap.add_argument("--cell", type=float, default=None, help="analysis grid cell in m (default about half the sample point spacing, 2-4 m)")
    ap.add_argument("--wake-frac", type=float, default=0.5, help="wake = wind below this share of the reference wind (default 0.5)")
    ap.add_argument("--hot-pct", type=float, default=80.0, help="heat spots: UTCI >= this percentile of the base (and >= 26 C)")
    ap.add_argument("--cold-pct", type=float, default=80.0, help="cold spots: UTCI <= the (100 - this) percentile of the base (and <= 9 C)")
    ap.add_argument("--u-ref", type=float, default=None, help="reference wind (m/s at 1.5 m) for the wake test; default = 90th percentile of the base wind field")
    ap.add_argument("--wind-factor", type=float, default=1.0, help="the factor compute_utci.py used on the CFD wind (default 1)")
    ap.add_argument("--min-region-m2", type=float, default=80.0, help="smaller regions are ignored")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--dpi", type=int, default=200, help="resolution of the two plain pictures (default 200 = about 3200 px wide)")
    ap.add_argument("--detailed", action="store_true", help="also draw the dense technical figures (maps, drivers, zone tables, categories); default: only the two plain pictures per layout and season")
    a = ap.parse_args(argv)
    t0 = time.time()
    root = a.root
    a.run_dir = a.run_dir or root / "run"
    a.input_dir = a.input_dir or root / "input"
    a.output_dir = a.output_dir or root / "output"
    outdir = a.outdir or a.output_dir / "analysis"
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    global WIND_FACTOR, SIMPLE_DPI
    WIND_FACTOR = a.wind_factor
    SIMPLE_DPI = a.dpi
    seasons = SEASONS if a.season == "both" else (a.season,)
    obj = read_objectives(a.output_dir)
    names = [n for n in select_names(a, obj) if n != a.base]
    log(f"base {a.base}; layouts {', '.join(names)}; seasons {', '.join(seasons)}")
    tn = {n: f"top_{i}_{n}" for i, n in enumerate(names, 1)}              # file names: top_1_<name>_..., top_2_<name>_... (rank = order of the list)
    log("files: " + ", ".join(tn.values()))

    results, grids, cases = [], {}, {}
    for s in seasons:
        try:
            base = Case(a.base, s, a.run_dir, a.input_dir)
        except Exception as e:
            log(f"base / {s}: {e} - season skipped")
            continue
        cases[("base", s)] = base
        log(f"{s}: utci_points.csv of the base: {base.point_note}")
        grid = Grid(base.layout.site, a.cell or float(np.clip(0.5 * base.spacing, 2.0, 4.0)))
        grids[s] = grid
        log(f"{s}: grid {grid.shape[1]} x {grid.shape[0]} cells of {grid.cell:.1f} m, {len(base.pts['x'])} sample points, base UTCI {base.summary.get('utci_C', float('nan')):.2f} °C")
        for n in names:
            try:
                cand = Case(n, s, a.run_dir, a.input_dir)
                R = analyse_pair(base, cand, grid, a)
                R.wake_frac = a.wake_frac
                log(f"{s} / {n}: ΔUTCI {R.dU_mean:+.2f} °C  (" + ", ".join(f"{d} {R.contrib_mean[d]:+.2f}" for d in DRIVERS) + f")  split error {R.split_err:.1e}")
                if not a.no_figures:
                    fig_simple_spots(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_1_{'hot' if s == 'heat' else 'cold'}_spots.png")
                    fig_simple_vent(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_2_ventilation.png")
                if not a.no_figures and a.detailed:
                    fig_maps(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_maps.png")
                    fig_drivers(R, grid, outdir / "figures" / f"{tn[n]}_{s}_drivers.png")
                    fig_wakes(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_wakes.png")
                    fig_zones(R, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_zones_spots.png", "spots")
                    fig_zones(R, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_zones_wakes.png", "wakes")
                    fig_categories(R, outdir / "figures" / f"{tn[n]}_{s}_categories.png")
                cases[(n, s)] = cand                       # a layout enters the report only when its numbers AND figures are complete
                results.append(R)
            except FileNotFoundError as e:
                log(f"{s} / {n}: skipped - {e}")
            except Exception as e:
                import traceback
                log(f"{s} / {n}: FAILED {type(e).__name__}: {e}")
                traceback.print_exc()
    if not results:
        log("nothing analysed")
        return 1
    if not a.no_figures and a.detailed:
        fig_overview(results, outdir / "figures" / "overview_drivers.png")

    # ------------------------------------------------------------------ report
    md = ["# Pareto analysis: why are the best layouts better than the base case?", "",
          f"Base case: `{a.base}`. Layouts analysed: " + ", ".join(f"`{n}`" for n in names) + ". Generated by `analyze_pareto.py` "
          f"({time.strftime('%Y-%m-%d %H:%M')}).", ""]
    md += ["## How to read this", "",
           "- **Benefit** is always in °C of UTCI in the direction that helps: heat hour = UTCI lower, cold hour = UTCI higher. Positive = good.",
           "- The change of UTCI is split **exactly** into four drivers (order-averaged substitution with the UTCI function): **Tmrt** (shade, sky view, surface temperatures), "
           "**wind** (ventilation / wind chill), **Ta** (air temperature), **RH** (humidity). The four add up to the total.",
           "- **Wake** = pedestrian-level wind below "
           f"{a.wake_frac:g} x the reference wind. **Hot / cold spots** = the worst parts of the base case (see each section for the threshold).",
           "- Values come from one hour per season at 1.5 m height; the statements are associations of layout changes with field changes.", ""]
    # objective table
    if obj:
        md += ["## Selected layouts", ""]
        rows = []
        for n in [a.base] + names:
            o = obj.get(n)
            if o:
                rows.append([n, f"{o['utci_heat']:.2f}", f"{o['utci_cold']:.2f}", f"{o['f1']:.2f}", f"{o['f2']:.2f}"])
        md += md_table(["layout", "UTCI heat °C", "UTCI cold °C", "f1 (heat − 26)", "f2 (9 − cold)"], rows) + [""]
    if (outdir / "figures" / "overview_drivers.png").is_file():
        md += ["![overview](figures/overview_drivers.png)", ""]
    # per season overview table
    for s in seasons:
        rs = [r for r in results if r.season == s]
        if not rs:
            continue
        md += [f"## Overview — {SEASON_TXT[s]}", ""]
        rows = []
        for r in rs:
            c = r.contrib_mean
            rows.append([r.name, f"{r.stats_b['UTCI_C']:.2f} → {r.stats_c['UTCI_C']:.2f}", f"{r.sgn * r.dU_mean:+.2f}"] +
                        [f"{r.sgn * c[d]:+.2f}" for d in DRIVERS] +
                        [f"{r.stats_b['wind_ms'] / r.u_ref:.2f} → {r.stats_c['wind_ms'] / r.u_ref:.2f}",
                         f"{r.vent['calm_area_b']:.0f} → {r.vent['calm_area_c']:.0f}",
                         f"{r.spot_area_b:.0f} → {r.spot_area_c:.0f}"])
        md += md_table(["layout", "site mean UTCI °C", "benefit °C"] + [f"{d} °C" for d in DRIVERS] +
                       ["wind / reference", "calm area m²", f"{'hot' if s == 'heat' else 'cold'}-spot area m²"], rows) + [""]
        md += ["Actual change of the inputs (mean over the ground that is open in both layouts) and, in brackets, the change of UTCI it caused (°C, − = cooler):", ""]
        rows = []
        for r in rs:
            sv = r.site_vals
            rows.append([r.name, f"{sv['UTCI']['b']:.2f} → {sv['UTCI']['c']:.2f} ({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d) for d in ORDER])
        md += md_table(["layout", "UTCI °C"] + [f"{d} {UNIT[d]}: base → layout (Δ) → effect" for d in ORDER], rows) + [""]
        L0 = rs[0]
        md += [f"Spot threshold: UTCI {'≥' if s == 'heat' else '≤'} {L0.thr:.1f} °C; reference wind {L0.u_ref:.2f} m/s at 1.5 m (90th percentile of the base; 10 m wind of the hour {L0.u10:.1f} m/s); analysis grid {grids[s].cell:.1f} m.", ""]
    # per layout
    for n in names:
        rs = [r for r in results if r.name == n]
        if not rs:
            continue
        md += [f"## Layout `{n}`", ""]
        d0 = rs[0].diff
        md += [f"*Layout changes vs base:* trees {d0['trees_base']} → {d0['trees_cand']} ({len(d0['trees_added'])} new, {len(d0['trees_removed'])} removed), canopy "
               f"{d0['canopy_base']:.0f} → {d0['canopy_cand']:.0f} m², mean tree height {d0['mean_height_base']:.1f} → {d0['mean_height_cand']:.1f} m, ponds {d0['ponds_base']} → "
               f"{d0['ponds_cand']} ({d0['pond_area_base']:.0f} → {d0['pond_area_cand']:.0f} m²), lawns {d0['lawn_area_base']:.0f} → {d0['lawn_area_cand']:.0f} m²; buildings moved: " +
               (", ".join(f"{b['id']} {b['shift']:.1f} m" for b in d0["buildings"] if b["shift"] > 0.5) or "none") + ".", ""]
        for R in rs:
            s = R.season
            base = cases[("base", s)]
            cand = cases[(n, s)]
            md += [f"### {n} — {SEASON_TXT[s]}", ""]
            for line in build_findings(R, base, cand, grids[s]):
                md += [line, ""]
            fg = outdir / "figures"
            kind = "hot" if s == "heat" else "cold"
            for tag, ttl in ((f"1_{kind}_spots", f"{kind} spots"), ("2_ventilation", "ventilation")):
                if (fg / f"{tn[n]}_{s}_{tag}.png").is_file():
                    md += [f"![{ttl}](figures/{tn[n]}_{s}_{tag}.png)", ""]
                    if (fg / f"{tn[n]}_{s}_{tag}.txt").is_file():
                        md += [" ".join((fg / f"{tn[n]}_{s}_{tag}.txt").read_text().splitlines()[2:]), ""]
            for stem, ttl in (("maps", "maps"), ("drivers", "drivers"), ("zones_spots", "zone values of the spots"), ("wakes", "wakes"), ("zones_wakes", "zone values of the wakes"), ("categories", "categories")):
                if (fg / f"{tn[n]}_{s}_{stem}.png").is_file():
                    md += [f"![{ttl}](figures/{tn[n]}_{s}_{stem}.png)", ""]
            centre = grids[s].site_mask
            sc = (float(grids[s].X[centre].mean()), float(grids[s].Y[centre].mean()))
            if R.regions or R.new_regions:
                md += [f"**Zone-averaged inputs of the {'hot' if s == 'heat' else 'cold'} spots** — each cell: base → layout (change) → effect of that change on UTCI in °C (− = cooler, + = warmer)", ""]
                md += md_zone_table(R.regions + R.new_regions, R.site_vals) + [""]
            if R.regions:
                md += [f"**{'Hot' if s == 'heat' else 'Cold'} spots of the base and what became of them**", ""]
                for r in R.regions:
                    md += [f"- **{r['id']}** ({fmt_area(r['area'])}, {where_text(r['centroid'], cand.layout, sc, R.flow)}): " + explain_region(r, R, s)]
                md += [""]
            if R.new_regions:
                md += ["**New stressed regions created by the layout**", ""]
                for r in R.new_regions:
                    ctx = r["context"]
                    extra = f" Dominant driver of the worsening: {r.get('dominant', 'n/a')}." if "dominant" in r else ""
                    md += [f"- **{r['id']}** ({fmt_area(r['area'])}, {where_text(r['centroid'], cand.layout, sc, R.flow)}): UTCI {r['vals']['UTCI']['b']:.1f} → {r['u_cand']:.1f} °C. "
                           f"{inputs_sentence(r['vals'])}. Sunlit {r['sunlit_c']:.2f}.{extra} Layout nearby: {ctx['trees_added_near']} new trees within 25 m, {ctx.get('trees_added_upwind', 0)} upwind."]
                md += [""]
            if R.wakes_b or R.wakes_c:
                md += ["**Wake regions (calm air at 1.5 m)**", ""]
                rows = []
                for w in R.wakes_b:
                    rows.append([f"{w['id']} (base)", fmt_area(w["area"]), f"{w['length_m']:.0f} × {w['width_m']:.0f} m", w["building"] or "-",
                                 f"{w['utci_mean']:.1f}", w["fate"] + (f" → {w['cand_match']} {w['cand_area']:.0f} m²" if w["cand_match"] else ""),
                                 f"{w.get('d_wind', float('nan')):+.2f}" if "d_wind" in w else "-",
                                 f"{w['d_u']:+.2f}" if "d_u" in w else "-"])
                for w in R.wakes_c:
                    if w.get("is_new"):
                        rows.append([f"{w['id']} (layout)", fmt_area(w["area"]), f"{w['length_m']:.0f} × {w['width_m']:.0f} m", w["building"] or "-", f"{w['utci_mean']:.1f}", "new", "-", "-"])
                md += md_table(["wake", "area", "length × width (along × across wind)", "trails building", "UTCI °C", "fate", "Δ wind m/s", "Δ UTCI °C"], rows) + [""]
                md += ["**Zone-averaged inputs of the wakes (same footprint in both layouts)** — each cell: base → layout (change) → effect on UTCI in °C", ""]
                md += md_zone_table(R.wakes_b + [w for w in R.wakes_c if w.get("is_new")], R.site_vals) + [""]
    # mitigation
    md += ["## Mitigation inference", ""]
    md += mitigation_text(results, cases.get(("base", seasons[0])), grids, a)
    md += ["", "### Rules behind these sentences (the template)", ""]
    md += md_table(["observation", "reading", "mitigation"], [list(r) for r in RULES_TABLE]) + [""]
    md += ["## Method and limits", "",
           f"- Sample points of `utci_points.csv` (z = 1.5 m) are interpolated linearly on a {', '.join(f'{grids[s].cell:.1f} m' for s in grids)} grid; cells inside buildings (of each layout), outside the site, "
           "or farther than 2.2 point spacings from a sample are left out. Changes are computed only where both layouts have open ground.",
           "- The driver split recomputes UTCI (pythermalcomfort, same call as `compute_utci.py`) for every combination of base / layout values of Tmrt, wind, Ta, RH "
           "(16 evaluations per cell) and averages over the orders: exact and additive. Recomputed UTCI of the base differs from the stored one by "
           + ", ".join(f"{np.mean([r.csv_err for r in results if r.season == s]):.2f} °C ({s})" for s in seasons if any(r.season == s for r in results)) + " (rounding / interpolation).",
           "- Zone values (Ta, Tmrt, RH, wind) are area means over the zone's cells, in the base and in the layout at the same footprint; the 'effect' next to each is the part of the "
           "UTCI change that this input alone caused (exact split), so a large change of an input with a small effect (or the reverse) shows how sensitive UTCI is to it in that zone.",
           "- The split of Tmrt is a linear regression on the sunlit share, the sky view and Ta; the remainder is attributed to surface temperatures / long-wave exchange.",
           "- One hour per season, steady flow field, one height. Wakes shown are pedestrian-level calm regions, not the full 3-D recirculation zones.",
           "- Layout context numbers (trees within 25 m, 70 m upwind corridor, ponds) show what changed near a region; they do not prove that the change caused the field change.", ""]
    md_text = "\n".join(md)
    hz = (cases[("base", seasons[0])].summary or {}).get("height_m") if ("base", seasons[0]) in cases else None
    if hz and abs(float(hz) - 1.5) > 1e-9:                    # the text says 1.5 m; the run used another height (compute_utci.py --height)
        md_text = md_text.replace("1.5 m", f"{float(hz):g} m")
    (outdir / "analysis_report.md").write_text(md_text)
    if not a.no_html:
        (outdir / "analysis_report.html").write_text(md_to_html(md_text, outdir))

    # ------------------------------------------------------------------ tables / json
    def drow(r):
        return dict(layout=r.name, season=r.season, utci_base=r.stats_b["UTCI_C"], utci_layout=r.stats_c["UTCI_C"], d_utci=r.dU_mean,
                    **{f"contrib_{d}": r.contrib_mean[d] for d in DRIVERS}, split_error=r.split_err,
                    wind_ratio_base=r.vent["wind_ratio_b"], wind_ratio_layout=r.vent["wind_ratio_c"], calm_area_base=r.vent["calm_area_b"], calm_area_layout=r.vent["calm_area_c"],
                    spot_threshold=r.thr, spot_area_base=r.spot_area_b, spot_area_layout=r.spot_area_c, sunlit_base=r.stats_b["sunlit"], sunlit_layout=r.stats_c["sunlit"],
                    sky_base=r.stats_b["sky_open"], sky_layout=r.stats_c["sky_open"], Tmrt_base=r.stats_b["Tmrt_C"], Tmrt_layout=r.stats_c["Tmrt_C"],
                    Ta_base=r.stats_b["Tair_C"], Ta_layout=r.stats_c["Tair_C"], RH_base=r.stats_b["RH_pct"], RH_layout=r.stats_c["RH_pct"],
                    **{f"{d}_change_site_mean": r.site_vals[d]["d"] for d in DRIVERS})

    def valcols(v):
        """flat columns: UTCI base / layout / change, and for Ta, Tmrt, RH, wind: base, layout, change, effect on UTCI"""
        if v is None:
            return {}
        o = dict(utci_base=v["UTCI"]["b"], utci_layout=v["UTCI"]["c"], utci_change=v["UTCI"]["d"])
        for d in ORDER:
            u = {"Ta": "C", "Tmrt": "C", "RH": "pct", "wind": "ms"}[d]
            o.update({f"{d}_base_{u}": v[d]["b"], f"{d}_layout_{u}": v[d]["c"], f"{d}_change_{u}": v[d]["d"], f"{d}_effect_on_UTCI_C": v[d]["eff"]})
        return o

    def write_csv(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    write_csv(outdir / "tables" / "drivers.csv", [drow(r) for r in results])
    reg_rows, wake_rows = [], []
    for r in results:
        for g in r.regions:
            reg_rows.append(dict(layout=r.name, season=r.season, region=g["id"], area_m2=g["area"], x=g["centroid"][0], y=g["centroid"][1], status=g["status"], utci_base=g["u_base"],
                                 utci_layout=g["u_cand"], resolved_share=g["resolved_share"], dominant=g["dominant"],
                                 **{f"contrib_{d}": g["contrib"][d] for d in DRIVERS}, trees_added_near=g["context"]["trees_added_near"],
                                 trees_added_upwind=g["context"].get("trees_added_upwind", 0)))
        for w in r.wakes_b:
            wake_rows.append(dict(layout=r.name, season=r.season, wake=w["id"], which="base", area_m2=w["area"], length_m=w["length_m"], width_m=w["width_m"], building=w["building"],
                                  utci=w["utci_mean"], fate=w["fate"]))
        for w in r.wakes_c:
            wake_rows.append(dict(layout=r.name, season=r.season, wake=w["id"], which="layout", area_m2=w["area"], length_m=w["length_m"], width_m=w["width_m"], building=w["building"],
                                  utci=w["utci_mean"], fate="new" if w.get("is_new") else ""))
    zone_rows = []
    for r in results:
        base_ = dict(layout=r.name, season=r.season)
        zone_rows.append(dict(base_, kind="whole site", zone="site", status="", area_m2=r.site_vals["area"], **valcols(r.site_vals)))
        for g in r.regions:
            zone_rows.append(dict(base_, kind="spot of the base", zone=g["id"], status=g["status"], area_m2=g["area"], **valcols(g["vals"])))
        for g in r.new_regions:
            zone_rows.append(dict(base_, kind="new stressed region", zone=g["id"], status="new", area_m2=g["area"], **valcols(g["vals"])))
        for w in r.wakes_b:
            zone_rows.append(dict(base_, kind="wake of the base", zone=w["id"], status=w["fate"], area_m2=w["area"], **valcols(w["vals"])))
        for w in r.wakes_c:
            if w.get("is_new"):
                zone_rows.append(dict(base_, kind="new wake", zone=w["id"], status="new", area_m2=w["area"], **valcols(w["vals"])))
    write_csv(outdir / "tables" / "zones.csv", zone_rows)
    write_csv(outdir / "tables" / "regions.csv", reg_rows)
    write_csv(outdir / "tables" / "wakes.csv", wake_rows)
    summ = dict(base=a.base, layouts=names, seasons=list(seasons), generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                results=[dict(drow(r), site_vals=r.site_vals, regions=[{k: v for k, v in g.items() if k not in ("mask", "context")} for g in r.regions]) for r in results])
    (outdir / "analysis_summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    log(f"done in {time.time() - t0:.0f} s -> {outdir}/analysis_report.md" + ("" if a.no_html else " and .html"))
    return 0


if __name__ == "__main__":
    sys.exit(main())









# #!/usr/bin/env python3
# """
# analyze_pareto.py  --  why are the best (Pareto) layouts better than the base case?   (post-processing, run it after the optimisation)

# Reads what the optimisation already wrote (nothing is simulated again):

#     output/top5.json  (or the Pareto front of output/objectives.csv)      which layouts to analyse
#     run/<name>/<heat|cold>/temperature_humidity/postProcessing/utci_points.csv + utci.json      pedestrian-height results of every layout
#     run/<name>/<season>/temperature_humidity/weatherData/target.json                            wind / sun / air of the hour
#     input/<name>.json                                                                           the layouts (buildings, trees, ponds, lawns)

# and compares every selected layout with the BASE case (individual_0 = base.json) for the heat hour and for the cold hour:

#   1. HOW MUCH, WHERE      UTCI maps of base / layout / difference on a common grid, category-area table, hot spots and cold spots.
#   2. WHY  (which driver)  The UTCI is a function of four things at the pedestrian: mean radiant temperature Tmrt, wind speed, air temperature Ta,
#                           relative humidity RH.  The change of UTCI is split EXACTLY into the four (Shapley / order-averaged substitution
#                           with the UTCI function itself, so the four add up to the change):  Tmrt (shade, sky view, hot / cool surfaces),
#                           wind (ventilation), Ta (evaporative / shade cooling of the air), RH (humidity).  Tmrt is split further, by a regression on the
#                           sunlit share, the sky view and Ta, into shade, sky view, air and "surfaces / long-wave".
#   3. WAKES, VENTILATION   Calm (wake) regions = wind below `--wake-frac` x the reference wind: area, length, width, which building they trail, what
#                           changed.  Mean wind ratio, calm share, fast channels.
#   4. HOT / COLD SPOTS     Regions of the base case where the stress is worst (heat: UTCI >= the `--hot-pct` percentile and >= 26 C; cold:
#                           <= the (100-`--cold-pct`) percentile and <= 9 C): were they resolved, reduced, still there?  New ones?  For every region the
#                           drivers and the layout changes (trees, ponds, lawns, building shifts) next to / upwind of it.
#   5. MITIGATION           Rule-based reading of the numbers: what worked, what is left, what to try next, and the trade-off between the hot and the cold hour.
#                           The rules are listed in the report, so you can see why each sentence is there.

# Output (default output/analysis/):  analysis_report.md  +  analysis_report.html (one file, pictures inside),  figures/*.png,  analysis_summary.json,
#                                     tables/*.csv

#     python3 python/analyze_pareto.py                                  # top5.json of ./output, base = individual_0, all seasons
#     python3 python/analyze_pareto.py --top 3 --season heat            # the first 3, heat hour only
#     python3 python/analyze_pareto.py --names offspring_10 individual_2 --base individual_0
#     python3 python/analyze_pareto.py --root /data/UHI_optimization_template_271026_scotch --outdir /tmp/analysis
#     python3 python/analyze_pareto.py --synthetic-demo ...             (see make_synthetic_demo.py: a made-up project to try the analyzer)

# Limits (also printed in the report): one hour per season, a steady frozen-flow run, results at ONE height (1.5 m): wakes and ventilation are those of the
# pedestrian level; a statement like "tree shade lowered UTCI" is an association between the layout change and the field change, not a controlled experiment.
# """
# import argparse
# import base64
# import csv
# import json
# import math
# import re
# import sys
# import time
# import warnings
# from pathlib import Path

# import numpy as np

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt                                   # noqa: E402
# from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
# from matplotlib.patches import Circle, Polygon as MPoly            # noqa: E402

# from scipy import ndimage as ndi                                  # noqa: E402
# from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator   # noqa: E402
# from scipy.spatial import cKDTree                                 # noqa: E402

# import shapely                                                    # noqa: E402
# from shapely.geometry import Point, Polygon                       # noqa: E402
# from shapely.ops import unary_union                               # noqa: E402

# warnings.filterwarnings("ignore")
# ROOT = Path(__file__).resolve().parent.parent

# SEASONS = ("heat", "cold")
# SEASON_TXT = {"heat": "heat-stress hour", "cold": "cold-stress hour"}
# CATEGORIES = [(46.0, "extreme heat stress"), (38.0, "very strong heat stress"), (32.0, "strong heat stress"),
#               (26.0, "moderate heat stress"), (9.0, "no thermal stress"), (0.0, "slight cold stress"),
#               (-13.0, "moderate cold stress"), (-27.0, "strong cold stress"), (-40.0, "very strong cold stress")]
# DRIVERS = ("Tmrt", "wind", "Ta", "RH")
# DRIVER_COL = {"Tmrt": "Tmrt_C", "wind": "wind_ms", "Ta": "Tair_C", "RH": "RH_pct"}
# DRIVER_LONG = {"Tmrt": "mean radiant temperature (shade / sky view / hot surfaces)", "wind": "wind speed (ventilation)",
#                "Ta": "air temperature (evaporation, shade, mixing)", "RH": "relative humidity"}
# COL = {"Tmrt": "#d9822b", "wind": "#2b7bba", "Ta": "#c0392b", "RH": "#4f9d69", "total": "#1c2830"}
# INK, MUTED = "#1c2830", "#6d7780"
# FIELDS = ("Tair_C", "RH_pct", "wind_ms", "Tmrt_C", "UTCI_C", "sunlit", "sky_open")

# RAINBOW = LinearSegmentedColormap.from_list("rainbow_desaturated", [
#     (0.000, (0.278431, 0.278431, 0.858824)), (0.143, (0.000000, 0.000000, 0.360784)), (0.285, (0.000000, 1.000000, 1.000000)),
#     (0.429, (0.000000, 0.501961, 0.000000)), (0.571, (1.000000, 1.000000, 0.000000)), (0.714, (1.000000, 0.380392, 0.000000)),
#     (0.857, (0.419608, 0.000000, 0.000000)), (1.000, (0.878431, 0.301961, 0.301961))])


# def category(v):
#     for lo, name in CATEGORIES:
#         if v > lo:
#             return name
#     return "extreme cold stress"


# def log(msg):
#     print(f"[analyze] {msg}", flush=True)


# # ============================================================================= reading the project
# def _xy(seq):
#     out = []
#     for p in seq or []:
#         if isinstance(p, dict):
#             out.append((float(p["x"]), float(p["y"] if "y" in p else p["z"])))      # osm buildings store the plan y as z
#         else:
#             out.append((float(p[0]), float(p[1])))
#     return out


# def _poly(seq):
#     pts = _xy(seq)
#     if len(pts) < 3:
#         return None
#     g = Polygon(pts)
#     if not g.is_valid:
#         g = g.buffer(0)
#     if g.is_empty:
#         return None
#     if g.geom_type != "Polygon":
#         g = max(getattr(g, "geoms", []), key=lambda q: q.area)
#     return g


# class Layout:
#     """site, buildings (id -> polygon, height), trees, ponds, lawns of one layout file"""

#     def __init__(self, path):
#         J = json.loads(Path(path).read_text())
#         J = J.get("siteOnly_json", J)
#         self.path = str(path)
#         self.site = _poly((J.get("site") or {}).get("coordinates"))
#         self.buildings, self.height = {}, {}
#         for b in J.get("buildings") or []:
#             vs = (b.get("dimensions") or {}).get("baseVertices") or (b.get("geometry") or {}).get("footprint")
#             g = _poly(vs)
#             if g is not None:
#                 self.buildings[b["id"]] = g
#                 self.height[b["id"]] = float(b.get("height") or (b.get("dimensions") or {}).get("height") or 0.0)
#         osm = J.get("osmBuildings") or J.get("osm_buildings") or []
#         for k, b in enumerate(osm):
#             g = _poly((b.get("dimensions") or {}).get("baseVertices") or (b.get("geometry") or {}).get("footprint") or b.get("coordinates"))
#             if g is not None:
#                 self.buildings[f"osm_{k}"] = g
#                 self.height[f"osm_{k}"] = float(b.get("height") or 10.0)
#         self.trees = []
#         for t in J.get("trees") or []:
#             r = (t.get("dimensions") or {}).get("crownRadius") or 0.5 * float(t.get("crownDiameter") or 4.0)
#             self.trees.append((str(t.get("id")), float(t["position"]["x"]), float(t["position"]["y"]), float(r), float(t.get("height") or 8.0)))
#         self.ponds = [g for g in (_poly(p.get("coordinates")) for p in J.get("ponds") or []) if g is not None]
#         self.lawns = [g for g in (_poly(p.get("coordinates")) for p in J.get("grassAreas") or []) if g is not None]
#         self.pond_ids = [p.get("id") for p in J.get("ponds") or []]

#     def building_union(self):
#         return unary_union(list(self.buildings.values())) if self.buildings else None


# def find_pp(run_dir, name, season):
#     """(postProcessing folder, job folder) of one layout / season, or None"""
#     job = Path(run_dir) / name / season
#     for pp in (job / "temperature_humidity" / "postProcessing", job / "postProcessing"):
#         if (pp / "utci_points.csv").is_file():
#             return pp, job
#     hits = sorted(job.glob("*/postProcessing/utci_points.csv"))
#     if hits:
#         return hits[0].parent, job
#     return None


# def read_points(csv_path, height=None):
#     """utci_points.csv -> dict of arrays; one row per (x, y).  Duplicated (x, y) rows (several heights, or the same point written twice) are collapsed:
#     the row whose z is closest to `height` is kept, else the rows are averaged.  Returns (points, note)"""
#     d = np.genfromtxt(csv_path, delimiter=",", names=True)
#     names = d.dtype.names
#     out = {k: np.atleast_1d(np.asarray(d[k], float)) for k in ("x", "y", "z") + FIELDS if k in names}
#     for k in FIELDS:
#         out.setdefault(k, np.full(len(out["x"]), np.nan))
#     n0 = len(out["x"])
#     xy = np.round(np.column_stack([out["x"], out["y"]]), 3)
#     _, first, inv, cnt = np.unique(xy, axis=0, return_index=True, return_inverse=True, return_counts=True)
#     inv = np.asarray(inv).ravel()
#     note = f"{n0} rows, {len(cnt)} distinct (x, y)"
#     if len(cnt) < n0:
#         z = out.get("z")
#         zl = np.unique(np.round(z, 3)) if z is not None else np.array([])
#         if z is not None and len(zl) > 1:
#             h = float(height) if height is not None else float(zl[np.argmin(np.abs(zl - 1.5))])
#             order = np.lexsort((np.abs(z - h), inv))                  # per (x, y): the row closest to the wanted height first
#             pick = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
#             out = {k: v[pick] for k, v in out.items()}
#             note += f"; z levels {', '.join(f'{v:g}' for v in zl[:6])} m -> kept the rows nearest z = {h:g} m"
#         else:
#             avg = {}
#             for k, v in out.items():
#                 sm = np.zeros(len(cnt))
#                 np.add.at(sm, inv, np.where(np.isfinite(v), v, 0.0))
#                 ct = np.zeros(len(cnt))
#                 np.add.at(ct, inv, np.isfinite(v).astype(float))
#                 avg[k] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
#             out = avg
#             note += "; duplicated points averaged"
#     return out, note


# def read_boundary(job, pp):
#     """boundary of the hour (wind speed / direction, air, sun): weatherData/target.json of the job"""
#     for p in (pp.parent / "weatherData" / "target.json", job / "weatherData" / "target.json", job / "temperature_humidity" / "weatherData" / "target.json"):
#         if p.is_file():
#             try:
#                 return json.loads(p.read_text()).get("boundary", {})
#             except Exception:
#                 pass
#     return {}


# def merge_near_points(pts, area):
#     """Sample points that lie almost on top of each other (several points per CFD cell, or the same cell written with tiny offsets) make the
#     spacing estimate collapse to a few cm; then every map keeps only the ground within ~2 x that distance of a point (a few % of the site) and
#     all areas / means / spots are computed on that sliver.  When most points have a neighbour much closer than sqrt(area / N) (the smallest
#     spacing the points can really have), every group of such points is replaced by one point: mean position, mean of every field.
#     Returns (pts, note)"""
#     n = len(pts["x"])
#     if n < 6 or not area or area <= 0:
#         return pts, ""
#     from scipy.sparse import coo_matrix
#     from scipy.sparse.csgraph import connected_components
#     xy = np.column_stack([pts["x"], pts["y"]])
#     tree = cKDTree(xy)
#     s0 = float(np.sqrt(area / n))
#     if np.median(tree.query(xy, k=2)[0][:, 1]) >= 0.3 * s0:
#         return pts, ""
#     pairs = tree.query_pairs(0.3 * s0, output_type="ndarray")
#     if len(pairs) == 0:
#         return pts, ""
#     g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
#     k, lab = connected_components(g, directed=False)
#     out = {}
#     for key, v in pts.items():
#         v = np.asarray(v, float)
#         ok = np.isfinite(v)
#         sm = np.bincount(lab, weights=np.where(ok, v, 0.0), minlength=k)
#         ct = np.bincount(lab, weights=ok.astype(float), minlength=k)
#         out[key] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
#     return out, f"; {n} points lie in clusters (several within {0.3 * s0:.2f} m), merged to {k} locations"


# class Case:
#     """one layout in one season"""

#     def __init__(self, name, season, run_dir, in_dir):
#         self.name, self.season = name, season
#         hit = find_pp(run_dir, name, season)
#         if hit is None:
#             raise FileNotFoundError(f"no utci_points.csv for {name} / {season} under {run_dir}")
#         self.pp, self.job = hit
#         self.summary = json.loads((self.pp / "utci.json").read_text()) if (self.pp / "utci.json").is_file() else {}
#         self.pts, self.point_note = read_points(self.pp / "utci_points.csv", self.summary.get("height_m"))
#         self.boundary = read_boundary(self.job, self.pp)
#         lj = Path(in_dir) / f"{name}.json"
#         if not lj.is_file():
#             c = sorted((self.job / "momentum").glob("*.json"))
#             lj = c[0] if c else None
#         if lj is None:
#             raise FileNotFoundError(f"no layout json for {name}")
#         self.layout = Layout(lj)
#         self.pts, merged_note = merge_near_points(self.pts, float(self.layout.site.area))
#         if merged_note:
#             self.point_note += merged_note
#             print(f"[analyze] {name}/{season}: {self.point_note}", file=sys.stderr)
#         xy = np.column_stack([self.pts["x"], self.pts["y"]])
#         d = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.array([0.0])
#         self.spacing = float(1.8 * np.median(d))
#         if not self.spacing > 0:
#             raise ValueError(f"{name}/{season}: the sample points of utci_points.csv have no spacing ({self.point_note}); check the file")

#     def flow(self):
#         """unit vector of the wind (towards where it blows), model frame"""
#         f = self.boundary.get("wind_flow_unit_vector_xy_model")
#         if f and len(f) == 2:
#             n = math.hypot(f[0], f[1])
#             if n > 1e-9:
#                 return np.array([f[0] / n, f[1] / n])
#         wd = self.boundary.get("wind_from_deg_model")
#         if wd is None:
#             return None
#         return np.array([-math.sin(math.radians(wd)), -math.cos(math.radians(wd))])


# # ============================================================================= grid
# class Grid:
#     def __init__(self, site, cell):
#         x0, y0, x1, y1 = site.bounds
#         self.cell = float(cell)
#         self.x = np.arange(x0 + 0.5 * cell, x1, cell)
#         self.y = np.arange(y0 + 0.5 * cell, y1, cell)
#         self.X, self.Y = np.meshgrid(self.x, self.y)
#         self.shape = self.X.shape
#         self.extent = (x0, x0 + self.cell * len(self.x), y0, y0 + self.cell * len(self.y))
#         self.area = self.cell ** 2
#         self.site_mask = shapely.contains_xy(site, self.X, self.Y)


# def rasterise(case, grid, gap_factor=2.2):
#     """fields of one case on the grid (linear interpolation of the sample points; NaN outside the site, in buildings and farther than
#     gap_factor x the point spacing from every sample)"""
#     p = case.pts
#     xy = np.column_stack([p["x"], p["y"]])
#     ok = np.isfinite(p["UTCI_C"])
#     xy = xy[ok]
#     tree = cKDTree(xy)
#     near = tree.query(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))[0].reshape(grid.shape)
#     bu = case.layout.building_union()
#     inb = shapely.contains_xy(bu, grid.X, grid.Y) if bu is not None else np.zeros(grid.shape, bool)
#     valid = grid.site_mask & ~inb & (near <= gap_factor * case.spacing)
#     F = {}
#     nn_idx = tree.query(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))[1].reshape(grid.shape)
#     for k in FIELDS:
#         v = p[k][ok]
#         if not np.isfinite(v).any():
#             F[k] = np.full(grid.shape, np.nan)
#             continue
#         good = np.isfinite(v)
#         try:
#             lin = LinearNDInterpolator(xy[good], v[good])(grid.X, grid.Y)
#         except Exception:
#             lin = np.full(grid.shape, np.nan)
#         nn = v[nn_idx]                                   # nearest sample of the same field (outside the convex hull of the samples)
#         a = np.where(np.isfinite(lin), lin, nn)
#         F[k] = np.where(valid, a, np.nan)
#     F["valid"] = valid
#     open_cells = int((grid.site_mask & ~inb).sum())
#     if open_cells and valid.sum() < 0.5 * open_cells:
#         print(f"[analyze] WARNING {case.name}/{case.season}: only {100 * valid.sum() / open_cells:.0f} % of the open ground is within {gap_factor} x "
#               f"{case.spacing:.2f} m of a sample point - the sample spacing looks wrong ({case.point_note}); maps and areas will be incomplete",
#               file=sys.stderr)
#     return F


# # ============================================================================= UTCI and its exact split
# WIND_FACTOR = 1.0                  # same as compute_utci.py --wind-factor (multiplies the CFD wind before UTCI)


# def utci_fn(Ta, tmrt, v, rh):
#     from pythermalcomfort.models import utci
#     Ta, tmrt, v, rh = (np.asarray(a, float) for a in (Ta, tmrt, v, rh))
#     v = np.clip(v * WIND_FACTOR, 0.5, 17.0)
#     tr = np.clip(tmrt, Ta - 30.0, Ta + 70.0)
#     rh = np.clip(rh, 1.0, 100.0)
#     res = utci(tdb=Ta, tr=tr, v=v, rh=rh, units="SI", limit_inputs=False)
#     return np.asarray(getattr(res, "utci", res), float)


# def shapley_split(B, C, mask):
#     """B, C: field dicts of base / candidate on the grid; mask: cells valid in both.  Returns dict(contrib={driver: array (cells,)},
#     u_base, u_cand (recomputed UTCI), total) with  sum(contrib) = u_cand - u_base  exactly (order-averaged substitution)."""
#     idx = np.where(mask)
#     cols = [DRIVER_COL[d] for d in DRIVERS]
#     b = [B[c][idx] for c in cols]
#     c = [C[c][idx] for c in cols]
#     n = len(DRIVERS)
#     cache = {}

#     def val(sub):
#         if sub not in cache:
#             args = [c[i] if (sub >> i) & 1 else b[i] for i in range(n)]
#             cache[sub] = utci_fn(Ta=args[2], tmrt=args[0], v=args[1], rh=args[3])
#         return cache[sub]

#     fact = [math.factorial(k) for k in range(n + 1)]
#     contrib = {}
#     for i, d in enumerate(DRIVERS):
#         acc = np.zeros(len(idx[0]))
#         for sub in range(1 << n):
#             if (sub >> i) & 1:
#                 continue
#             k = bin(sub).count("1")
#             w = fact[k] * fact[n - k - 1] / fact[n]
#             acc += w * (val(sub | (1 << i)) - val(sub))
#         full = np.full(B["valid"].shape, np.nan)
#         full[idx] = acc
#         contrib[d] = full
#     u0, u1 = np.full(B["valid"].shape, np.nan), np.full(B["valid"].shape, np.nan)
#     u0[idx], u1[idx] = val(0), val((1 << n) - 1)
#     return dict(contrib=contrib, u_base=u0, u_cand=u1)


# def excess_split(F, region_mask, valid):
#     """why is this region hotter / colder than the REST of the site in the same layout?  Region means vs the means of the rest of the open ground,
#     split over Tmrt, wind, Ta, RH (same exact method).  Returns dict(contrib={driver: dC}, total, u_region, u_rest)"""
#     rest = valid & ~region_mask
#     if not region_mask.any() or not rest.any():
#         return None
#     mk = lambda m: {c: np.array([[float(np.nanmean(F[c][m]))]]) for c in DRIVER_COL.values()}      # noqa: E731
#     Bm, Cm = mk(rest), mk(region_mask)
#     ones = np.ones((1, 1), bool)
#     Bm["valid"] = Cm["valid"] = ones
#     sp = shapley_split(Bm, Cm, ones)
#     return dict(contrib={d: float(sp["contrib"][d][0, 0]) for d in DRIVERS}, total=float(sp["u_cand"][0, 0] - sp["u_base"][0, 0]),
#                 u_region=float(sp["u_cand"][0, 0]), u_rest=float(sp["u_base"][0, 0]))


# def tmrt_attribution(B, C, mask):
#     """split the change of Tmrt into shade (sunlit share), sky view, air temperature and the rest (surfaces / long-wave): linear regression
#     Tmrt ~ 1 + sunlit + sky_open + Ta on the cells of both cases.  Returns dict(coef, r2, parts={name: array})"""
#     idx = np.where(mask)
#     A, y = [], []
#     for F in (B, C):
#         A.append(np.column_stack([np.ones(len(idx[0])), F["sunlit"][idx], F["sky_open"][idx], F["Tair_C"][idx]]))
#         y.append(F["Tmrt_C"][idx])
#     A, y = np.vstack(A), np.concatenate(y)
#     okk = np.isfinite(A).all(1) & np.isfinite(y)
#     if okk.sum() < 20 or np.ptp(A[okk, 1]) < 1e-6 and np.ptp(A[okk, 2]) < 1e-6:
#         return None
#     coef, *_ = np.linalg.lstsq(A[okk], y[okk], rcond=None)
#     res = y[okk] - A[okk] @ coef
#     r2 = 1.0 - float(res.var() / max(y[okk].var(), 1e-9))
#     parts = {}
#     for name, j, key in (("shade (sunlit share)", 1, "sunlit"), ("sky view", 2, "sky_open"), ("air temperature", 3, "Tair_C")):
#         full = np.full(B["valid"].shape, np.nan)
#         full[idx] = coef[j] * (C[key][idx] - B[key][idx])
#         parts[name] = full
#     dt = np.full(B["valid"].shape, np.nan)
#     dt[idx] = C["Tmrt_C"][idx] - B["Tmrt_C"][idx]
#     rest = dt.copy()
#     for v in parts.values():
#         rest = rest - v
#     parts["surfaces / long-wave (rest)"] = rest
#     return dict(coef=[float(c) for c in coef], r2=r2, parts=parts, dTmrt=dt)


# # ============================================================================= regions
# def label_mask(mask, grid, min_area):
#     """connected regions (8-neighbour, one closing step so a 1-cell gap does not split a region) larger than min_area m2:
#     list of dict(mask, area, centroid, ...) sorted by area"""
#     if not mask.any():
#         return []
#     m = ndi.binary_closing(mask, structure=np.ones((3, 3), bool)) & (mask | ndi.binary_dilation(mask, iterations=1))
#     lab, n = ndi.label(m, structure=np.ones((3, 3), int))
#     out = []
#     for i in range(1, n + 1):
#         mm = (lab == i) & mask
#         a = float(mm.sum() * grid.area)
#         if a < min_area:
#             continue
#         out.append(dict(mask=mm, area=a, centroid=(float(grid.X[mm].mean()), float(grid.Y[mm].mean()))))
#     out.sort(key=lambda r: -r["area"])
#     return out


# def compass(dx, dy):
#     """name of the direction of (dx, dy) in the model frame (+y north, +x east)"""
#     if math.hypot(dx, dy) < 1e-9:
#         return "centre"
#     names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
#     return names[int(round(math.degrees(math.atan2(dy, dx)) / 45.0)) % 8]


# def region_extent(mask, grid, flow):
#     """length along the flow and width across it (m) of a region, from its cells"""
#     xs, ys = grid.X[mask], grid.Y[mask]
#     if flow is None:
#         return float(np.ptp(xs)), float(np.ptp(ys))
#     u = xs * flow[0] + ys * flow[1]
#     v = -xs * flow[1] + ys * flow[0]
#     return float(np.ptp(u) + grid.cell), float(np.ptp(v) + grid.cell)


# def nearest_building(layout, pt, flow=None):
#     """(id, distance) of the building nearest to pt; with flow: the nearest building UPWIND of pt (the one whose wake it may be), else the nearest"""
#     best = None
#     P = Point(pt)
#     for bid, g in layout.buildings.items():
#         d = g.distance(P)
#         if flow is not None:
#             c = g.centroid
#             if (pt[0] - c.x) * flow[0] + (pt[1] - c.y) * flow[1] <= 0:
#                 continue
#         if best is None or d < best[1]:
#             best = (bid, d)
#     if best is None and flow is not None:
#         return nearest_building(layout, pt, None)
#     return best


# # ============================================================================= layout differences
# def layout_diff(Lb, Lc):
#     """what changed between two layouts: building shifts, trees added / removed, canopy, ponds, lawns"""
#     out = {"buildings": []}
#     for bid, g in Lc.buildings.items():
#         if bid in Lb.buildings and not bid.startswith("osm_"):
#             d = (g.centroid.x - Lb.buildings[bid].centroid.x, g.centroid.y - Lb.buildings[bid].centroid.y)
#             out["buildings"].append(dict(id=bid, dx=round(d[0], 2), dy=round(d[1], 2), shift=round(math.hypot(*d), 2)))
#     sh = [b["shift"] for b in out["buildings"]]
#     out["building_shift_mean"] = float(np.mean(sh)) if sh else 0.0
#     out["building_shift_max"] = float(np.max(sh)) if sh else 0.0

#     def key(t):
#         return (round(t[1], 2), round(t[2], 2), round(t[3], 2))
#     kb, kc = {key(t) for t in Lb.trees}, {key(t) for t in Lc.trees}
#     out["trees_base"], out["trees_cand"] = len(Lb.trees), len(Lc.trees)
#     out["trees_added"] = [t for t in Lc.trees if key(t) not in kb]
#     out["trees_removed"] = [t for t in Lb.trees if key(t) not in kc]
#     out["canopy_base"] = float(sum(math.pi * t[3] ** 2 for t in Lb.trees))
#     out["canopy_cand"] = float(sum(math.pi * t[3] ** 2 for t in Lc.trees))
#     out["mean_height_base"] = float(np.mean([t[4] for t in Lb.trees])) if Lb.trees else 0.0
#     out["mean_height_cand"] = float(np.mean([t[4] for t in Lc.trees])) if Lc.trees else 0.0
#     out["pond_area_base"] = float(sum(p.area for p in Lb.ponds))
#     out["pond_area_cand"] = float(sum(p.area for p in Lc.ponds))
#     out["lawn_area_base"] = float(sum(p.area for p in Lb.lawns))
#     out["lawn_area_cand"] = float(sum(p.area for p in Lc.lawns))
#     out["ponds_cand"], out["ponds_base"] = len(Lc.ponds), len(Lb.ponds)
#     return out


# def region_layout_context(region_mask, grid, flow, Lb, Lc, diff, near_m=25.0, upwind_len=70.0):
#     """layout changes around one region: new / removed trees within near_m of it, trees added in a corridor upwind of it, ponds / lawns in the
#     candidate near it (and in the base), buildings that moved near it"""
#     from shapely.geometry import MultiPoint
#     pts = np.column_stack([grid.X[region_mask], grid.Y[region_mask]])
#     reg = MultiPoint(pts[::max(1, len(pts) // 400)]).buffer(0.75 * grid.cell)
#     ctx = {}
#     near = reg.buffer(near_m)
#     shapely.prepare(near)
#     added = [t for t in diff["trees_added"] if near.contains(Point(t[1], t[2]))]
#     removed = [t for t in diff["trees_removed"] if near.contains(Point(t[1], t[2]))]
#     ctx["trees_added_near"], ctx["trees_removed_near"] = len(added), len(removed)
#     ctx["canopy_added_near_m2"] = float(sum(math.pi * t[3] ** 2 for t in added))
#     ctx["canopy_removed_near_m2"] = float(sum(math.pi * t[3] ** 2 for t in removed))
#     if flow is not None:
#         c = reg.centroid
#         u = lambda x, y: (x - c.x) * flow[0] + (y - c.y) * flow[1]            # noqa: E731
#         v = lambda x, y: -(x - c.x) * flow[1] + (y - c.y) * flow[0]           # noqa: E731
#         half = 0.5 * region_extent(region_mask, grid, flow)[1] + 10.0
#         up_added = [t for t in diff["trees_added"] if -upwind_len <= u(t[1], t[2]) <= 0 and abs(v(t[1], t[2])) <= half]
#         up_removed = [t for t in diff["trees_removed"] if -upwind_len <= u(t[1], t[2]) <= 0 and abs(v(t[1], t[2])) <= half]
#         ctx["trees_added_upwind"], ctx["trees_removed_upwind"] = len(up_added), len(up_removed)

#         def pond_upwind(L):
#             n = 0.0
#             for p in L.ponds:
#                 q = p.centroid
#                 if -upwind_len * 1.5 <= u(q.x, q.y) <= 0 and abs(v(q.x, q.y)) <= half + 15.0:
#                     n += p.area
#             return n
#         ctx["pond_area_upwind_base"], ctx["pond_area_upwind_cand"] = pond_upwind(Lb), pond_upwind(Lc)
#     ctx["pond_area_near_base"] = float(sum(p.intersection(near).area for p in Lb.ponds))
#     ctx["pond_area_near_cand"] = float(sum(p.intersection(near).area for p in Lc.ponds))
#     ctx["lawn_area_near_base"] = float(sum(p.intersection(near).area for p in Lb.lawns))
#     ctx["lawn_area_near_cand"] = float(sum(p.intersection(near).area for p in Lc.lawns))
#     moved = [b for b in diff["buildings"] if b["shift"] > 0.5 and Lc.buildings[b["id"]].distance(reg) <= near_m + 15.0]
#     ctx["buildings_moved_near"] = [(b["id"], b["shift"]) for b in moved]
#     return ctx


# # ============================================================================= the analysis of one candidate in one season
# def stress_areas(U, valid, grid):
#     """area (m2) per UTCI category"""
#     out = {}
#     u = U[valid]
#     edges = sorted(c[0] for c in CATEGORIES)
#     names = {c[0]: c[1] for c in CATEGORIES}
#     lo = -1e9
#     for e in edges + [1e9]:
#         nm = "extreme cold stress" if lo == -1e9 else names[lo]
#         out[nm] = float(((u > lo) & (u <= e)).sum() * grid.area)
#         lo = e
#     return out


# def field_stats(F, valid, grid):
#     s = {}
#     for k in FIELDS:
#         v = F[k][valid]
#         s[k] = float(np.nanmean(v)) if v.size else float("nan")
#     s["utci_p10"], s["utci_p90"] = (float(np.nanpercentile(F["UTCI_C"][valid], q)) for q in (10, 90))
#     s["area_m2"] = float(valid.sum() * grid.area)
#     return s


# def wake_regions(F, valid, grid, flow, u_ref, frac, min_area, layout):
#     """calm regions: wind < frac * u_ref"""
#     calm = valid & (F["wind_ms"] < frac * u_ref)
#     regs = label_mask(calm, grid, min_area)
#     out = []
#     for k, r in enumerate(regs, 1):
#         m = r["mask"]
#         ln, wd = region_extent(m, grid, flow)
#         nb = nearest_building(layout, r["centroid"], flow)
#         out.append(dict(id=f"W{k}", area=r["area"], centroid=r["centroid"], length_m=ln, width_m=wd, mask=m,
#                         wind_mean=float(np.nanmean(F["wind_ms"][m])), utci_mean=float(np.nanmean(F["UTCI_C"][m])),
#                         building=None if nb is None else nb[0], building_dist=None if nb is None else float(nb[1])))
#     return calm, out


# def match_regions(regs_a, regs_b, min_overlap=0.2):
#     """for every region of A: (index of the best overlapping region of B or None, overlap share of the smaller of the two)"""
#     out = []
#     for r in regs_a:
#         best, bo = None, 0.0
#         for j, s in enumerate(regs_b):
#             ov = float((r["mask"] & s["mask"]).sum()) / max(min(float(r["mask"].sum()), float(s["mask"].sum())), 1.0)
#             if ov > bo:
#                 best, bo = j, ov
#         out.append((best if bo >= min_overlap else None, bo))
#     return out


# def match_wakes(wa, wb, max_shift_m=45.0):
#     """base wake -> layout wake: first the wake that trails the same building (nearest centroid within max_shift_m, since a moved building drags its wake
#     along), else the one it overlaps.  Returns (list of index or None per wake of wa, set of indices of wb that were matched)"""
#     res, used = [], set()
#     ov = match_regions(wa, wb)
#     for k, w in enumerate(wa):
#         cand = [j for j, v in enumerate(wb) if j not in used and v["building"] is not None and v["building"] == w["building"]
#                 and math.hypot(v["centroid"][0] - w["centroid"][0], v["centroid"][1] - w["centroid"][1]) <= max_shift_m]
#         if cand:
#             j = min(cand, key=lambda j: math.hypot(wb[j]["centroid"][0] - w["centroid"][0], wb[j]["centroid"][1] - w["centroid"][1]))
#         else:
#             j = ov[k][0]
#             if j in used:
#                 j = None
#         res.append(j)
#         if j is not None:
#             used.add(j)
#     return res, used


# UNIT = {"Tmrt": "°C", "wind": "m/s", "Ta": "°C", "RH": "%"}
# NDEC = {"Tmrt": 1, "wind": 2, "Ta": 1, "RH": 1}
# ORDER = ("Ta", "Tmrt", "RH", "wind")                    # order of the columns of the value tables


# def zone_vals(mask, B, C, both, sp, grid):
#     """area-averaged inputs of a zone in the base and in the layout, their change, and the part of the UTCI change each one caused (exact split):
#     dict(area, n, UTCI=dict(b, c, d), Ta=dict(b, c, d, eff), Tmrt=..., RH=..., wind=...)"""
#     m = mask & both
#     if not m.any():
#         return None
#     v = dict(area=float(mask.sum() * grid.area), n=int(m.sum()))
#     ub, uc = float(np.nanmean(B["UTCI_C"][m])), float(np.nanmean(C["UTCI_C"][m]))
#     v["UTCI"] = dict(b=ub, c=uc, d=uc - ub)
#     for dr in DRIVERS:
#         col = DRIVER_COL[dr]
#         b, c = float(np.nanmean(B[col][m])), float(np.nanmean(C[col][m]))
#         v[dr] = dict(b=b, c=c, d=c - b, eff=float(np.nanmean(sp["contrib"][dr][m])))
#     return v


# def vtxt(v, dr, eff=True, nl=False):
#     """'36.1 → 35.9 (−0.2) → −0.27' : base → layout (change) → effect on UTCI (°C)"""
#     n = NDEC[dr]
#     x = v[dr]
#     s = f"{x['b']:.{n}f} → {x['c']:.{n}f} ({x['d']:+.{n}f})"
#     if eff:
#         s += ("\n" if nl else " → ") + f"{x['eff']:+.2f}"
#     return s


# def ctxt(v, dr):
#     """compact label text for drawing on a map"""
#     n = NDEC[dr] if dr != "RH" else 0
#     return f"{dr if dr != 'wind' else 'U'} {v[dr]['b']:.{n}f}→{v[dr]['c']:.{n}f}"


# def inputs_sentence(v, sgn=None):
#     """'Ta 36.1 → 35.9 °C (−0.2) lowers UTCI by 0.27 °C; Tmrt ...' for the four inputs in order of their effect"""
#     items = sorted(DRIVERS, key=lambda d: -abs(v[d]["eff"]))
#     parts = []
#     for d in items:
#         x = v[d]
#         n = NDEC[d]
#         verb = "raises" if x["eff"] > 0.005 else ("lowers" if x["eff"] < -0.005 else "no effect on")
#         tail = f" {verb} UTCI by {abs(x['eff']):.2f} °C" if abs(x["eff"]) >= 0.005 else " no effect on UTCI"
#         parts.append(f"{d} {x['b']:.{n}f} → {x['c']:.{n}f} {UNIT[d]} ({x['d']:+.{n}f}){tail}")
#     return "; ".join(parts)


# class SeasonResult:
#     pass


# def analyse_pair(base, cand, grid, a):
#     """everything for one candidate / one season"""
#     R = SeasonResult()
#     R.season, R.name = cand.season, cand.name
#     heat = cand.season == "heat"
#     sgn = -1.0 if heat else 1.0                       # benefit = sgn * dUTCI
#     B, C = rasterise(base, grid), rasterise(cand, grid)
#     both = B["valid"] & C["valid"]
#     if not B["valid"].any() or not C["valid"].any() or not both.any():
#         raise ValueError(f"{cand.name}/{cand.season}: no open ground on the analysis grid (valid cells base {int(B['valid'].sum())}, layout {int(C['valid'].sum())}, both {int(both.sum())}); "
#                          f"grid {grid.shape[1]} x {grid.shape[0]} cells of {grid.cell:g} m, site bounds {tuple(round(v, 1) for v in cand.layout.site.bounds) if cand.layout.site is not None else None}, "
#                          f"sample points x {np.nanmin(cand.pts['x']):.1f}..{np.nanmax(cand.pts['x']):.1f}, y {np.nanmin(cand.pts['y']):.1f}..{np.nanmax(cand.pts['y']):.1f}")
#     R.B, R.C, R.both, R.sgn = B, C, both, sgn
#     R.flow = cand.flow() if cand.flow() is not None else base.flow()
#     u10 = cand.boundary.get("wind_speed_10m_ms") or base.boundary.get("wind_speed_10m_ms")
#     R.u10 = float(u10) if u10 else float("nan")
#     # reference wind = free pedestrian-level wind of the BASE case (90th percentile of its 1.5 m field), same value for both layouts
#     R.u_ref = float(a.u_ref) if a.u_ref else float(np.nanpercentile(B["wind_ms"][B["valid"]], 90))
#     R.stats_b, R.stats_c = field_stats(B, B["valid"], grid), field_stats(C, C["valid"], grid)
#     R.stats_b_both, R.stats_c_both = field_stats(B, both, grid), field_stats(C, both, grid)
#     R.cat_b, R.cat_c = stress_areas(B["UTCI_C"], B["valid"], grid), stress_areas(C["UTCI_C"], C["valid"], grid)
#     R.obj_b = base.summary.get("utci_C")
#     R.obj_c = cand.summary.get("utci_C")
#     R.diff = layout_diff(base.layout, cand.layout)

#     # ---- exact split of the UTCI change
#     sp = shapley_split(B, C, both)
#     R.split = sp
#     tot = sp["u_cand"] - sp["u_base"]
#     R.dU = tot
#     R.contrib_mean = {d: float(np.nanmean(sp["contrib"][d][both])) for d in DRIVERS}
#     R.dU_mean = float(np.nanmean(tot[both]))
#     R.split_err = float(np.nanmax(np.abs(sum(sp["contrib"][d] for d in DRIVERS) - tot)[both])) if both.any() else 0.0
#     R.csv_err = float(np.nanmean(np.abs(sp["u_base"][both] - B["UTCI_C"][both])))
#     R.tmrt = tmrt_attribution(B, C, both)
#     if R.tmrt is not None:
#         R.tmrt_parts_mean = {k: float(np.nanmean(v[both])) for k, v in R.tmrt["parts"].items()}
#         R.dTmrt_mean = float(np.nanmean(R.tmrt["dTmrt"][both]))

#     R.site_vals = zone_vals(both, B, C, both, sp, grid)

#     # ---- hot / cold spots of the base case
#     ub = B["UTCI_C"]
#     if heat:
#         thr = max(float(np.nanpercentile(ub[B["valid"]], a.hot_pct)), 26.0)
#         spot_b = B["valid"] & (ub >= thr)
#         spot_c = C["valid"] & (C["UTCI_C"] >= thr)
#     else:
#         thr = min(float(np.nanpercentile(ub[B["valid"]], 100 - a.cold_pct)), 9.0)
#         spot_b = B["valid"] & (ub <= thr)
#         spot_c = C["valid"] & (C["UTCI_C"] <= thr)
#     R.thr = thr
#     R.spot_b, R.spot_c = spot_b, spot_c
#     R.spot_area_b, R.spot_area_c = float(spot_b.sum() * grid.area), float(spot_c.sum() * grid.area)
#     regs = label_mask(spot_b, grid, a.min_region_m2)
#     R.regions = []
#     for k, r in enumerate(regs, 1):
#         m = r["mask"] & both
#         if m.sum() < 3:
#             continue
#         info = dict(id=("H" if heat else "C") + str(k), area=r["area"], centroid=r["centroid"], mask=r["mask"])
#         info["u_base"] = float(np.nanmean(B["UTCI_C"][m]))
#         info["u_cand"] = float(np.nanmean(C["UTCI_C"][m]))
#         info["d_u"] = info["u_cand"] - info["u_base"]
#         info["benefit"] = sgn * info["d_u"]
#         if heat:
#             still = (C["UTCI_C"][m] >= thr)
#         else:
#             still = (C["UTCI_C"][m] <= thr)
#         info["resolved_share"] = float(1.0 - still.mean())
#         info["contrib"] = {d: float(np.nanmean(sp["contrib"][d][m])) for d in DRIVERS}
#         for key, lab in (("Tair_C", "Ta"), ("RH_pct", "RH"), ("wind_ms", "wind"), ("Tmrt_C", "Tmrt"), ("sunlit", "sunlit"), ("sky_open", "sky_open")):
#             info[lab + "_b"], info[lab + "_c"] = float(np.nanmean(B[key][m])), float(np.nanmean(C[key][m]))
#         if R.tmrt is not None:
#             info["tmrt_parts"] = {k2: float(np.nanmean(v[m])) for k2, v in R.tmrt["parts"].items()}
#         dom = max(DRIVERS, key=lambda d: sgn * info["contrib"][d])
#         info["dominant"] = dom
#         if info["benefit"] > 0.3 and info["resolved_share"] >= 0.7:
#             info["status"] = "resolved"
#         elif info["benefit"] > 0.3 and info["resolved_share"] >= 0.3:
#             info["status"] = "partly resolved"
#         elif info["benefit"] > 0.3:
#             info["status"] = "reduced"
#         elif info["benefit"] < -0.3:
#             info["status"] = "worse"
#         else:
#             info["status"] = "unchanged"
#         info["context"] = region_layout_context(r["mask"], grid, R.flow, base.layout, cand.layout, R.diff)
#         nb = nearest_building(cand.layout, r["centroid"], R.flow)
#         info["near_building"] = None if nb is None else nb[0]
#         info["near_building_dist"] = None if nb is None else float(nb[1])
#         info["cause_b"], info["cause_c"] = excess_split(B, r["mask"] & B["valid"], B["valid"]), excess_split(C, r["mask"] & C["valid"], C["valid"])
#         info["wind_ratio_b"], info["wind_ratio_c"] = info["wind_b"] / R.u_ref, info["wind_c"] / R.u_ref
#         info["vals"] = zone_vals(r["mask"], B, C, both, sp, grid)
#         R.regions.append(info)
#     new_mask = spot_c & ~spot_b & both & (sgn * tot < -0.3)          # stressed in the layout, not in the base, and clearly worse than the base
#     new_regs = label_mask(new_mask, grid, a.min_region_m2)
#     R.new_regions = []
#     for k, r in enumerate(new_regs, 1):
#         m = r["mask"]
#         info = dict(id=("nH" if heat else "nC") + str(k), area=r["area"], centroid=r["centroid"], mask=m)
#         info["u_cand"] = float(np.nanmean(C["UTCI_C"][m]))
#         for key, lab in (("Tair_C", "Ta"), ("wind_ms", "wind"), ("Tmrt_C", "Tmrt"), ("sunlit", "sunlit")):
#             info[lab + "_c"] = float(np.nanmean(C[key][m]))
#         b2 = (m & both)
#         if b2.any():
#             info["u_base"] = float(np.nanmean(B["UTCI_C"][b2]))
#             info["contrib"] = {d: float(np.nanmean(sp["contrib"][d][b2])) for d in DRIVERS}
#             info["dominant"] = max(DRIVERS, key=lambda d: -sgn * info["contrib"][d])        # the driver that made it worse
#         info["context"] = region_layout_context(m, grid, R.flow, base.layout, cand.layout, R.diff)
#         nb = nearest_building(cand.layout, r["centroid"], R.flow)
#         info["near_building"] = None if nb is None else nb[0]
#         info["cause_c"] = excess_split(C, m & C["valid"], C["valid"])
#         info["sunlit_c"], info["sky_open_c"] = float(np.nanmean(C["sunlit"][m])), float(np.nanmean(C["sky_open"][m]))
#         info["RH_c"] = float(np.nanmean(C["RH_pct"][m]))
#         info["vals"] = zone_vals(m, B, C, both, sp, grid)
#         R.new_regions.append(info)

#     # ---- wakes / ventilation
#     calm_b, wb = wake_regions(B, B["valid"], grid, R.flow, R.u_ref, a.wake_frac, a.min_region_m2, base.layout)
#     calm_c, wc = wake_regions(C, C["valid"], grid, R.flow, R.u_ref, a.wake_frac, a.min_region_m2, cand.layout)
#     R.calm_b, R.calm_c, R.wakes_b, R.wakes_c = calm_b, calm_c, wb, wc
#     R.vent = dict(
#         wind_ratio_b=R.stats_b["wind_ms"] / R.u_ref, wind_ratio_c=R.stats_c["wind_ms"] / R.u_ref,
#         calm_area_b=float(calm_b.sum() * grid.area), calm_area_c=float(calm_c.sum() * grid.area),
#         calm_share_b=float(calm_b.sum() / max(B["valid"].sum(), 1)), calm_share_c=float(calm_c.sum() / max(C["valid"].sum(), 1)),
#         fast_share_b=float(((B["wind_ms"] > 1.3 * R.u_ref) & B["valid"]).sum() / max(B["valid"].sum(), 1)),
#         fast_share_c=float(((C["wind_ms"] > 1.3 * R.u_ref) & C["valid"]).sum() / max(C["valid"].sum(), 1)),
#         longest_wake_b=max([w["length_m"] for w in wb], default=0.0), longest_wake_c=max([w["length_m"] for w in wc], default=0.0),
#         wind_ratio_in_spots_b=float(np.nanmean(B["wind_ms"][spot_b]) / R.u_ref) if spot_b.any() else float("nan"),
#         wind_ratio_in_spots_c=float(np.nanmean(C["wind_ms"][spot_c]) / R.u_ref) if spot_c.any() else float("nan"))
#     mt, used = match_wakes(wb, wc)
#     for w, j in zip(wb, mt):
#         w["fate"] = "gone" if j is None else ("smaller" if wc[j]["area"] < 0.8 * w["area"] else ("larger" if wc[j]["area"] > 1.25 * w["area"] else "similar"))
#         w["cand_match"] = None if j is None else wc[j]["id"]
#         w["cand_area"] = 0.0 if j is None else wc[j]["area"]
#         m = w["mask"] & both
#         if m.any():
#             w["d_wind"] = float(np.nanmean(C["wind_ms"][m] - B["wind_ms"][m]))
#             w["d_u"] = float(np.nanmean(C["UTCI_C"][m] - B["UTCI_C"][m]))
#             w["contrib"] = {d: float(np.nanmean(sp["contrib"][d][m])) for d in DRIVERS}
#         w["context"] = region_layout_context(w["mask"], grid, R.flow, base.layout, cand.layout, R.diff)
#         w["vals"] = zone_vals(w["mask"], B, C, both, sp, grid)
#     for j, w in enumerate(wc):
#         w["is_new"] = j not in used
#         w["vals"] = zone_vals(w["mask"], B, C, both, sp, grid)
#     # ---- fast channels
#     return R


# # ============================================================================= drawing
# def _footprint_patches(ax, layout, tree_alpha=1.0, faces=True, tree_lw=0.5):
#     if layout.site is not None:
#         ax.plot(*layout.site.exterior.xy, color=INK, lw=1.0)
#     for g in layout.lawns:
#         ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#d8ecd0" if faces else "none", ec="#8fb98a", lw=0.5, alpha=0.8, zorder=3))
#     for g in layout.ponds:
#         ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#9fcbff" if faces else "none", ec="#2b6cb0", lw=0.9, alpha=0.85 if faces else 1.0, zorder=4))
#     for bid, g in layout.buildings.items():
#         ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#8a8f94", ec=INK, lw=0.7, zorder=6))
#     for t in layout.trees:
#         ax.add_patch(Circle((t[1], t[2]), t[3], fc="none", ec="#2f7d32", lw=tree_lw, alpha=tree_alpha, zorder=5))


# def _setup(ax, grid, title, layout=None, small=False):
#     ax.set_aspect("equal")
#     ax.set_xlim(grid.extent[0], grid.extent[1])
#     ax.set_ylim(grid.extent[2], grid.extent[3])
#     ax.set_title(title, fontsize=9 if small else 10, color=INK)
#     ax.tick_params(labelsize=7, colors=MUTED)
#     for s in ax.spines.values():
#         s.set_color("#c8ced3")
#     if layout is not None:
#         _footprint_patches(ax, layout)


# def _flow_arrow(ax, grid, flow, label=True):
#     if flow is None:
#         return
#     x0, x1, y0, y1 = grid.extent
#     L = 0.12 * (x1 - x0)
#     cx, cy = x0 + 0.1 * (x1 - x0), y1 - 0.07 * (y1 - y0)
#     ax.annotate("", xy=(cx + L * flow[0], cy + L * flow[1]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#222"), zorder=20)
#     if label:
#         ax.text(cx, cy - 0.025 * (y1 - y0), "wind", fontsize=7, color="#222", ha="left", va="top", zorder=20)


# def _nice_levels(vmin, vmax, n=12):
#     span = max(vmax - vmin, 1e-9)
#     raw = span / n
#     mag = 10 ** math.floor(math.log10(raw))
#     step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
#     lo, hi = math.floor(vmin / step + 1e-9) * step, math.ceil(vmax / step - 1e-9) * step
#     return np.arange(lo, hi + 0.5 * step, step)


# def _field(ax, grid, Z, vmin, vmax, cmap, levels=None, lines=None, fill_levels=12):
#     Zm = np.ma.masked_invalid(Z)
#     lv = _nice_levels(vmin, vmax, fill_levels)
#     cf = ax.contourf(grid.X, grid.Y, np.clip(Zm, lv[0], lv[-1]), levels=lv, cmap=cmap, extend="neither", zorder=1)
#     if lines is not None and len(lines):
#         try:
#             cs = ax.contour(grid.X, grid.Y, Zm, levels=lines, colors="#222", linewidths=0.5, alpha=0.6, zorder=2)
#             ax.clabel(cs, fontsize=6, fmt="%g")
#         except Exception:
#             pass
#     return cf


# def _outline(ax, grid, mask, **kw):
#     if mask is None or not mask.any():
#         return
#     try:
#         ax.contour(grid.X, grid.Y, mask.astype(float), levels=[0.5], **kw)
#     except Exception:
#         pass


# def _cbar(fig, cf, ax, label):
#     cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
#     cb.set_label(label, fontsize=7)
#     cb.ax.tick_params(labelsize=6)
#     return cb


# def _label_regions(ax, regs, color="#111", prefix=None, fs=7):
#     for r in regs:
#         ax.text(r["centroid"][0], r["centroid"][1], r["id"], fontsize=fs, ha="center", va="center", color=color, weight="bold", zorder=25,
#                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


# def _label_zones(ax, zones, fs=5.3, color="#111"):
#     """write the area-averaged UTCI, Ta, Tmrt, RH and wind (base → layout) of every zone on the map; boxes that would overlap are pushed apart
#     and joined to their zone by a thin line"""
#     anns = []
#     for z in zones:
#         v = z.get("vals")
#         if v is None:
#             continue
#         txt = f"{z['id']}\nUTCI {v['UTCI']['b']:.1f}→{v['UTCI']['c']:.1f}\n" + "\n".join(ctxt(v, d) for d in ORDER)
#         an = ax.annotate(txt, xy=z["centroid"], xytext=z["centroid"], textcoords="data", fontsize=fs, ha="center", va="center", color=color, zorder=25, linespacing=1.05,
#                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", lw=0.4, alpha=0.9), arrowprops=dict(arrowstyle="-", lw=0.5, color="#444", shrinkA=0, shrinkB=0))
#         anns.append((an, z["centroid"]))
#     if len(anns) < 2:
#         return
#     fig = ax.figure
#     fig.canvas.draw()
#     rend = fig.canvas.get_renderer()
#     inv = ax.transData.inverted()
#     axbb = ax.get_window_extent(rend)
#     pos = [np.array(ax.transData.transform(c), float) for _, c in anns]            # display coordinates of the label centres
#     size = []
#     for an, _ in anns:
#         bb = matplotlib.text.Text.get_window_extent(an, rend)
#         size.append((bb.width + 6.0, bb.height + 6.0))
#     for _ in range(80):
#         moved = False
#         for i in range(len(anns)):
#             for j in range(i + 1, len(anns)):
#                 dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
#                 ox = 0.5 * (size[i][0] + size[j][0]) - abs(dx)
#                 oy = 0.5 * (size[i][1] + size[j][1]) - abs(dy)
#                 if ox > 0 and oy > 0:
#                     moved = True
#                     if oy <= ox:
#                         sgn_ = 1.0 if dy >= 0 else -1.0
#                         pos[i][1] -= sgn_ * 0.5 * (oy + 1.0)
#                         pos[j][1] += sgn_ * 0.5 * (oy + 1.0)
#                     else:
#                         sgn_ = 1.0 if dx >= 0 else -1.0
#                         pos[i][0] -= sgn_ * 0.5 * (ox + 1.0)
#                         pos[j][0] += sgn_ * 0.5 * (ox + 1.0)
#         for i in range(len(anns)):                                               # stay inside the axes
#             pos[i][0] = float(np.clip(pos[i][0], axbb.x0 + 0.5 * size[i][0], axbb.x1 - 0.5 * size[i][0]))
#             pos[i][1] = float(np.clip(pos[i][1], axbb.y0 + 0.5 * size[i][1], axbb.y1 - 0.5 * size[i][1]))
#         if not moved:
#             break
#     for (an, c), pp in zip(anns, pos):
#         an.xyann = tuple(inv.transform(pp))


# def fig_maps(R, base, cand, grid, path):
#     """2 x 4 panels: UTCI base / layout / change / spots, and the four drivers"""
#     heat = R.season == "heat"
#     ub, uc = R.B["UTCI_C"], R.C["UTCI_C"]
#     allv = np.concatenate([ub[R.B["valid"]], uc[R.C["valid"]]])
#     vmin, vmax = float(np.floor(np.nanpercentile(allv, 1) - 0.5)), float(np.ceil(np.nanpercentile(allv, 99) + 0.5))
#     if vmax - vmin < 4:
#         vmax = vmin + 4
#     cat_lines = [c[0] for c in CATEGORIES if vmin < c[0] < vmax]
#     fig, axs = plt.subplots(2, 4, figsize=(20, 9.6))
#     # row 1
#     cf = _field(axs[0, 0], grid, ub, vmin, vmax, RAINBOW, lines=cat_lines)
#     _setup(axs[0, 0], grid, f"UTCI base  ({base.name})  mean {R.stats_b['UTCI_C']:.1f} °C", base.layout)
#     _cbar(fig, cf, axs[0, 0], "UTCI °C")
#     cf = _field(axs[0, 1], grid, uc, vmin, vmax, RAINBOW, lines=cat_lines)
#     _setup(axs[0, 1], grid, f"UTCI  {cand.name}  mean {R.stats_c['UTCI_C']:.1f} °C", cand.layout)
#     _cbar(fig, cf, axs[0, 1], "UTCI °C")
#     d = np.where(R.both, R.dU, np.nan)
#     lim = max(0.5, float(np.nanpercentile(np.abs(d[R.both]), 98)))
#     cf = _field(axs[0, 2], grid, d, -lim, lim, "RdBu_r", lines=None)
#     _setup(axs[0, 2], grid, "change of UTCI (layout − base)\nblue = cooler, red = warmer", cand.layout)
#     _outline(axs[0, 2], grid, R.spot_b, colors="k", linewidths=1.2)
#     _outline(axs[0, 2], grid, R.spot_c, colors="k", linewidths=1.2, linestyles="--")
#     _cbar(fig, cf, axs[0, 2], "ΔUTCI °C")
#     # spots panel
#     ax = axs[0, 3]
#     _setup(ax, grid, f"{'hot' if heat else 'cold'} spots of the base (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C)\nand what became of them\n(zone means, base → layout: UTCI, Ta °C, Tmrt °C, RH %, U m/s)", cand.layout, small=True)
#     colors = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c"}
#     for r in R.regions:
#         ax.contourf(grid.X, grid.Y, np.where(r["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[colors[r["status"]]], alpha=0.75, zorder=2)
#     for r in R.new_regions:
#         ax.contourf(grid.X, grid.Y, np.where(r["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#8e44ad"], alpha=0.8, zorder=2)
#     _label_zones(ax, R.regions)
#     _label_zones(ax, R.new_regions, color="#4a235a")
#     import matplotlib.patches as mp
#     ax.legend(handles=[mp.Patch(color=c, label=k) for k, c in colors.items()] + [mp.Patch(color="#8e44ad", label="new in layout")],
#               fontsize=6, loc="lower left", framealpha=0.85)
#     # row 2: drivers
#     def diffpanel(ax, key, title, unit, scale=1.0):
#         dd = np.where(R.both, (R.C[key] - R.B[key]) * scale, np.nan)
#         lm = max(0.01, float(np.nanpercentile(np.abs(dd[R.both]), 98)))
#         cf_ = _field(ax, grid, dd, -lm, lm, "RdBu_r")
#         _setup(ax, grid, title, cand.layout)
#         _cbar(fig, cf_, ax, unit)
#         return cf_
#     diffpanel(axs[1, 0], "Tmrt_C", "Δ Tmrt (layout − base)", "°C")
#     ax = axs[1, 1]
#     diffpanel(ax, "wind_ms", f"Δ wind speed (reference wind {R.u_ref:.1f} m/s)\nwake: solid = base, dashed = layout", "m/s")
#     _outline(ax, grid, R.calm_b, colors="k", linewidths=1.1)
#     _outline(ax, grid, R.calm_c, colors="k", linewidths=1.1, linestyles="--")
#     _flow_arrow(ax, grid, R.flow)
#     diffpanel(axs[1, 2], "Tair_C", "Δ air temperature", "°C")
#     diffpanel(axs[1, 3], "RH_pct", "Δ relative humidity", "%")
#     fig.suptitle(f"{cand.name} vs base — {SEASON_TXT[R.season]}   (site mean UTCI {R.stats_b['UTCI_C']:.2f} → {R.stats_c['UTCI_C']:.2f} °C)", fontsize=12, color=INK)
#     fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
#     fig.savefig(path, dpi=80)
#     plt.close(fig)


# def fig_drivers(R, grid, path):
#     """why: Shapley split of the UTCI change (site mean and the regions), Tmrt split"""
#     sgn = R.sgn
#     fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1.6, 1.1]))
#     ax = axs[0]
#     names = list(DRIVERS)
#     vals = [sgn * R.contrib_mean[d] for d in names]
#     ax.barh(range(len(names)), vals, color=[COL[d] for d in names])
#     ax.set_yticks(range(len(names)))
#     ax.set_yticklabels([f"{d}\nΔ {R.site_vals[d]['d']:+.{NDEC[d]}f} {UNIT[d]}" for d in names], fontsize=8)
#     for i, v in enumerate(vals):
#         ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
#     ax.axvline(0, color=INK, lw=0.8)
#     ax.set_title(f"site mean: benefit {sgn * R.dU_mean:+.2f} °C\n(benefit = {'cooler' if sgn < 0 else 'warmer'})", fontsize=10)
#     ax.margins(x=0.3)
#     ax.invert_yaxis()
#     ax.set_xlabel("°C of UTCI benefit  (label: actual change of the input)")
#     ax = axs[1]
#     labels, per = [], []
#     for r in R.regions:
#         vv = r["vals"]
#         labels.append(f"{r['id']}\n{r['area']:.0f} m²\n" + "\n".join(f"Δ{'U' if d == 'wind' else d} {vv[d]['d']:+.{NDEC[d]}f}" for d in ORDER))
#         per.append([sgn * r["contrib"][d] for d in DRIVERS])
#     if per:
#         per = np.array(per)
#         x = np.arange(len(labels))
#         pos, neg = np.zeros(len(labels)), np.zeros(len(labels))
#         for j, d in enumerate(DRIVERS):
#             v = per[:, j]
#             bot = np.where(v >= 0, pos, neg)
#             ax.bar(x, v, bottom=bot, color=COL[d], label=d, width=0.6)
#             pos += np.where(v >= 0, v, 0)
#             neg += np.where(v < 0, v, 0)
#         ax.plot(x, [r["benefit"] for r in R.regions], "kD", ms=5, label="total")
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, fontsize=7)
#         ax.axhline(0, color=INK, lw=0.8)
#         ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
#     ax.set_title(f"base {'hot' if R.season == 'heat' else 'cold'}-spot regions: benefit by driver", fontsize=10)
#     ax.set_ylabel("°C of UTCI benefit")
#     ax = axs[2]
#     if R.tmrt is not None:
#         parts = R.tmrt_parts_mean
#         names2 = list(parts.keys())
#         v2 = [-p if R.season == "heat" else p for p in parts.values()]
#         ax.barh(range(len(names2)), v2, color="#d9822b", alpha=0.85)
#         ax.set_yticks(range(len(names2)))
#         ax.set_yticklabels(names2, fontsize=8)
#         for i, v in enumerate(v2):
#             ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
#         ax.axvline(0, color=INK, lw=0.8)
#         ax.invert_yaxis()
#         ax.margins(x=0.25)
#         ax.set_title(f"inside Tmrt: Δ Tmrt = {R.dTmrt_mean:+.2f} °C  (regression R² {R.tmrt['r2']:.2f})\nbenefit in Tmrt °C", fontsize=9)
#     fig.tight_layout()
#     fig.savefig(path, dpi=80)
#     plt.close(fig)


# def fig_wakes(R, base, cand, grid, path):
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
#     ratio_b, ratio_c = R.B["wind_ms"] / R.u_ref, R.C["wind_ms"] / R.u_ref
#     vmax = float(np.ceil(10 * max(1.6, np.nanpercentile(np.concatenate([ratio_b[R.B["valid"]], ratio_c[R.C["valid"]]]), 99))) / 10)
#     for ax, ratio, F, name, lay, wk in ((axs[0], ratio_b, R.B, base.name, base.layout, R.wakes_b), (axs[1], ratio_c, R.C, cand.name, cand.layout, R.wakes_c)):
#         cf = _field(ax, grid, ratio, 0.0, vmax, "viridis", lines=[0.5, 1.0, 1.3])
#         _setup(ax, grid, f"wind speed / reference wind — {name}\ncalm (< {R.wake_frac:g}) regions outlined", lay)
#         _outline(ax, grid, (F["wind_ms"] < R.wake_frac * R.u_ref) & F["valid"], colors="w", linewidths=1.4)
#         _label_regions(ax, wk, color="#7a1f1f")
#         _flow_arrow(ax, grid, R.flow)
#         _cbar(fig, cf, ax, "U / U_ref")
#     ax = axs[2]
#     _setup(ax, grid, "wake regions: what became of the base wakes\n(zone means, base → layout: UTCI, Ta, Tmrt, RH, U)", cand.layout)
#     fate_col = {"gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b"}
#     for w in R.wakes_b:
#         ax.contourf(grid.X, grid.Y, np.where(w["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[fate_col[w["fate"]]], alpha=0.75, zorder=2)
#     for w in R.wakes_c:
#         if w.get("is_new"):
#             ax.contourf(grid.X, grid.Y, np.where(w["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#8e44ad"], alpha=0.8, zorder=2)
#     _label_zones(ax, R.wakes_b)
#     _label_zones(ax, [w for w in R.wakes_c if w.get("is_new")], color="#4a235a")
#     import matplotlib.patches as mp
#     ax.legend(handles=[mp.Patch(color=c, label=f"base wake: {k}") for k, c in fate_col.items()] + [mp.Patch(color="#8e44ad", label="new wake in layout")],
#               fontsize=6, loc="lower left", framealpha=0.85)
#     _flow_arrow(ax, grid, R.flow)
#     fig.suptitle(f"ventilation and wakes — {cand.name} vs base — {SEASON_TXT[R.season]}", fontsize=12, color=INK)
#     fig.tight_layout(rect=(0, 0, 1, 0.95))
#     fig.savefig(path, dpi=80)
#     plt.close(fig)


# def zone_table_rows(zones, with_status=True):
#     rows = []
#     for z in zones:
#         v = z.get("vals")
#         if v is None:
#             continue
#         tag = z.get("status") or z.get("fate") or ("new" if z.get("is_new") else "")
#         if z["id"].startswith("n"):
#             tag = "new"
#         rows.append([f"{z['id']}\n{tag}", f"{z['area']:.0f}", f"{v['UTCI']['b']:.1f} → {v['UTCI']['c']:.1f}\n({v['UTCI']['d']:+.2f})"] + [vtxt(v, d, True, True) for d in ORDER])
#     return rows


# def fig_zones(R, cand, grid, path, kind):
#     """map of the zones with their area-mean values written on them + a table: for Ta, Tmrt, RH and wind the base value → layout value (change)
#     and the effect of that change on UTCI.  kind = 'spots' (hot / cold spots of the base and new ones) or 'wakes' (calm regions)"""
#     heat = R.season == "heat"
#     if kind == "spots":
#         zones = R.regions + R.new_regions
#         ttl = f"{'hot' if heat else 'cold'} spots of the base (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C) and new ones"
#         colr = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c", "new": "#8e44ad"}
#         colof = lambda z: colr["new"] if z["id"].startswith("n") else colr[z["status"]]            # noqa: E731
#     else:
#         zones = R.wakes_b + [w for w in R.wakes_c if w.get("is_new")]
#         ttl = f"calm regions (wind < {R.wake_frac:g} x {R.u_ref:.2f} m/s): base wakes and new ones"
#         colr = {"gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b", "new": "#8e44ad"}
#         colof = lambda z: colr["new"] if z.get("is_new") else colr[z["fate"]]                     # noqa: E731
#     zones = [z for z in zones if z.get("vals") is not None]
#     if not zones:
#         return False
#     rows = zone_table_rows(zones)
#     sv = R.site_vals
#     rows.append(["whole site\n(open ground)", f"{sv['area']:.0f}", f"{sv['UTCI']['b']:.1f} → {sv['UTCI']['c']:.1f}\n({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d, True, True) for d in ORDER])
#     nrow = len(rows)
#     fig = plt.figure(figsize=(21, max(8.0, 1.2 + 0.62 * nrow)))
#     gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
#     axm, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
#     _setup(axm, grid, ttl, cand.layout)
#     for z in zones:
#         axm.contourf(grid.X, grid.Y, np.where(z["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[colof(z)], alpha=0.7, zorder=2)
#     _label_zones(axm, zones, fs=5.6)
#     _flow_arrow(axm, grid, R.flow)
#     import matplotlib.patches as mp
#     axm.legend(handles=[mp.Patch(color=c, label=k) for k, c in colr.items()], fontsize=7, loc="lower left", framealpha=0.9)
#     axt.axis("off")
#     cols = ["zone", "area m²", "UTCI °C\n(change)", "Ta °C\nbase → layout (change)\neffect on UTCI °C", "Tmrt °C\nbase → layout (change)\neffect on UTCI °C",
#             "RH %\nbase → layout (change)\neffect on UTCI °C", "wind m/s\nbase → layout (change)\neffect on UTCI °C"]
#     tb = axt.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colWidths=[0.1, 0.07, 0.12, 0.18, 0.18, 0.18, 0.18])
#     tb.auto_set_font_size(False)
#     tb.set_fontsize(7.6)
#     tb.scale(1, 2.55)
#     for (r, c), cell in tb.get_celld().items():
#         cell.set_edgecolor("#c8ced3")
#         if r == 0:
#             cell.set_facecolor("#e8eef2")
#             cell.set_text_props(weight="bold", fontsize=7)
#             cell.set_height(cell.get_height() * 1.25)
#         elif r == nrow:
#             cell.set_facecolor("#f2f4f6")
#             cell.set_text_props(weight="bold")
#         elif c == 0:
#             cell.set_facecolor(colof(zones[r - 1]))
#             cell.set_text_props(color="white", weight="bold")
#         elif c >= 3:                                         # shade the cell by the effect on UTCI (blue cooler, red warmer)
#             eff = (sv if r == nrow else zones[r - 1]["vals"])[ORDER[c - 3]]["eff"]
#             a_ = min(abs(eff) / 1.5, 1.0) * 0.55
#             cell.set_facecolor((0.84, 0.19, 0.15, a_) if eff > 0 else (0.17, 0.48, 0.73, a_))
#     fig.suptitle(f"{cand.name} vs base — {SEASON_TXT[R.season]}: zone-averaged inputs and what they did to UTCI   (effect: − = cooler, + = warmer)", fontsize=12, color=INK)
#     fig.tight_layout(rect=(0, 0, 1, 0.96))
#     fig.savefig(path, dpi=80)
#     plt.close(fig)
#     return True


# # ============================================================================= plain two-picture view: heat (or cold) spots, ventilation
# # Each picture: base map | layout map | key, then a table of the spots / calm zones with their Ta, Tmrt, RH and wind, then a few plain sentences.
# # Everything is placed in inches on a fixed canvas, so boxes cannot overlap.
# import textwrap                                                       # noqa: E402

# BAND_EDGES = [-80, -40, -27, -13, 0, 9, 26, 32, 38, 46, 80]
# BAND_NAMES = ["extreme cold stress", "very strong cold stress", "strong cold stress", "moderate cold stress", "slight cold stress", "no thermal stress",
#               "moderate heat stress", "strong heat stress", "very strong heat stress", "extreme heat stress"]
# BAND_RANGE = ["below −40", "−40 to −27", "−27 to −13", "−13 to 0", "0 to 9", "9 to 26", "26 to 32", "32 to 38", "38 to 46", "above 46"]
# BAND_COL = ["#08306b", "#2b5fa8", "#4a8bc9", "#86b6e0", "#c5dcf0", "#e6f2dc", "#fde0a1", "#fbb04e", "#e5612a", "#9c2412"]
# STATUS_COL = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c", "new": "#8e44ad",
#               "gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b"}
# STATUS_TXT = {"resolved": "gone", "partly resolved": "partly gone", "reduced": "reduced", "unchanged": "unchanged", "worse": "worse", "new": "NEW",
#               "gone": "gone", "smaller": "smaller", "similar": "similar", "larger": "larger"}
# SIMPLE_W = 16.0
# SIMPLE_DPI = 200                                                      # --dpi
# MAX_ROWS = 8
# BLUE_TXT, RED_TXT, GREY_TXT = "#1f5f99", "#b3261e", "#5b666e"


# def _inch_axes(fig, W, H, x, y, w, h):
#     """axes covering the box (x, y, w, h) given in inches from the top-left of the canvas; its data units are inches too (y downwards)"""
#     ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
#     ax.set_xlim(0, w)
#     ax.set_ylim(h, 0)
#     ax.axis("off")
#     return ax


# def _short_where(centroid, layout, site_centre, flow=None, near=45.0):
#     d = compass(centroid[0] - site_centre[0], centroid[1] - site_centre[1])
#     s = f"{d} part" if d != "centre" else "centre"
#     nb = nearest_building(layout, centroid, None)
#     if nb is not None and nb[1] <= near:
#         s += f", next to {_bn(nb[0])}" if nb[1] < 12 else f", {nb[1]:.0f} m from {_bn(nb[0])}"
#     return s


# def _bn(bid):
#     """short, readable building name: 'building_3_internal_floor_0' -> 'B3'"""
#     t = str(bid)
#     t = re.sub(r"[_\- ]?(internal|external|inner|outer)?[_\- ]?floor[_\- ]?\d+$", "", t, flags=re.I)
#     t = re.sub(r"[_\- ](internal|external)$", "", t, flags=re.I)
#     t = re.sub(r"^building[_\- ]?", "B", t, flags=re.I) if re.match(r"^building[_\-]", t, flags=re.I) else t
#     return t or str(bid)


# def _building_groups(layout):
#     """buildings that share a footprint (e.g. the floors of one building) -> list of (label, polygon of the largest member)"""
#     ids = list(layout.buildings)
#     polys = [layout.buildings[i] for i in ids]
#     parent = list(range(len(ids)))

#     def find(i):
#         while parent[i] != i:
#             parent[i] = parent[parent[i]]
#             i = parent[i]
#         return i
#     try:
#         from shapely.strtree import STRtree
#         tree = STRtree(polys)
#         pairs = ((i, int(j)) for i in range(len(polys)) for j in tree.query(polys[i]) if int(j) > i)
#     except Exception:
#         pairs = ((i, j) for i in range(len(polys)) for j in range(i + 1, len(polys)))
#     for i, j in pairs:
#         try:
#             inter = polys[i].intersection(polys[j]).area
#         except Exception:
#             continue
#         if inter > 0.5 * min(polys[i].area, polys[j].area):
#             parent[find(j)] = find(i)
#     groups = {}
#     for k in range(len(ids)):
#         groups.setdefault(find(k), []).append(k)
#     out = []
#     for mem in groups.values():
#         big = max(mem, key=lambda k: polys[k].area)
#         names = []
#         for k in sorted(mem, key=lambda k: -polys[k].area):
#             n = _bn(ids[k])
#             if n not in names:
#                 names.append(n)
#         out.append(("\n".join(names[:2]), polys[big]))
#     return out


# def _building_labels(ax, layout, fs=8.0):
#     for label, g in _building_groups(layout):
#         if g.area < 120:
#             continue
#         p = g.representative_point()
#         ax.text(p.x, p.y, label, fontsize=fs, color="white", ha="center", va="center", weight="bold", zorder=8, clip_on=True, linespacing=1.0)


# def _scale_bar(ax, grid, length=50.0):
#     x0, x1, y0, y1 = grid.extent
#     xs, ys = x1 - 0.06 * (x1 - x0) - length, y0 + 0.05 * (y1 - y0)
#     ax.plot([xs, xs + length], [ys, ys], color=INK, lw=2.2, solid_capstyle="butt", zorder=30)
#     ax.text(xs + length / 2, ys + 0.012 * (y1 - y0), f"{length:g} m", fontsize=8, ha="center", va="bottom", color=INK, zorder=30,
#             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


# def _map_panel(ax, grid, layout, title, subtitle):
#     ax.set_aspect("equal", adjustable="box")
#     ax.set_xlim(grid.extent[0], grid.extent[1])
#     ax.set_ylim(grid.extent[2], grid.extent[3])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for s in ax.spines.values():
#         s.set_color("#aab2b8")
#     ax.set_title(title, fontsize=12.5, color=INK, weight="bold", loc="left", pad=22)
#     ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5, color=GREY_TXT, ha="left", va="bottom")
#     _footprint_patches(ax, layout, tree_alpha=0.55, faces=False, tree_lw=0.4)
#     _building_labels(ax, layout)
#     _scale_bar(ax, grid)


# def _wind_arrow_simple(ax, grid, flow):
#     if flow is None:
#         return
#     x0, x1, y0, y1 = grid.extent
#     L = 0.14 * (x1 - x0)
#     cx, cy = x0 + 0.12 * (x1 - x0), y1 - 0.09 * (y1 - y0)
#     ax.annotate("", xy=(cx + L * flow[0], cy + L * flow[1]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=INK), zorder=30)
#     ax.text(cx - 0.01 * (x1 - x0), cy + 0.02 * (y1 - y0), "wind", fontsize=9, color=INK, ha="center", va="bottom", zorder=30, weight="bold",
#             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


# def _place_badges(fig, ax, items, r_px=12.0):
#     """numbered circles on a map: one per zone at its centre; circles that would touch are pushed apart and joined to their zone by a thin line"""
#     if not items:
#         return
#     fig.canvas.draw()
#     inv = ax.transData.inverted()
#     true = [np.array(ax.transData.transform(it["xy"]), float) for it in items]
#     pos = [p.copy() for p in true]
#     axbb = ax.get_window_extent()
#     d_min = 2 * r_px + 4
#     for _ in range(200):
#         moved = False
#         for i in range(len(pos)):
#             for j in range(i + 1, len(pos)):
#                 v = pos[j] - pos[i]
#                 dist = float(np.hypot(*v))
#                 if dist < d_min:
#                     moved = True
#                     if dist < 1e-6:
#                         v, dist = np.array([1.0, 0.3]), 1.04
#                     push = 0.5 * (d_min - dist + 1.0) * v / dist
#                     pos[i] -= push
#                     pos[j] += push
#         for i in range(len(pos)):
#             pos[i][0] = float(np.clip(pos[i][0], axbb.x0 + r_px + 2, axbb.x1 - r_px - 2))
#             pos[i][1] = float(np.clip(pos[i][1], axbb.y0 + r_px + 2, axbb.y1 - r_px - 2))
#         if not moved:
#             break
#     for it, t, p in zip(items, true, pos):
#         xy = it["xy"]
#         xyt = tuple(inv.transform(p))
#         if np.hypot(*(p - t)) > 0.6 * r_px:
#             ax.plot([xy[0], xyt[0]], [xy[1], xyt[1]], color="#111", lw=0.9, zorder=26)
#         ax.plot([xy[0]], [xy[1]], "o", ms=3.2, color="#111", zorder=27)
#         ax.text(xyt[0], xyt[1], it["label"], fontsize=10, color="white", ha="center", va="center", weight="bold", zorder=28,
#                 bbox=dict(boxstyle="circle,pad=0.28", fc=it["color"], ec="#111", lw=1.0))


# def _rows_of_zones(zones, cap=MAX_ROWS):
#     """largest zones first; returns (shown, hidden)"""
#     z = sorted([x for x in zones if x.get("vals") is not None], key=lambda g: -g["area"])
#     return z[:cap], z[cap:]


# def _fmt_eff(x, dr):
#     if abs(x) < 0.05:
#         return "UTCI ±0", GREY_TXT
#     return f"UTCI {x:+.1f}".replace("-", "−"), (BLUE_TXT if x < 0 else RED_TXT)


# def _num(x, nd):
#     return f"{x:.{nd}f}".replace("-", "−") if round(x, nd) != 0 else f"{0:.{nd}f}"


# def _delta(x, nd):
#     if round(x, nd) == 0:
#         return f"{0:.{nd}f}"
#     return f"{x:+.{nd}f}".replace("-", "−")


# def _draw_zone_table(ax, W, rows, site, header_first="Spot", with_site=True, cell_h=0.56):
#     """rows: list of dict(num, color, status, where, area, vals);  columns: spot | where | UTCI | Ta | Tmrt | RH | wind.  Returns the height used (inches)"""
#     colw = [1.5, 3.05, 1.8, 2.2, 2.2, 2.2, 2.2]
#     x = [0.0]
#     for w in colw:
#         x.append(x[-1] + w)
#     hdr = [(header_first, ""), ("Where, how big", ""), ("UTCI °C", "feels-like temp."), ("Air temp. Ta °C", "base → layout"), ("Radiant temp. Tmrt °C", "base → layout"),
#            ("Humidity RH %", "base → layout"), ("Wind m/s", "base → layout")]
#     hh = 0.62
#     ax.add_patch(plt.Rectangle((0, 0), x[-1], hh, fc="#e8eef2", ec="#c8ced3", lw=0.8))
#     for k, (t, s) in enumerate(hdr):
#         cx = 0.5 * (x[k] + x[k + 1])
#         ax.text(cx, hh * (0.40 if s else 0.5), t, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
#         if s:
#             ax.text(cx, hh * 0.78, s, ha="center", va="center", fontsize=8.5, color=GREY_TXT)
#     y = hh
#     allrows = list(rows) + ([dict(num=None, status="", where="Whole site (open ground)", area=site["area"], vals=site, site=True)] if with_site and site else [])
#     for rw in allrows:
#         v = rw["vals"]
#         is_site = rw.get("site", False)
#         ax.add_patch(plt.Rectangle((0, y), x[-1], cell_h, fc="#f1f3f5" if is_site else "white", ec="#c8ced3", lw=0.8))
#         # first column: numbered badge + status
#         if not is_site:
#             ax.text(x[0] + 0.32, y + cell_h / 2, str(rw["num"]), ha="center", va="center", fontsize=10.5, color="white", weight="bold",
#                     bbox=dict(boxstyle="circle,pad=0.28", fc=rw["color"], ec="#111", lw=0.9))
#             ax.text(x[0] + 0.62, y + cell_h / 2, rw["status"], ha="left", va="center", fontsize=8.3, color=INK)
#         # where / how big
#         ax.text(x[1] + 0.08, y + cell_h * 0.34, rw["where"], ha="left", va="center", fontsize=9.5, color=INK, weight="bold" if is_site else "normal")
#         ax.text(x[1] + 0.08, y + cell_h * 0.72, f"{rw['area']:,.0f} m²", ha="left", va="center", fontsize=9, color=GREY_TXT)
#         # UTCI
#         u = v["UTCI"]
#         ax.text(0.5 * (x[2] + x[3]), y + cell_h * 0.34, f"{_num(u['b'], 1)} → {_num(u['c'], 1)}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
#         ax.text(0.5 * (x[2] + x[3]), y + cell_h * 0.73, f"{_delta(u['d'], 1)} °C", ha="center", va="center", fontsize=9, color=BLUE_TXT if u["d"] < -0.05 else (RED_TXT if u["d"] > 0.05 else GREY_TXT))
#         for k, d in enumerate(ORDER):
#             c = 3 + k
#             vv = v[d]
#             nd = NDEC[d] if d != "RH" else 1
#             eff = vv["eff"]
#             a_ = min(abs(eff) / 1.5, 1.0) * 0.5
#             if a_ > 0.02:
#                 ax.add_patch(plt.Rectangle((x[c], y), colw[c], cell_h, fc=(0.84, 0.19, 0.15, a_) if eff > 0 else (0.17, 0.48, 0.73, a_), ec="none"))
#             cx = 0.5 * (x[c] + x[c + 1])
#             ax.text(cx, y + cell_h * 0.30, f"{_num(vv['b'], nd)} → {_num(vv['c'], nd)}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
#             t2, c2 = _fmt_eff(eff, d)
#             ax.text(cx, y + cell_h * 0.72, f"({_delta(vv['d'], nd)})   {t2}", ha="center", va="center", fontsize=8.8, color=c2)
#         y += cell_h
#     return y


# GLOSSARY = ("UTCI = the 'feels-like' temperature a person experiences (it combines the four columns on its right).   Ta = air temperature.   "
#             "Tmrt = radiant temperature: the heat a person receives from the sun and from hot surfaces (shade lowers it).   RH = relative humidity.   "
#             "Each cell: base → layout, (change), and what that change alone did to UTCI (blue = lowers UTCI, red = raises it).")


# def _band_legend(ax, y, present, w):
#     """colour chips of the UTCI categories that occur on the maps"""
#     ax.text(0.0, y, "Colour on the maps = UTCI category", fontsize=10.5, weight="bold", color=INK, va="top")
#     y += 0.34
#     for k in range(len(BAND_NAMES) - 1, -1, -1):
#         if k not in present:
#             continue
#         ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc=BAND_COL[k], ec="#777", lw=0.5))
#         ax.text(0.45, y + 0.1, f"{BAND_NAMES[k]}", fontsize=8.8, va="center", color=INK)
#         ax.text(w, y + 0.1, BAND_RANGE[k] + " °C", fontsize=8.3, va="center", ha="right", color=GREY_TXT)
#         y += 0.27
#     return y


# def _symbol_legend(ax, y, items):
#     """items: [(kind, colour, text)] kind in patch / circle / line"""
#     for kind, col, txt in items:
#         if kind == "patch":
#             ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc=col, ec="#555", lw=0.6))
#         elif kind == "outline":
#             ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc="none", ec=col, lw=1.4))
#         elif kind == "dash":
#             ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc="none", ec=col, lw=1.4, ls="--"))
#         elif kind == "circle":
#             ax.plot([0.17], [y + 0.1], "o", ms=9, mfc=col, mec="#111", mew=0.9)
#         ax.text(0.45, y + 0.1, txt, fontsize=8.8, va="center", color=INK)
#         y += 0.28
#     return y


# def _site_centre(grid):
#     m = grid.site_mask
#     return (float(grid.X[m].mean()), float(grid.Y[m].mean()))


# # ---------------------------------------------------------------------------------------- the words (2-3 sentences each)
# def _drv_plain(d):
#     return {"Tmrt": "radiant heat from sun and hot surfaces (shade)", "wind": "wind", "Ta": "air temperature", "RH": "humidity"}[d]


# def _area_weighted(zones, fn):
#     tot = sum(z["area"] for z in zones)
#     return sum(z["area"] * fn(z) for z in zones) / tot if tot > 0 else 0.0


# def _vtxt_short(d, b, c):
#     nd = NDEC[d] if d != "RH" else 1
#     lab = {"Ta": "Ta", "Tmrt": "Tmrt", "RH": "RH", "wind": "wind"}[d]
#     return f"{lab} {_num(b, nd)} → {_num(c, nd)} {UNIT[d]}"


# def short_spot_inference(R, base, cand, shown_index):
#     """three sentences about the hot / cold spots (list of str)"""
#     heat = R.season == "heat"
#     word = "hot" if heat else "cold"
#     sb, sc = R.spot_area_b, R.spot_area_c
#     n_all = len(R.regions)
#     n_res = sum(1 for g in R.regions if g["status"] == "resolved")
#     pct = 100 * (sb - sc) / sb if sb > 0 else 0.0
#     out = []
#     verb = "shrinks" if sc < sb else ("grows" if sc > sb else "stays")
#     s1 = (f"{word.capitalize()} ground (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C) {verb} from {sb:,.0f} to {sc:,.0f} m² ({'−' if sc <= sb else '+'}{abs(pct):.0f} %)")
#     if n_all:
#         s1 += f": {n_res} of {n_all} base {word} spots are gone"
#     if R.regions:
#         zs = R.regions
#         ub, uc = _area_weighted(zs, lambda z: z["u_base"]), _area_weighted(zs, lambda z: z["u_cand"])
#         s1 += f", and UTCI inside them goes {ub:.1f} → {uc:.1f} °C ({_delta(uc - ub, 1)})"
#     out.append(s1 + ".")
#     if R.regions:
#         contrib = {d: _area_weighted(zs, lambda z, d=d: z["contrib"][d]) for d in DRIVERS}
#         vb = {d: _area_weighted(zs, lambda z, d=d: z["vals"][d]["b"]) for d in DRIVERS}
#         vc = {d: _area_weighted(zs, lambda z, d=d: z["vals"][d]["c"]) for d in DRIVERS}
#         order = [d for d in sorted(DRIVERS, key=lambda d: -abs(contrib[d])) if abs(contrib[d]) >= 0.1][:2]
#         if order:
#             bits = [f"{_drv_plain(d)} ({_vtxt_short(d, vb[d], vc[d])}, worth {_delta(contrib[d], 1)} °C of UTCI)" for d in order]
#             s2 = "Main cause: " + " and ".join(bits)
#             dd = R.diff
#             if dd["trees_cand"] != dd["trees_base"]:
#                 s2 += f"; the layout has {dd['trees_cand']} trees instead of {dd['trees_base']} (canopy {dd['canopy_base']:.0f} → {dd['canopy_cand']:.0f} m²)"
#             out.append(s2 + ".")
#     left = sorted([g for g in R.regions if g["status"] in ("unchanged", "worse", "reduced", "partly resolved")], key=lambda g: -g["area"])
#     s3 = ""
#     if left:
#         g = left[0]
#         cause, _act = mitigation_for_region(g, R, R.season)
#         s3 = f"Still {word}: spot {shown_index.get(id(g), '?')} ({g['area']:,.0f} m², UTCI {g['u_cand']:.1f} °C), cause: {cause}"
#     if R.new_regions:
#         g = max(R.new_regions, key=lambda x: x["area"])
#         nb = g.get("near_building")
#         s3 += ("; " if s3 else "") + f"new {word} ground of {sum(x['area'] for x in R.new_regions):,.0f} m²" + (f" appears next to {_bn(nb)}" if nb else " appears")
#     if not s3:
#         s3 = f"No {word} spot of at least 80 m² remains and no new one appears" if n_all else f"The base case has no {word} spot of at least 80 m², and the layout adds none"
#     out.append(s3.rstrip(".") + ".")
#     return out


# def short_vent_inference(R, base, cand, shown_index):
#     heat = R.season == "heat"
#     v = R.vent
#     wb = R.wakes_b
#     newc = [w for w in R.wakes_c if w.get("is_new")]
#     thr = R.wake_frac * R.u_ref
#     n_gone = sum(1 for w in wb if w["fate"] == "gone")
#     n_small = sum(1 for w in wb if w["fate"] == "smaller")
#     out = [f"Calm ground (wind < {thr:.2f} m/s) goes from {v['calm_area_b']:,.0f} to {v['calm_area_c']:,.0f} m² ({100 * v['calm_share_b']:.0f} → {100 * v['calm_share_c']:.0f} % of the open ground) "
#            f"and the mean wind from {R.stats_b['wind_ms']:.2f} to {R.stats_c['wind_ms']:.2f} m/s: of {len(wb)} base calm zones {n_gone} are gone and {n_small} smaller"
#            + (f", {len(newc)} new one(s) appear" if newc else "") + "."]
#     cw = R.sgn * R.contrib_mean["wind"]
#     cm = R.calm_b & R.B["valid"]
#     if heat:
#         s2 = ""
#         if cm.any():
#             pen = float(np.nanmean(R.B["UTCI_C"][cm])) - R.stats_b["UTCI_C"]
#             s2 = f"In the hot hour still air costs heat: the base calm zones were {_delta(pen, 1)} °C in UTCI against the site mean. "
#         if cw > 0.15:
#             s2 += f"The wind change alone lowers the site-mean UTCI by {cw:.1f} °C, so ventilation is part of the benefit."
#         elif cw < -0.15:
#             s2 += f"The wind change alone raises the site-mean UTCI by {abs(cw):.1f} °C; the benefit comes from shade and air temperature instead."
#         else:
#             s2 += "The wind change alone hardly moves the site-mean UTCI, so the benefit does not come from ventilation."
#     else:
#         s2 = ("In the cold hour calm air is shelter from the wind chill. " + (f"The wind change alone {'raises' if cw > 0 else 'lowers'} the site-mean UTCI by {abs(cw):.1f} °C "
#               f"({'less' if cw > 0 else 'more'} wind chill)."))
#     out.append(s2)
#     shown, _ = _rows_of_zones(wb + newc, cap=1)
#     s3 = ""
#     if shown:
#         w = shown[0]
#         vv = w["vals"]
#         fate = "new in the layout" if w.get("is_new") else {"gone": "gone in the layout", "smaller": "smaller in the layout", "similar": "about the same", "larger": "larger in the layout"}[w["fate"]]
#         s3 = (f"Largest zone (zone {shown_index.get(id(w), '?')}{', behind ' + _bn(w['building']) if w.get('building') else ''}, {w['area']:,.0f} m², {w['length_m']:.0f} m long): {fate}; "
#               f"wind {vv['wind']['b']:.2f} → {vv['wind']['c']:.2f} m/s, UTCI {vv['UTCI']['b']:.1f} → {vv['UTCI']['c']:.1f} °C")
#         if heat and not w.get("is_new") and w["fate"] in ("similar", "larger") and w.get("building"):
#             s3 += f"; to fix it shift {_bn(w['building'])} off the wind line or leave a gap upwind"
#     if s3:
#         out.append(s3 + ".")
#     return out


# def _write_inference(path, title, sentences):
#     path.write_text(title + "\n\n" + "\n".join(sentences) + "\n")


# def _inference_box(ax, wbox, hbox, sentences, fs=11.0):
#     ax.add_patch(plt.Rectangle((0, 0), wbox, hbox, fc="#f6f8f9", ec="#d5dbdf", lw=0.8))
#     ax.text(0.15, 0.12, "What it means", fontsize=12.5, weight="bold", color=INK, va="top")
#     nch = int((wbox - 0.6) / (fs * 0.0070))
#     y = 0.5
#     for s in sentences:
#         ls = textwrap.wrap(s, width=nch)
#         ax.text(0.15, y, "\n".join(ls), fontsize=fs, color="#222", va="top", linespacing=1.3)
#         y += len(ls) * 0.215 + 0.1


# def _inference_height(wbox, sentences, fs=11.0):
#     nch = int((wbox - 0.6) / (fs * 0.0070))
#     return 0.5 + sum(len(textwrap.wrap(s, width=nch)) * 0.215 + 0.1 for s in sentences) + 0.15


# def _index_map(zones):
#     z, _ = _rows_of_zones(zones, cap=10 ** 6)
#     return {id(x): k for k, x in enumerate(z, 1)}


# # ---------------------------------------------------------------------------------------- the two figures
# def fig_simple_spots(R, base, cand, grid, path):
#     heat = R.season == "heat"
#     word = "Hot" if heat else "Cold"
#     zones_all = R.regions + R.new_regions
#     shown, hidden = _rows_of_zones(zones_all)
#     centre = _site_centre(grid)
#     rows = []
#     for k, z in enumerate(shown, 1):
#         is_new = z["id"].startswith("n")
#         st = "new" if is_new else z["status"]
#         rows.append(dict(num=k, color=STATUS_COL[st], status=STATUS_TXT[st], where=_short_where(z["centroid"], cand.layout, centre), area=z["area"], vals=z["vals"], zone=z, is_new=is_new))
#     paras = short_spot_inference(R, base, cand, _index_map(zones_all))
#     W = SIMPLE_W
#     M = 0.4
#     ms = 5.55
#     n_rows = len(rows) + 1
#     tab_h = 0.62 + 0.56 * n_rows + (0.32 if not rows else 0.0)
#     hid_txt = f"{len(hidden)} smaller {word.lower()} spot(s) ({sum(h['area'] for h in hidden):,.0f} m²) are not shown; see tables/regions.csv." if hidden else ""
#     para_h = _inference_height(W - 2 * M, paras)
#     H = 0.95 + 0.55 + ms + 0.3 + tab_h + 0.75 + para_h + 0.25
#     fig = plt.figure(figsize=(W, H), facecolor="white")
#     fig.text(M / W, 1 - 0.38 / H, f"{word} spots: {cand.name} compared with the base case — {SEASON_TXT[R.season]}", fontsize=17, weight="bold", color=INK, va="center")
#     fig.text(M / W, 1 - 0.74 / H, f"Pedestrian height; {word.lower()} spot = UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C in the base case.  Numbers on the maps are the spots listed in the table below.",
#              fontsize=10.5, color=GREY_TXT, va="center")
#     y0 = 0.95 + 0.55
#     xs = [M, M + ms + 0.3]
#     allu = np.concatenate([R.B["UTCI_C"][R.B["valid"]], R.C["UTCI_C"][R.C["valid"]]])
#     uv0, uv1 = float(np.nanpercentile(allu, 1)), float(np.nanpercentile(allu, 99))
#     lvu = _nice_levels(uv0, max(uv1, uv0 + 3.0), 12)
#     cmu = "YlOrRd" if heat else "Blues_r"
#     axes = []
#     for k, (Fld, lay, name, stats, spot) in enumerate(((R.B, base.layout, "BASE CASE", R.stats_b, R.spot_b), (R.C, cand.layout, f"{cand.name.upper()}  (best layout)", R.stats_c, R.spot_c))):
#         ax = fig.add_axes([xs[k] / W, 1 - (y0 + ms) / H, ms / W, ms / H])
#         Z = np.ma.masked_invalid(np.where(Fld["valid"], Fld["UTCI_C"], np.nan))
#         cfu = ax.contourf(grid.X, grid.Y, np.clip(Z, lvu[0], lvu[-1]), levels=lvu, cmap=cmu, zorder=1)
#         sub = f"mean UTCI {stats['UTCI_C']:.1f} °C   ·   {word.lower()} ground {(R.spot_area_b if k == 0 else R.spot_area_c):,.0f} m²"
#         _map_panel(ax, grid, lay, name, sub)
#         _outline(ax, grid, R.spot_b, colors="k", linewidths=1.6, linestyles="-" if k == 0 else "--")
#         if k == 1 and R.new_regions:
#             for z in R.new_regions:
#                 _outline(ax, grid, z["mask"], colors="#8e44ad", linewidths=1.6, linestyles="--")
#         axes.append(ax)
#     # key
#     kx = M + 2 * ms + 0.3 + 0.45
#     kw = W - M - kx
#     axk = _inch_axes(fig, W, H, kx, y0 - 0.05, kw, ms + 0.1)
#     axk.text(0.0, 0.05, "Colour on the maps = UTCI, °C", fontsize=10.5, weight="bold", color=INK, va="top")
#     cax = fig.add_axes([kx / W, 1 - (y0 - 0.05 + 0.78) / H, kw / W, 0.2 / H])
#     cb = fig.colorbar(cfu, cax=cax, orientation="horizontal")
#     cb.ax.tick_params(labelsize=8.5)
#     present = [k for k in range(len(BAND_NAMES)) if BAND_EDGES[k + 1] > lvu[0] and BAND_EDGES[k] < lvu[-1]]
#     yk = 1.2
#     axk.text(0.0, yk, "UTCI categories (stress level)", fontsize=10.0, weight="bold", color=INK, va="top")
#     yk += 0.3
#     for k in reversed(present):
#         axk.text(0.0, yk + 0.095, BAND_RANGE[k] + " °C", fontsize=8.5, va="center", color=GREY_TXT)
#         axk.text(1.15, yk + 0.095, BAND_NAMES[k], fontsize=8.5, va="center", color=INK)
#         yk += 0.25
#     yk += 0.12
#     axk.text(0.0, yk, "On the maps", fontsize=10.0, weight="bold", color=INK, va="top")
#     yk += 0.3
#     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="#8a8f94", ec="#555", lw=0.6))
#     axk.text(0.45, yk + 0.095, "building (name inside)", fontsize=8.5, va="center", color=INK)
#     yk += 0.25
#     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="none", ec="#2b6cb0", lw=1.2))
#     axk.text(0.45, yk + 0.095, "ponds, lawns, tree crowns (thin lines)", fontsize=8.5, va="center", color=INK)
#     yk += 0.25
#     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="none", ec="k", lw=1.5))
#     axk.text(0.45, yk + 0.095, "black outline = the spot (dashed in the layout)", fontsize=8.5, va="center", color=INK)
#     yk += 0.37
#     axk.text(0.0, yk, f"Circle colour = what became of the spot", fontsize=10.0, weight="bold", color=INK, va="top")
#     yk += 0.3
#     for col, txt in [(STATUS_COL["resolved"], "gone (≥ 70 % no longer " + ("hot)" if heat else "cold)")), (STATUS_COL["partly resolved"], "partly gone"),
#                      (STATUS_COL["reduced"], "reduced, still " + word.lower()), (STATUS_COL["unchanged"], "unchanged"), (STATUS_COL["worse"], "worse"),
#                      (STATUS_COL["new"], "new in the layout (purple outline)")]:
#         axk.plot([0.17], [yk + 0.095], "o", ms=8.5, mfc=col, mec="#111", mew=0.9)
#         axk.text(0.45, yk + 0.095, txt, fontsize=8.5, va="center", color=INK)
#         yk += 0.25
#     # badges on both maps
#     badge_items = [dict(label=str(r["num"]), xy=r["zone"]["centroid"], color=STATUS_COL["new" if r["is_new"] else r["zone"]["status"]]) for r in rows]
#     for ax in axes:
#         _place_badges(fig, ax, badge_items)
#     # table
#     ty = y0 + ms + 0.3
#     axt = _inch_axes(fig, W, H, M, ty, W - 2 * M, tab_h + 0.75)
#     used = _draw_zone_table(axt, W, rows, R.site_vals)
#     gy = used + 0.1
#     if not rows:
#         axt.text(0.0, used + 0.08, f"No {word.lower()} spot of at least {80:.0f} m² in the base case.", fontsize=11, color=INK, va="top")
#         gy = used + 0.42
#     axt.text(0.0, gy, "\n".join(textwrap.wrap(GLOSSARY, 205)) + ("\n" + hid_txt if hid_txt else ""), fontsize=8.6, color=GREY_TXT, va="top", linespacing=1.3)
#     # sentences
#     py = ty + tab_h + 0.75
#     axp = _inch_axes(fig, W, H, M, py, W - 2 * M, para_h)
#     _inference_box(axp, W - 2 * M, para_h, paras)
#     fig.savefig(path, dpi=SIMPLE_DPI, facecolor="white")
#     plt.close(fig)
#     _write_inference(Path(path).with_suffix(".txt"), f"{word} spots - {cand.name} vs base - {SEASON_TXT[R.season]}", paras)
#     return True


# def fig_simple_vent(R, base, cand, grid, path):
#     heat = R.season == "heat"
#     wb = R.wakes_b
#     newc = [w for w in R.wakes_c if w.get("is_new")]
#     zones = wb + newc
#     shown, hidden = _rows_of_zones(zones)
#     centre = _site_centre(grid)
#     rows = []
#     for k, z in enumerate(shown, 1):
#         is_new = bool(z.get("is_new"))
#         st = "new" if is_new else z["fate"]
#         w = z
#         where = _short_where(z["centroid"], cand.layout if is_new else base.layout, centre)
#         if z.get("building"):
#             where = f"behind {_bn(z['building'])}, {compass(z['centroid'][0] - centre[0], centre[1] * 0 + z['centroid'][1] - centre[1])} part"
#         rows.append(dict(num=k, color=STATUS_COL[st], status=STATUS_TXT[st], where=where, area=z["area"], vals=z["vals"], zone=z, is_new=is_new))
#     paras = short_vent_inference(R, base, cand, _index_map(zones))
#     W = SIMPLE_W
#     M = 0.4
#     ms = 5.55
#     n_rows = len(rows) + 1
#     tab_h = 0.62 + 0.56 * n_rows + (0.32 if not rows else 0.0)
#     hid_txt = f"{len(hidden)} smaller calm zone(s) ({sum(h['area'] for h in hidden):,.0f} m²) are not shown; see tables/wakes.csv." if hidden else ""
#     para_h = _inference_height(W - 2 * M, paras)
#     H = 0.95 + 0.55 + ms + 0.3 + tab_h + 0.75 + para_h + 0.25
#     fig = plt.figure(figsize=(W, H), facecolor="white")
#     thr = R.wake_frac * R.u_ref
#     fig.text(M / W, 1 - 0.38 / H, f"Ventilation: {cand.name} compared with the base case — {SEASON_TXT[R.season]}", fontsize=17, weight="bold", color=INK, va="center")
#     fig.text(M / W, 1 - 0.74 / H, f"Wind speed at pedestrian height.  Calm zone = wind below {thr:.2f} m/s ({R.wake_frac:g} × the free wind of the base case, {R.u_ref:.2f} m/s); "
#                                   f"{'still air traps heat, so calm zones are a problem in this hour.' if heat else 'calm air is shelter from the wind chill in this hour.'}",
#              fontsize=10.5, color=GREY_TXT, va="center")
#     y0 = 0.95 + 0.55
#     xs = [M, M + ms + 0.3]
#     vmax = float(np.ceil(10 * max(1.0, np.nanpercentile(np.concatenate([R.B["wind_ms"][R.B["valid"]], R.C["wind_ms"][R.C["valid"]]]), 99))) / 10)
#     lv = _nice_levels(0.0, vmax, 10)
#     axes, cf = [], None
#     for k, (Fld, lay, name, stats, calm, wk) in enumerate(((R.B, base.layout, "BASE CASE", R.stats_b, R.calm_b, R.wakes_b), (R.C, cand.layout, f"{cand.name.upper()}  (best layout)", R.stats_c, R.calm_c, R.wakes_c))):
#         ax = fig.add_axes([xs[k] / W, 1 - (y0 + ms) / H, ms / W, ms / H])
#         Z = np.ma.masked_invalid(np.where(Fld["valid"], Fld["wind_ms"], np.nan))
#         cf = ax.contourf(grid.X, grid.Y, np.clip(Z, lv[0], lv[-1]), levels=lv, cmap="Blues", zorder=1)
#         sub = f"mean wind {stats['wind_ms']:.2f} m/s   ·   calm ground {(R.vent['calm_area_b'] if k == 0 else R.vent['calm_area_c']):,.0f} m²"
#         _map_panel(ax, grid, lay, name, sub)
#         ax.contourf(grid.X, grid.Y, np.where(calm, 1.0, np.nan), levels=[0.5, 1.5], colors=["#c0392b"], alpha=0.35, zorder=2)
#         _outline(ax, grid, calm, colors="#9c1c10", linewidths=1.4)
#         _wind_arrow_simple(ax, grid, R.flow)
#         axes.append(ax)
#     kx = M + 2 * ms + 0.3 + 0.45
#     kw = W - M - kx
#     axk = _inch_axes(fig, W, H, kx, y0 - 0.05, kw, ms + 0.1)
#     axk.text(0.0, 0.05, "Colour on the maps = wind speed", fontsize=10.5, weight="bold", color=INK, va="top")
#     cax = fig.add_axes([(kx) / W, 1 - (y0 - 0.05 + 0.75) / H, kw / W, 0.2 / H])
#     cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
#     cb.set_label("m/s", fontsize=9)
#     cb.ax.tick_params(labelsize=8.5)
#     yk = 1.35
#     axk.text(0.0, yk, "Symbols", fontsize=10.5, weight="bold", color=INK, va="top")
#     yk += 0.34
#     yk = _symbol_legend(axk, yk, [("patch", "#8a8f94", "building (name inside)"), ("outline", "#2b6cb0", "pond outline"), ("outline", "#8fb98a", "lawn outline"), ("outline", "#2f7d32", "tree crown")])
#     yk += 0.12
#     axk.text(0.0, yk, "Calm zones", fontsize=10.5, weight="bold", color=INK, va="top")
#     yk += 0.34
#     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.2, fc="#c0392b", alpha=0.35, ec="#9c1c10", lw=1.4))
#     axk.text(0.45, yk + 0.1, "calm zone: wind < %.2f m/s" % thr, fontsize=8.6, va="center", color=INK)
#     yk += 0.28
#     for kind, col, txt in [("circle", STATUS_COL["gone"], "base calm zone: gone in the layout"), ("circle", STATUS_COL["smaller"], "smaller"), ("circle", STATUS_COL["similar"], "about the same"),
#                            ("circle", STATUS_COL["larger"], "larger"), ("circle", STATUS_COL["new"], "new in the layout")]:
#         axk.plot([0.17], [yk + 0.1], "o", ms=9, mfc=col, mec="#111", mew=0.9)
#         axk.text(0.45, yk + 0.1, txt, fontsize=8.6, va="center", color=INK)
#         yk += 0.28
#     badge_items = [dict(label=str(r["num"]), xy=r["zone"]["centroid"], color=r["color"]) for r in rows]
#     for ax in axes:
#         _place_badges(fig, ax, badge_items)
#     ty = y0 + ms + 0.3
#     axt = _inch_axes(fig, W, H, M, ty, W - 2 * M, tab_h + 0.75)
#     used = _draw_zone_table(axt, W, rows, R.site_vals, header_first="Calm zone")
#     gy = used + 0.1
#     if not rows:
#         axt.text(0.0, used + 0.08, "No calm zone of at least 80 m² in either layout.", fontsize=11, color=INK, va="top")
#         gy = used + 0.42
#     axt.text(0.0, gy, "\n".join(textwrap.wrap(GLOSSARY, 205)) + ("\n" + hid_txt if hid_txt else ""), fontsize=8.6, color=GREY_TXT, va="top", linespacing=1.3)
#     py = ty + tab_h + 0.75
#     axp = _inch_axes(fig, W, H, M, py, W - 2 * M, para_h)
#     _inference_box(axp, W - 2 * M, para_h, paras)
#     fig.savefig(path, dpi=SIMPLE_DPI, facecolor="white")
#     plt.close(fig)
#     _write_inference(Path(path).with_suffix(".txt"), f"ventilation - {cand.name} vs base - {SEASON_TXT[R.season]}", paras)
#     return True


# def fig_categories(R, path):
#     names = [c[1] for c in CATEGORIES][::-1]
#     names = ["extreme cold stress"] + [c[1] for c in CATEGORIES][::-1][:0] + [c[1] for c in CATEGORIES][::-1]
#     cats = ["extreme cold stress", "very strong cold stress", "strong cold stress", "moderate cold stress", "slight cold stress", "no thermal stress",
#             "moderate heat stress", "strong heat stress", "very strong heat stress", "extreme heat stress"]
#     cats = [c for c in cats if R.cat_b.get(c, 0) > 0 or R.cat_c.get(c, 0) > 0]
#     fig, ax = plt.subplots(figsize=(8.5, 3.8))
#     x = np.arange(len(cats))
#     ax.bar(x - 0.2, [100 * R.cat_b[c] / R.stats_b["area_m2"] for c in cats], 0.4, color="#8a9aa6", label="base")
#     ax.bar(x + 0.2, [100 * R.cat_c[c] / R.stats_c["area_m2"] for c in cats], 0.4, color="#2b7bba", label=R.name)
#     ax.set_xticks(x)
#     ax.set_xticklabels([c.replace(" stress", "\nstress").replace("no thermal\nstress", "no thermal\nstress") for c in cats], fontsize=7)
#     ax.set_ylabel("% of the walkable area")
#     ax.legend(fontsize=8)
#     ax.set_title(f"share of the area per UTCI category — {SEASON_TXT[R.season]}", fontsize=10)
#     fig.tight_layout()
#     fig.savefig(path, dpi=80)
#     plt.close(fig)


# def fig_overview(results, path):
#     """all selected layouts: benefit by driver, per season"""
#     names = sorted({r.name for r in results})
#     seasons = [s for s in SEASONS if any(r.season == s for r in results)]
#     fig, axs = plt.subplots(1, len(seasons), figsize=(6.5 * len(seasons) + 1, 4.8), squeeze=False)
#     for ax, s in zip(axs[0], seasons):
#         rs = {r.name: r for r in results if r.season == s}
#         x = np.arange(len(rs))
#         pos, neg = np.zeros(len(rs)), np.zeros(len(rs))
#         for d in DRIVERS:
#             v = np.array([rs[n].sgn * rs[n].contrib_mean[d] for n in rs])
#             ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), color=COL[d], label=d, width=0.6)
#             pos += np.where(v >= 0, v, 0)
#             neg += np.where(v < 0, v, 0)
#         ax.plot(x, [rs[n].sgn * rs[n].dU_mean for n in rs], "kD", ms=6, label="total")
#         ax.set_xticks(x)
#         ax.set_xticklabels(list(rs), rotation=20, fontsize=8)
#         ax.axhline(0, color=INK, lw=0.8)
#         ax.set_ylabel("°C of UTCI benefit (site mean)")
#         ax.set_title(f"{SEASON_TXT[s]}: what produced the benefit", fontsize=10)
#         ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
#     fig.tight_layout()
#     fig.savefig(path, dpi=80)
#     plt.close(fig)


# # ============================================================================= text: findings and mitigation
# def fmt_area(a):
#     return f"{a:,.0f} m²"


# def where_text(centroid, layout, site_centre, flow=None):
#     d = compass(centroid[0] - site_centre[0], centroid[1] - site_centre[1])
#     nb = nearest_building(layout, centroid, flow)
#     s = f"{d} part of the site"
#     if nb is not None:
#         s += f", {nb[1]:.0f} m from {nb[0]}"
#     return s


# def driver_sentence(contrib, sgn, total=None, drivers=DRIVERS):
#     """'Tmrt -3.1 °C (82 %), wind ...' in order of importance, signed as UTCI change; the share is shown only for terms that go the same way as the total"""
#     tot = sum(contrib[d] for d in drivers) if total is None else total
#     items = sorted(drivers, key=lambda d: -abs(contrib[d]))
#     parts = []
#     for d in items:
#         if abs(tot) > 1e-6 and contrib[d] * tot > 0:
#             parts.append(f"{d} {contrib[d]:+.2f} °C ({100 * contrib[d] / tot:.0f} %)")
#         elif abs(contrib[d]) >= 0.005:
#             parts.append(f"{d} {contrib[d]:+.2f} °C (opposes)")
#         else:
#             parts.append(f"{d} {contrib[d]:+.2f} °C")
#     return ", ".join(parts)


# def explain_region(r, R, season):
#     """one paragraph for a hot / cold region"""
#     heat = season == "heat"
#     ctx = r["context"]
#     s = (f"{r['status'].upper()}: UTCI {r['u_base']:.1f} → {r['u_cand']:.1f} °C ({r['d_u']:+.1f} °C); {100 * r['resolved_share']:.0f} % of its area is no longer "
#          f"{'above' if heat else 'below'} {R.thr:.1f} °C" + (" (cells just beyond the threshold flip with a small change)" if r["status"] in ("unchanged", "reduced") and r["resolved_share"] >= 0.5 else "") + ". Zone means, base → layout, and what each change did to UTCI: " + inputs_sentence(r["vals"]) + ".")
#     cb, _vb = cause_of(r.get("cause_b"), heat)
#     if cb:
#         s += f" In the base it was {'hotter' if heat else 'colder'} than the rest of the site mainly because of {cb} ({(_vb if heat else -_vb):+.1f} °C)."
#     det = []
#     if abs(r["sunlit_c"] - r["sunlit_b"]) >= 0.1:
#         det.append(f"sunlit share {r['sunlit_b']:.2f} → {r['sunlit_c']:.2f}")
#     if abs(r["sky_open_c"] - r["sky_open_b"]) >= 0.05:
#         det.append(f"sky view {r['sky_open_b']:.2f} → {r['sky_open_c']:.2f}")
#     det.append(f"wind {r['wind_ratio_b']:.2f} → {r['wind_ratio_c']:.2f} of the reference wind")
#     s += " (" + "; ".join(det) + ")."
#     ch = []
#     if ctx["trees_added_near"]:
#         ch.append(f"{ctx['trees_added_near']} new trees within 25 m (+{ctx['canopy_added_near_m2']:.0f} m² canopy)")
#     if ctx["trees_removed_near"]:
#         ch.append(f"{ctx['trees_removed_near']} trees removed within 25 m")
#     if ctx.get("trees_added_upwind"):
#         ch.append(f"{ctx['trees_added_upwind']} new trees in the 70 m upwind corridor")
#     if ctx.get("trees_removed_upwind"):
#         ch.append(f"{ctx['trees_removed_upwind']} trees removed from the upwind corridor")
#     dp = ctx["pond_area_near_cand"] - ctx["pond_area_near_base"]
#     if abs(dp) > 30:
#         ch.append(f"pond area within 25 m {ctx['pond_area_near_base']:.0f} → {ctx['pond_area_near_cand']:.0f} m²")
#     dpu = ctx.get("pond_area_upwind_cand", 0) - ctx.get("pond_area_upwind_base", 0)
#     if abs(dpu) > 30:
#         ch.append(f"pond area upwind {ctx['pond_area_upwind_base']:.0f} → {ctx['pond_area_upwind_cand']:.0f} m²")
#     dl = ctx["lawn_area_near_cand"] - ctx["lawn_area_near_base"]
#     if abs(dl) > 30:
#         ch.append(f"lawn within 25 m {ctx['lawn_area_near_base']:.0f} → {ctx['lawn_area_near_cand']:.0f} m²")
#     for bid, sh in ctx["buildings_moved_near"]:
#         ch.append(f"{bid} moved {sh:.1f} m")
#     s += (" Layout change near it: " + "; ".join(ch) + ".") if ch else " No layout change within 25 m (the change comes from elsewhere in the flow or from the whole site)."
#     return s


# def cause_of(cc, heat, min_c=0.3):
#     """dominant driver of the excess of a region over the rest of the site: (driver or None, its contribution in the stress direction)"""
#     if not cc:
#         return None, 0.0
#     sc = {d: (cc["contrib"][d] if heat else -cc["contrib"][d]) for d in DRIVERS}
#     d = max(DRIVERS, key=lambda k: sc[k])
#     return (d if sc[d] >= min_c else None), sc[d]


# def mitigation_for_region(r, R, season):
#     """(cause sentence, action sentence) for a region that is still stressed in the layout"""
#     heat = season == "heat"
#     d, val = cause_of(r.get("cause_c"), heat)
#     wr = r.get("wind_c", float("nan")) / R.u_ref if "wind_c" in r else float("nan")
#     sun, sky, tm, ta = r.get("sunlit_c", float("nan")), r.get("sky_open_c", float("nan")), r.get("Tmrt_c", float("nan")), r.get("Ta_c", float("nan"))
#     nb = r.get("near_building")
#     if d is None:
#         return ("no single driver explains it (every driver within 0.3 °C of the rest of the site)",
#                 "mixed: the whole site is stressed here rather than a pocket; use the site-wide levers (shade and ventilation for heat, shelter and sun for cold).")
#     if heat:
#         if d == "Tmrt" and sun > 0.5:
#             return (f"radiant load, +{val:.1f} °C of UTCI over the rest of the site (sunlit {sun:.2f}, Tmrt {tm:.0f} °C vs air {ta:.0f} °C)",
#                     "shade it: wide, tall crowns (radius 4-6 m, height 10-15 m) on the sun side of the spot, or put a building / canopy so that its shadow falls on it; lawn or water instead of dark paving.")
#         if d == "Tmrt":
#             return (f"radiation although shaded, +{val:.1f} °C (sunlit {sun:.2f}, sky view {sky:.2f}, Tmrt {tm:.0f} °C)",
#                     "hot surfaces / trapped long-wave: open the sky a little (thin or lift the canopy), cool the surfaces (lawn, water, light paving).")
#         if d == "wind":
#             return (f"calm air, +{val:.1f} °C (wind {wr:.2f} of the reference)",
#                     "open a ventilation path: no tree row or wall directly upwind, line a gap between buildings up with the wind"
#                     + (f", shift the building upwind of it ({_bn(nb)}) sideways" if nb else "") + ".")
#         if d == "Ta":
#             return (f"warm air, +{val:.1f} °C (Ta {ta:.1f} °C)",
#                     "water or lawn UPWIND of the spot with a clear path to it (cooled air travels with the wind) rather than at the spot itself.")
#         return (f"humid air, +{val:.1f} °C", "keep ponds downwind of where people stay, or make them smaller / further away.")
#     if d == "wind":
#         return (f"wind chill, {val:.1f} °C colder than the rest of the site (wind {wr:.2f} of the reference)",
#                 "shelter: 2-3 staggered tree rows (crown radius 3-5 m) across the wind 10-25 m upwind of it; close channels between buildings that point along the wind.")
#     if d == "Tmrt":
#         return (f"radiation, {val:.1f} °C colder (sunlit {sun:.2f}, Tmrt {tm:.1f} °C vs air {ta:.1f} °C)",
#                 "keep winter sun access: no dense canopy on the sun side, a warm / low-albedo surface where the sun reaches it.")
#     if d == "Ta":
#         return (f"cold air, {val:.1f} °C below the rest (Ta {ta:.1f} °C)", "exposed to the approach air: shelter it from the approach direction so the air mixes less.")
#     return (f"humidity, {val:.1f} °C", "small effect in the cold hour; check the numbers before acting.")


# def build_findings(R, base, cand, grid):
#     """list of strings (markdown) that explain one candidate / season"""
#     sgn, heat = R.sgn, R.season == "heat"
#     L = []
#     obj = ""
#     if R.obj_b is not None and R.obj_c is not None:
#         obj = f" (objective, area-mean of the sample points: {R.obj_b:.2f} → {R.obj_c:.2f} °C, {R.obj_c - R.obj_b:+.2f})"
#     L.append(f"**{SEASON_TXT[R.season].capitalize()}.** Site-mean UTCI {R.stats_b['UTCI_C']:.2f} → {R.stats_c['UTCI_C']:.2f} °C on the common grid{obj}; "
#              f"category {category(R.stats_b['UTCI_C'])} → {category(R.stats_c['UTCI_C'])}. Where both layouts have open ground the mean change is "
#              f"{R.dU_mean:+.2f} °C.")
#     c = R.contrib_mean
#     ranking = sorted(DRIVERS, key=lambda d: -sgn * c[d])
#     top = ranking[0]
#     pos = [d for d in DRIVERS if sgn * c[d] > 0.05]
#     neg = [d for d in DRIVERS if sgn * c[d] < -0.05]
#     L.append("*Site mean, inputs and their effect on UTCI* (base → layout; the effect is the part of the UTCI change that this input alone caused): " + inputs_sentence(R.site_vals) + ".")
#     L.append(f"*Why:* the change of UTCI splits into  {driver_sentence(c, sgn, R.dU_mean)}.")
#     if abs(R.dU_mean) < 0.15:
#         L.append("The site mean hardly changes; the benefit, if any, is local (see the regions below).")
#     else:
#         word = "lowered" if R.dU_mean < 0 else "raised"
#         s = f"The mean UTCI was {word} mainly by **{DRIVER_LONG[top]}**"
#         if sgn * c[top] > 0:
#             s += f" (benefit {sgn * c[top]:+.2f} °C of the net {sgn * R.dU_mean:+.2f} °C)."
#         else:
#             s += "."
#         if neg:
#             s += " Working against the benefit: " + ", ".join(f"{d} ({sgn * c[d]:+.2f} °C)" for d in neg) + "."
#         L.append(s)
#     if R.tmrt is not None and abs(R.dTmrt_mean) > 0.1:
#         p = R.tmrt_parts_mean
#         items = sorted(p.items(), key=lambda kv: -abs(kv[1]))
#         L.append(f"*Inside Tmrt* (Δ Tmrt {R.dTmrt_mean:+.2f} °C): " + ", ".join(f"{k} {v:+.2f}" for k, v in items) +
#                  f" °C. Regression of Tmrt on sunlit share, sky view and Ta explains R² = {R.tmrt['r2']:.2f}. Sunlit share {R.stats_b['sunlit']:.2f} → {R.stats_c['sunlit']:.2f}, "
#                  f"sky view {R.stats_b['sky_open']:.2f} → {R.stats_c['sky_open']:.2f}.")
#     v = R.vent
#     L.append(f"*Ventilation* at 1.5 m: mean wind {R.stats_b['wind_ms']:.2f} → {R.stats_c['wind_ms']:.2f} m/s ({v['wind_ratio_b']:.2f} → {v['wind_ratio_c']:.2f} of the reference wind "
#              f"{R.u_ref:.1f} m/s); calm area (< {R.wake_frac:g} of reference) {fmt_area(v['calm_area_b'])} → {fmt_area(v['calm_area_c'])} "
#              f"({100 * v['calm_share_b']:.0f} → {100 * v['calm_share_c']:.0f} % of the open ground); longest wake {v['longest_wake_b']:.0f} → {v['longest_wake_c']:.0f} m.")
#     if heat:
#         wind_help = sgn * c["wind"]
#         if wind_help > 0.15:
#             L.append(f"→ Ventilation **improved** the heat case: it accounts for {wind_help:+.2f} °C of the benefit.")
#         elif wind_help < -0.15:
#             L.append(f"→ Ventilation got **worse** in the heat case ({wind_help:+.2f} °C); the benefit came from the other drivers.")
#         else:
#             L.append("→ Ventilation changed little in the heat case; the benefit does not come from wind.")
#     else:
#         wind_help = sgn * c["wind"]
#         if wind_help > 0.15:
#             L.append(f"→ Less wind at the pedestrian reduced the cold stress: wind accounts for {wind_help:+.2f} °C of the benefit (wind chill reduced).")
#         elif wind_help < -0.15:
#             L.append(f"→ More wind increased the cold stress ({wind_help:+.2f} °C); the other drivers compensated.")
#         else:
#             L.append("→ Wind chill hardly changed in the cold case; the benefit is from radiation / air temperature.")
#     # regions
#     L.append(f"*{'Hot' if heat else 'Cold'} spots of the base* (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C): {fmt_area(R.spot_area_b)} in the base, {fmt_area(R.spot_area_c)} in the layout "
#              f"({R.spot_area_c - R.spot_area_b:+,.0f} m²); {len(R.regions)} region(s), {len(R.new_regions)} new region(s) appear.")
#     return L


# def mitigation_text(results, base, grids, a):
#     """consolidated inference over all analysed layouts / seasons (list of markdown lines)"""
#     out = []
#     by = {s: [r for r in results if r.season == s] for s in SEASONS}
#     for s in SEASONS:
#         rs = by[s]
#         if not rs:
#             continue
#         heat = s == "heat"
#         sgn = -1.0 if heat else 1.0
#         out += [f"### {SEASON_TXT[s].capitalize()}", ""]
#         m = {d: float(np.mean([sgn * r.contrib_mean[d] for r in rs])) for d in DRIVERS}
#         tot = float(np.mean([sgn * r.dU_mean for r in rs]))
#         out += [f"**What worked.** Average over the {len(rs)} layouts: benefit {tot:+.2f} °C of site-mean UTCI = " +
#                 ", ".join(f"{d} {m[d]:+.2f}" for d in sorted(DRIVERS, key=lambda d: -abs(m[d]))) + " °C.", ""]
#         lead = max(DRIVERS, key=lambda d: m[d])
#         if heat:
#             tips = {"Tmrt": "The heat stress is dissolved mainly by cutting the radiant load: keep shade (tree crowns, building shadow) on the walkable ground and do not lose canopy to the other objective.",
#                     "wind": "The heat stress is dissolved mainly by ventilation: protect the open corridors along the wind and keep tree rows and walls out of the approach path of the hot spots.",
#                     "Ta": "The heat stress is dissolved mainly by cooler air: evaporation (ponds, lawns) and canopy cooling upwind of the spots is working; keep a clear wind path from the water to the places that need it.",
#                     "RH": "Humidity is the leading term: check that ponds do not raise RH where people stay."}
#         else:
#             tips = {"Tmrt": "The cold stress is reduced mainly by radiation: sun reaches the walkable ground; keep sun access when adding trees.",
#                     "wind": "The cold stress is reduced mainly by wind shelter: keep the shelter, and keep it from closing the heat-hour ventilation (see the trade-off below).",
#                     "Ta": "The cold stress is reduced mainly by warmer air (less mixing with the cold approach air, sheltered pockets).",
#                     "RH": "Humidity is the leading term; its effect on cold stress is small, check the numbers before drawing conclusions."}
#         out += [f"**Leading driver: {lead}.** " + tips[lead], ""]
#         # what is left: area-weighted causes over all leftover regions
#         causes, tot_area = {}, 0.0
#         for r in rs:
#             for reg in r.regions:
#                 if reg["status"] in ("unchanged", "worse", "reduced", "partly resolved"):
#                     d, _ = cause_of(reg.get("cause_c"), heat)
#                     causes[d or "mixed"] = causes.get(d or "mixed", 0.0) + reg["area"]
#                     tot_area += reg["area"]
#         if tot_area > 0:
#             out += ["**What is left.** Cause of the remaining " + ("hot" if heat else "cold") + " spots, weighted by area over all layouts: " +
#                     ", ".join(f"{k} {100 * v / tot_area:.0f} %" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) + ".", ""]
#         out += ["**Per layout: what to try next.**", ""]
#         for r in rs:
#             left = [g for g in r.regions if g["status"] in ("unchanged", "worse", "reduced", "partly resolved")]
#             left.sort(key=lambda g: -g["area"])
#             big = [g for g in left if g["area"] >= 150.0 or g["status"] == "worse"][:4]
#             n_res = sum(1 for g in r.regions if g["status"] == "resolved")
#             head = f"- **{r.name}**: {n_res} of {len(r.regions)} base spots resolved, {len(left)} left ({sum(g['area'] for g in left):,.0f} m²)"
#             if r.new_regions:
#                 head += f", {len(r.new_regions)} new ({sum(g['area'] for g in r.new_regions):,.0f} m²)"
#             out.append(head + ".")
#             seen = {}
#             for g in big:
#                 cause, act = mitigation_for_region(g, r, s)
#                 if act in seen:
#                     act = f"same action as {seen[act]}."
#                 else:
#                     seen[act] = g["id"]
#                 out.append(f"  - {g['id']} ({g['area']:,.0f} m², {g['status']}, UTCI {g['u_cand']:.1f} °C): cause — {cause}. Try: {act}")
#             if len(left) > len(big):
#                 rest = [g for g in left if g not in big]
#                 out.append(f"  - {len(rest)} smaller region(s) ({sum(g['area'] for g in rest):,.0f} m²): same levers as above.")
#             if r.new_regions:
#                 dom = {}
#                 for g in r.new_regions:
#                     dom[g.get("dominant", "?")] = dom.get(g.get("dominant", "?"), 0.0) + g["area"]
#                 nbs = sorted({g.get("near_building") for g in r.new_regions if g.get("near_building")})
#                 up = sum(g["context"].get("trees_added_upwind", 0) for g in r.new_regions)
#                 out.append(f"  - new stressed ground, worsened mainly by {', '.join(k for k, v in sorted(dom.items(), key=lambda kv: -kv[1]))}, next to {', '.join(map(str, nbs[:4]))}: "
#                            f"check what was added upwind of it ({up} trees added in its upwind corridors) and what moved; it is the price of the change that helped elsewhere.")
#             if heat:
#                 hot_w = [w for w in r.wakes_c if w["utci_mean"] > r.stats_c["UTCI_C"] + 0.3 and w["area"] >= 150.0]
#                 hot_w.sort(key=lambda w: -w["area"])
#                 for w in hot_w[:2]:
#                     out.append(f"  - wake {w['id']} ({w['area']:,.0f} m², {w['length_m']:.0f} m long, UTCI {w['utci_mean']:.1f} °C vs site {r.stats_c['UTCI_C']:.1f}), trails {w['building']}"
#                                f"{' (new in this layout)' if w.get('is_new') else ''}: shift {w['building']} off the wind line, narrow it across the wind, or leave a gap upwind.")
#         out.append("")
#     # trade-off
#     if by["heat"] and by["cold"]:
#         out += ["### Trade-off between the hot and the cold hour", ""]
#         for h in by["heat"]:
#             c_ = next((x for x in by["cold"] if x.name == h.name), None)
#             if c_ is None:
#                 continue
#             hw, cw = -h.contrib_mean["wind"], c_.contrib_mean["wind"]
#             ht, ct = -h.contrib_mean["Tmrt"], c_.contrib_mean["Tmrt"]
#             msg = (f"- **{h.name}**: heat benefit {-h.dU_mean:+.2f} °C (wind {hw:+.2f}, Tmrt {ht:+.2f}), cold benefit {c_.dU_mean:+.2f} °C (wind {cw:+.2f}, Tmrt {ct:+.2f}).")
#             if hw < -0.1 and cw > 0.1:
#                 msg += " The windbreak helps the cold hour but costs ventilation in the heat hour: a belt with gaps towards the heat-wind direction would keep both."
#             elif hw > 0.1 and cw < -0.1:
#                 msg += " The ventilation gained for the heat hour is paid for by more wind chill in the cold hour: shelter the cold-wind side separately."
#             elif ht > 0.1 and ct < -0.1:
#                 msg += " Shade that cools the heat hour also lowers Tmrt in the cold hour: keep the shade where the heat spots are, and keep the sun side open in winter."
#             elif ht > 0.1 and cw > 0.1:
#                 msg += " Shade for the heat hour and shelter for the cold hour pull in the same direction: trees do both here."
#             else:
#                 msg += " No direct conflict between the two hours in the drivers."
#             out.append(msg)
#         out.append("")
#     return out


# RULES_TABLE = [
#     ("Heat: Tmrt dominates the benefit, sunlit share fell", "Shade (tree crowns, building shadow) did the work", "More / larger crowns at the remaining sunlit spots; protect existing canopy"),
#     ("Heat: wind dominates, calm area shrank", "Ventilation did the work", "Keep corridors along the heat wind open; no tree rows / walls directly upwind of spots"),
#     ("Heat: Ta dominates (air cooler)", "Evaporation / canopy cooling reached the spots", "Water and lawn upwind of the spots, open path from water to spot"),
#     ("Heat: wake region behind a building with UTCI above the site mean", "Wake traps heat", "Shift the building off the wind line, split / narrow it across the wind, leave a gap upwind"),
#     ("Heat: RH term negative", "Humid air from ponds is carried onto people", "Ponds downwind of seating, or smaller ponds, or more evaporation distance"),
#     ("Cold: wind dominates, wind ratio fell", "Windbreak / building shelter did the work", "Keep the belt; gaps only where heat ventilation needs them"),
#     ("Cold: spot with wind ratio > 1.15", "Wind channel / exposure", "Staggered rows 10-25 m upwind across the wind; close channels aligned with the wind"),
#     ("Cold: spot shaded and Tmrt < Ta", "Missing sun / radiant heat", "Sun access (no dense canopy on the sun side), warm surfaces"),
#     ("A region got worse", "Something placed upwind or beside it", "Check the layout changes listed for the region and undo that one"),
# ]


# # ============================================================================= report
# def md_table(header, rows):
#     out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
#     for r in rows:
#         out.append("| " + " | ".join(str(c) for c in r) + " |")
#     return out


# def md_zone_table(zones, site_vals):
#     rows = []
#     for z in zones:
#         v = z.get("vals")
#         if v is None:
#             continue
#         tag = "new" if (z["id"].startswith("n") or z.get("is_new")) else (z.get("status") or z.get("fate") or "")
#         rows.append([f"{z['id']}" + ("" if z["id"][0] in "HCn" else (" (layout)" if z.get("is_new") else " (base)")), tag, f"{z['area']:.0f}", f"{v['UTCI']['b']:.1f} → {v['UTCI']['c']:.1f} ({v['UTCI']['d']:+.2f})"] +
#                     [vtxt(v, d) for d in ORDER])
#     sv = site_vals
#     rows.append(["whole site", "", f"{sv['area']:.0f}", f"{sv['UTCI']['b']:.1f} → {sv['UTCI']['c']:.1f} ({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d) for d in ORDER])
#     return md_table(["zone", "status", "area m²", "UTCI °C"] + [f"{d} {UNIT[d]}" for d in ORDER], rows)


# def md_to_html(md, img_dir=None, embed=True):
#     """small converter for the subset used here (headings, paragraphs, bullets, tables, images, **bold**, *italic*, `code`)"""
#     try:
#         import markdown as mdl
#         html = mdl.markdown(md, extensions=["tables"])
#     except Exception:
#         html = _mini_md(md)
#     if embed and img_dir is not None:
#         tot = sum(f.stat().st_size for f in Path(img_dir).glob("figures/*.png"))
#         if tot > 30e6:                                    # HD pictures: do not inline them (the html would be huge); link the files instead
#             embed = False
#     if embed and img_dir is not None:
#         def rep(m):
#             p = Path(img_dir) / m.group(1)
#             if p.is_file():
#                 return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
#             return m.group(0)
#         html = re.sub(r'src="([^"]+\.png)"', rep, html)
#     css = ("body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#1c2830;line-height:1.5}"
#            "h1,h2,h3{color:#14323f}table{border-collapse:collapse;margin:10px 0;font-size:13px}td,th{border:1px solid #cfd6dc;padding:4px 8px;text-align:left}"
#            "th{background:#eef2f5}img{max-width:100%;border:1px solid #dde3e8;margin:8px 0}code{background:#eef2f5;padding:1px 4px;border-radius:3px}"
#            "li{margin:3px 0}@media(prefers-color-scheme:dark){body{background:#12181c;color:#dfe6ea}h1,h2,h3{color:#bcd6e2}th{background:#1d272d}td,th{border-color:#34424a}code{background:#1d272d}}")
#     return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>UHI Pareto analysis</title><style>{css}</style></head><body>{html}</body></html>"


# def _inline(s):
#     s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
#     s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
#     s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
#     s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
#     s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
#     return s


# def _mini_md(md):
#     out, lines, i = [], md.splitlines(), 0
#     while i < len(lines):
#         ln = lines[i]
#         if ln.startswith("#"):
#             n = len(ln) - len(ln.lstrip("#"))
#             out.append(f"<h{n}>{_inline(ln[n:].strip())}</h{n}>")
#         elif ln.startswith("|"):
#             rows = []
#             while i < len(lines) and lines[i].startswith("|"):
#                 rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
#                 i += 1
#             i -= 1
#             t = "<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr>"
#             for r in rows[2:]:
#                 t += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
#             out.append(t + "</table>")
#         elif ln.startswith("- "):
#             out.append("<ul>")
#             while i < len(lines) and lines[i].startswith("- "):
#                 out.append(f"<li>{_inline(lines[i][2:])}</li>")
#                 i += 1
#             i -= 1
#             out.append("</ul>")
#         elif ln.strip():
#             out.append(f"<p>{_inline(ln)}</p>")
#         i += 1
#     return "\n".join(out)


# # ============================================================================= selection
# def read_objectives(out_dir):
#     p = Path(out_dir) / "objectives.csv"
#     if not p.is_file():
#         return {}
#     rows = {}
#     for r in csv.DictReader(p.open()):
#         try:
#             rows[r["name"]] = dict(f1=float(r["f1"]), f2=float(r["f2"]), utci_heat=float(r["utci_heat"]), utci_cold=float(r["utci_cold"]), status=r.get("status", "ok"))
#         except (KeyError, ValueError):
#             continue
#     return rows


# def pareto_front(obj):
#     names = [n for n, r in obj.items() if r.get("status", "ok") == "ok"]
#     front = []
#     for n in names:
#         a = obj[n]
#         if not any((obj[m]["f1"] <= a["f1"] and obj[m]["f2"] <= a["f2"]) and (obj[m]["f1"] < a["f1"] or obj[m]["f2"] < a["f2"]) for m in names if m != n):
#             front.append(n)
#     return sorted(front, key=lambda n: obj[n]["f1"] + obj[n]["f2"])


# def select_names(a, obj):
#     if a.names:
#         return list(a.names)
#     t5 = Path(a.output_dir) / "top5.json"
#     if t5.is_file():
#         try:
#             names = [r["name"] for r in json.loads(t5.read_text())]
#             return names[:a.top]
#         except Exception:
#             pass
#     if obj:
#         return pareto_front(obj)[:a.top]
#     raise SystemExit("no top5.json / objectives.csv found: give --names")


# # ============================================================================= main
# def main(argv=None):
#     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--root", type=Path, default=ROOT, help="project folder (contains input/, run/, output/)")
#     ap.add_argument("--run-dir", type=Path)
#     ap.add_argument("--input-dir", type=Path, help="folder with the layout jsons (default <root>/input)")
#     ap.add_argument("--output-dir", type=Path, help="folder with top5.json / objectives.csv (default <root>/output)")
#     ap.add_argument("--outdir", type=Path, help="where the analysis is written (default <output-dir>/analysis)")
#     ap.add_argument("--base", default="individual_0", help="name of the base case (default individual_0 = base.json)")
#     ap.add_argument("--names", nargs="+", help="layouts to analyse (default: top5.json, else the Pareto front)")
#     ap.add_argument("--top", type=int, default=5, help="how many of top5.json / of the Pareto front")
#     ap.add_argument("--season", choices=("heat", "cold", "both"), default="both")
#     ap.add_argument("--cell", type=float, default=None, help="analysis grid cell in m (default about half the sample point spacing, 2-4 m)")
#     ap.add_argument("--wake-frac", type=float, default=0.5, help="wake = wind below this share of the reference wind (default 0.5)")
#     ap.add_argument("--hot-pct", type=float, default=80.0, help="heat spots: UTCI >= this percentile of the base (and >= 26 C)")
#     ap.add_argument("--cold-pct", type=float, default=80.0, help="cold spots: UTCI <= the (100 - this) percentile of the base (and <= 9 C)")
#     ap.add_argument("--u-ref", type=float, default=None, help="reference wind (m/s at 1.5 m) for the wake test; default = 90th percentile of the base wind field")
#     ap.add_argument("--wind-factor", type=float, default=1.0, help="the factor compute_utci.py used on the CFD wind (default 1)")
#     ap.add_argument("--min-region-m2", type=float, default=80.0, help="smaller regions are ignored")
#     ap.add_argument("--no-html", action="store_true")
#     ap.add_argument("--no-figures", action="store_true")
#     ap.add_argument("--dpi", type=int, default=200, help="resolution of the two plain pictures (default 200 = about 3200 px wide)")
#     ap.add_argument("--detailed", action="store_true", help="also draw the dense technical figures (maps, drivers, zone tables, categories); default: only the two plain pictures per layout and season")
#     a = ap.parse_args(argv)
#     t0 = time.time()
#     root = a.root
#     a.run_dir = a.run_dir or root / "run"
#     a.input_dir = a.input_dir or root / "input"
#     a.output_dir = a.output_dir or root / "output"
#     outdir = a.outdir or a.output_dir / "analysis"
#     (outdir / "figures").mkdir(parents=True, exist_ok=True)
#     (outdir / "tables").mkdir(parents=True, exist_ok=True)
#     global WIND_FACTOR, SIMPLE_DPI
#     WIND_FACTOR = a.wind_factor
#     SIMPLE_DPI = a.dpi
#     seasons = SEASONS if a.season == "both" else (a.season,)
#     obj = read_objectives(a.output_dir)
#     names = [n for n in select_names(a, obj) if n != a.base]
#     log(f"base {a.base}; layouts {', '.join(names)}; seasons {', '.join(seasons)}")
#     tn = {n: f"top_{i}_{n}" for i, n in enumerate(names, 1)}              # file names: top_1_<name>_..., top_2_<name>_... (rank = order of the list)
#     log("files: " + ", ".join(tn.values()))

#     results, grids, cases = [], {}, {}
#     for s in seasons:
#         try:
#             base = Case(a.base, s, a.run_dir, a.input_dir)
#         except Exception as e:
#             log(f"base / {s}: {e} - season skipped")
#             continue
#         cases[("base", s)] = base
#         log(f"{s}: utci_points.csv of the base: {base.point_note}")
#         grid = Grid(base.layout.site, a.cell or float(np.clip(0.5 * base.spacing, 2.0, 4.0)))
#         grids[s] = grid
#         log(f"{s}: grid {grid.shape[1]} x {grid.shape[0]} cells of {grid.cell:.1f} m, {len(base.pts['x'])} sample points, base UTCI {base.summary.get('utci_C', float('nan')):.2f} °C")
#         for n in names:
#             try:
#                 cand = Case(n, s, a.run_dir, a.input_dir)
#                 R = analyse_pair(base, cand, grid, a)
#                 R.wake_frac = a.wake_frac
#                 log(f"{s} / {n}: ΔUTCI {R.dU_mean:+.2f} °C  (" + ", ".join(f"{d} {R.contrib_mean[d]:+.2f}" for d in DRIVERS) + f")  split error {R.split_err:.1e}")
#                 if not a.no_figures:
#                     fig_simple_spots(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_1_{'hot' if s == 'heat' else 'cold'}_spots.png")
#                     fig_simple_vent(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_2_ventilation.png")
#                 if not a.no_figures and a.detailed:
#                     fig_maps(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_maps.png")
#                     fig_drivers(R, grid, outdir / "figures" / f"{tn[n]}_{s}_drivers.png")
#                     fig_wakes(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_wakes.png")
#                     fig_zones(R, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_zones_spots.png", "spots")
#                     fig_zones(R, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_zones_wakes.png", "wakes")
#                     fig_categories(R, outdir / "figures" / f"{tn[n]}_{s}_categories.png")
#                 cases[(n, s)] = cand                       # a layout enters the report only when its numbers AND figures are complete
#                 results.append(R)
#             except FileNotFoundError as e:
#                 log(f"{s} / {n}: skipped - {e}")
#             except Exception as e:
#                 import traceback
#                 log(f"{s} / {n}: FAILED {type(e).__name__}: {e}")
#                 traceback.print_exc()
#     if not results:
#         log("nothing analysed")
#         return 1
#     if not a.no_figures and a.detailed:
#         fig_overview(results, outdir / "figures" / "overview_drivers.png")

#     # ------------------------------------------------------------------ report
#     md = ["# Pareto analysis: why are the best layouts better than the base case?", "",
#           f"Base case: `{a.base}`. Layouts analysed: " + ", ".join(f"`{n}`" for n in names) + ". Generated by `analyze_pareto.py` "
#           f"({time.strftime('%Y-%m-%d %H:%M')}).", ""]
#     md += ["## How to read this", "",
#            "- **Benefit** is always in °C of UTCI in the direction that helps: heat hour = UTCI lower, cold hour = UTCI higher. Positive = good.",
#            "- The change of UTCI is split **exactly** into four drivers (order-averaged substitution with the UTCI function): **Tmrt** (shade, sky view, surface temperatures), "
#            "**wind** (ventilation / wind chill), **Ta** (air temperature), **RH** (humidity). The four add up to the total.",
#            "- **Wake** = pedestrian-level wind below "
#            f"{a.wake_frac:g} x the reference wind. **Hot / cold spots** = the worst parts of the base case (see each section for the threshold).",
#            "- Values come from one hour per season at 1.5 m height; the statements are associations of layout changes with field changes.", ""]
#     # objective table
#     if obj:
#         md += ["## Selected layouts", ""]
#         rows = []
#         for n in [a.base] + names:
#             o = obj.get(n)
#             if o:
#                 rows.append([n, f"{o['utci_heat']:.2f}", f"{o['utci_cold']:.2f}", f"{o['f1']:.2f}", f"{o['f2']:.2f}"])
#         md += md_table(["layout", "UTCI heat °C", "UTCI cold °C", "f1 (heat − 26)", "f2 (9 − cold)"], rows) + [""]
#     if (outdir / "figures" / "overview_drivers.png").is_file():
#         md += ["![overview](figures/overview_drivers.png)", ""]
#     # per season overview table
#     for s in seasons:
#         rs = [r for r in results if r.season == s]
#         if not rs:
#             continue
#         md += [f"## Overview — {SEASON_TXT[s]}", ""]
#         rows = []
#         for r in rs:
#             c = r.contrib_mean
#             rows.append([r.name, f"{r.stats_b['UTCI_C']:.2f} → {r.stats_c['UTCI_C']:.2f}", f"{r.sgn * r.dU_mean:+.2f}"] +
#                         [f"{r.sgn * c[d]:+.2f}" for d in DRIVERS] +
#                         [f"{r.stats_b['wind_ms'] / r.u_ref:.2f} → {r.stats_c['wind_ms'] / r.u_ref:.2f}",
#                          f"{r.vent['calm_area_b']:.0f} → {r.vent['calm_area_c']:.0f}",
#                          f"{r.spot_area_b:.0f} → {r.spot_area_c:.0f}"])
#         md += md_table(["layout", "site mean UTCI °C", "benefit °C"] + [f"{d} °C" for d in DRIVERS] +
#                        ["wind / reference", "calm area m²", f"{'hot' if s == 'heat' else 'cold'}-spot area m²"], rows) + [""]
#         md += ["Actual change of the inputs (mean over the ground that is open in both layouts) and, in brackets, the change of UTCI it caused (°C, − = cooler):", ""]
#         rows = []
#         for r in rs:
#             sv = r.site_vals
#             rows.append([r.name, f"{sv['UTCI']['b']:.2f} → {sv['UTCI']['c']:.2f} ({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d) for d in ORDER])
#         md += md_table(["layout", "UTCI °C"] + [f"{d} {UNIT[d]}: base → layout (Δ) → effect" for d in ORDER], rows) + [""]
#         L0 = rs[0]
#         md += [f"Spot threshold: UTCI {'≥' if s == 'heat' else '≤'} {L0.thr:.1f} °C; reference wind {L0.u_ref:.2f} m/s at 1.5 m (90th percentile of the base; 10 m wind of the hour {L0.u10:.1f} m/s); analysis grid {grids[s].cell:.1f} m.", ""]
#     # per layout
#     for n in names:
#         rs = [r for r in results if r.name == n]
#         if not rs:
#             continue
#         md += [f"## Layout `{n}`", ""]
#         d0 = rs[0].diff
#         md += [f"*Layout changes vs base:* trees {d0['trees_base']} → {d0['trees_cand']} ({len(d0['trees_added'])} new, {len(d0['trees_removed'])} removed), canopy "
#                f"{d0['canopy_base']:.0f} → {d0['canopy_cand']:.0f} m², mean tree height {d0['mean_height_base']:.1f} → {d0['mean_height_cand']:.1f} m, ponds {d0['ponds_base']} → "
#                f"{d0['ponds_cand']} ({d0['pond_area_base']:.0f} → {d0['pond_area_cand']:.0f} m²), lawns {d0['lawn_area_base']:.0f} → {d0['lawn_area_cand']:.0f} m²; buildings moved: " +
#                (", ".join(f"{b['id']} {b['shift']:.1f} m" for b in d0["buildings"] if b["shift"] > 0.5) or "none") + ".", ""]
#         for R in rs:
#             s = R.season
#             base = cases[("base", s)]
#             cand = cases[(n, s)]
#             md += [f"### {n} — {SEASON_TXT[s]}", ""]
#             for line in build_findings(R, base, cand, grids[s]):
#                 md += [line, ""]
#             fg = outdir / "figures"
#             kind = "hot" if s == "heat" else "cold"
#             for tag, ttl in ((f"1_{kind}_spots", f"{kind} spots"), ("2_ventilation", "ventilation")):
#                 if (fg / f"{tn[n]}_{s}_{tag}.png").is_file():
#                     md += [f"![{ttl}](figures/{tn[n]}_{s}_{tag}.png)", ""]
#                     if (fg / f"{tn[n]}_{s}_{tag}.txt").is_file():
#                         md += [" ".join((fg / f"{tn[n]}_{s}_{tag}.txt").read_text().splitlines()[2:]), ""]
#             for stem, ttl in (("maps", "maps"), ("drivers", "drivers"), ("zones_spots", "zone values of the spots"), ("wakes", "wakes"), ("zones_wakes", "zone values of the wakes"), ("categories", "categories")):
#                 if (fg / f"{tn[n]}_{s}_{stem}.png").is_file():
#                     md += [f"![{ttl}](figures/{tn[n]}_{s}_{stem}.png)", ""]
#             centre = grids[s].site_mask
#             sc = (float(grids[s].X[centre].mean()), float(grids[s].Y[centre].mean()))
#             if R.regions or R.new_regions:
#                 md += [f"**Zone-averaged inputs of the {'hot' if s == 'heat' else 'cold'} spots** — each cell: base → layout (change) → effect of that change on UTCI in °C (− = cooler, + = warmer)", ""]
#                 md += md_zone_table(R.regions + R.new_regions, R.site_vals) + [""]
#             if R.regions:
#                 md += [f"**{'Hot' if s == 'heat' else 'Cold'} spots of the base and what became of them**", ""]
#                 for r in R.regions:
#                     md += [f"- **{r['id']}** ({fmt_area(r['area'])}, {where_text(r['centroid'], cand.layout, sc, R.flow)}): " + explain_region(r, R, s)]
#                 md += [""]
#             if R.new_regions:
#                 md += ["**New stressed regions created by the layout**", ""]
#                 for r in R.new_regions:
#                     ctx = r["context"]
#                     extra = f" Dominant driver of the worsening: {r.get('dominant', 'n/a')}." if "dominant" in r else ""
#                     md += [f"- **{r['id']}** ({fmt_area(r['area'])}, {where_text(r['centroid'], cand.layout, sc, R.flow)}): UTCI {r['vals']['UTCI']['b']:.1f} → {r['u_cand']:.1f} °C. "
#                            f"{inputs_sentence(r['vals'])}. Sunlit {r['sunlit_c']:.2f}.{extra} Layout nearby: {ctx['trees_added_near']} new trees within 25 m, {ctx.get('trees_added_upwind', 0)} upwind."]
#                 md += [""]
#             if R.wakes_b or R.wakes_c:
#                 md += ["**Wake regions (calm air at 1.5 m)**", ""]
#                 rows = []
#                 for w in R.wakes_b:
#                     rows.append([f"{w['id']} (base)", fmt_area(w["area"]), f"{w['length_m']:.0f} × {w['width_m']:.0f} m", w["building"] or "-",
#                                  f"{w['utci_mean']:.1f}", w["fate"] + (f" → {w['cand_match']} {w['cand_area']:.0f} m²" if w["cand_match"] else ""),
#                                  f"{w.get('d_wind', float('nan')):+.2f}" if "d_wind" in w else "-",
#                                  f"{w['d_u']:+.2f}" if "d_u" in w else "-"])
#                 for w in R.wakes_c:
#                     if w.get("is_new"):
#                         rows.append([f"{w['id']} (layout)", fmt_area(w["area"]), f"{w['length_m']:.0f} × {w['width_m']:.0f} m", w["building"] or "-", f"{w['utci_mean']:.1f}", "new", "-", "-"])
#                 md += md_table(["wake", "area", "length × width (along × across wind)", "trails building", "UTCI °C", "fate", "Δ wind m/s", "Δ UTCI °C"], rows) + [""]
#                 md += ["**Zone-averaged inputs of the wakes (same footprint in both layouts)** — each cell: base → layout (change) → effect on UTCI in °C", ""]
#                 md += md_zone_table(R.wakes_b + [w for w in R.wakes_c if w.get("is_new")], R.site_vals) + [""]
#     # mitigation
#     md += ["## Mitigation inference", ""]
#     md += mitigation_text(results, cases.get(("base", seasons[0])), grids, a)
#     md += ["", "### Rules behind these sentences (the template)", ""]
#     md += md_table(["observation", "reading", "mitigation"], [list(r) for r in RULES_TABLE]) + [""]
#     md += ["## Method and limits", "",
#            f"- Sample points of `utci_points.csv` (z = 1.5 m) are interpolated linearly on a {', '.join(f'{grids[s].cell:.1f} m' for s in grids)} grid; cells inside buildings (of each layout), outside the site, "
#            "or farther than 2.2 point spacings from a sample are left out. Changes are computed only where both layouts have open ground.",
#            "- The driver split recomputes UTCI (pythermalcomfort, same call as `compute_utci.py`) for every combination of base / layout values of Tmrt, wind, Ta, RH "
#            "(16 evaluations per cell) and averages over the orders: exact and additive. Recomputed UTCI of the base differs from the stored one by "
#            + ", ".join(f"{np.mean([r.csv_err for r in results if r.season == s]):.2f} °C ({s})" for s in seasons if any(r.season == s for r in results)) + " (rounding / interpolation).",
#            "- Zone values (Ta, Tmrt, RH, wind) are area means over the zone's cells, in the base and in the layout at the same footprint; the 'effect' next to each is the part of the "
#            "UTCI change that this input alone caused (exact split), so a large change of an input with a small effect (or the reverse) shows how sensitive UTCI is to it in that zone.",
#            "- The split of Tmrt is a linear regression on the sunlit share, the sky view and Ta; the remainder is attributed to surface temperatures / long-wave exchange.",
#            "- One hour per season, steady flow field, one height. Wakes shown are pedestrian-level calm regions, not the full 3-D recirculation zones.",
#            "- Layout context numbers (trees within 25 m, 70 m upwind corridor, ponds) show what changed near a region; they do not prove that the change caused the field change.", ""]
#     md_text = "\n".join(md)
#     hz = (cases[("base", seasons[0])].summary or {}).get("height_m") if ("base", seasons[0]) in cases else None
#     if hz and abs(float(hz) - 1.5) > 1e-9:                    # the text says 1.5 m; the run used another height (compute_utci.py --height)
#         md_text = md_text.replace("1.5 m", f"{float(hz):g} m")
#     (outdir / "analysis_report.md").write_text(md_text)
#     if not a.no_html:
#         (outdir / "analysis_report.html").write_text(md_to_html(md_text, outdir))

#     # ------------------------------------------------------------------ tables / json
#     def drow(r):
#         return dict(layout=r.name, season=r.season, utci_base=r.stats_b["UTCI_C"], utci_layout=r.stats_c["UTCI_C"], d_utci=r.dU_mean,
#                     **{f"contrib_{d}": r.contrib_mean[d] for d in DRIVERS}, split_error=r.split_err,
#                     wind_ratio_base=r.vent["wind_ratio_b"], wind_ratio_layout=r.vent["wind_ratio_c"], calm_area_base=r.vent["calm_area_b"], calm_area_layout=r.vent["calm_area_c"],
#                     spot_threshold=r.thr, spot_area_base=r.spot_area_b, spot_area_layout=r.spot_area_c, sunlit_base=r.stats_b["sunlit"], sunlit_layout=r.stats_c["sunlit"],
#                     sky_base=r.stats_b["sky_open"], sky_layout=r.stats_c["sky_open"], Tmrt_base=r.stats_b["Tmrt_C"], Tmrt_layout=r.stats_c["Tmrt_C"],
#                     Ta_base=r.stats_b["Tair_C"], Ta_layout=r.stats_c["Tair_C"], RH_base=r.stats_b["RH_pct"], RH_layout=r.stats_c["RH_pct"],
#                     **{f"{d}_change_site_mean": r.site_vals[d]["d"] for d in DRIVERS})

#     def valcols(v):
#         """flat columns: UTCI base / layout / change, and for Ta, Tmrt, RH, wind: base, layout, change, effect on UTCI"""
#         if v is None:
#             return {}
#         o = dict(utci_base=v["UTCI"]["b"], utci_layout=v["UTCI"]["c"], utci_change=v["UTCI"]["d"])
#         for d in ORDER:
#             u = {"Ta": "C", "Tmrt": "C", "RH": "pct", "wind": "ms"}[d]
#             o.update({f"{d}_base_{u}": v[d]["b"], f"{d}_layout_{u}": v[d]["c"], f"{d}_change_{u}": v[d]["d"], f"{d}_effect_on_UTCI_C": v[d]["eff"]})
#         return o

#     def write_csv(path, rows):
#         if not rows:
#             return
#         keys = list(rows[0].keys())
#         with open(path, "w", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=keys)
#             w.writeheader()
#             w.writerows(rows)
#     write_csv(outdir / "tables" / "drivers.csv", [drow(r) for r in results])
#     reg_rows, wake_rows = [], []
#     for r in results:
#         for g in r.regions:
#             reg_rows.append(dict(layout=r.name, season=r.season, region=g["id"], area_m2=g["area"], x=g["centroid"][0], y=g["centroid"][1], status=g["status"], utci_base=g["u_base"],
#                                  utci_layout=g["u_cand"], resolved_share=g["resolved_share"], dominant=g["dominant"],
#                                  **{f"contrib_{d}": g["contrib"][d] for d in DRIVERS}, trees_added_near=g["context"]["trees_added_near"],
#                                  trees_added_upwind=g["context"].get("trees_added_upwind", 0)))
#         for w in r.wakes_b:
#             wake_rows.append(dict(layout=r.name, season=r.season, wake=w["id"], which="base", area_m2=w["area"], length_m=w["length_m"], width_m=w["width_m"], building=w["building"],
#                                   utci=w["utci_mean"], fate=w["fate"]))
#         for w in r.wakes_c:
#             wake_rows.append(dict(layout=r.name, season=r.season, wake=w["id"], which="layout", area_m2=w["area"], length_m=w["length_m"], width_m=w["width_m"], building=w["building"],
#                                   utci=w["utci_mean"], fate="new" if w.get("is_new") else ""))
#     zone_rows = []
#     for r in results:
#         base_ = dict(layout=r.name, season=r.season)
#         zone_rows.append(dict(base_, kind="whole site", zone="site", status="", area_m2=r.site_vals["area"], **valcols(r.site_vals)))
#         for g in r.regions:
#             zone_rows.append(dict(base_, kind="spot of the base", zone=g["id"], status=g["status"], area_m2=g["area"], **valcols(g["vals"])))
#         for g in r.new_regions:
#             zone_rows.append(dict(base_, kind="new stressed region", zone=g["id"], status="new", area_m2=g["area"], **valcols(g["vals"])))
#         for w in r.wakes_b:
#             zone_rows.append(dict(base_, kind="wake of the base", zone=w["id"], status=w["fate"], area_m2=w["area"], **valcols(w["vals"])))
#         for w in r.wakes_c:
#             if w.get("is_new"):
#                 zone_rows.append(dict(base_, kind="new wake", zone=w["id"], status="new", area_m2=w["area"], **valcols(w["vals"])))
#     write_csv(outdir / "tables" / "zones.csv", zone_rows)
#     write_csv(outdir / "tables" / "regions.csv", reg_rows)
#     write_csv(outdir / "tables" / "wakes.csv", wake_rows)
#     summ = dict(base=a.base, layouts=names, seasons=list(seasons), generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
#                 results=[dict(drow(r), site_vals=r.site_vals, regions=[{k: v for k, v in g.items() if k not in ("mask", "context")} for g in r.regions]) for r in results])
#     (outdir / "analysis_summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
#     log(f"done in {time.time() - t0:.0f} s -> {outdir}/analysis_report.md" + ("" if a.no_html else " and .html"))
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())



# # #!/usr/bin/env python3
# # """
# # analyze_pareto.py  --  why are the best (Pareto) layouts better than the base case?   (post-processing, run it after the optimisation)

# # Reads what the optimisation already wrote (nothing is simulated again):

# #     output/top5.json  (or the Pareto front of output/objectives.csv)      which layouts to analyse
# #     run/<name>/<heat|cold>/temperature_humidity/postProcessing/utci_points.csv + utci.json      pedestrian-height results of every layout
# #     run/<name>/<season>/temperature_humidity/weatherData/target.json                            wind / sun / air of the hour
# #     input/<name>.json                                                                           the layouts (buildings, trees, ponds, lawns)

# # and compares every selected layout with the BASE case (individual_0 = base.json) for the heat hour and for the cold hour:

# #   1. HOW MUCH, WHERE      UTCI maps of base / layout / difference on a common grid, category-area table, hot spots and cold spots.
# #   2. WHY  (which driver)  The UTCI is a function of four things at the pedestrian: mean radiant temperature Tmrt, wind speed, air temperature Ta,
# #                           relative humidity RH.  The change of UTCI is split EXACTLY into the four (Shapley / order-averaged substitution
# #                           with the UTCI function itself, so the four add up to the change):  Tmrt (shade, sky view, hot / cool surfaces),
# #                           wind (ventilation), Ta (evaporative / shade cooling of the air), RH (humidity).  Tmrt is split further, by a regression on the
# #                           sunlit share, the sky view and Ta, into shade, sky view, air and "surfaces / long-wave".
# #   3. WAKES, VENTILATION   Calm (wake) regions = wind below `--wake-frac` x the reference wind: area, length, width, which building they trail, what
# #                           changed.  Mean wind ratio, calm share, fast channels.
# #   4. HOT / COLD SPOTS     Regions of the base case where the stress is worst (heat: UTCI >= the `--hot-pct` percentile and >= 26 C; cold:
# #                           <= the (100-`--cold-pct`) percentile and <= 9 C): were they resolved, reduced, still there?  New ones?  For every region the
# #                           drivers and the layout changes (trees, ponds, lawns, building shifts) next to / upwind of it.
# #   5. MITIGATION           Rule-based reading of the numbers: what worked, what is left, what to try next, and the trade-off between the hot and the cold hour.
# #                           The rules are listed in the report, so you can see why each sentence is there.

# # Output (default output/analysis/):  analysis_report.md  +  analysis_report.html (one file, pictures inside),  figures/*.png,  analysis_summary.json,
# #                                     tables/*.csv

# #     python3 python/analyze_pareto.py                                  # top5.json of ./output, base = individual_0, all seasons
# #     python3 python/analyze_pareto.py --top 3 --season heat            # the first 3, heat hour only
# #     python3 python/analyze_pareto.py --names offspring_10 individual_2 --base individual_0
# #     python3 python/analyze_pareto.py --root /data/UHI_optimization_template_271026_scotch --outdir /tmp/analysis
# #     python3 python/analyze_pareto.py --synthetic-demo ...             (see make_synthetic_demo.py: a made-up project to try the analyzer)

# # Limits (also printed in the report): one hour per season, a steady frozen-flow run, results at ONE height (1.5 m): wakes and ventilation are those of the
# # pedestrian level; a statement like "tree shade lowered UTCI" is an association between the layout change and the field change, not a controlled experiment.
# # """
# # import argparse
# # import base64
# # import csv
# # import json
# # import math
# # import re
# # import sys
# # import time
# # import warnings
# # from pathlib import Path

# # import numpy as np

# # import matplotlib

# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt                                   # noqa: E402
# # from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
# # from matplotlib.patches import Circle, Polygon as MPoly            # noqa: E402

# # from scipy import ndimage as ndi                                  # noqa: E402
# # from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator   # noqa: E402
# # from scipy.spatial import cKDTree                                 # noqa: E402

# # import shapely                                                    # noqa: E402
# # from shapely.geometry import Point, Polygon                       # noqa: E402
# # from shapely.ops import unary_union                               # noqa: E402

# # warnings.filterwarnings("ignore")
# # ROOT = Path(__file__).resolve().parent.parent

# # SEASONS = ("heat", "cold")
# # SEASON_TXT = {"heat": "heat-stress hour", "cold": "cold-stress hour"}
# # CATEGORIES = [(46.0, "extreme heat stress"), (38.0, "very strong heat stress"), (32.0, "strong heat stress"),
# #               (26.0, "moderate heat stress"), (9.0, "no thermal stress"), (0.0, "slight cold stress"),
# #               (-13.0, "moderate cold stress"), (-27.0, "strong cold stress"), (-40.0, "very strong cold stress")]
# # DRIVERS = ("Tmrt", "wind", "Ta", "RH")
# # DRIVER_COL = {"Tmrt": "Tmrt_C", "wind": "wind_ms", "Ta": "Tair_C", "RH": "RH_pct"}
# # DRIVER_LONG = {"Tmrt": "mean radiant temperature (shade / sky view / hot surfaces)", "wind": "wind speed (ventilation)",
# #                "Ta": "air temperature (evaporation, shade, mixing)", "RH": "relative humidity"}
# # COL = {"Tmrt": "#d9822b", "wind": "#2b7bba", "Ta": "#c0392b", "RH": "#4f9d69", "total": "#1c2830"}
# # INK, MUTED = "#1c2830", "#6d7780"
# # FIELDS = ("Tair_C", "RH_pct", "wind_ms", "Tmrt_C", "UTCI_C", "sunlit", "sky_open")

# # RAINBOW = LinearSegmentedColormap.from_list("rainbow_desaturated", [
# #     (0.000, (0.278431, 0.278431, 0.858824)), (0.143, (0.000000, 0.000000, 0.360784)), (0.285, (0.000000, 1.000000, 1.000000)),
# #     (0.429, (0.000000, 0.501961, 0.000000)), (0.571, (1.000000, 1.000000, 0.000000)), (0.714, (1.000000, 0.380392, 0.000000)),
# #     (0.857, (0.419608, 0.000000, 0.000000)), (1.000, (0.878431, 0.301961, 0.301961))])


# # def category(v):
# #     for lo, name in CATEGORIES:
# #         if v > lo:
# #             return name
# #     return "extreme cold stress"


# # def log(msg):
# #     print(f"[analyze] {msg}", flush=True)


# # # ============================================================================= reading the project
# # def _xy(seq):
# #     out = []
# #     for p in seq or []:
# #         if isinstance(p, dict):
# #             out.append((float(p["x"]), float(p["y"] if "y" in p else p["z"])))      # osm buildings store the plan y as z
# #         else:
# #             out.append((float(p[0]), float(p[1])))
# #     return out


# # def _poly(seq):
# #     pts = _xy(seq)
# #     if len(pts) < 3:
# #         return None
# #     g = Polygon(pts)
# #     if not g.is_valid:
# #         g = g.buffer(0)
# #     if g.is_empty:
# #         return None
# #     if g.geom_type != "Polygon":
# #         g = max(getattr(g, "geoms", []), key=lambda q: q.area)
# #     return g


# # class Layout:
# #     """site, buildings (id -> polygon, height), trees, ponds, lawns of one layout file"""

# #     def __init__(self, path):
# #         J = json.loads(Path(path).read_text())
# #         J = J.get("siteOnly_json", J)
# #         self.path = str(path)
# #         self.site = _poly((J.get("site") or {}).get("coordinates"))
# #         self.buildings, self.height = {}, {}
# #         for b in J.get("buildings") or []:
# #             vs = (b.get("dimensions") or {}).get("baseVertices") or (b.get("geometry") or {}).get("footprint")
# #             g = _poly(vs)
# #             if g is not None:
# #                 self.buildings[b["id"]] = g
# #                 self.height[b["id"]] = float(b.get("height") or (b.get("dimensions") or {}).get("height") or 0.0)
# #         osm = J.get("osmBuildings") or J.get("osm_buildings") or []
# #         for k, b in enumerate(osm):
# #             g = _poly((b.get("dimensions") or {}).get("baseVertices") or (b.get("geometry") or {}).get("footprint") or b.get("coordinates"))
# #             if g is not None:
# #                 self.buildings[f"osm_{k}"] = g
# #                 self.height[f"osm_{k}"] = float(b.get("height") or 10.0)
# #         self.trees = []
# #         for t in J.get("trees") or []:
# #             r = (t.get("dimensions") or {}).get("crownRadius") or 0.5 * float(t.get("crownDiameter") or 4.0)
# #             self.trees.append((str(t.get("id")), float(t["position"]["x"]), float(t["position"]["y"]), float(r), float(t.get("height") or 8.0)))
# #         self.ponds = [g for g in (_poly(p.get("coordinates")) for p in J.get("ponds") or []) if g is not None]
# #         self.lawns = [g for g in (_poly(p.get("coordinates")) for p in J.get("grassAreas") or []) if g is not None]
# #         self.pond_ids = [p.get("id") for p in J.get("ponds") or []]

# #     def building_union(self):
# #         return unary_union(list(self.buildings.values())) if self.buildings else None


# # def find_pp(run_dir, name, season):
# #     """(postProcessing folder, job folder) of one layout / season, or None"""
# #     job = Path(run_dir) / name / season
# #     for pp in (job / "temperature_humidity" / "postProcessing", job / "postProcessing"):
# #         if (pp / "utci_points.csv").is_file():
# #             return pp, job
# #     hits = sorted(job.glob("*/postProcessing/utci_points.csv"))
# #     if hits:
# #         return hits[0].parent, job
# #     return None


# # def read_points(csv_path, height=None):
# #     """utci_points.csv -> dict of arrays; one row per (x, y).  Duplicated (x, y) rows (several heights, or the same point written twice) are collapsed:
# #     the row whose z is closest to `height` is kept, else the rows are averaged.  Returns (points, note)"""
# #     d = np.genfromtxt(csv_path, delimiter=",", names=True)
# #     names = d.dtype.names
# #     out = {k: np.atleast_1d(np.asarray(d[k], float)) for k in ("x", "y", "z") + FIELDS if k in names}
# #     for k in FIELDS:
# #         out.setdefault(k, np.full(len(out["x"]), np.nan))
# #     n0 = len(out["x"])
# #     xy = np.round(np.column_stack([out["x"], out["y"]]), 3)
# #     _, first, inv, cnt = np.unique(xy, axis=0, return_index=True, return_inverse=True, return_counts=True)
# #     inv = np.asarray(inv).ravel()
# #     note = f"{n0} rows, {len(cnt)} distinct (x, y)"
# #     if len(cnt) < n0:
# #         z = out.get("z")
# #         zl = np.unique(np.round(z, 3)) if z is not None else np.array([])
# #         if z is not None and len(zl) > 1:
# #             h = float(height) if height is not None else float(zl[np.argmin(np.abs(zl - 1.5))])
# #             order = np.lexsort((np.abs(z - h), inv))                  # per (x, y): the row closest to the wanted height first
# #             pick = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
# #             out = {k: v[pick] for k, v in out.items()}
# #             note += f"; z levels {', '.join(f'{v:g}' for v in zl[:6])} m -> kept the rows nearest z = {h:g} m"
# #         else:
# #             avg = {}
# #             for k, v in out.items():
# #                 sm = np.zeros(len(cnt))
# #                 np.add.at(sm, inv, np.where(np.isfinite(v), v, 0.0))
# #                 ct = np.zeros(len(cnt))
# #                 np.add.at(ct, inv, np.isfinite(v).astype(float))
# #                 avg[k] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
# #             out = avg
# #             note += "; duplicated points averaged"
# #     return out, note


# # def read_boundary(job, pp):
# #     """boundary of the hour (wind speed / direction, air, sun): weatherData/target.json of the job"""
# #     for p in (pp.parent / "weatherData" / "target.json", job / "weatherData" / "target.json", job / "temperature_humidity" / "weatherData" / "target.json"):
# #         if p.is_file():
# #             try:
# #                 return json.loads(p.read_text()).get("boundary", {})
# #             except Exception:
# #                 pass
# #     return {}


# # class Case:
# #     """one layout in one season"""

# #     def __init__(self, name, season, run_dir, in_dir):
# #         self.name, self.season = name, season
# #         hit = find_pp(run_dir, name, season)
# #         if hit is None:
# #             raise FileNotFoundError(f"no utci_points.csv for {name} / {season} under {run_dir}")
# #         self.pp, self.job = hit
# #         self.summary = json.loads((self.pp / "utci.json").read_text()) if (self.pp / "utci.json").is_file() else {}
# #         self.pts, self.point_note = read_points(self.pp / "utci_points.csv", self.summary.get("height_m"))
# #         self.boundary = read_boundary(self.job, self.pp)
# #         lj = Path(in_dir) / f"{name}.json"
# #         if not lj.is_file():
# #             c = sorted((self.job / "momentum").glob("*.json"))
# #             lj = c[0] if c else None
# #         if lj is None:
# #             raise FileNotFoundError(f"no layout json for {name}")
# #         self.layout = Layout(lj)
# #         xy = np.column_stack([self.pts["x"], self.pts["y"]])
# #         d = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.array([0.0])
# #         self.spacing = float(1.8 * np.median(d))
# #         if not self.spacing > 0:
# #             raise ValueError(f"{name}/{season}: the sample points of utci_points.csv have no spacing ({self.point_note}); check the file")

# #     def flow(self):
# #         """unit vector of the wind (towards where it blows), model frame"""
# #         f = self.boundary.get("wind_flow_unit_vector_xy_model")
# #         if f and len(f) == 2:
# #             n = math.hypot(f[0], f[1])
# #             if n > 1e-9:
# #                 return np.array([f[0] / n, f[1] / n])
# #         wd = self.boundary.get("wind_from_deg_model")
# #         if wd is None:
# #             return None
# #         return np.array([-math.sin(math.radians(wd)), -math.cos(math.radians(wd))])


# # # ============================================================================= grid
# # class Grid:
# #     def __init__(self, site, cell):
# #         x0, y0, x1, y1 = site.bounds
# #         self.cell = float(cell)
# #         self.x = np.arange(x0 + 0.5 * cell, x1, cell)
# #         self.y = np.arange(y0 + 0.5 * cell, y1, cell)
# #         self.X, self.Y = np.meshgrid(self.x, self.y)
# #         self.shape = self.X.shape
# #         self.extent = (x0, x0 + self.cell * len(self.x), y0, y0 + self.cell * len(self.y))
# #         self.area = self.cell ** 2
# #         self.site_mask = shapely.contains_xy(site, self.X, self.Y)


# # def rasterise(case, grid, gap_factor=2.2):
# #     """fields of one case on the grid (linear interpolation of the sample points; NaN outside the site, in buildings and farther than
# #     gap_factor x the point spacing from every sample)"""
# #     p = case.pts
# #     xy = np.column_stack([p["x"], p["y"]])
# #     ok = np.isfinite(p["UTCI_C"])
# #     xy = xy[ok]
# #     tree = cKDTree(xy)
# #     near = tree.query(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))[0].reshape(grid.shape)
# #     bu = case.layout.building_union()
# #     inb = shapely.contains_xy(bu, grid.X, grid.Y) if bu is not None else np.zeros(grid.shape, bool)
# #     valid = grid.site_mask & ~inb & (near <= gap_factor * case.spacing)
# #     F = {}
# #     nn_idx = tree.query(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))[1].reshape(grid.shape)
# #     for k in FIELDS:
# #         v = p[k][ok]
# #         if not np.isfinite(v).any():
# #             F[k] = np.full(grid.shape, np.nan)
# #             continue
# #         good = np.isfinite(v)
# #         try:
# #             lin = LinearNDInterpolator(xy[good], v[good])(grid.X, grid.Y)
# #         except Exception:
# #             lin = np.full(grid.shape, np.nan)
# #         nn = v[nn_idx]                                   # nearest sample of the same field (outside the convex hull of the samples)
# #         a = np.where(np.isfinite(lin), lin, nn)
# #         F[k] = np.where(valid, a, np.nan)
# #     F["valid"] = valid
# #     return F


# # # ============================================================================= UTCI and its exact split
# # WIND_FACTOR = 1.0                  # same as compute_utci.py --wind-factor (multiplies the CFD wind before UTCI)


# # def utci_fn(Ta, tmrt, v, rh):
# #     from pythermalcomfort.models import utci
# #     Ta, tmrt, v, rh = (np.asarray(a, float) for a in (Ta, tmrt, v, rh))
# #     v = np.clip(v * WIND_FACTOR, 0.5, 17.0)
# #     tr = np.clip(tmrt, Ta - 30.0, Ta + 70.0)
# #     rh = np.clip(rh, 1.0, 100.0)
# #     res = utci(tdb=Ta, tr=tr, v=v, rh=rh, units="SI", limit_inputs=False)
# #     return np.asarray(getattr(res, "utci", res), float)


# # def shapley_split(B, C, mask):
# #     """B, C: field dicts of base / candidate on the grid; mask: cells valid in both.  Returns dict(contrib={driver: array (cells,)},
# #     u_base, u_cand (recomputed UTCI), total) with  sum(contrib) = u_cand - u_base  exactly (order-averaged substitution)."""
# #     idx = np.where(mask)
# #     cols = [DRIVER_COL[d] for d in DRIVERS]
# #     b = [B[c][idx] for c in cols]
# #     c = [C[c][idx] for c in cols]
# #     n = len(DRIVERS)
# #     cache = {}

# #     def val(sub):
# #         if sub not in cache:
# #             args = [c[i] if (sub >> i) & 1 else b[i] for i in range(n)]
# #             cache[sub] = utci_fn(Ta=args[2], tmrt=args[0], v=args[1], rh=args[3])
# #         return cache[sub]

# #     fact = [math.factorial(k) for k in range(n + 1)]
# #     contrib = {}
# #     for i, d in enumerate(DRIVERS):
# #         acc = np.zeros(len(idx[0]))
# #         for sub in range(1 << n):
# #             if (sub >> i) & 1:
# #                 continue
# #             k = bin(sub).count("1")
# #             w = fact[k] * fact[n - k - 1] / fact[n]
# #             acc += w * (val(sub | (1 << i)) - val(sub))
# #         full = np.full(B["valid"].shape, np.nan)
# #         full[idx] = acc
# #         contrib[d] = full
# #     u0, u1 = np.full(B["valid"].shape, np.nan), np.full(B["valid"].shape, np.nan)
# #     u0[idx], u1[idx] = val(0), val((1 << n) - 1)
# #     return dict(contrib=contrib, u_base=u0, u_cand=u1)


# # def excess_split(F, region_mask, valid):
# #     """why is this region hotter / colder than the REST of the site in the same layout?  Region means vs the means of the rest of the open ground,
# #     split over Tmrt, wind, Ta, RH (same exact method).  Returns dict(contrib={driver: dC}, total, u_region, u_rest)"""
# #     rest = valid & ~region_mask
# #     if not region_mask.any() or not rest.any():
# #         return None
# #     mk = lambda m: {c: np.array([[float(np.nanmean(F[c][m]))]]) for c in DRIVER_COL.values()}      # noqa: E731
# #     Bm, Cm = mk(rest), mk(region_mask)
# #     ones = np.ones((1, 1), bool)
# #     Bm["valid"] = Cm["valid"] = ones
# #     sp = shapley_split(Bm, Cm, ones)
# #     return dict(contrib={d: float(sp["contrib"][d][0, 0]) for d in DRIVERS}, total=float(sp["u_cand"][0, 0] - sp["u_base"][0, 0]),
# #                 u_region=float(sp["u_cand"][0, 0]), u_rest=float(sp["u_base"][0, 0]))


# # def tmrt_attribution(B, C, mask):
# #     """split the change of Tmrt into shade (sunlit share), sky view, air temperature and the rest (surfaces / long-wave): linear regression
# #     Tmrt ~ 1 + sunlit + sky_open + Ta on the cells of both cases.  Returns dict(coef, r2, parts={name: array})"""
# #     idx = np.where(mask)
# #     A, y = [], []
# #     for F in (B, C):
# #         A.append(np.column_stack([np.ones(len(idx[0])), F["sunlit"][idx], F["sky_open"][idx], F["Tair_C"][idx]]))
# #         y.append(F["Tmrt_C"][idx])
# #     A, y = np.vstack(A), np.concatenate(y)
# #     okk = np.isfinite(A).all(1) & np.isfinite(y)
# #     if okk.sum() < 20 or np.ptp(A[okk, 1]) < 1e-6 and np.ptp(A[okk, 2]) < 1e-6:
# #         return None
# #     coef, *_ = np.linalg.lstsq(A[okk], y[okk], rcond=None)
# #     res = y[okk] - A[okk] @ coef
# #     r2 = 1.0 - float(res.var() / max(y[okk].var(), 1e-9))
# #     parts = {}
# #     for name, j, key in (("shade (sunlit share)", 1, "sunlit"), ("sky view", 2, "sky_open"), ("air temperature", 3, "Tair_C")):
# #         full = np.full(B["valid"].shape, np.nan)
# #         full[idx] = coef[j] * (C[key][idx] - B[key][idx])
# #         parts[name] = full
# #     dt = np.full(B["valid"].shape, np.nan)
# #     dt[idx] = C["Tmrt_C"][idx] - B["Tmrt_C"][idx]
# #     rest = dt.copy()
# #     for v in parts.values():
# #         rest = rest - v
# #     parts["surfaces / long-wave (rest)"] = rest
# #     return dict(coef=[float(c) for c in coef], r2=r2, parts=parts, dTmrt=dt)


# # # ============================================================================= regions
# # def label_mask(mask, grid, min_area):
# #     """connected regions (8-neighbour, one closing step so a 1-cell gap does not split a region) larger than min_area m2:
# #     list of dict(mask, area, centroid, ...) sorted by area"""
# #     if not mask.any():
# #         return []
# #     m = ndi.binary_closing(mask, structure=np.ones((3, 3), bool)) & (mask | ndi.binary_dilation(mask, iterations=1))
# #     lab, n = ndi.label(m, structure=np.ones((3, 3), int))
# #     out = []
# #     for i in range(1, n + 1):
# #         mm = (lab == i) & mask
# #         a = float(mm.sum() * grid.area)
# #         if a < min_area:
# #             continue
# #         out.append(dict(mask=mm, area=a, centroid=(float(grid.X[mm].mean()), float(grid.Y[mm].mean()))))
# #     out.sort(key=lambda r: -r["area"])
# #     return out


# # def compass(dx, dy):
# #     """name of the direction of (dx, dy) in the model frame (+y north, +x east)"""
# #     if math.hypot(dx, dy) < 1e-9:
# #         return "centre"
# #     names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
# #     return names[int(round(math.degrees(math.atan2(dy, dx)) / 45.0)) % 8]


# # def region_extent(mask, grid, flow):
# #     """length along the flow and width across it (m) of a region, from its cells"""
# #     xs, ys = grid.X[mask], grid.Y[mask]
# #     if flow is None:
# #         return float(np.ptp(xs)), float(np.ptp(ys))
# #     u = xs * flow[0] + ys * flow[1]
# #     v = -xs * flow[1] + ys * flow[0]
# #     return float(np.ptp(u) + grid.cell), float(np.ptp(v) + grid.cell)


# # def nearest_building(layout, pt, flow=None):
# #     """(id, distance) of the building nearest to pt; with flow: the nearest building UPWIND of pt (the one whose wake it may be), else the nearest"""
# #     best = None
# #     P = Point(pt)
# #     for bid, g in layout.buildings.items():
# #         d = g.distance(P)
# #         if flow is not None:
# #             c = g.centroid
# #             if (pt[0] - c.x) * flow[0] + (pt[1] - c.y) * flow[1] <= 0:
# #                 continue
# #         if best is None or d < best[1]:
# #             best = (bid, d)
# #     if best is None and flow is not None:
# #         return nearest_building(layout, pt, None)
# #     return best


# # # ============================================================================= layout differences
# # def layout_diff(Lb, Lc):
# #     """what changed between two layouts: building shifts, trees added / removed, canopy, ponds, lawns"""
# #     out = {"buildings": []}
# #     for bid, g in Lc.buildings.items():
# #         if bid in Lb.buildings and not bid.startswith("osm_"):
# #             d = (g.centroid.x - Lb.buildings[bid].centroid.x, g.centroid.y - Lb.buildings[bid].centroid.y)
# #             out["buildings"].append(dict(id=bid, dx=round(d[0], 2), dy=round(d[1], 2), shift=round(math.hypot(*d), 2)))
# #     sh = [b["shift"] for b in out["buildings"]]
# #     out["building_shift_mean"] = float(np.mean(sh)) if sh else 0.0
# #     out["building_shift_max"] = float(np.max(sh)) if sh else 0.0

# #     def key(t):
# #         return (round(t[1], 2), round(t[2], 2), round(t[3], 2))
# #     kb, kc = {key(t) for t in Lb.trees}, {key(t) for t in Lc.trees}
# #     out["trees_base"], out["trees_cand"] = len(Lb.trees), len(Lc.trees)
# #     out["trees_added"] = [t for t in Lc.trees if key(t) not in kb]
# #     out["trees_removed"] = [t for t in Lb.trees if key(t) not in kc]
# #     out["canopy_base"] = float(sum(math.pi * t[3] ** 2 for t in Lb.trees))
# #     out["canopy_cand"] = float(sum(math.pi * t[3] ** 2 for t in Lc.trees))
# #     out["mean_height_base"] = float(np.mean([t[4] for t in Lb.trees])) if Lb.trees else 0.0
# #     out["mean_height_cand"] = float(np.mean([t[4] for t in Lc.trees])) if Lc.trees else 0.0
# #     out["pond_area_base"] = float(sum(p.area for p in Lb.ponds))
# #     out["pond_area_cand"] = float(sum(p.area for p in Lc.ponds))
# #     out["lawn_area_base"] = float(sum(p.area for p in Lb.lawns))
# #     out["lawn_area_cand"] = float(sum(p.area for p in Lc.lawns))
# #     out["ponds_cand"], out["ponds_base"] = len(Lc.ponds), len(Lb.ponds)
# #     return out


# # def region_layout_context(region_mask, grid, flow, Lb, Lc, diff, near_m=25.0, upwind_len=70.0):
# #     """layout changes around one region: new / removed trees within near_m of it, trees added in a corridor upwind of it, ponds / lawns in the
# #     candidate near it (and in the base), buildings that moved near it"""
# #     from shapely.geometry import MultiPoint
# #     pts = np.column_stack([grid.X[region_mask], grid.Y[region_mask]])
# #     reg = MultiPoint(pts[::max(1, len(pts) // 400)]).buffer(0.75 * grid.cell)
# #     ctx = {}
# #     near = reg.buffer(near_m)
# #     shapely.prepare(near)
# #     added = [t for t in diff["trees_added"] if near.contains(Point(t[1], t[2]))]
# #     removed = [t for t in diff["trees_removed"] if near.contains(Point(t[1], t[2]))]
# #     ctx["trees_added_near"], ctx["trees_removed_near"] = len(added), len(removed)
# #     ctx["canopy_added_near_m2"] = float(sum(math.pi * t[3] ** 2 for t in added))
# #     ctx["canopy_removed_near_m2"] = float(sum(math.pi * t[3] ** 2 for t in removed))
# #     if flow is not None:
# #         c = reg.centroid
# #         u = lambda x, y: (x - c.x) * flow[0] + (y - c.y) * flow[1]            # noqa: E731
# #         v = lambda x, y: -(x - c.x) * flow[1] + (y - c.y) * flow[0]           # noqa: E731
# #         half = 0.5 * region_extent(region_mask, grid, flow)[1] + 10.0
# #         up_added = [t for t in diff["trees_added"] if -upwind_len <= u(t[1], t[2]) <= 0 and abs(v(t[1], t[2])) <= half]
# #         up_removed = [t for t in diff["trees_removed"] if -upwind_len <= u(t[1], t[2]) <= 0 and abs(v(t[1], t[2])) <= half]
# #         ctx["trees_added_upwind"], ctx["trees_removed_upwind"] = len(up_added), len(up_removed)

# #         def pond_upwind(L):
# #             n = 0.0
# #             for p in L.ponds:
# #                 q = p.centroid
# #                 if -upwind_len * 1.5 <= u(q.x, q.y) <= 0 and abs(v(q.x, q.y)) <= half + 15.0:
# #                     n += p.area
# #             return n
# #         ctx["pond_area_upwind_base"], ctx["pond_area_upwind_cand"] = pond_upwind(Lb), pond_upwind(Lc)
# #     ctx["pond_area_near_base"] = float(sum(p.intersection(near).area for p in Lb.ponds))
# #     ctx["pond_area_near_cand"] = float(sum(p.intersection(near).area for p in Lc.ponds))
# #     ctx["lawn_area_near_base"] = float(sum(p.intersection(near).area for p in Lb.lawns))
# #     ctx["lawn_area_near_cand"] = float(sum(p.intersection(near).area for p in Lc.lawns))
# #     moved = [b for b in diff["buildings"] if b["shift"] > 0.5 and Lc.buildings[b["id"]].distance(reg) <= near_m + 15.0]
# #     ctx["buildings_moved_near"] = [(b["id"], b["shift"]) for b in moved]
# #     return ctx


# # # ============================================================================= the analysis of one candidate in one season
# # def stress_areas(U, valid, grid):
# #     """area (m2) per UTCI category"""
# #     out = {}
# #     u = U[valid]
# #     edges = sorted(c[0] for c in CATEGORIES)
# #     names = {c[0]: c[1] for c in CATEGORIES}
# #     lo = -1e9
# #     for e in edges + [1e9]:
# #         nm = "extreme cold stress" if lo == -1e9 else names[lo]
# #         out[nm] = float(((u > lo) & (u <= e)).sum() * grid.area)
# #         lo = e
# #     return out


# # def field_stats(F, valid, grid):
# #     s = {}
# #     for k in FIELDS:
# #         v = F[k][valid]
# #         s[k] = float(np.nanmean(v)) if v.size else float("nan")
# #     s["utci_p10"], s["utci_p90"] = (float(np.nanpercentile(F["UTCI_C"][valid], q)) for q in (10, 90))
# #     s["area_m2"] = float(valid.sum() * grid.area)
# #     return s


# # def wake_regions(F, valid, grid, flow, u_ref, frac, min_area, layout):
# #     """calm regions: wind < frac * u_ref"""
# #     calm = valid & (F["wind_ms"] < frac * u_ref)
# #     regs = label_mask(calm, grid, min_area)
# #     out = []
# #     for k, r in enumerate(regs, 1):
# #         m = r["mask"]
# #         ln, wd = region_extent(m, grid, flow)
# #         nb = nearest_building(layout, r["centroid"], flow)
# #         out.append(dict(id=f"W{k}", area=r["area"], centroid=r["centroid"], length_m=ln, width_m=wd, mask=m,
# #                         wind_mean=float(np.nanmean(F["wind_ms"][m])), utci_mean=float(np.nanmean(F["UTCI_C"][m])),
# #                         building=None if nb is None else nb[0], building_dist=None if nb is None else float(nb[1])))
# #     return calm, out


# # def match_regions(regs_a, regs_b, min_overlap=0.2):
# #     """for every region of A: (index of the best overlapping region of B or None, overlap share of the smaller of the two)"""
# #     out = []
# #     for r in regs_a:
# #         best, bo = None, 0.0
# #         for j, s in enumerate(regs_b):
# #             ov = float((r["mask"] & s["mask"]).sum()) / max(min(float(r["mask"].sum()), float(s["mask"].sum())), 1.0)
# #             if ov > bo:
# #                 best, bo = j, ov
# #         out.append((best if bo >= min_overlap else None, bo))
# #     return out


# # def match_wakes(wa, wb, max_shift_m=45.0):
# #     """base wake -> layout wake: first the wake that trails the same building (nearest centroid within max_shift_m, since a moved building drags its wake
# #     along), else the one it overlaps.  Returns (list of index or None per wake of wa, set of indices of wb that were matched)"""
# #     res, used = [], set()
# #     ov = match_regions(wa, wb)
# #     for k, w in enumerate(wa):
# #         cand = [j for j, v in enumerate(wb) if j not in used and v["building"] is not None and v["building"] == w["building"]
# #                 and math.hypot(v["centroid"][0] - w["centroid"][0], v["centroid"][1] - w["centroid"][1]) <= max_shift_m]
# #         if cand:
# #             j = min(cand, key=lambda j: math.hypot(wb[j]["centroid"][0] - w["centroid"][0], wb[j]["centroid"][1] - w["centroid"][1]))
# #         else:
# #             j = ov[k][0]
# #             if j in used:
# #                 j = None
# #         res.append(j)
# #         if j is not None:
# #             used.add(j)
# #     return res, used


# # UNIT = {"Tmrt": "°C", "wind": "m/s", "Ta": "°C", "RH": "%"}
# # NDEC = {"Tmrt": 1, "wind": 2, "Ta": 1, "RH": 1}
# # ORDER = ("Ta", "Tmrt", "RH", "wind")                    # order of the columns of the value tables


# # def zone_vals(mask, B, C, both, sp, grid):
# #     """area-averaged inputs of a zone in the base and in the layout, their change, and the part of the UTCI change each one caused (exact split):
# #     dict(area, n, UTCI=dict(b, c, d), Ta=dict(b, c, d, eff), Tmrt=..., RH=..., wind=...)"""
# #     m = mask & both
# #     if not m.any():
# #         return None
# #     v = dict(area=float(mask.sum() * grid.area), n=int(m.sum()))
# #     ub, uc = float(np.nanmean(B["UTCI_C"][m])), float(np.nanmean(C["UTCI_C"][m]))
# #     v["UTCI"] = dict(b=ub, c=uc, d=uc - ub)
# #     for dr in DRIVERS:
# #         col = DRIVER_COL[dr]
# #         b, c = float(np.nanmean(B[col][m])), float(np.nanmean(C[col][m]))
# #         v[dr] = dict(b=b, c=c, d=c - b, eff=float(np.nanmean(sp["contrib"][dr][m])))
# #     return v


# # def vtxt(v, dr, eff=True, nl=False):
# #     """'36.1 → 35.9 (−0.2) → −0.27' : base → layout (change) → effect on UTCI (°C)"""
# #     n = NDEC[dr]
# #     x = v[dr]
# #     s = f"{x['b']:.{n}f} → {x['c']:.{n}f} ({x['d']:+.{n}f})"
# #     if eff:
# #         s += ("\n" if nl else " → ") + f"{x['eff']:+.2f}"
# #     return s


# # def ctxt(v, dr):
# #     """compact label text for drawing on a map"""
# #     n = NDEC[dr] if dr != "RH" else 0
# #     return f"{dr if dr != 'wind' else 'U'} {v[dr]['b']:.{n}f}→{v[dr]['c']:.{n}f}"


# # def inputs_sentence(v, sgn=None):
# #     """'Ta 36.1 → 35.9 °C (−0.2) lowers UTCI by 0.27 °C; Tmrt ...' for the four inputs in order of their effect"""
# #     items = sorted(DRIVERS, key=lambda d: -abs(v[d]["eff"]))
# #     parts = []
# #     for d in items:
# #         x = v[d]
# #         n = NDEC[d]
# #         verb = "raises" if x["eff"] > 0.005 else ("lowers" if x["eff"] < -0.005 else "no effect on")
# #         tail = f" {verb} UTCI by {abs(x['eff']):.2f} °C" if abs(x["eff"]) >= 0.005 else " no effect on UTCI"
# #         parts.append(f"{d} {x['b']:.{n}f} → {x['c']:.{n}f} {UNIT[d]} ({x['d']:+.{n}f}){tail}")
# #     return "; ".join(parts)


# # class SeasonResult:
# #     pass


# # def analyse_pair(base, cand, grid, a):
# #     """everything for one candidate / one season"""
# #     R = SeasonResult()
# #     R.season, R.name = cand.season, cand.name
# #     heat = cand.season == "heat"
# #     sgn = -1.0 if heat else 1.0                       # benefit = sgn * dUTCI
# #     B, C = rasterise(base, grid), rasterise(cand, grid)
# #     both = B["valid"] & C["valid"]
# #     if not B["valid"].any() or not C["valid"].any() or not both.any():
# #         raise ValueError(f"{cand.name}/{cand.season}: no open ground on the analysis grid (valid cells base {int(B['valid'].sum())}, layout {int(C['valid'].sum())}, both {int(both.sum())}); "
# #                          f"grid {grid.shape[1]} x {grid.shape[0]} cells of {grid.cell:g} m, site bounds {tuple(round(v, 1) for v in cand.layout.site.bounds) if cand.layout.site is not None else None}, "
# #                          f"sample points x {np.nanmin(cand.pts['x']):.1f}..{np.nanmax(cand.pts['x']):.1f}, y {np.nanmin(cand.pts['y']):.1f}..{np.nanmax(cand.pts['y']):.1f}")
# #     R.B, R.C, R.both, R.sgn = B, C, both, sgn
# #     R.flow = cand.flow() if cand.flow() is not None else base.flow()
# #     u10 = cand.boundary.get("wind_speed_10m_ms") or base.boundary.get("wind_speed_10m_ms")
# #     R.u10 = float(u10) if u10 else float("nan")
# #     # reference wind = free pedestrian-level wind of the BASE case (90th percentile of its 1.5 m field), same value for both layouts
# #     R.u_ref = float(a.u_ref) if a.u_ref else float(np.nanpercentile(B["wind_ms"][B["valid"]], 90))
# #     R.stats_b, R.stats_c = field_stats(B, B["valid"], grid), field_stats(C, C["valid"], grid)
# #     R.stats_b_both, R.stats_c_both = field_stats(B, both, grid), field_stats(C, both, grid)
# #     R.cat_b, R.cat_c = stress_areas(B["UTCI_C"], B["valid"], grid), stress_areas(C["UTCI_C"], C["valid"], grid)
# #     R.obj_b = base.summary.get("utci_C")
# #     R.obj_c = cand.summary.get("utci_C")
# #     R.diff = layout_diff(base.layout, cand.layout)

# #     # ---- exact split of the UTCI change
# #     sp = shapley_split(B, C, both)
# #     R.split = sp
# #     tot = sp["u_cand"] - sp["u_base"]
# #     R.dU = tot
# #     R.contrib_mean = {d: float(np.nanmean(sp["contrib"][d][both])) for d in DRIVERS}
# #     R.dU_mean = float(np.nanmean(tot[both]))
# #     R.split_err = float(np.nanmax(np.abs(sum(sp["contrib"][d] for d in DRIVERS) - tot)[both])) if both.any() else 0.0
# #     R.csv_err = float(np.nanmean(np.abs(sp["u_base"][both] - B["UTCI_C"][both])))
# #     R.tmrt = tmrt_attribution(B, C, both)
# #     if R.tmrt is not None:
# #         R.tmrt_parts_mean = {k: float(np.nanmean(v[both])) for k, v in R.tmrt["parts"].items()}
# #         R.dTmrt_mean = float(np.nanmean(R.tmrt["dTmrt"][both]))

# #     R.site_vals = zone_vals(both, B, C, both, sp, grid)

# #     # ---- hot / cold spots of the base case
# #     ub = B["UTCI_C"]
# #     if heat:
# #         thr = max(float(np.nanpercentile(ub[B["valid"]], a.hot_pct)), 26.0)
# #         spot_b = B["valid"] & (ub >= thr)
# #         spot_c = C["valid"] & (C["UTCI_C"] >= thr)
# #     else:
# #         thr = min(float(np.nanpercentile(ub[B["valid"]], 100 - a.cold_pct)), 9.0)
# #         spot_b = B["valid"] & (ub <= thr)
# #         spot_c = C["valid"] & (C["UTCI_C"] <= thr)
# #     R.thr = thr
# #     R.spot_b, R.spot_c = spot_b, spot_c
# #     R.spot_area_b, R.spot_area_c = float(spot_b.sum() * grid.area), float(spot_c.sum() * grid.area)
# #     regs = label_mask(spot_b, grid, a.min_region_m2)
# #     R.regions = []
# #     for k, r in enumerate(regs, 1):
# #         m = r["mask"] & both
# #         if m.sum() < 3:
# #             continue
# #         info = dict(id=("H" if heat else "C") + str(k), area=r["area"], centroid=r["centroid"], mask=r["mask"])
# #         info["u_base"] = float(np.nanmean(B["UTCI_C"][m]))
# #         info["u_cand"] = float(np.nanmean(C["UTCI_C"][m]))
# #         info["d_u"] = info["u_cand"] - info["u_base"]
# #         info["benefit"] = sgn * info["d_u"]
# #         if heat:
# #             still = (C["UTCI_C"][m] >= thr)
# #         else:
# #             still = (C["UTCI_C"][m] <= thr)
# #         info["resolved_share"] = float(1.0 - still.mean())
# #         info["contrib"] = {d: float(np.nanmean(sp["contrib"][d][m])) for d in DRIVERS}
# #         for key, lab in (("Tair_C", "Ta"), ("RH_pct", "RH"), ("wind_ms", "wind"), ("Tmrt_C", "Tmrt"), ("sunlit", "sunlit"), ("sky_open", "sky_open")):
# #             info[lab + "_b"], info[lab + "_c"] = float(np.nanmean(B[key][m])), float(np.nanmean(C[key][m]))
# #         if R.tmrt is not None:
# #             info["tmrt_parts"] = {k2: float(np.nanmean(v[m])) for k2, v in R.tmrt["parts"].items()}
# #         dom = max(DRIVERS, key=lambda d: sgn * info["contrib"][d])
# #         info["dominant"] = dom
# #         if info["benefit"] > 0.3 and info["resolved_share"] >= 0.7:
# #             info["status"] = "resolved"
# #         elif info["benefit"] > 0.3 and info["resolved_share"] >= 0.3:
# #             info["status"] = "partly resolved"
# #         elif info["benefit"] > 0.3:
# #             info["status"] = "reduced"
# #         elif info["benefit"] < -0.3:
# #             info["status"] = "worse"
# #         else:
# #             info["status"] = "unchanged"
# #         info["context"] = region_layout_context(r["mask"], grid, R.flow, base.layout, cand.layout, R.diff)
# #         nb = nearest_building(cand.layout, r["centroid"], R.flow)
# #         info["near_building"] = None if nb is None else nb[0]
# #         info["near_building_dist"] = None if nb is None else float(nb[1])
# #         info["cause_b"], info["cause_c"] = excess_split(B, r["mask"] & B["valid"], B["valid"]), excess_split(C, r["mask"] & C["valid"], C["valid"])
# #         info["wind_ratio_b"], info["wind_ratio_c"] = info["wind_b"] / R.u_ref, info["wind_c"] / R.u_ref
# #         info["vals"] = zone_vals(r["mask"], B, C, both, sp, grid)
# #         R.regions.append(info)
# #     new_mask = spot_c & ~spot_b & both & (sgn * tot < -0.3)          # stressed in the layout, not in the base, and clearly worse than the base
# #     new_regs = label_mask(new_mask, grid, a.min_region_m2)
# #     R.new_regions = []
# #     for k, r in enumerate(new_regs, 1):
# #         m = r["mask"]
# #         info = dict(id=("nH" if heat else "nC") + str(k), area=r["area"], centroid=r["centroid"], mask=m)
# #         info["u_cand"] = float(np.nanmean(C["UTCI_C"][m]))
# #         for key, lab in (("Tair_C", "Ta"), ("wind_ms", "wind"), ("Tmrt_C", "Tmrt"), ("sunlit", "sunlit")):
# #             info[lab + "_c"] = float(np.nanmean(C[key][m]))
# #         b2 = (m & both)
# #         if b2.any():
# #             info["u_base"] = float(np.nanmean(B["UTCI_C"][b2]))
# #             info["contrib"] = {d: float(np.nanmean(sp["contrib"][d][b2])) for d in DRIVERS}
# #             info["dominant"] = max(DRIVERS, key=lambda d: -sgn * info["contrib"][d])        # the driver that made it worse
# #         info["context"] = region_layout_context(m, grid, R.flow, base.layout, cand.layout, R.diff)
# #         nb = nearest_building(cand.layout, r["centroid"], R.flow)
# #         info["near_building"] = None if nb is None else nb[0]
# #         info["cause_c"] = excess_split(C, m & C["valid"], C["valid"])
# #         info["sunlit_c"], info["sky_open_c"] = float(np.nanmean(C["sunlit"][m])), float(np.nanmean(C["sky_open"][m]))
# #         info["RH_c"] = float(np.nanmean(C["RH_pct"][m]))
# #         info["vals"] = zone_vals(m, B, C, both, sp, grid)
# #         R.new_regions.append(info)

# #     # ---- wakes / ventilation
# #     calm_b, wb = wake_regions(B, B["valid"], grid, R.flow, R.u_ref, a.wake_frac, a.min_region_m2, base.layout)
# #     calm_c, wc = wake_regions(C, C["valid"], grid, R.flow, R.u_ref, a.wake_frac, a.min_region_m2, cand.layout)
# #     R.calm_b, R.calm_c, R.wakes_b, R.wakes_c = calm_b, calm_c, wb, wc
# #     R.vent = dict(
# #         wind_ratio_b=R.stats_b["wind_ms"] / R.u_ref, wind_ratio_c=R.stats_c["wind_ms"] / R.u_ref,
# #         calm_area_b=float(calm_b.sum() * grid.area), calm_area_c=float(calm_c.sum() * grid.area),
# #         calm_share_b=float(calm_b.sum() / max(B["valid"].sum(), 1)), calm_share_c=float(calm_c.sum() / max(C["valid"].sum(), 1)),
# #         fast_share_b=float(((B["wind_ms"] > 1.3 * R.u_ref) & B["valid"]).sum() / max(B["valid"].sum(), 1)),
# #         fast_share_c=float(((C["wind_ms"] > 1.3 * R.u_ref) & C["valid"]).sum() / max(C["valid"].sum(), 1)),
# #         longest_wake_b=max([w["length_m"] for w in wb], default=0.0), longest_wake_c=max([w["length_m"] for w in wc], default=0.0),
# #         wind_ratio_in_spots_b=float(np.nanmean(B["wind_ms"][spot_b]) / R.u_ref) if spot_b.any() else float("nan"),
# #         wind_ratio_in_spots_c=float(np.nanmean(C["wind_ms"][spot_c]) / R.u_ref) if spot_c.any() else float("nan"))
# #     mt, used = match_wakes(wb, wc)
# #     for w, j in zip(wb, mt):
# #         w["fate"] = "gone" if j is None else ("smaller" if wc[j]["area"] < 0.8 * w["area"] else ("larger" if wc[j]["area"] > 1.25 * w["area"] else "similar"))
# #         w["cand_match"] = None if j is None else wc[j]["id"]
# #         w["cand_area"] = 0.0 if j is None else wc[j]["area"]
# #         m = w["mask"] & both
# #         if m.any():
# #             w["d_wind"] = float(np.nanmean(C["wind_ms"][m] - B["wind_ms"][m]))
# #             w["d_u"] = float(np.nanmean(C["UTCI_C"][m] - B["UTCI_C"][m]))
# #             w["contrib"] = {d: float(np.nanmean(sp["contrib"][d][m])) for d in DRIVERS}
# #         w["context"] = region_layout_context(w["mask"], grid, R.flow, base.layout, cand.layout, R.diff)
# #         w["vals"] = zone_vals(w["mask"], B, C, both, sp, grid)
# #     for j, w in enumerate(wc):
# #         w["is_new"] = j not in used
# #         w["vals"] = zone_vals(w["mask"], B, C, both, sp, grid)
# #     # ---- fast channels
# #     return R


# # # ============================================================================= drawing
# # def _footprint_patches(ax, layout, tree_alpha=1.0, faces=True, tree_lw=0.5):
# #     if layout.site is not None:
# #         ax.plot(*layout.site.exterior.xy, color=INK, lw=1.0)
# #     for g in layout.lawns:
# #         ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#d8ecd0" if faces else "none", ec="#8fb98a", lw=0.5, alpha=0.8, zorder=3))
# #     for g in layout.ponds:
# #         ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#9fcbff" if faces else "none", ec="#2b6cb0", lw=0.9, alpha=0.85 if faces else 1.0, zorder=4))
# #     for bid, g in layout.buildings.items():
# #         ax.add_patch(MPoly(np.array(g.exterior.coords), fc="#8a8f94", ec=INK, lw=0.7, zorder=6))
# #     for t in layout.trees:
# #         ax.add_patch(Circle((t[1], t[2]), t[3], fc="none", ec="#2f7d32", lw=tree_lw, alpha=tree_alpha, zorder=5))


# # def _setup(ax, grid, title, layout=None, small=False):
# #     ax.set_aspect("equal")
# #     ax.set_xlim(grid.extent[0], grid.extent[1])
# #     ax.set_ylim(grid.extent[2], grid.extent[3])
# #     ax.set_title(title, fontsize=9 if small else 10, color=INK)
# #     ax.tick_params(labelsize=7, colors=MUTED)
# #     for s in ax.spines.values():
# #         s.set_color("#c8ced3")
# #     if layout is not None:
# #         _footprint_patches(ax, layout)


# # def _flow_arrow(ax, grid, flow, label=True):
# #     if flow is None:
# #         return
# #     x0, x1, y0, y1 = grid.extent
# #     L = 0.12 * (x1 - x0)
# #     cx, cy = x0 + 0.1 * (x1 - x0), y1 - 0.07 * (y1 - y0)
# #     ax.annotate("", xy=(cx + L * flow[0], cy + L * flow[1]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#222"), zorder=20)
# #     if label:
# #         ax.text(cx, cy - 0.025 * (y1 - y0), "wind", fontsize=7, color="#222", ha="left", va="top", zorder=20)


# # def _nice_levels(vmin, vmax, n=12):
# #     span = max(vmax - vmin, 1e-9)
# #     raw = span / n
# #     mag = 10 ** math.floor(math.log10(raw))
# #     step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
# #     lo, hi = math.floor(vmin / step + 1e-9) * step, math.ceil(vmax / step - 1e-9) * step
# #     return np.arange(lo, hi + 0.5 * step, step)


# # def _field(ax, grid, Z, vmin, vmax, cmap, levels=None, lines=None, fill_levels=12):
# #     Zm = np.ma.masked_invalid(Z)
# #     lv = _nice_levels(vmin, vmax, fill_levels)
# #     cf = ax.contourf(grid.X, grid.Y, np.clip(Zm, lv[0], lv[-1]), levels=lv, cmap=cmap, extend="neither", zorder=1)
# #     if lines is not None and len(lines):
# #         try:
# #             cs = ax.contour(grid.X, grid.Y, Zm, levels=lines, colors="#222", linewidths=0.5, alpha=0.6, zorder=2)
# #             ax.clabel(cs, fontsize=6, fmt="%g")
# #         except Exception:
# #             pass
# #     return cf


# # def _outline(ax, grid, mask, **kw):
# #     if mask is None or not mask.any():
# #         return
# #     try:
# #         ax.contour(grid.X, grid.Y, mask.astype(float), levels=[0.5], **kw)
# #     except Exception:
# #         pass


# # def _cbar(fig, cf, ax, label):
# #     cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
# #     cb.set_label(label, fontsize=7)
# #     cb.ax.tick_params(labelsize=6)
# #     return cb


# # def _label_regions(ax, regs, color="#111", prefix=None, fs=7):
# #     for r in regs:
# #         ax.text(r["centroid"][0], r["centroid"][1], r["id"], fontsize=fs, ha="center", va="center", color=color, weight="bold", zorder=25,
# #                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


# # def _label_zones(ax, zones, fs=5.3, color="#111"):
# #     """write the area-averaged UTCI, Ta, Tmrt, RH and wind (base → layout) of every zone on the map; boxes that would overlap are pushed apart
# #     and joined to their zone by a thin line"""
# #     anns = []
# #     for z in zones:
# #         v = z.get("vals")
# #         if v is None:
# #             continue
# #         txt = f"{z['id']}\nUTCI {v['UTCI']['b']:.1f}→{v['UTCI']['c']:.1f}\n" + "\n".join(ctxt(v, d) for d in ORDER)
# #         an = ax.annotate(txt, xy=z["centroid"], xytext=z["centroid"], textcoords="data", fontsize=fs, ha="center", va="center", color=color, zorder=25, linespacing=1.05,
# #                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", lw=0.4, alpha=0.9), arrowprops=dict(arrowstyle="-", lw=0.5, color="#444", shrinkA=0, shrinkB=0))
# #         anns.append((an, z["centroid"]))
# #     if len(anns) < 2:
# #         return
# #     fig = ax.figure
# #     fig.canvas.draw()
# #     rend = fig.canvas.get_renderer()
# #     inv = ax.transData.inverted()
# #     axbb = ax.get_window_extent(rend)
# #     pos = [np.array(ax.transData.transform(c), float) for _, c in anns]            # display coordinates of the label centres
# #     size = []
# #     for an, _ in anns:
# #         bb = matplotlib.text.Text.get_window_extent(an, rend)
# #         size.append((bb.width + 6.0, bb.height + 6.0))
# #     for _ in range(80):
# #         moved = False
# #         for i in range(len(anns)):
# #             for j in range(i + 1, len(anns)):
# #                 dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
# #                 ox = 0.5 * (size[i][0] + size[j][0]) - abs(dx)
# #                 oy = 0.5 * (size[i][1] + size[j][1]) - abs(dy)
# #                 if ox > 0 and oy > 0:
# #                     moved = True
# #                     if oy <= ox:
# #                         sgn_ = 1.0 if dy >= 0 else -1.0
# #                         pos[i][1] -= sgn_ * 0.5 * (oy + 1.0)
# #                         pos[j][1] += sgn_ * 0.5 * (oy + 1.0)
# #                     else:
# #                         sgn_ = 1.0 if dx >= 0 else -1.0
# #                         pos[i][0] -= sgn_ * 0.5 * (ox + 1.0)
# #                         pos[j][0] += sgn_ * 0.5 * (ox + 1.0)
# #         for i in range(len(anns)):                                               # stay inside the axes
# #             pos[i][0] = float(np.clip(pos[i][0], axbb.x0 + 0.5 * size[i][0], axbb.x1 - 0.5 * size[i][0]))
# #             pos[i][1] = float(np.clip(pos[i][1], axbb.y0 + 0.5 * size[i][1], axbb.y1 - 0.5 * size[i][1]))
# #         if not moved:
# #             break
# #     for (an, c), pp in zip(anns, pos):
# #         an.xyann = tuple(inv.transform(pp))


# # def fig_maps(R, base, cand, grid, path):
# #     """2 x 4 panels: UTCI base / layout / change / spots, and the four drivers"""
# #     heat = R.season == "heat"
# #     ub, uc = R.B["UTCI_C"], R.C["UTCI_C"]
# #     allv = np.concatenate([ub[R.B["valid"]], uc[R.C["valid"]]])
# #     vmin, vmax = float(np.floor(np.nanpercentile(allv, 1) - 0.5)), float(np.ceil(np.nanpercentile(allv, 99) + 0.5))
# #     if vmax - vmin < 4:
# #         vmax = vmin + 4
# #     cat_lines = [c[0] for c in CATEGORIES if vmin < c[0] < vmax]
# #     fig, axs = plt.subplots(2, 4, figsize=(20, 9.6))
# #     # row 1
# #     cf = _field(axs[0, 0], grid, ub, vmin, vmax, RAINBOW, lines=cat_lines)
# #     _setup(axs[0, 0], grid, f"UTCI base  ({base.name})  mean {R.stats_b['UTCI_C']:.1f} °C", base.layout)
# #     _cbar(fig, cf, axs[0, 0], "UTCI °C")
# #     cf = _field(axs[0, 1], grid, uc, vmin, vmax, RAINBOW, lines=cat_lines)
# #     _setup(axs[0, 1], grid, f"UTCI  {cand.name}  mean {R.stats_c['UTCI_C']:.1f} °C", cand.layout)
# #     _cbar(fig, cf, axs[0, 1], "UTCI °C")
# #     d = np.where(R.both, R.dU, np.nan)
# #     lim = max(0.5, float(np.nanpercentile(np.abs(d[R.both]), 98)))
# #     cf = _field(axs[0, 2], grid, d, -lim, lim, "RdBu_r", lines=None)
# #     _setup(axs[0, 2], grid, "change of UTCI (layout − base)\nblue = cooler, red = warmer", cand.layout)
# #     _outline(axs[0, 2], grid, R.spot_b, colors="k", linewidths=1.2)
# #     _outline(axs[0, 2], grid, R.spot_c, colors="k", linewidths=1.2, linestyles="--")
# #     _cbar(fig, cf, axs[0, 2], "ΔUTCI °C")
# #     # spots panel
# #     ax = axs[0, 3]
# #     _setup(ax, grid, f"{'hot' if heat else 'cold'} spots of the base (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C)\nand what became of them\n(zone means, base → layout: UTCI, Ta °C, Tmrt °C, RH %, U m/s)", cand.layout, small=True)
# #     colors = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c"}
# #     for r in R.regions:
# #         ax.contourf(grid.X, grid.Y, np.where(r["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[colors[r["status"]]], alpha=0.75, zorder=2)
# #     for r in R.new_regions:
# #         ax.contourf(grid.X, grid.Y, np.where(r["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#8e44ad"], alpha=0.8, zorder=2)
# #     _label_zones(ax, R.regions)
# #     _label_zones(ax, R.new_regions, color="#4a235a")
# #     import matplotlib.patches as mp
# #     ax.legend(handles=[mp.Patch(color=c, label=k) for k, c in colors.items()] + [mp.Patch(color="#8e44ad", label="new in layout")],
# #               fontsize=6, loc="lower left", framealpha=0.85)
# #     # row 2: drivers
# #     def diffpanel(ax, key, title, unit, scale=1.0):
# #         dd = np.where(R.both, (R.C[key] - R.B[key]) * scale, np.nan)
# #         lm = max(0.01, float(np.nanpercentile(np.abs(dd[R.both]), 98)))
# #         cf_ = _field(ax, grid, dd, -lm, lm, "RdBu_r")
# #         _setup(ax, grid, title, cand.layout)
# #         _cbar(fig, cf_, ax, unit)
# #         return cf_
# #     diffpanel(axs[1, 0], "Tmrt_C", "Δ Tmrt (layout − base)", "°C")
# #     ax = axs[1, 1]
# #     diffpanel(ax, "wind_ms", f"Δ wind speed (reference wind {R.u_ref:.1f} m/s)\nwake: solid = base, dashed = layout", "m/s")
# #     _outline(ax, grid, R.calm_b, colors="k", linewidths=1.1)
# #     _outline(ax, grid, R.calm_c, colors="k", linewidths=1.1, linestyles="--")
# #     _flow_arrow(ax, grid, R.flow)
# #     diffpanel(axs[1, 2], "Tair_C", "Δ air temperature", "°C")
# #     diffpanel(axs[1, 3], "RH_pct", "Δ relative humidity", "%")
# #     fig.suptitle(f"{cand.name} vs base — {SEASON_TXT[R.season]}   (site mean UTCI {R.stats_b['UTCI_C']:.2f} → {R.stats_c['UTCI_C']:.2f} °C)", fontsize=12, color=INK)
# #     fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
# #     fig.savefig(path, dpi=80)
# #     plt.close(fig)


# # def fig_drivers(R, grid, path):
# #     """why: Shapley split of the UTCI change (site mean and the regions), Tmrt split"""
# #     sgn = R.sgn
# #     fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1.6, 1.1]))
# #     ax = axs[0]
# #     names = list(DRIVERS)
# #     vals = [sgn * R.contrib_mean[d] for d in names]
# #     ax.barh(range(len(names)), vals, color=[COL[d] for d in names])
# #     ax.set_yticks(range(len(names)))
# #     ax.set_yticklabels([f"{d}\nΔ {R.site_vals[d]['d']:+.{NDEC[d]}f} {UNIT[d]}" for d in names], fontsize=8)
# #     for i, v in enumerate(vals):
# #         ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
# #     ax.axvline(0, color=INK, lw=0.8)
# #     ax.set_title(f"site mean: benefit {sgn * R.dU_mean:+.2f} °C\n(benefit = {'cooler' if sgn < 0 else 'warmer'})", fontsize=10)
# #     ax.margins(x=0.3)
# #     ax.invert_yaxis()
# #     ax.set_xlabel("°C of UTCI benefit  (label: actual change of the input)")
# #     ax = axs[1]
# #     labels, per = [], []
# #     for r in R.regions:
# #         vv = r["vals"]
# #         labels.append(f"{r['id']}\n{r['area']:.0f} m²\n" + "\n".join(f"Δ{'U' if d == 'wind' else d} {vv[d]['d']:+.{NDEC[d]}f}" for d in ORDER))
# #         per.append([sgn * r["contrib"][d] for d in DRIVERS])
# #     if per:
# #         per = np.array(per)
# #         x = np.arange(len(labels))
# #         pos, neg = np.zeros(len(labels)), np.zeros(len(labels))
# #         for j, d in enumerate(DRIVERS):
# #             v = per[:, j]
# #             bot = np.where(v >= 0, pos, neg)
# #             ax.bar(x, v, bottom=bot, color=COL[d], label=d, width=0.6)
# #             pos += np.where(v >= 0, v, 0)
# #             neg += np.where(v < 0, v, 0)
# #         ax.plot(x, [r["benefit"] for r in R.regions], "kD", ms=5, label="total")
# #         ax.set_xticks(x)
# #         ax.set_xticklabels(labels, fontsize=7)
# #         ax.axhline(0, color=INK, lw=0.8)
# #         ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
# #     ax.set_title(f"base {'hot' if R.season == 'heat' else 'cold'}-spot regions: benefit by driver", fontsize=10)
# #     ax.set_ylabel("°C of UTCI benefit")
# #     ax = axs[2]
# #     if R.tmrt is not None:
# #         parts = R.tmrt_parts_mean
# #         names2 = list(parts.keys())
# #         v2 = [-p if R.season == "heat" else p for p in parts.values()]
# #         ax.barh(range(len(names2)), v2, color="#d9822b", alpha=0.85)
# #         ax.set_yticks(range(len(names2)))
# #         ax.set_yticklabels(names2, fontsize=8)
# #         for i, v in enumerate(v2):
# #             ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
# #         ax.axvline(0, color=INK, lw=0.8)
# #         ax.invert_yaxis()
# #         ax.margins(x=0.25)
# #         ax.set_title(f"inside Tmrt: Δ Tmrt = {R.dTmrt_mean:+.2f} °C  (regression R² {R.tmrt['r2']:.2f})\nbenefit in Tmrt °C", fontsize=9)
# #     fig.tight_layout()
# #     fig.savefig(path, dpi=80)
# #     plt.close(fig)


# # def fig_wakes(R, base, cand, grid, path):
# #     fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
# #     ratio_b, ratio_c = R.B["wind_ms"] / R.u_ref, R.C["wind_ms"] / R.u_ref
# #     vmax = float(np.ceil(10 * max(1.6, np.nanpercentile(np.concatenate([ratio_b[R.B["valid"]], ratio_c[R.C["valid"]]]), 99))) / 10)
# #     for ax, ratio, F, name, lay, wk in ((axs[0], ratio_b, R.B, base.name, base.layout, R.wakes_b), (axs[1], ratio_c, R.C, cand.name, cand.layout, R.wakes_c)):
# #         cf = _field(ax, grid, ratio, 0.0, vmax, "viridis", lines=[0.5, 1.0, 1.3])
# #         _setup(ax, grid, f"wind speed / reference wind — {name}\ncalm (< {R.wake_frac:g}) regions outlined", lay)
# #         _outline(ax, grid, (F["wind_ms"] < R.wake_frac * R.u_ref) & F["valid"], colors="w", linewidths=1.4)
# #         _label_regions(ax, wk, color="#7a1f1f")
# #         _flow_arrow(ax, grid, R.flow)
# #         _cbar(fig, cf, ax, "U / U_ref")
# #     ax = axs[2]
# #     _setup(ax, grid, "wake regions: what became of the base wakes\n(zone means, base → layout: UTCI, Ta, Tmrt, RH, U)", cand.layout)
# #     fate_col = {"gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b"}
# #     for w in R.wakes_b:
# #         ax.contourf(grid.X, grid.Y, np.where(w["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[fate_col[w["fate"]]], alpha=0.75, zorder=2)
# #     for w in R.wakes_c:
# #         if w.get("is_new"):
# #             ax.contourf(grid.X, grid.Y, np.where(w["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#8e44ad"], alpha=0.8, zorder=2)
# #     _label_zones(ax, R.wakes_b)
# #     _label_zones(ax, [w for w in R.wakes_c if w.get("is_new")], color="#4a235a")
# #     import matplotlib.patches as mp
# #     ax.legend(handles=[mp.Patch(color=c, label=f"base wake: {k}") for k, c in fate_col.items()] + [mp.Patch(color="#8e44ad", label="new wake in layout")],
# #               fontsize=6, loc="lower left", framealpha=0.85)
# #     _flow_arrow(ax, grid, R.flow)
# #     fig.suptitle(f"ventilation and wakes — {cand.name} vs base — {SEASON_TXT[R.season]}", fontsize=12, color=INK)
# #     fig.tight_layout(rect=(0, 0, 1, 0.95))
# #     fig.savefig(path, dpi=80)
# #     plt.close(fig)


# # def zone_table_rows(zones, with_status=True):
# #     rows = []
# #     for z in zones:
# #         v = z.get("vals")
# #         if v is None:
# #             continue
# #         tag = z.get("status") or z.get("fate") or ("new" if z.get("is_new") else "")
# #         if z["id"].startswith("n"):
# #             tag = "new"
# #         rows.append([f"{z['id']}\n{tag}", f"{z['area']:.0f}", f"{v['UTCI']['b']:.1f} → {v['UTCI']['c']:.1f}\n({v['UTCI']['d']:+.2f})"] + [vtxt(v, d, True, True) for d in ORDER])
# #     return rows


# # def fig_zones(R, cand, grid, path, kind):
# #     """map of the zones with their area-mean values written on them + a table: for Ta, Tmrt, RH and wind the base value → layout value (change)
# #     and the effect of that change on UTCI.  kind = 'spots' (hot / cold spots of the base and new ones) or 'wakes' (calm regions)"""
# #     heat = R.season == "heat"
# #     if kind == "spots":
# #         zones = R.regions + R.new_regions
# #         ttl = f"{'hot' if heat else 'cold'} spots of the base (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C) and new ones"
# #         colr = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c", "new": "#8e44ad"}
# #         colof = lambda z: colr["new"] if z["id"].startswith("n") else colr[z["status"]]            # noqa: E731
# #     else:
# #         zones = R.wakes_b + [w for w in R.wakes_c if w.get("is_new")]
# #         ttl = f"calm regions (wind < {R.wake_frac:g} x {R.u_ref:.2f} m/s): base wakes and new ones"
# #         colr = {"gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b", "new": "#8e44ad"}
# #         colof = lambda z: colr["new"] if z.get("is_new") else colr[z["fate"]]                     # noqa: E731
# #     zones = [z for z in zones if z.get("vals") is not None]
# #     if not zones:
# #         return False
# #     rows = zone_table_rows(zones)
# #     sv = R.site_vals
# #     rows.append(["whole site\n(open ground)", f"{sv['area']:.0f}", f"{sv['UTCI']['b']:.1f} → {sv['UTCI']['c']:.1f}\n({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d, True, True) for d in ORDER])
# #     nrow = len(rows)
# #     fig = plt.figure(figsize=(21, max(8.0, 1.2 + 0.62 * nrow)))
# #     gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
# #     axm, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
# #     _setup(axm, grid, ttl, cand.layout)
# #     for z in zones:
# #         axm.contourf(grid.X, grid.Y, np.where(z["mask"], 1.0, np.nan), levels=[0.5, 1.5], colors=[colof(z)], alpha=0.7, zorder=2)
# #     _label_zones(axm, zones, fs=5.6)
# #     _flow_arrow(axm, grid, R.flow)
# #     import matplotlib.patches as mp
# #     axm.legend(handles=[mp.Patch(color=c, label=k) for k, c in colr.items()], fontsize=7, loc="lower left", framealpha=0.9)
# #     axt.axis("off")
# #     cols = ["zone", "area m²", "UTCI °C\n(change)", "Ta °C\nbase → layout (change)\neffect on UTCI °C", "Tmrt °C\nbase → layout (change)\neffect on UTCI °C",
# #             "RH %\nbase → layout (change)\neffect on UTCI °C", "wind m/s\nbase → layout (change)\neffect on UTCI °C"]
# #     tb = axt.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colWidths=[0.1, 0.07, 0.12, 0.18, 0.18, 0.18, 0.18])
# #     tb.auto_set_font_size(False)
# #     tb.set_fontsize(7.6)
# #     tb.scale(1, 2.55)
# #     for (r, c), cell in tb.get_celld().items():
# #         cell.set_edgecolor("#c8ced3")
# #         if r == 0:
# #             cell.set_facecolor("#e8eef2")
# #             cell.set_text_props(weight="bold", fontsize=7)
# #             cell.set_height(cell.get_height() * 1.25)
# #         elif r == nrow:
# #             cell.set_facecolor("#f2f4f6")
# #             cell.set_text_props(weight="bold")
# #         elif c == 0:
# #             cell.set_facecolor(colof(zones[r - 1]))
# #             cell.set_text_props(color="white", weight="bold")
# #         elif c >= 3:                                         # shade the cell by the effect on UTCI (blue cooler, red warmer)
# #             eff = (sv if r == nrow else zones[r - 1]["vals"])[ORDER[c - 3]]["eff"]
# #             a_ = min(abs(eff) / 1.5, 1.0) * 0.55
# #             cell.set_facecolor((0.84, 0.19, 0.15, a_) if eff > 0 else (0.17, 0.48, 0.73, a_))
# #     fig.suptitle(f"{cand.name} vs base — {SEASON_TXT[R.season]}: zone-averaged inputs and what they did to UTCI   (effect: − = cooler, + = warmer)", fontsize=12, color=INK)
# #     fig.tight_layout(rect=(0, 0, 1, 0.96))
# #     fig.savefig(path, dpi=80)
# #     plt.close(fig)
# #     return True


# # # ============================================================================= plain two-picture view: heat (or cold) spots, ventilation
# # # Each picture: base map | layout map | key, then a table of the spots / calm zones with their Ta, Tmrt, RH and wind, then a few plain sentences.
# # # Everything is placed in inches on a fixed canvas, so boxes cannot overlap.
# # import textwrap                                                       # noqa: E402

# # BAND_EDGES = [-80, -40, -27, -13, 0, 9, 26, 32, 38, 46, 80]
# # BAND_NAMES = ["extreme cold stress", "very strong cold stress", "strong cold stress", "moderate cold stress", "slight cold stress", "no thermal stress",
# #               "moderate heat stress", "strong heat stress", "very strong heat stress", "extreme heat stress"]
# # BAND_RANGE = ["below −40", "−40 to −27", "−27 to −13", "−13 to 0", "0 to 9", "9 to 26", "26 to 32", "32 to 38", "38 to 46", "above 46"]
# # BAND_COL = ["#08306b", "#2b5fa8", "#4a8bc9", "#86b6e0", "#c5dcf0", "#e6f2dc", "#fde0a1", "#fbb04e", "#e5612a", "#9c2412"]
# # STATUS_COL = {"resolved": "#2e9d57", "partly resolved": "#9bc53d", "reduced": "#e8b230", "unchanged": "#c0392b", "worse": "#7b241c", "new": "#8e44ad",
# #               "gone": "#2e9d57", "smaller": "#9bc53d", "similar": "#e8b230", "larger": "#c0392b"}
# # STATUS_TXT = {"resolved": "gone", "partly resolved": "partly gone", "reduced": "reduced", "unchanged": "unchanged", "worse": "worse", "new": "NEW",
# #               "gone": "gone", "smaller": "smaller", "similar": "similar", "larger": "larger"}
# # SIMPLE_W = 16.0
# # SIMPLE_DPI = 200                                                      # --dpi
# # MAX_ROWS = 8
# # BLUE_TXT, RED_TXT, GREY_TXT = "#1f5f99", "#b3261e", "#5b666e"


# # def _inch_axes(fig, W, H, x, y, w, h):
# #     """axes covering the box (x, y, w, h) given in inches from the top-left of the canvas; its data units are inches too (y downwards)"""
# #     ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
# #     ax.set_xlim(0, w)
# #     ax.set_ylim(h, 0)
# #     ax.axis("off")
# #     return ax


# # def _short_where(centroid, layout, site_centre, flow=None, near=45.0):
# #     d = compass(centroid[0] - site_centre[0], centroid[1] - site_centre[1])
# #     s = f"{d} part" if d != "centre" else "centre"
# #     nb = nearest_building(layout, centroid, None)
# #     if nb is not None and nb[1] <= near:
# #         s += f", next to {_bn(nb[0])}" if nb[1] < 12 else f", {nb[1]:.0f} m from {_bn(nb[0])}"
# #     return s


# # def _bn(bid):
# #     """short, readable building name: 'building_3_internal_floor_0' -> 'B3'"""
# #     t = str(bid)
# #     t = re.sub(r"[_\- ]?(internal|external|inner|outer)?[_\- ]?floor[_\- ]?\d+$", "", t, flags=re.I)
# #     t = re.sub(r"[_\- ](internal|external)$", "", t, flags=re.I)
# #     t = re.sub(r"^building[_\- ]?", "B", t, flags=re.I) if re.match(r"^building[_\-]", t, flags=re.I) else t
# #     return t or str(bid)


# # def _building_groups(layout):
# #     """buildings that share a footprint (e.g. the floors of one building) -> list of (label, polygon of the largest member)"""
# #     ids = list(layout.buildings)
# #     polys = [layout.buildings[i] for i in ids]
# #     parent = list(range(len(ids)))

# #     def find(i):
# #         while parent[i] != i:
# #             parent[i] = parent[parent[i]]
# #             i = parent[i]
# #         return i
# #     try:
# #         from shapely.strtree import STRtree
# #         tree = STRtree(polys)
# #         pairs = ((i, int(j)) for i in range(len(polys)) for j in tree.query(polys[i]) if int(j) > i)
# #     except Exception:
# #         pairs = ((i, j) for i in range(len(polys)) for j in range(i + 1, len(polys)))
# #     for i, j in pairs:
# #         try:
# #             inter = polys[i].intersection(polys[j]).area
# #         except Exception:
# #             continue
# #         if inter > 0.5 * min(polys[i].area, polys[j].area):
# #             parent[find(j)] = find(i)
# #     groups = {}
# #     for k in range(len(ids)):
# #         groups.setdefault(find(k), []).append(k)
# #     out = []
# #     for mem in groups.values():
# #         big = max(mem, key=lambda k: polys[k].area)
# #         names = []
# #         for k in sorted(mem, key=lambda k: -polys[k].area):
# #             n = _bn(ids[k])
# #             if n not in names:
# #                 names.append(n)
# #         out.append(("\n".join(names[:2]), polys[big]))
# #     return out


# # def _building_labels(ax, layout, fs=8.0):
# #     for label, g in _building_groups(layout):
# #         if g.area < 120:
# #             continue
# #         p = g.representative_point()
# #         ax.text(p.x, p.y, label, fontsize=fs, color="white", ha="center", va="center", weight="bold", zorder=8, clip_on=True, linespacing=1.0)


# # def _scale_bar(ax, grid, length=50.0):
# #     x0, x1, y0, y1 = grid.extent
# #     xs, ys = x1 - 0.06 * (x1 - x0) - length, y0 + 0.05 * (y1 - y0)
# #     ax.plot([xs, xs + length], [ys, ys], color=INK, lw=2.2, solid_capstyle="butt", zorder=30)
# #     ax.text(xs + length / 2, ys + 0.012 * (y1 - y0), f"{length:g} m", fontsize=8, ha="center", va="bottom", color=INK, zorder=30,
# #             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


# # def _map_panel(ax, grid, layout, title, subtitle):
# #     ax.set_aspect("equal", adjustable="box")
# #     ax.set_xlim(grid.extent[0], grid.extent[1])
# #     ax.set_ylim(grid.extent[2], grid.extent[3])
# #     ax.set_xticks([])
# #     ax.set_yticks([])
# #     for s in ax.spines.values():
# #         s.set_color("#aab2b8")
# #     ax.set_title(title, fontsize=12.5, color=INK, weight="bold", loc="left", pad=22)
# #     ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5, color=GREY_TXT, ha="left", va="bottom")
# #     _footprint_patches(ax, layout, tree_alpha=0.55, faces=False, tree_lw=0.4)
# #     _building_labels(ax, layout)
# #     _scale_bar(ax, grid)


# # def _wind_arrow_simple(ax, grid, flow):
# #     if flow is None:
# #         return
# #     x0, x1, y0, y1 = grid.extent
# #     L = 0.14 * (x1 - x0)
# #     cx, cy = x0 + 0.12 * (x1 - x0), y1 - 0.09 * (y1 - y0)
# #     ax.annotate("", xy=(cx + L * flow[0], cy + L * flow[1]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=INK), zorder=30)
# #     ax.text(cx - 0.01 * (x1 - x0), cy + 0.02 * (y1 - y0), "wind", fontsize=9, color=INK, ha="center", va="bottom", zorder=30, weight="bold",
# #             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


# # def _place_badges(fig, ax, items, r_px=12.0):
# #     """numbered circles on a map: one per zone at its centre; circles that would touch are pushed apart and joined to their zone by a thin line"""
# #     if not items:
# #         return
# #     fig.canvas.draw()
# #     inv = ax.transData.inverted()
# #     true = [np.array(ax.transData.transform(it["xy"]), float) for it in items]
# #     pos = [p.copy() for p in true]
# #     axbb = ax.get_window_extent()
# #     d_min = 2 * r_px + 4
# #     for _ in range(200):
# #         moved = False
# #         for i in range(len(pos)):
# #             for j in range(i + 1, len(pos)):
# #                 v = pos[j] - pos[i]
# #                 dist = float(np.hypot(*v))
# #                 if dist < d_min:
# #                     moved = True
# #                     if dist < 1e-6:
# #                         v, dist = np.array([1.0, 0.3]), 1.04
# #                     push = 0.5 * (d_min - dist + 1.0) * v / dist
# #                     pos[i] -= push
# #                     pos[j] += push
# #         for i in range(len(pos)):
# #             pos[i][0] = float(np.clip(pos[i][0], axbb.x0 + r_px + 2, axbb.x1 - r_px - 2))
# #             pos[i][1] = float(np.clip(pos[i][1], axbb.y0 + r_px + 2, axbb.y1 - r_px - 2))
# #         if not moved:
# #             break
# #     for it, t, p in zip(items, true, pos):
# #         xy = it["xy"]
# #         xyt = tuple(inv.transform(p))
# #         if np.hypot(*(p - t)) > 0.6 * r_px:
# #             ax.plot([xy[0], xyt[0]], [xy[1], xyt[1]], color="#111", lw=0.9, zorder=26)
# #         ax.plot([xy[0]], [xy[1]], "o", ms=3.2, color="#111", zorder=27)
# #         ax.text(xyt[0], xyt[1], it["label"], fontsize=10, color="white", ha="center", va="center", weight="bold", zorder=28,
# #                 bbox=dict(boxstyle="circle,pad=0.28", fc=it["color"], ec="#111", lw=1.0))


# # def _rows_of_zones(zones, cap=MAX_ROWS):
# #     """largest zones first; returns (shown, hidden)"""
# #     z = sorted([x for x in zones if x.get("vals") is not None], key=lambda g: -g["area"])
# #     return z[:cap], z[cap:]


# # def _fmt_eff(x, dr):
# #     if abs(x) < 0.05:
# #         return "UTCI ±0", GREY_TXT
# #     return f"UTCI {x:+.1f}".replace("-", "−"), (BLUE_TXT if x < 0 else RED_TXT)


# # def _num(x, nd):
# #     return f"{x:.{nd}f}".replace("-", "−") if round(x, nd) != 0 else f"{0:.{nd}f}"


# # def _delta(x, nd):
# #     if round(x, nd) == 0:
# #         return f"{0:.{nd}f}"
# #     return f"{x:+.{nd}f}".replace("-", "−")


# # def _draw_zone_table(ax, W, rows, site, header_first="Spot", with_site=True, cell_h=0.56):
# #     """rows: list of dict(num, color, status, where, area, vals);  columns: spot | where | UTCI | Ta | Tmrt | RH | wind.  Returns the height used (inches)"""
# #     colw = [1.5, 3.05, 1.8, 2.2, 2.2, 2.2, 2.2]
# #     x = [0.0]
# #     for w in colw:
# #         x.append(x[-1] + w)
# #     hdr = [(header_first, ""), ("Where, how big", ""), ("UTCI °C", "feels-like temp."), ("Air temp. Ta °C", "base → layout"), ("Radiant temp. Tmrt °C", "base → layout"),
# #            ("Humidity RH %", "base → layout"), ("Wind m/s", "base → layout")]
# #     hh = 0.62
# #     ax.add_patch(plt.Rectangle((0, 0), x[-1], hh, fc="#e8eef2", ec="#c8ced3", lw=0.8))
# #     for k, (t, s) in enumerate(hdr):
# #         cx = 0.5 * (x[k] + x[k + 1])
# #         ax.text(cx, hh * (0.40 if s else 0.5), t, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
# #         if s:
# #             ax.text(cx, hh * 0.78, s, ha="center", va="center", fontsize=8.5, color=GREY_TXT)
# #     y = hh
# #     allrows = list(rows) + ([dict(num=None, status="", where="Whole site (open ground)", area=site["area"], vals=site, site=True)] if with_site and site else [])
# #     for rw in allrows:
# #         v = rw["vals"]
# #         is_site = rw.get("site", False)
# #         ax.add_patch(plt.Rectangle((0, y), x[-1], cell_h, fc="#f1f3f5" if is_site else "white", ec="#c8ced3", lw=0.8))
# #         # first column: numbered badge + status
# #         if not is_site:
# #             ax.text(x[0] + 0.32, y + cell_h / 2, str(rw["num"]), ha="center", va="center", fontsize=10.5, color="white", weight="bold",
# #                     bbox=dict(boxstyle="circle,pad=0.28", fc=rw["color"], ec="#111", lw=0.9))
# #             ax.text(x[0] + 0.62, y + cell_h / 2, rw["status"], ha="left", va="center", fontsize=8.3, color=INK)
# #         # where / how big
# #         ax.text(x[1] + 0.08, y + cell_h * 0.34, rw["where"], ha="left", va="center", fontsize=9.5, color=INK, weight="bold" if is_site else "normal")
# #         ax.text(x[1] + 0.08, y + cell_h * 0.72, f"{rw['area']:,.0f} m²", ha="left", va="center", fontsize=9, color=GREY_TXT)
# #         # UTCI
# #         u = v["UTCI"]
# #         ax.text(0.5 * (x[2] + x[3]), y + cell_h * 0.34, f"{_num(u['b'], 1)} → {_num(u['c'], 1)}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
# #         ax.text(0.5 * (x[2] + x[3]), y + cell_h * 0.73, f"{_delta(u['d'], 1)} °C", ha="center", va="center", fontsize=9, color=BLUE_TXT if u["d"] < -0.05 else (RED_TXT if u["d"] > 0.05 else GREY_TXT))
# #         for k, d in enumerate(ORDER):
# #             c = 3 + k
# #             vv = v[d]
# #             nd = NDEC[d] if d != "RH" else 1
# #             eff = vv["eff"]
# #             a_ = min(abs(eff) / 1.5, 1.0) * 0.5
# #             if a_ > 0.02:
# #                 ax.add_patch(plt.Rectangle((x[c], y), colw[c], cell_h, fc=(0.84, 0.19, 0.15, a_) if eff > 0 else (0.17, 0.48, 0.73, a_), ec="none"))
# #             cx = 0.5 * (x[c] + x[c + 1])
# #             ax.text(cx, y + cell_h * 0.30, f"{_num(vv['b'], nd)} → {_num(vv['c'], nd)}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
# #             t2, c2 = _fmt_eff(eff, d)
# #             ax.text(cx, y + cell_h * 0.72, f"({_delta(vv['d'], nd)})   {t2}", ha="center", va="center", fontsize=8.8, color=c2)
# #         y += cell_h
# #     return y


# # GLOSSARY = ("UTCI = the 'feels-like' temperature a person experiences (it combines the four columns on its right).   Ta = air temperature.   "
# #             "Tmrt = radiant temperature: the heat a person receives from the sun and from hot surfaces (shade lowers it).   RH = relative humidity.   "
# #             "Each cell: base → layout, (change), and what that change alone did to UTCI (blue = lowers UTCI, red = raises it).")


# # def _band_legend(ax, y, present, w):
# #     """colour chips of the UTCI categories that occur on the maps"""
# #     ax.text(0.0, y, "Colour on the maps = UTCI category", fontsize=10.5, weight="bold", color=INK, va="top")
# #     y += 0.34
# #     for k in range(len(BAND_NAMES) - 1, -1, -1):
# #         if k not in present:
# #             continue
# #         ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc=BAND_COL[k], ec="#777", lw=0.5))
# #         ax.text(0.45, y + 0.1, f"{BAND_NAMES[k]}", fontsize=8.8, va="center", color=INK)
# #         ax.text(w, y + 0.1, BAND_RANGE[k] + " °C", fontsize=8.3, va="center", ha="right", color=GREY_TXT)
# #         y += 0.27
# #     return y


# # def _symbol_legend(ax, y, items):
# #     """items: [(kind, colour, text)] kind in patch / circle / line"""
# #     for kind, col, txt in items:
# #         if kind == "patch":
# #             ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc=col, ec="#555", lw=0.6))
# #         elif kind == "outline":
# #             ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc="none", ec=col, lw=1.4))
# #         elif kind == "dash":
# #             ax.add_patch(plt.Rectangle((0.0, y), 0.34, 0.2, fc="none", ec=col, lw=1.4, ls="--"))
# #         elif kind == "circle":
# #             ax.plot([0.17], [y + 0.1], "o", ms=9, mfc=col, mec="#111", mew=0.9)
# #         ax.text(0.45, y + 0.1, txt, fontsize=8.8, va="center", color=INK)
# #         y += 0.28
# #     return y


# # def _site_centre(grid):
# #     m = grid.site_mask
# #     return (float(grid.X[m].mean()), float(grid.Y[m].mean()))


# # # ---------------------------------------------------------------------------------------- the words (2-3 sentences each)
# # def _drv_plain(d):
# #     return {"Tmrt": "radiant heat from sun and hot surfaces (shade)", "wind": "wind", "Ta": "air temperature", "RH": "humidity"}[d]


# # def _area_weighted(zones, fn):
# #     tot = sum(z["area"] for z in zones)
# #     return sum(z["area"] * fn(z) for z in zones) / tot if tot > 0 else 0.0


# # def _vtxt_short(d, b, c):
# #     nd = NDEC[d] if d != "RH" else 1
# #     lab = {"Ta": "Ta", "Tmrt": "Tmrt", "RH": "RH", "wind": "wind"}[d]
# #     return f"{lab} {_num(b, nd)} → {_num(c, nd)} {UNIT[d]}"


# # def short_spot_inference(R, base, cand, shown_index):
# #     """three sentences about the hot / cold spots (list of str)"""
# #     heat = R.season == "heat"
# #     word = "hot" if heat else "cold"
# #     sb, sc = R.spot_area_b, R.spot_area_c
# #     n_all = len(R.regions)
# #     n_res = sum(1 for g in R.regions if g["status"] == "resolved")
# #     pct = 100 * (sb - sc) / sb if sb > 0 else 0.0
# #     out = []
# #     s1 = (f"{word.capitalize()} ground (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C) shrinks from {sb:,.0f} to {sc:,.0f} m² ({'−' if sc <= sb else '+'}{abs(pct):.0f} %): "
# #           f"{n_res} of {n_all} base {word} spots are gone")
# #     if R.regions:
# #         zs = R.regions
# #         ub, uc = _area_weighted(zs, lambda z: z["u_base"]), _area_weighted(zs, lambda z: z["u_cand"])
# #         s1 += f", and UTCI inside them goes {ub:.1f} → {uc:.1f} °C ({_delta(uc - ub, 1)})"
# #     out.append(s1 + ".")
# #     if R.regions:
# #         contrib = {d: _area_weighted(zs, lambda z, d=d: z["contrib"][d]) for d in DRIVERS}
# #         vb = {d: _area_weighted(zs, lambda z, d=d: z["vals"][d]["b"]) for d in DRIVERS}
# #         vc = {d: _area_weighted(zs, lambda z, d=d: z["vals"][d]["c"]) for d in DRIVERS}
# #         order = [d for d in sorted(DRIVERS, key=lambda d: -abs(contrib[d])) if abs(contrib[d]) >= 0.1][:2]
# #         if order:
# #             bits = [f"{_drv_plain(d)} ({_vtxt_short(d, vb[d], vc[d])}, worth {_delta(contrib[d], 1)} °C of UTCI)" for d in order]
# #             s2 = "Main cause: " + " and ".join(bits)
# #             dd = R.diff
# #             if dd["trees_cand"] != dd["trees_base"]:
# #                 s2 += f"; the layout has {dd['trees_cand']} trees instead of {dd['trees_base']} (canopy {dd['canopy_base']:.0f} → {dd['canopy_cand']:.0f} m²)"
# #             out.append(s2 + ".")
# #     left = sorted([g for g in R.regions if g["status"] in ("unchanged", "worse", "reduced", "partly resolved")], key=lambda g: -g["area"])
# #     s3 = ""
# #     if left:
# #         g = left[0]
# #         cause, _act = mitigation_for_region(g, R, R.season)
# #         s3 = f"Still {word}: spot {shown_index.get(id(g), '?')} ({g['area']:,.0f} m², UTCI {g['u_cand']:.1f} °C), cause: {cause}"
# #     if R.new_regions:
# #         g = max(R.new_regions, key=lambda x: x["area"])
# #         nb = g.get("near_building")
# #         s3 += ("; " if s3 else "") + f"new {word} ground of {sum(x['area'] for x in R.new_regions):,.0f} m²" + (f" appears next to {_bn(nb)}" if nb else " appears")
# #     if not s3:
# #         s3 = f"No {word} spot remains and no new one appears."
# #     out.append(s3 + ".")
# #     return out


# # def short_vent_inference(R, base, cand, shown_index):
# #     heat = R.season == "heat"
# #     v = R.vent
# #     wb = R.wakes_b
# #     newc = [w for w in R.wakes_c if w.get("is_new")]
# #     thr = R.wake_frac * R.u_ref
# #     n_gone = sum(1 for w in wb if w["fate"] == "gone")
# #     n_small = sum(1 for w in wb if w["fate"] == "smaller")
# #     out = [f"Calm ground (wind < {thr:.2f} m/s) goes from {v['calm_area_b']:,.0f} to {v['calm_area_c']:,.0f} m² ({100 * v['calm_share_b']:.0f} → {100 * v['calm_share_c']:.0f} % of the open ground) "
# #            f"and the mean wind from {R.stats_b['wind_ms']:.2f} to {R.stats_c['wind_ms']:.2f} m/s: of {len(wb)} base calm zones {n_gone} are gone and {n_small} smaller"
# #            + (f", {len(newc)} new one(s) appear" if newc else "") + "."]
# #     cw = R.sgn * R.contrib_mean["wind"]
# #     cm = R.calm_b & R.B["valid"]
# #     if heat:
# #         s2 = ""
# #         if cm.any():
# #             pen = float(np.nanmean(R.B["UTCI_C"][cm])) - R.stats_b["UTCI_C"]
# #             s2 = f"In the hot hour still air costs heat: the base calm zones were {_delta(pen, 1)} °C in UTCI against the site mean. "
# #         if cw > 0.15:
# #             s2 += f"The wind change alone lowers the site-mean UTCI by {cw:.1f} °C, so ventilation is part of the benefit."
# #         elif cw < -0.15:
# #             s2 += f"The wind change alone raises the site-mean UTCI by {abs(cw):.1f} °C; the benefit comes from shade and air temperature instead."
# #         else:
# #             s2 += "The wind change alone hardly moves the site-mean UTCI, so the benefit does not come from ventilation."
# #     else:
# #         s2 = ("In the cold hour calm air is shelter from the wind chill. " + (f"The wind change alone {'raises' if cw > 0 else 'lowers'} the site-mean UTCI by {abs(cw):.1f} °C "
# #               f"({'less' if cw > 0 else 'more'} wind chill)."))
# #     out.append(s2)
# #     shown, _ = _rows_of_zones(wb + newc, cap=1)
# #     s3 = ""
# #     if shown:
# #         w = shown[0]
# #         vv = w["vals"]
# #         fate = "new in the layout" if w.get("is_new") else {"gone": "gone in the layout", "smaller": "smaller in the layout", "similar": "about the same", "larger": "larger in the layout"}[w["fate"]]
# #         s3 = (f"Largest zone (zone {shown_index.get(id(w), '?')}{', behind ' + _bn(w['building']) if w.get('building') else ''}, {w['area']:,.0f} m², {w['length_m']:.0f} m long): {fate}; "
# #               f"wind {vv['wind']['b']:.2f} → {vv['wind']['c']:.2f} m/s, UTCI {vv['UTCI']['b']:.1f} → {vv['UTCI']['c']:.1f} °C")
# #         if heat and not w.get("is_new") and w["fate"] in ("similar", "larger") and w.get("building"):
# #             s3 += f"; to fix it shift {_bn(w['building'])} off the wind line or leave a gap upwind"
# #     if s3:
# #         out.append(s3 + ".")
# #     return out


# # def _write_inference(path, title, sentences):
# #     path.write_text(title + "\n\n" + "\n".join(sentences) + "\n")


# # def _inference_box(ax, wbox, hbox, sentences, fs=11.0):
# #     ax.add_patch(plt.Rectangle((0, 0), wbox, hbox, fc="#f6f8f9", ec="#d5dbdf", lw=0.8))
# #     ax.text(0.15, 0.12, "What it means", fontsize=12.5, weight="bold", color=INK, va="top")
# #     nch = int((wbox - 0.6) / (fs * 0.0070))
# #     y = 0.5
# #     for s in sentences:
# #         ls = textwrap.wrap(s, width=nch)
# #         ax.text(0.15, y, "\n".join(ls), fontsize=fs, color="#222", va="top", linespacing=1.3)
# #         y += len(ls) * 0.215 + 0.1


# # def _inference_height(wbox, sentences, fs=11.0):
# #     nch = int((wbox - 0.6) / (fs * 0.0070))
# #     return 0.5 + sum(len(textwrap.wrap(s, width=nch)) * 0.215 + 0.1 for s in sentences) + 0.15


# # def _index_map(zones):
# #     z, _ = _rows_of_zones(zones, cap=10 ** 6)
# #     return {id(x): k for k, x in enumerate(z, 1)}


# # # ---------------------------------------------------------------------------------------- the two figures
# # def fig_simple_spots(R, base, cand, grid, path):
# #     heat = R.season == "heat"
# #     word = "Hot" if heat else "Cold"
# #     zones_all = R.regions + R.new_regions
# #     shown, hidden = _rows_of_zones(zones_all)
# #     centre = _site_centre(grid)
# #     rows = []
# #     for k, z in enumerate(shown, 1):
# #         is_new = z["id"].startswith("n")
# #         st = "new" if is_new else z["status"]
# #         rows.append(dict(num=k, color=STATUS_COL[st], status=STATUS_TXT[st], where=_short_where(z["centroid"], cand.layout, centre), area=z["area"], vals=z["vals"], zone=z, is_new=is_new))
# #     paras = short_spot_inference(R, base, cand, _index_map(zones_all))
# #     W = SIMPLE_W
# #     M = 0.4
# #     ms = 5.55
# #     n_rows = len(rows) + 1
# #     tab_h = 0.62 + 0.56 * n_rows
# #     hid_txt = f"{len(hidden)} smaller {word.lower()} spot(s) ({sum(h['area'] for h in hidden):,.0f} m²) are not shown; see tables/regions.csv." if hidden else ""
# #     para_h = _inference_height(W - 2 * M, paras)
# #     H = 0.95 + 0.55 + ms + 0.3 + tab_h + 0.75 + para_h + 0.25
# #     fig = plt.figure(figsize=(W, H), facecolor="white")
# #     fig.text(M / W, 1 - 0.38 / H, f"{word} spots: {cand.name} compared with the base case — {SEASON_TXT[R.season]}", fontsize=17, weight="bold", color=INK, va="center")
# #     fig.text(M / W, 1 - 0.74 / H, f"Pedestrian height; {word.lower()} spot = UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C in the base case.  Numbers on the maps are the spots listed in the table below.",
# #              fontsize=10.5, color=GREY_TXT, va="center")
# #     y0 = 0.95 + 0.55
# #     xs = [M, M + ms + 0.3]
# #     allu = np.concatenate([R.B["UTCI_C"][R.B["valid"]], R.C["UTCI_C"][R.C["valid"]]])
# #     uv0, uv1 = float(np.nanpercentile(allu, 1)), float(np.nanpercentile(allu, 99))
# #     lvu = _nice_levels(uv0, max(uv1, uv0 + 3.0), 12)
# #     cmu = "YlOrRd" if heat else "Blues_r"
# #     axes = []
# #     for k, (Fld, lay, name, stats, spot) in enumerate(((R.B, base.layout, "BASE CASE", R.stats_b, R.spot_b), (R.C, cand.layout, f"{cand.name.upper()}  (best layout)", R.stats_c, R.spot_c))):
# #         ax = fig.add_axes([xs[k] / W, 1 - (y0 + ms) / H, ms / W, ms / H])
# #         Z = np.ma.masked_invalid(np.where(Fld["valid"], Fld["UTCI_C"], np.nan))
# #         cfu = ax.contourf(grid.X, grid.Y, np.clip(Z, lvu[0], lvu[-1]), levels=lvu, cmap=cmu, zorder=1)
# #         sub = f"mean UTCI {stats['UTCI_C']:.1f} °C   ·   {word.lower()} ground {(R.spot_area_b if k == 0 else R.spot_area_c):,.0f} m²"
# #         _map_panel(ax, grid, lay, name, sub)
# #         _outline(ax, grid, R.spot_b, colors="k", linewidths=1.6, linestyles="-" if k == 0 else "--")
# #         if k == 1 and R.new_regions:
# #             for z in R.new_regions:
# #                 _outline(ax, grid, z["mask"], colors="#8e44ad", linewidths=1.6, linestyles="--")
# #         axes.append(ax)
# #     # key
# #     kx = M + 2 * ms + 0.3 + 0.45
# #     kw = W - M - kx
# #     axk = _inch_axes(fig, W, H, kx, y0 - 0.05, kw, ms + 0.1)
# #     axk.text(0.0, 0.05, "Colour on the maps = UTCI, °C", fontsize=10.5, weight="bold", color=INK, va="top")
# #     cax = fig.add_axes([kx / W, 1 - (y0 - 0.05 + 0.78) / H, kw / W, 0.2 / H])
# #     cb = fig.colorbar(cfu, cax=cax, orientation="horizontal")
# #     cb.ax.tick_params(labelsize=8.5)
# #     present = [k for k in range(len(BAND_NAMES)) if BAND_EDGES[k + 1] > lvu[0] and BAND_EDGES[k] < lvu[-1]]
# #     yk = 1.2
# #     axk.text(0.0, yk, "UTCI categories (stress level)", fontsize=10.0, weight="bold", color=INK, va="top")
# #     yk += 0.3
# #     for k in reversed(present):
# #         axk.text(0.0, yk + 0.095, BAND_RANGE[k] + " °C", fontsize=8.5, va="center", color=GREY_TXT)
# #         axk.text(1.15, yk + 0.095, BAND_NAMES[k], fontsize=8.5, va="center", color=INK)
# #         yk += 0.25
# #     yk += 0.12
# #     axk.text(0.0, yk, "On the maps", fontsize=10.0, weight="bold", color=INK, va="top")
# #     yk += 0.3
# #     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="#8a8f94", ec="#555", lw=0.6))
# #     axk.text(0.45, yk + 0.095, "building (name inside)", fontsize=8.5, va="center", color=INK)
# #     yk += 0.25
# #     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="none", ec="#2b6cb0", lw=1.2))
# #     axk.text(0.45, yk + 0.095, "ponds, lawns, tree crowns (thin lines)", fontsize=8.5, va="center", color=INK)
# #     yk += 0.25
# #     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.19, fc="none", ec="k", lw=1.5))
# #     axk.text(0.45, yk + 0.095, "black outline = the spot (dashed in the layout)", fontsize=8.5, va="center", color=INK)
# #     yk += 0.37
# #     axk.text(0.0, yk, f"Circle colour = what became of the spot", fontsize=10.0, weight="bold", color=INK, va="top")
# #     yk += 0.3
# #     for col, txt in [(STATUS_COL["resolved"], "gone (≥ 70 % no longer " + ("hot)" if heat else "cold)")), (STATUS_COL["partly resolved"], "partly gone"),
# #                      (STATUS_COL["reduced"], "reduced, still " + word.lower()), (STATUS_COL["unchanged"], "unchanged"), (STATUS_COL["worse"], "worse"),
# #                      (STATUS_COL["new"], "new in the layout (purple outline)")]:
# #         axk.plot([0.17], [yk + 0.095], "o", ms=8.5, mfc=col, mec="#111", mew=0.9)
# #         axk.text(0.45, yk + 0.095, txt, fontsize=8.5, va="center", color=INK)
# #         yk += 0.25
# #     # badges on both maps
# #     badge_items = [dict(label=str(r["num"]), xy=r["zone"]["centroid"], color=STATUS_COL["new" if r["is_new"] else r["zone"]["status"]]) for r in rows]
# #     for ax in axes:
# #         _place_badges(fig, ax, badge_items)
# #     # table
# #     ty = y0 + ms + 0.3
# #     axt = _inch_axes(fig, W, H, M, ty, W - 2 * M, tab_h + 0.75)
# #     used = _draw_zone_table(axt, W, rows, R.site_vals)
# #     axt.text(0.0, used + 0.1, "\n".join(textwrap.wrap(GLOSSARY, 205)) + ("\n" + hid_txt if hid_txt else ""), fontsize=8.6, color=GREY_TXT, va="top", linespacing=1.3)
# #     if not rows:
# #         axt.text(0.0, 0.8, f"No {word.lower()} spot of at least {80:.0f} m² in the base case.", fontsize=11, color=INK)
# #     # sentences
# #     py = ty + tab_h + 0.75
# #     axp = _inch_axes(fig, W, H, M, py, W - 2 * M, para_h)
# #     _inference_box(axp, W - 2 * M, para_h, paras)
# #     fig.savefig(path, dpi=SIMPLE_DPI, facecolor="white")
# #     plt.close(fig)
# #     _write_inference(Path(path).with_suffix(".txt"), f"{word} spots - {cand.name} vs base - {SEASON_TXT[R.season]}", paras)
# #     return True


# # def fig_simple_vent(R, base, cand, grid, path):
# #     heat = R.season == "heat"
# #     wb = R.wakes_b
# #     newc = [w for w in R.wakes_c if w.get("is_new")]
# #     zones = wb + newc
# #     shown, hidden = _rows_of_zones(zones)
# #     centre = _site_centre(grid)
# #     rows = []
# #     for k, z in enumerate(shown, 1):
# #         is_new = bool(z.get("is_new"))
# #         st = "new" if is_new else z["fate"]
# #         w = z
# #         where = _short_where(z["centroid"], cand.layout if is_new else base.layout, centre)
# #         if z.get("building"):
# #             where = f"behind {_bn(z['building'])}, {compass(z['centroid'][0] - centre[0], centre[1] * 0 + z['centroid'][1] - centre[1])} part"
# #         rows.append(dict(num=k, color=STATUS_COL[st], status=STATUS_TXT[st], where=where, area=z["area"], vals=z["vals"], zone=z, is_new=is_new))
# #     paras = short_vent_inference(R, base, cand, _index_map(zones))
# #     W = SIMPLE_W
# #     M = 0.4
# #     ms = 5.55
# #     n_rows = len(rows) + 1
# #     tab_h = 0.62 + 0.56 * n_rows
# #     hid_txt = f"{len(hidden)} smaller calm zone(s) ({sum(h['area'] for h in hidden):,.0f} m²) are not shown; see tables/wakes.csv." if hidden else ""
# #     para_h = _inference_height(W - 2 * M, paras)
# #     H = 0.95 + 0.55 + ms + 0.3 + tab_h + 0.75 + para_h + 0.25
# #     fig = plt.figure(figsize=(W, H), facecolor="white")
# #     thr = R.wake_frac * R.u_ref
# #     fig.text(M / W, 1 - 0.38 / H, f"Ventilation: {cand.name} compared with the base case — {SEASON_TXT[R.season]}", fontsize=17, weight="bold", color=INK, va="center")
# #     fig.text(M / W, 1 - 0.74 / H, f"Wind speed at pedestrian height.  Calm zone = wind below {thr:.2f} m/s ({R.wake_frac:g} × the free wind of the base case, {R.u_ref:.2f} m/s); "
# #                                   f"{'still air traps heat, so calm zones are a problem in this hour.' if heat else 'calm air is shelter from the wind chill in this hour.'}",
# #              fontsize=10.5, color=GREY_TXT, va="center")
# #     y0 = 0.95 + 0.55
# #     xs = [M, M + ms + 0.3]
# #     vmax = float(np.ceil(10 * max(1.0, np.nanpercentile(np.concatenate([R.B["wind_ms"][R.B["valid"]], R.C["wind_ms"][R.C["valid"]]]), 99))) / 10)
# #     lv = _nice_levels(0.0, vmax, 10)
# #     axes, cf = [], None
# #     for k, (Fld, lay, name, stats, calm, wk) in enumerate(((R.B, base.layout, "BASE CASE", R.stats_b, R.calm_b, R.wakes_b), (R.C, cand.layout, f"{cand.name.upper()}  (best layout)", R.stats_c, R.calm_c, R.wakes_c))):
# #         ax = fig.add_axes([xs[k] / W, 1 - (y0 + ms) / H, ms / W, ms / H])
# #         Z = np.ma.masked_invalid(np.where(Fld["valid"], Fld["wind_ms"], np.nan))
# #         cf = ax.contourf(grid.X, grid.Y, np.clip(Z, lv[0], lv[-1]), levels=lv, cmap="Blues", zorder=1)
# #         sub = f"mean wind {stats['wind_ms']:.2f} m/s   ·   calm ground {(R.vent['calm_area_b'] if k == 0 else R.vent['calm_area_c']):,.0f} m²"
# #         _map_panel(ax, grid, lay, name, sub)
# #         ax.contourf(grid.X, grid.Y, np.where(calm, 1.0, np.nan), levels=[0.5, 1.5], colors=["#c0392b"], alpha=0.35, zorder=2)
# #         _outline(ax, grid, calm, colors="#9c1c10", linewidths=1.4)
# #         _wind_arrow_simple(ax, grid, R.flow)
# #         axes.append(ax)
# #     kx = M + 2 * ms + 0.3 + 0.45
# #     kw = W - M - kx
# #     axk = _inch_axes(fig, W, H, kx, y0 - 0.05, kw, ms + 0.1)
# #     axk.text(0.0, 0.05, "Colour on the maps = wind speed", fontsize=10.5, weight="bold", color=INK, va="top")
# #     cax = fig.add_axes([(kx) / W, 1 - (y0 - 0.05 + 0.75) / H, kw / W, 0.2 / H])
# #     cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
# #     cb.set_label("m/s", fontsize=9)
# #     cb.ax.tick_params(labelsize=8.5)
# #     yk = 1.35
# #     axk.text(0.0, yk, "Symbols", fontsize=10.5, weight="bold", color=INK, va="top")
# #     yk += 0.34
# #     yk = _symbol_legend(axk, yk, [("patch", "#8a8f94", "building (name inside)"), ("outline", "#2b6cb0", "pond outline"), ("outline", "#8fb98a", "lawn outline"), ("outline", "#2f7d32", "tree crown")])
# #     yk += 0.12
# #     axk.text(0.0, yk, "Calm zones", fontsize=10.5, weight="bold", color=INK, va="top")
# #     yk += 0.34
# #     axk.add_patch(plt.Rectangle((0.0, yk), 0.34, 0.2, fc="#c0392b", alpha=0.35, ec="#9c1c10", lw=1.4))
# #     axk.text(0.45, yk + 0.1, "calm zone: wind < %.2f m/s" % thr, fontsize=8.6, va="center", color=INK)
# #     yk += 0.28
# #     for kind, col, txt in [("circle", STATUS_COL["gone"], "base calm zone: gone in the layout"), ("circle", STATUS_COL["smaller"], "smaller"), ("circle", STATUS_COL["similar"], "about the same"),
# #                            ("circle", STATUS_COL["larger"], "larger"), ("circle", STATUS_COL["new"], "new in the layout")]:
# #         axk.plot([0.17], [yk + 0.1], "o", ms=9, mfc=col, mec="#111", mew=0.9)
# #         axk.text(0.45, yk + 0.1, txt, fontsize=8.6, va="center", color=INK)
# #         yk += 0.28
# #     badge_items = [dict(label=str(r["num"]), xy=r["zone"]["centroid"], color=r["color"]) for r in rows]
# #     for ax in axes:
# #         _place_badges(fig, ax, badge_items)
# #     ty = y0 + ms + 0.3
# #     axt = _inch_axes(fig, W, H, M, ty, W - 2 * M, tab_h + 0.75)
# #     used = _draw_zone_table(axt, W, rows, R.site_vals, header_first="Calm zone")
# #     axt.text(0.0, used + 0.1, "\n".join(textwrap.wrap(GLOSSARY, 205)) + ("\n" + hid_txt if hid_txt else ""), fontsize=8.6, color=GREY_TXT, va="top", linespacing=1.3)
# #     if not rows:
# #         axt.text(0.0, 0.8, "No calm zone of at least 80 m² in either layout.", fontsize=11, color=INK)
# #     py = ty + tab_h + 0.75
# #     axp = _inch_axes(fig, W, H, M, py, W - 2 * M, para_h)
# #     _inference_box(axp, W - 2 * M, para_h, paras)
# #     fig.savefig(path, dpi=SIMPLE_DPI, facecolor="white")
# #     plt.close(fig)
# #     _write_inference(Path(path).with_suffix(".txt"), f"ventilation - {cand.name} vs base - {SEASON_TXT[R.season]}", paras)
# #     return True


# # def fig_categories(R, path):
# #     names = [c[1] for c in CATEGORIES][::-1]
# #     names = ["extreme cold stress"] + [c[1] for c in CATEGORIES][::-1][:0] + [c[1] for c in CATEGORIES][::-1]
# #     cats = ["extreme cold stress", "very strong cold stress", "strong cold stress", "moderate cold stress", "slight cold stress", "no thermal stress",
# #             "moderate heat stress", "strong heat stress", "very strong heat stress", "extreme heat stress"]
# #     cats = [c for c in cats if R.cat_b.get(c, 0) > 0 or R.cat_c.get(c, 0) > 0]
# #     fig, ax = plt.subplots(figsize=(8.5, 3.8))
# #     x = np.arange(len(cats))
# #     ax.bar(x - 0.2, [100 * R.cat_b[c] / R.stats_b["area_m2"] for c in cats], 0.4, color="#8a9aa6", label="base")
# #     ax.bar(x + 0.2, [100 * R.cat_c[c] / R.stats_c["area_m2"] for c in cats], 0.4, color="#2b7bba", label=R.name)
# #     ax.set_xticks(x)
# #     ax.set_xticklabels([c.replace(" stress", "\nstress").replace("no thermal\nstress", "no thermal\nstress") for c in cats], fontsize=7)
# #     ax.set_ylabel("% of the walkable area")
# #     ax.legend(fontsize=8)
# #     ax.set_title(f"share of the area per UTCI category — {SEASON_TXT[R.season]}", fontsize=10)
# #     fig.tight_layout()
# #     fig.savefig(path, dpi=80)
# #     plt.close(fig)


# # def fig_overview(results, path):
# #     """all selected layouts: benefit by driver, per season"""
# #     names = sorted({r.name for r in results})
# #     seasons = [s for s in SEASONS if any(r.season == s for r in results)]
# #     fig, axs = plt.subplots(1, len(seasons), figsize=(6.5 * len(seasons) + 1, 4.8), squeeze=False)
# #     for ax, s in zip(axs[0], seasons):
# #         rs = {r.name: r for r in results if r.season == s}
# #         x = np.arange(len(rs))
# #         pos, neg = np.zeros(len(rs)), np.zeros(len(rs))
# #         for d in DRIVERS:
# #             v = np.array([rs[n].sgn * rs[n].contrib_mean[d] for n in rs])
# #             ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), color=COL[d], label=d, width=0.6)
# #             pos += np.where(v >= 0, v, 0)
# #             neg += np.where(v < 0, v, 0)
# #         ax.plot(x, [rs[n].sgn * rs[n].dU_mean for n in rs], "kD", ms=6, label="total")
# #         ax.set_xticks(x)
# #         ax.set_xticklabels(list(rs), rotation=20, fontsize=8)
# #         ax.axhline(0, color=INK, lw=0.8)
# #         ax.set_ylabel("°C of UTCI benefit (site mean)")
# #         ax.set_title(f"{SEASON_TXT[s]}: what produced the benefit", fontsize=10)
# #         ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
# #     fig.tight_layout()
# #     fig.savefig(path, dpi=80)
# #     plt.close(fig)


# # # ============================================================================= text: findings and mitigation
# # def fmt_area(a):
# #     return f"{a:,.0f} m²"


# # def where_text(centroid, layout, site_centre, flow=None):
# #     d = compass(centroid[0] - site_centre[0], centroid[1] - site_centre[1])
# #     nb = nearest_building(layout, centroid, flow)
# #     s = f"{d} part of the site"
# #     if nb is not None:
# #         s += f", {nb[1]:.0f} m from {nb[0]}"
# #     return s


# # def driver_sentence(contrib, sgn, total=None, drivers=DRIVERS):
# #     """'Tmrt -3.1 °C (82 %), wind ...' in order of importance, signed as UTCI change; the share is shown only for terms that go the same way as the total"""
# #     tot = sum(contrib[d] for d in drivers) if total is None else total
# #     items = sorted(drivers, key=lambda d: -abs(contrib[d]))
# #     parts = []
# #     for d in items:
# #         if abs(tot) > 1e-6 and contrib[d] * tot > 0:
# #             parts.append(f"{d} {contrib[d]:+.2f} °C ({100 * contrib[d] / tot:.0f} %)")
# #         elif abs(contrib[d]) >= 0.005:
# #             parts.append(f"{d} {contrib[d]:+.2f} °C (opposes)")
# #         else:
# #             parts.append(f"{d} {contrib[d]:+.2f} °C")
# #     return ", ".join(parts)


# # def explain_region(r, R, season):
# #     """one paragraph for a hot / cold region"""
# #     heat = season == "heat"
# #     ctx = r["context"]
# #     s = (f"{r['status'].upper()}: UTCI {r['u_base']:.1f} → {r['u_cand']:.1f} °C ({r['d_u']:+.1f} °C); {100 * r['resolved_share']:.0f} % of its area is no longer "
# #          f"{'above' if heat else 'below'} {R.thr:.1f} °C" + (" (cells just beyond the threshold flip with a small change)" if r["status"] in ("unchanged", "reduced") and r["resolved_share"] >= 0.5 else "") + ". Zone means, base → layout, and what each change did to UTCI: " + inputs_sentence(r["vals"]) + ".")
# #     cb, _vb = cause_of(r.get("cause_b"), heat)
# #     if cb:
# #         s += f" In the base it was {'hotter' if heat else 'colder'} than the rest of the site mainly because of {cb} ({(_vb if heat else -_vb):+.1f} °C)."
# #     det = []
# #     if abs(r["sunlit_c"] - r["sunlit_b"]) >= 0.1:
# #         det.append(f"sunlit share {r['sunlit_b']:.2f} → {r['sunlit_c']:.2f}")
# #     if abs(r["sky_open_c"] - r["sky_open_b"]) >= 0.05:
# #         det.append(f"sky view {r['sky_open_b']:.2f} → {r['sky_open_c']:.2f}")
# #     det.append(f"wind {r['wind_ratio_b']:.2f} → {r['wind_ratio_c']:.2f} of the reference wind")
# #     s += " (" + "; ".join(det) + ")."
# #     ch = []
# #     if ctx["trees_added_near"]:
# #         ch.append(f"{ctx['trees_added_near']} new trees within 25 m (+{ctx['canopy_added_near_m2']:.0f} m² canopy)")
# #     if ctx["trees_removed_near"]:
# #         ch.append(f"{ctx['trees_removed_near']} trees removed within 25 m")
# #     if ctx.get("trees_added_upwind"):
# #         ch.append(f"{ctx['trees_added_upwind']} new trees in the 70 m upwind corridor")
# #     if ctx.get("trees_removed_upwind"):
# #         ch.append(f"{ctx['trees_removed_upwind']} trees removed from the upwind corridor")
# #     dp = ctx["pond_area_near_cand"] - ctx["pond_area_near_base"]
# #     if abs(dp) > 30:
# #         ch.append(f"pond area within 25 m {ctx['pond_area_near_base']:.0f} → {ctx['pond_area_near_cand']:.0f} m²")
# #     dpu = ctx.get("pond_area_upwind_cand", 0) - ctx.get("pond_area_upwind_base", 0)
# #     if abs(dpu) > 30:
# #         ch.append(f"pond area upwind {ctx['pond_area_upwind_base']:.0f} → {ctx['pond_area_upwind_cand']:.0f} m²")
# #     dl = ctx["lawn_area_near_cand"] - ctx["lawn_area_near_base"]
# #     if abs(dl) > 30:
# #         ch.append(f"lawn within 25 m {ctx['lawn_area_near_base']:.0f} → {ctx['lawn_area_near_cand']:.0f} m²")
# #     for bid, sh in ctx["buildings_moved_near"]:
# #         ch.append(f"{bid} moved {sh:.1f} m")
# #     s += (" Layout change near it: " + "; ".join(ch) + ".") if ch else " No layout change within 25 m (the change comes from elsewhere in the flow or from the whole site)."
# #     return s


# # def cause_of(cc, heat, min_c=0.3):
# #     """dominant driver of the excess of a region over the rest of the site: (driver or None, its contribution in the stress direction)"""
# #     if not cc:
# #         return None, 0.0
# #     sc = {d: (cc["contrib"][d] if heat else -cc["contrib"][d]) for d in DRIVERS}
# #     d = max(DRIVERS, key=lambda k: sc[k])
# #     return (d if sc[d] >= min_c else None), sc[d]


# # def mitigation_for_region(r, R, season):
# #     """(cause sentence, action sentence) for a region that is still stressed in the layout"""
# #     heat = season == "heat"
# #     d, val = cause_of(r.get("cause_c"), heat)
# #     wr = r.get("wind_c", float("nan")) / R.u_ref if "wind_c" in r else float("nan")
# #     sun, sky, tm, ta = r.get("sunlit_c", float("nan")), r.get("sky_open_c", float("nan")), r.get("Tmrt_c", float("nan")), r.get("Ta_c", float("nan"))
# #     nb = r.get("near_building")
# #     if d is None:
# #         return ("no single driver explains it (every driver within 0.3 °C of the rest of the site)",
# #                 "mixed: the whole site is stressed here rather than a pocket; use the site-wide levers (shade and ventilation for heat, shelter and sun for cold).")
# #     if heat:
# #         if d == "Tmrt" and sun > 0.5:
# #             return (f"radiant load, +{val:.1f} °C of UTCI over the rest of the site (sunlit {sun:.2f}, Tmrt {tm:.0f} °C vs air {ta:.0f} °C)",
# #                     "shade it: wide, tall crowns (radius 4-6 m, height 10-15 m) on the sun side of the spot, or put a building / canopy so that its shadow falls on it; lawn or water instead of dark paving.")
# #         if d == "Tmrt":
# #             return (f"radiation although shaded, +{val:.1f} °C (sunlit {sun:.2f}, sky view {sky:.2f}, Tmrt {tm:.0f} °C)",
# #                     "hot surfaces / trapped long-wave: open the sky a little (thin or lift the canopy), cool the surfaces (lawn, water, light paving).")
# #         if d == "wind":
# #             return (f"calm air, +{val:.1f} °C (wind {wr:.2f} of the reference)",
# #                     "open a ventilation path: no tree row or wall directly upwind, line a gap between buildings up with the wind"
# #                     + (f", shift the building upwind of it ({_bn(nb)}) sideways" if nb else "") + ".")
# #         if d == "Ta":
# #             return (f"warm air, +{val:.1f} °C (Ta {ta:.1f} °C)",
# #                     "water or lawn UPWIND of the spot with a clear path to it (cooled air travels with the wind) rather than at the spot itself.")
# #         return (f"humid air, +{val:.1f} °C", "keep ponds downwind of where people stay, or make them smaller / further away.")
# #     if d == "wind":
# #         return (f"wind chill, {val:.1f} °C colder than the rest of the site (wind {wr:.2f} of the reference)",
# #                 "shelter: 2-3 staggered tree rows (crown radius 3-5 m) across the wind 10-25 m upwind of it; close channels between buildings that point along the wind.")
# #     if d == "Tmrt":
# #         return (f"radiation, {val:.1f} °C colder (sunlit {sun:.2f}, Tmrt {tm:.1f} °C vs air {ta:.1f} °C)",
# #                 "keep winter sun access: no dense canopy on the sun side, a warm / low-albedo surface where the sun reaches it.")
# #     if d == "Ta":
# #         return (f"cold air, {val:.1f} °C below the rest (Ta {ta:.1f} °C)", "exposed to the approach air: shelter it from the approach direction so the air mixes less.")
# #     return (f"humidity, {val:.1f} °C", "small effect in the cold hour; check the numbers before acting.")


# # def build_findings(R, base, cand, grid):
# #     """list of strings (markdown) that explain one candidate / season"""
# #     sgn, heat = R.sgn, R.season == "heat"
# #     L = []
# #     obj = ""
# #     if R.obj_b is not None and R.obj_c is not None:
# #         obj = f" (objective, area-mean of the sample points: {R.obj_b:.2f} → {R.obj_c:.2f} °C, {R.obj_c - R.obj_b:+.2f})"
# #     L.append(f"**{SEASON_TXT[R.season].capitalize()}.** Site-mean UTCI {R.stats_b['UTCI_C']:.2f} → {R.stats_c['UTCI_C']:.2f} °C on the common grid{obj}; "
# #              f"category {category(R.stats_b['UTCI_C'])} → {category(R.stats_c['UTCI_C'])}. Where both layouts have open ground the mean change is "
# #              f"{R.dU_mean:+.2f} °C.")
# #     c = R.contrib_mean
# #     ranking = sorted(DRIVERS, key=lambda d: -sgn * c[d])
# #     top = ranking[0]
# #     pos = [d for d in DRIVERS if sgn * c[d] > 0.05]
# #     neg = [d for d in DRIVERS if sgn * c[d] < -0.05]
# #     L.append("*Site mean, inputs and their effect on UTCI* (base → layout; the effect is the part of the UTCI change that this input alone caused): " + inputs_sentence(R.site_vals) + ".")
# #     L.append(f"*Why:* the change of UTCI splits into  {driver_sentence(c, sgn, R.dU_mean)}.")
# #     if abs(R.dU_mean) < 0.15:
# #         L.append("The site mean hardly changes; the benefit, if any, is local (see the regions below).")
# #     else:
# #         word = "lowered" if R.dU_mean < 0 else "raised"
# #         s = f"The mean UTCI was {word} mainly by **{DRIVER_LONG[top]}**"
# #         if sgn * c[top] > 0:
# #             s += f" (benefit {sgn * c[top]:+.2f} °C of the net {sgn * R.dU_mean:+.2f} °C)."
# #         else:
# #             s += "."
# #         if neg:
# #             s += " Working against the benefit: " + ", ".join(f"{d} ({sgn * c[d]:+.2f} °C)" for d in neg) + "."
# #         L.append(s)
# #     if R.tmrt is not None and abs(R.dTmrt_mean) > 0.1:
# #         p = R.tmrt_parts_mean
# #         items = sorted(p.items(), key=lambda kv: -abs(kv[1]))
# #         L.append(f"*Inside Tmrt* (Δ Tmrt {R.dTmrt_mean:+.2f} °C): " + ", ".join(f"{k} {v:+.2f}" for k, v in items) +
# #                  f" °C. Regression of Tmrt on sunlit share, sky view and Ta explains R² = {R.tmrt['r2']:.2f}. Sunlit share {R.stats_b['sunlit']:.2f} → {R.stats_c['sunlit']:.2f}, "
# #                  f"sky view {R.stats_b['sky_open']:.2f} → {R.stats_c['sky_open']:.2f}.")
# #     v = R.vent
# #     L.append(f"*Ventilation* at 1.5 m: mean wind {R.stats_b['wind_ms']:.2f} → {R.stats_c['wind_ms']:.2f} m/s ({v['wind_ratio_b']:.2f} → {v['wind_ratio_c']:.2f} of the reference wind "
# #              f"{R.u_ref:.1f} m/s); calm area (< {R.wake_frac:g} of reference) {fmt_area(v['calm_area_b'])} → {fmt_area(v['calm_area_c'])} "
# #              f"({100 * v['calm_share_b']:.0f} → {100 * v['calm_share_c']:.0f} % of the open ground); longest wake {v['longest_wake_b']:.0f} → {v['longest_wake_c']:.0f} m.")
# #     if heat:
# #         wind_help = sgn * c["wind"]
# #         if wind_help > 0.15:
# #             L.append(f"→ Ventilation **improved** the heat case: it accounts for {wind_help:+.2f} °C of the benefit.")
# #         elif wind_help < -0.15:
# #             L.append(f"→ Ventilation got **worse** in the heat case ({wind_help:+.2f} °C); the benefit came from the other drivers.")
# #         else:
# #             L.append("→ Ventilation changed little in the heat case; the benefit does not come from wind.")
# #     else:
# #         wind_help = sgn * c["wind"]
# #         if wind_help > 0.15:
# #             L.append(f"→ Less wind at the pedestrian reduced the cold stress: wind accounts for {wind_help:+.2f} °C of the benefit (wind chill reduced).")
# #         elif wind_help < -0.15:
# #             L.append(f"→ More wind increased the cold stress ({wind_help:+.2f} °C); the other drivers compensated.")
# #         else:
# #             L.append("→ Wind chill hardly changed in the cold case; the benefit is from radiation / air temperature.")
# #     # regions
# #     L.append(f"*{'Hot' if heat else 'Cold'} spots of the base* (UTCI {'≥' if heat else '≤'} {R.thr:.1f} °C): {fmt_area(R.spot_area_b)} in the base, {fmt_area(R.spot_area_c)} in the layout "
# #              f"({R.spot_area_c - R.spot_area_b:+,.0f} m²); {len(R.regions)} region(s), {len(R.new_regions)} new region(s) appear.")
# #     return L


# # def mitigation_text(results, base, grids, a):
# #     """consolidated inference over all analysed layouts / seasons (list of markdown lines)"""
# #     out = []
# #     by = {s: [r for r in results if r.season == s] for s in SEASONS}
# #     for s in SEASONS:
# #         rs = by[s]
# #         if not rs:
# #             continue
# #         heat = s == "heat"
# #         sgn = -1.0 if heat else 1.0
# #         out += [f"### {SEASON_TXT[s].capitalize()}", ""]
# #         m = {d: float(np.mean([sgn * r.contrib_mean[d] for r in rs])) for d in DRIVERS}
# #         tot = float(np.mean([sgn * r.dU_mean for r in rs]))
# #         out += [f"**What worked.** Average over the {len(rs)} layouts: benefit {tot:+.2f} °C of site-mean UTCI = " +
# #                 ", ".join(f"{d} {m[d]:+.2f}" for d in sorted(DRIVERS, key=lambda d: -abs(m[d]))) + " °C.", ""]
# #         lead = max(DRIVERS, key=lambda d: m[d])
# #         if heat:
# #             tips = {"Tmrt": "The heat stress is dissolved mainly by cutting the radiant load: keep shade (tree crowns, building shadow) on the walkable ground and do not lose canopy to the other objective.",
# #                     "wind": "The heat stress is dissolved mainly by ventilation: protect the open corridors along the wind and keep tree rows and walls out of the approach path of the hot spots.",
# #                     "Ta": "The heat stress is dissolved mainly by cooler air: evaporation (ponds, lawns) and canopy cooling upwind of the spots is working; keep a clear wind path from the water to the places that need it.",
# #                     "RH": "Humidity is the leading term: check that ponds do not raise RH where people stay."}
# #         else:
# #             tips = {"Tmrt": "The cold stress is reduced mainly by radiation: sun reaches the walkable ground; keep sun access when adding trees.",
# #                     "wind": "The cold stress is reduced mainly by wind shelter: keep the shelter, and keep it from closing the heat-hour ventilation (see the trade-off below).",
# #                     "Ta": "The cold stress is reduced mainly by warmer air (less mixing with the cold approach air, sheltered pockets).",
# #                     "RH": "Humidity is the leading term; its effect on cold stress is small, check the numbers before drawing conclusions."}
# #         out += [f"**Leading driver: {lead}.** " + tips[lead], ""]
# #         # what is left: area-weighted causes over all leftover regions
# #         causes, tot_area = {}, 0.0
# #         for r in rs:
# #             for reg in r.regions:
# #                 if reg["status"] in ("unchanged", "worse", "reduced", "partly resolved"):
# #                     d, _ = cause_of(reg.get("cause_c"), heat)
# #                     causes[d or "mixed"] = causes.get(d or "mixed", 0.0) + reg["area"]
# #                     tot_area += reg["area"]
# #         if tot_area > 0:
# #             out += ["**What is left.** Cause of the remaining " + ("hot" if heat else "cold") + " spots, weighted by area over all layouts: " +
# #                     ", ".join(f"{k} {100 * v / tot_area:.0f} %" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) + ".", ""]
# #         out += ["**Per layout: what to try next.**", ""]
# #         for r in rs:
# #             left = [g for g in r.regions if g["status"] in ("unchanged", "worse", "reduced", "partly resolved")]
# #             left.sort(key=lambda g: -g["area"])
# #             big = [g for g in left if g["area"] >= 150.0 or g["status"] == "worse"][:4]
# #             n_res = sum(1 for g in r.regions if g["status"] == "resolved")
# #             head = f"- **{r.name}**: {n_res} of {len(r.regions)} base spots resolved, {len(left)} left ({sum(g['area'] for g in left):,.0f} m²)"
# #             if r.new_regions:
# #                 head += f", {len(r.new_regions)} new ({sum(g['area'] for g in r.new_regions):,.0f} m²)"
# #             out.append(head + ".")
# #             seen = {}
# #             for g in big:
# #                 cause, act = mitigation_for_region(g, r, s)
# #                 if act in seen:
# #                     act = f"same action as {seen[act]}."
# #                 else:
# #                     seen[act] = g["id"]
# #                 out.append(f"  - {g['id']} ({g['area']:,.0f} m², {g['status']}, UTCI {g['u_cand']:.1f} °C): cause — {cause}. Try: {act}")
# #             if len(left) > len(big):
# #                 rest = [g for g in left if g not in big]
# #                 out.append(f"  - {len(rest)} smaller region(s) ({sum(g['area'] for g in rest):,.0f} m²): same levers as above.")
# #             if r.new_regions:
# #                 dom = {}
# #                 for g in r.new_regions:
# #                     dom[g.get("dominant", "?")] = dom.get(g.get("dominant", "?"), 0.0) + g["area"]
# #                 nbs = sorted({g.get("near_building") for g in r.new_regions if g.get("near_building")})
# #                 up = sum(g["context"].get("trees_added_upwind", 0) for g in r.new_regions)
# #                 out.append(f"  - new stressed ground, worsened mainly by {', '.join(k for k, v in sorted(dom.items(), key=lambda kv: -kv[1]))}, next to {', '.join(map(str, nbs[:4]))}: "
# #                            f"check what was added upwind of it ({up} trees added in its upwind corridors) and what moved; it is the price of the change that helped elsewhere.")
# #             if heat:
# #                 hot_w = [w for w in r.wakes_c if w["utci_mean"] > r.stats_c["UTCI_C"] + 0.3 and w["area"] >= 150.0]
# #                 hot_w.sort(key=lambda w: -w["area"])
# #                 for w in hot_w[:2]:
# #                     out.append(f"  - wake {w['id']} ({w['area']:,.0f} m², {w['length_m']:.0f} m long, UTCI {w['utci_mean']:.1f} °C vs site {r.stats_c['UTCI_C']:.1f}), trails {w['building']}"
# #                                f"{' (new in this layout)' if w.get('is_new') else ''}: shift {w['building']} off the wind line, narrow it across the wind, or leave a gap upwind.")
# #         out.append("")
# #     # trade-off
# #     if by["heat"] and by["cold"]:
# #         out += ["### Trade-off between the hot and the cold hour", ""]
# #         for h in by["heat"]:
# #             c_ = next((x for x in by["cold"] if x.name == h.name), None)
# #             if c_ is None:
# #                 continue
# #             hw, cw = -h.contrib_mean["wind"], c_.contrib_mean["wind"]
# #             ht, ct = -h.contrib_mean["Tmrt"], c_.contrib_mean["Tmrt"]
# #             msg = (f"- **{h.name}**: heat benefit {-h.dU_mean:+.2f} °C (wind {hw:+.2f}, Tmrt {ht:+.2f}), cold benefit {c_.dU_mean:+.2f} °C (wind {cw:+.2f}, Tmrt {ct:+.2f}).")
# #             if hw < -0.1 and cw > 0.1:
# #                 msg += " The windbreak helps the cold hour but costs ventilation in the heat hour: a belt with gaps towards the heat-wind direction would keep both."
# #             elif hw > 0.1 and cw < -0.1:
# #                 msg += " The ventilation gained for the heat hour is paid for by more wind chill in the cold hour: shelter the cold-wind side separately."
# #             elif ht > 0.1 and ct < -0.1:
# #                 msg += " Shade that cools the heat hour also lowers Tmrt in the cold hour: keep the shade where the heat spots are, and keep the sun side open in winter."
# #             elif ht > 0.1 and cw > 0.1:
# #                 msg += " Shade for the heat hour and shelter for the cold hour pull in the same direction: trees do both here."
# #             else:
# #                 msg += " No direct conflict between the two hours in the drivers."
# #             out.append(msg)
# #         out.append("")
# #     return out


# # RULES_TABLE = [
# #     ("Heat: Tmrt dominates the benefit, sunlit share fell", "Shade (tree crowns, building shadow) did the work", "More / larger crowns at the remaining sunlit spots; protect existing canopy"),
# #     ("Heat: wind dominates, calm area shrank", "Ventilation did the work", "Keep corridors along the heat wind open; no tree rows / walls directly upwind of spots"),
# #     ("Heat: Ta dominates (air cooler)", "Evaporation / canopy cooling reached the spots", "Water and lawn upwind of the spots, open path from water to spot"),
# #     ("Heat: wake region behind a building with UTCI above the site mean", "Wake traps heat", "Shift the building off the wind line, split / narrow it across the wind, leave a gap upwind"),
# #     ("Heat: RH term negative", "Humid air from ponds is carried onto people", "Ponds downwind of seating, or smaller ponds, or more evaporation distance"),
# #     ("Cold: wind dominates, wind ratio fell", "Windbreak / building shelter did the work", "Keep the belt; gaps only where heat ventilation needs them"),
# #     ("Cold: spot with wind ratio > 1.15", "Wind channel / exposure", "Staggered rows 10-25 m upwind across the wind; close channels aligned with the wind"),
# #     ("Cold: spot shaded and Tmrt < Ta", "Missing sun / radiant heat", "Sun access (no dense canopy on the sun side), warm surfaces"),
# #     ("A region got worse", "Something placed upwind or beside it", "Check the layout changes listed for the region and undo that one"),
# # ]


# # # ============================================================================= report
# # def md_table(header, rows):
# #     out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
# #     for r in rows:
# #         out.append("| " + " | ".join(str(c) for c in r) + " |")
# #     return out


# # def md_zone_table(zones, site_vals):
# #     rows = []
# #     for z in zones:
# #         v = z.get("vals")
# #         if v is None:
# #             continue
# #         tag = "new" if (z["id"].startswith("n") or z.get("is_new")) else (z.get("status") or z.get("fate") or "")
# #         rows.append([f"{z['id']}" + ("" if z["id"][0] in "HCn" else (" (layout)" if z.get("is_new") else " (base)")), tag, f"{z['area']:.0f}", f"{v['UTCI']['b']:.1f} → {v['UTCI']['c']:.1f} ({v['UTCI']['d']:+.2f})"] +
# #                     [vtxt(v, d) for d in ORDER])
# #     sv = site_vals
# #     rows.append(["whole site", "", f"{sv['area']:.0f}", f"{sv['UTCI']['b']:.1f} → {sv['UTCI']['c']:.1f} ({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d) for d in ORDER])
# #     return md_table(["zone", "status", "area m²", "UTCI °C"] + [f"{d} {UNIT[d]}" for d in ORDER], rows)


# # def md_to_html(md, img_dir=None, embed=True):
# #     """small converter for the subset used here (headings, paragraphs, bullets, tables, images, **bold**, *italic*, `code`)"""
# #     try:
# #         import markdown as mdl
# #         html = mdl.markdown(md, extensions=["tables"])
# #     except Exception:
# #         html = _mini_md(md)
# #     if embed and img_dir is not None:
# #         tot = sum(f.stat().st_size for f in Path(img_dir).glob("figures/*.png"))
# #         if tot > 30e6:                                    # HD pictures: do not inline them (the html would be huge); link the files instead
# #             embed = False
# #     if embed and img_dir is not None:
# #         def rep(m):
# #             p = Path(img_dir) / m.group(1)
# #             if p.is_file():
# #                 return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
# #             return m.group(0)
# #         html = re.sub(r'src="([^"]+\.png)"', rep, html)
# #     css = ("body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#1c2830;line-height:1.5}"
# #            "h1,h2,h3{color:#14323f}table{border-collapse:collapse;margin:10px 0;font-size:13px}td,th{border:1px solid #cfd6dc;padding:4px 8px;text-align:left}"
# #            "th{background:#eef2f5}img{max-width:100%;border:1px solid #dde3e8;margin:8px 0}code{background:#eef2f5;padding:1px 4px;border-radius:3px}"
# #            "li{margin:3px 0}@media(prefers-color-scheme:dark){body{background:#12181c;color:#dfe6ea}h1,h2,h3{color:#bcd6e2}th{background:#1d272d}td,th{border-color:#34424a}code{background:#1d272d}}")
# #     return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>UHI Pareto analysis</title><style>{css}</style></head><body>{html}</body></html>"


# # def _inline(s):
# #     s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
# #     s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
# #     s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
# #     s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
# #     s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
# #     return s


# # def _mini_md(md):
# #     out, lines, i = [], md.splitlines(), 0
# #     while i < len(lines):
# #         ln = lines[i]
# #         if ln.startswith("#"):
# #             n = len(ln) - len(ln.lstrip("#"))
# #             out.append(f"<h{n}>{_inline(ln[n:].strip())}</h{n}>")
# #         elif ln.startswith("|"):
# #             rows = []
# #             while i < len(lines) and lines[i].startswith("|"):
# #                 rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
# #                 i += 1
# #             i -= 1
# #             t = "<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr>"
# #             for r in rows[2:]:
# #                 t += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
# #             out.append(t + "</table>")
# #         elif ln.startswith("- "):
# #             out.append("<ul>")
# #             while i < len(lines) and lines[i].startswith("- "):
# #                 out.append(f"<li>{_inline(lines[i][2:])}</li>")
# #                 i += 1
# #             i -= 1
# #             out.append("</ul>")
# #         elif ln.strip():
# #             out.append(f"<p>{_inline(ln)}</p>")
# #         i += 1
# #     return "\n".join(out)


# # # ============================================================================= selection
# # def read_objectives(out_dir):
# #     p = Path(out_dir) / "objectives.csv"
# #     if not p.is_file():
# #         return {}
# #     rows = {}
# #     for r in csv.DictReader(p.open()):
# #         try:
# #             rows[r["name"]] = dict(f1=float(r["f1"]), f2=float(r["f2"]), utci_heat=float(r["utci_heat"]), utci_cold=float(r["utci_cold"]), status=r.get("status", "ok"))
# #         except (KeyError, ValueError):
# #             continue
# #     return rows


# # def pareto_front(obj):
# #     names = [n for n, r in obj.items() if r.get("status", "ok") == "ok"]
# #     front = []
# #     for n in names:
# #         a = obj[n]
# #         if not any((obj[m]["f1"] <= a["f1"] and obj[m]["f2"] <= a["f2"]) and (obj[m]["f1"] < a["f1"] or obj[m]["f2"] < a["f2"]) for m in names if m != n):
# #             front.append(n)
# #     return sorted(front, key=lambda n: obj[n]["f1"] + obj[n]["f2"])


# # def select_names(a, obj):
# #     if a.names:
# #         return list(a.names)
# #     t5 = Path(a.output_dir) / "top5.json"
# #     if t5.is_file():
# #         try:
# #             names = [r["name"] for r in json.loads(t5.read_text())]
# #             return names[:a.top]
# #         except Exception:
# #             pass
# #     if obj:
# #         return pareto_front(obj)[:a.top]
# #     raise SystemExit("no top5.json / objectives.csv found: give --names")


# # # ============================================================================= main
# # def main(argv=None):
# #     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
# #     ap.add_argument("--root", type=Path, default=ROOT, help="project folder (contains input/, run/, output/)")
# #     ap.add_argument("--run-dir", type=Path)
# #     ap.add_argument("--input-dir", type=Path, help="folder with the layout jsons (default <root>/input)")
# #     ap.add_argument("--output-dir", type=Path, help="folder with top5.json / objectives.csv (default <root>/output)")
# #     ap.add_argument("--outdir", type=Path, help="where the analysis is written (default <output-dir>/analysis)")
# #     ap.add_argument("--base", default="individual_0", help="name of the base case (default individual_0 = base.json)")
# #     ap.add_argument("--names", nargs="+", help="layouts to analyse (default: top5.json, else the Pareto front)")
# #     ap.add_argument("--top", type=int, default=5, help="how many of top5.json / of the Pareto front")
# #     ap.add_argument("--season", choices=("heat", "cold", "both"), default="both")
# #     ap.add_argument("--cell", type=float, default=None, help="analysis grid cell in m (default about half the sample point spacing, 2-4 m)")
# #     ap.add_argument("--wake-frac", type=float, default=0.5, help="wake = wind below this share of the reference wind (default 0.5)")
# #     ap.add_argument("--hot-pct", type=float, default=80.0, help="heat spots: UTCI >= this percentile of the base (and >= 26 C)")
# #     ap.add_argument("--cold-pct", type=float, default=80.0, help="cold spots: UTCI <= the (100 - this) percentile of the base (and <= 9 C)")
# #     ap.add_argument("--u-ref", type=float, default=None, help="reference wind (m/s at 1.5 m) for the wake test; default = 90th percentile of the base wind field")
# #     ap.add_argument("--wind-factor", type=float, default=1.0, help="the factor compute_utci.py used on the CFD wind (default 1)")
# #     ap.add_argument("--min-region-m2", type=float, default=80.0, help="smaller regions are ignored")
# #     ap.add_argument("--no-html", action="store_true")
# #     ap.add_argument("--no-figures", action="store_true")
# #     ap.add_argument("--dpi", type=int, default=200, help="resolution of the two plain pictures (default 200 = about 3200 px wide)")
# #     ap.add_argument("--detailed", action="store_true", help="also draw the dense technical figures (maps, drivers, zone tables, categories); default: only the two plain pictures per layout and season")
# #     a = ap.parse_args(argv)
# #     t0 = time.time()
# #     root = a.root
# #     a.run_dir = a.run_dir or root / "run"
# #     a.input_dir = a.input_dir or root / "input"
# #     a.output_dir = a.output_dir or root / "output"
# #     outdir = a.outdir or a.output_dir / "analysis"
# #     (outdir / "figures").mkdir(parents=True, exist_ok=True)
# #     (outdir / "tables").mkdir(parents=True, exist_ok=True)
# #     global WIND_FACTOR, SIMPLE_DPI
# #     WIND_FACTOR = a.wind_factor
# #     SIMPLE_DPI = a.dpi
# #     seasons = SEASONS if a.season == "both" else (a.season,)
# #     obj = read_objectives(a.output_dir)
# #     names = [n for n in select_names(a, obj) if n != a.base]
# #     log(f"base {a.base}; layouts {', '.join(names)}; seasons {', '.join(seasons)}")
# #     tn = {n: f"top_{i}_{n}" for i, n in enumerate(names, 1)}              # file names: top_1_<name>_..., top_2_<name>_... (rank = order of the list)
# #     log("files: " + ", ".join(tn.values()))

# #     results, grids, cases = [], {}, {}
# #     for s in seasons:
# #         try:
# #             base = Case(a.base, s, a.run_dir, a.input_dir)
# #         except Exception as e:
# #             log(f"base / {s}: {e} - season skipped")
# #             continue
# #         cases[("base", s)] = base
# #         log(f"{s}: utci_points.csv of the base: {base.point_note}")
# #         grid = Grid(base.layout.site, a.cell or float(np.clip(0.5 * base.spacing, 2.0, 4.0)))
# #         grids[s] = grid
# #         log(f"{s}: grid {grid.shape[1]} x {grid.shape[0]} cells of {grid.cell:.1f} m, {len(base.pts['x'])} sample points, base UTCI {base.summary.get('utci_C', float('nan')):.2f} °C")
# #         for n in names:
# #             try:
# #                 cand = Case(n, s, a.run_dir, a.input_dir)
# #                 R = analyse_pair(base, cand, grid, a)
# #                 R.wake_frac = a.wake_frac
# #                 log(f"{s} / {n}: ΔUTCI {R.dU_mean:+.2f} °C  (" + ", ".join(f"{d} {R.contrib_mean[d]:+.2f}" for d in DRIVERS) + f")  split error {R.split_err:.1e}")
# #                 if not a.no_figures:
# #                     fig_simple_spots(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_1_{'hot' if s == 'heat' else 'cold'}_spots.png")
# #                     fig_simple_vent(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_2_ventilation.png")
# #                 if not a.no_figures and a.detailed:
# #                     fig_maps(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_maps.png")
# #                     fig_drivers(R, grid, outdir / "figures" / f"{tn[n]}_{s}_drivers.png")
# #                     fig_wakes(R, base, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_wakes.png")
# #                     fig_zones(R, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_zones_spots.png", "spots")
# #                     fig_zones(R, cand, grid, outdir / "figures" / f"{tn[n]}_{s}_zones_wakes.png", "wakes")
# #                     fig_categories(R, outdir / "figures" / f"{tn[n]}_{s}_categories.png")
# #                 cases[(n, s)] = cand                       # a layout enters the report only when its numbers AND figures are complete
# #                 results.append(R)
# #             except FileNotFoundError as e:
# #                 log(f"{s} / {n}: skipped - {e}")
# #             except Exception as e:
# #                 import traceback
# #                 log(f"{s} / {n}: FAILED {type(e).__name__}: {e}")
# #                 traceback.print_exc()
# #     if not results:
# #         log("nothing analysed")
# #         return 1
# #     if not a.no_figures and a.detailed:
# #         fig_overview(results, outdir / "figures" / "overview_drivers.png")

# #     # ------------------------------------------------------------------ report
# #     md = ["# Pareto analysis: why are the best layouts better than the base case?", "",
# #           f"Base case: `{a.base}`. Layouts analysed: " + ", ".join(f"`{n}`" for n in names) + ". Generated by `analyze_pareto.py` "
# #           f"({time.strftime('%Y-%m-%d %H:%M')}).", ""]
# #     md += ["## How to read this", "",
# #            "- **Benefit** is always in °C of UTCI in the direction that helps: heat hour = UTCI lower, cold hour = UTCI higher. Positive = good.",
# #            "- The change of UTCI is split **exactly** into four drivers (order-averaged substitution with the UTCI function): **Tmrt** (shade, sky view, surface temperatures), "
# #            "**wind** (ventilation / wind chill), **Ta** (air temperature), **RH** (humidity). The four add up to the total.",
# #            "- **Wake** = pedestrian-level wind below "
# #            f"{a.wake_frac:g} x the reference wind. **Hot / cold spots** = the worst parts of the base case (see each section for the threshold).",
# #            "- Values come from one hour per season at 1.5 m height; the statements are associations of layout changes with field changes.", ""]
# #     # objective table
# #     if obj:
# #         md += ["## Selected layouts", ""]
# #         rows = []
# #         for n in [a.base] + names:
# #             o = obj.get(n)
# #             if o:
# #                 rows.append([n, f"{o['utci_heat']:.2f}", f"{o['utci_cold']:.2f}", f"{o['f1']:.2f}", f"{o['f2']:.2f}"])
# #         md += md_table(["layout", "UTCI heat °C", "UTCI cold °C", "f1 (heat − 26)", "f2 (9 − cold)"], rows) + [""]
# #     if (outdir / "figures" / "overview_drivers.png").is_file():
# #         md += ["![overview](figures/overview_drivers.png)", ""]
# #     # per season overview table
# #     for s in seasons:
# #         rs = [r for r in results if r.season == s]
# #         if not rs:
# #             continue
# #         md += [f"## Overview — {SEASON_TXT[s]}", ""]
# #         rows = []
# #         for r in rs:
# #             c = r.contrib_mean
# #             rows.append([r.name, f"{r.stats_b['UTCI_C']:.2f} → {r.stats_c['UTCI_C']:.2f}", f"{r.sgn * r.dU_mean:+.2f}"] +
# #                         [f"{r.sgn * c[d]:+.2f}" for d in DRIVERS] +
# #                         [f"{r.stats_b['wind_ms'] / r.u_ref:.2f} → {r.stats_c['wind_ms'] / r.u_ref:.2f}",
# #                          f"{r.vent['calm_area_b']:.0f} → {r.vent['calm_area_c']:.0f}",
# #                          f"{r.spot_area_b:.0f} → {r.spot_area_c:.0f}"])
# #         md += md_table(["layout", "site mean UTCI °C", "benefit °C"] + [f"{d} °C" for d in DRIVERS] +
# #                        ["wind / reference", "calm area m²", f"{'hot' if s == 'heat' else 'cold'}-spot area m²"], rows) + [""]
# #         md += ["Actual change of the inputs (mean over the ground that is open in both layouts) and, in brackets, the change of UTCI it caused (°C, − = cooler):", ""]
# #         rows = []
# #         for r in rs:
# #             sv = r.site_vals
# #             rows.append([r.name, f"{sv['UTCI']['b']:.2f} → {sv['UTCI']['c']:.2f} ({sv['UTCI']['d']:+.2f})"] + [vtxt(sv, d) for d in ORDER])
# #         md += md_table(["layout", "UTCI °C"] + [f"{d} {UNIT[d]}: base → layout (Δ) → effect" for d in ORDER], rows) + [""]
# #         L0 = rs[0]
# #         md += [f"Spot threshold: UTCI {'≥' if s == 'heat' else '≤'} {L0.thr:.1f} °C; reference wind {L0.u_ref:.2f} m/s at 1.5 m (90th percentile of the base; 10 m wind of the hour {L0.u10:.1f} m/s); analysis grid {grids[s].cell:.1f} m.", ""]
# #     # per layout
# #     for n in names:
# #         rs = [r for r in results if r.name == n]
# #         if not rs:
# #             continue
# #         md += [f"## Layout `{n}`", ""]
# #         d0 = rs[0].diff
# #         md += [f"*Layout changes vs base:* trees {d0['trees_base']} → {d0['trees_cand']} ({len(d0['trees_added'])} new, {len(d0['trees_removed'])} removed), canopy "
# #                f"{d0['canopy_base']:.0f} → {d0['canopy_cand']:.0f} m², mean tree height {d0['mean_height_base']:.1f} → {d0['mean_height_cand']:.1f} m, ponds {d0['ponds_base']} → "
# #                f"{d0['ponds_cand']} ({d0['pond_area_base']:.0f} → {d0['pond_area_cand']:.0f} m²), lawns {d0['lawn_area_base']:.0f} → {d0['lawn_area_cand']:.0f} m²; buildings moved: " +
# #                (", ".join(f"{b['id']} {b['shift']:.1f} m" for b in d0["buildings"] if b["shift"] > 0.5) or "none") + ".", ""]
# #         for R in rs:
# #             s = R.season
# #             base = cases[("base", s)]
# #             cand = cases[(n, s)]
# #             md += [f"### {n} — {SEASON_TXT[s]}", ""]
# #             for line in build_findings(R, base, cand, grids[s]):
# #                 md += [line, ""]
# #             fg = outdir / "figures"
# #             kind = "hot" if s == "heat" else "cold"
# #             for tag, ttl in ((f"1_{kind}_spots", f"{kind} spots"), ("2_ventilation", "ventilation")):
# #                 if (fg / f"{tn[n]}_{s}_{tag}.png").is_file():
# #                     md += [f"![{ttl}](figures/{tn[n]}_{s}_{tag}.png)", ""]
# #                     if (fg / f"{tn[n]}_{s}_{tag}.txt").is_file():
# #                         md += [" ".join((fg / f"{tn[n]}_{s}_{tag}.txt").read_text().splitlines()[2:]), ""]
# #             for stem, ttl in (("maps", "maps"), ("drivers", "drivers"), ("zones_spots", "zone values of the spots"), ("wakes", "wakes"), ("zones_wakes", "zone values of the wakes"), ("categories", "categories")):
# #                 if (fg / f"{tn[n]}_{s}_{stem}.png").is_file():
# #                     md += [f"![{ttl}](figures/{tn[n]}_{s}_{stem}.png)", ""]
# #             centre = grids[s].site_mask
# #             sc = (float(grids[s].X[centre].mean()), float(grids[s].Y[centre].mean()))
# #             if R.regions or R.new_regions:
# #                 md += [f"**Zone-averaged inputs of the {'hot' if s == 'heat' else 'cold'} spots** — each cell: base → layout (change) → effect of that change on UTCI in °C (− = cooler, + = warmer)", ""]
# #                 md += md_zone_table(R.regions + R.new_regions, R.site_vals) + [""]
# #             if R.regions:
# #                 md += [f"**{'Hot' if s == 'heat' else 'Cold'} spots of the base and what became of them**", ""]
# #                 for r in R.regions:
# #                     md += [f"- **{r['id']}** ({fmt_area(r['area'])}, {where_text(r['centroid'], cand.layout, sc, R.flow)}): " + explain_region(r, R, s)]
# #                 md += [""]
# #             if R.new_regions:
# #                 md += ["**New stressed regions created by the layout**", ""]
# #                 for r in R.new_regions:
# #                     ctx = r["context"]
# #                     extra = f" Dominant driver of the worsening: {r.get('dominant', 'n/a')}." if "dominant" in r else ""
# #                     md += [f"- **{r['id']}** ({fmt_area(r['area'])}, {where_text(r['centroid'], cand.layout, sc, R.flow)}): UTCI {r['vals']['UTCI']['b']:.1f} → {r['u_cand']:.1f} °C. "
# #                            f"{inputs_sentence(r['vals'])}. Sunlit {r['sunlit_c']:.2f}.{extra} Layout nearby: {ctx['trees_added_near']} new trees within 25 m, {ctx.get('trees_added_upwind', 0)} upwind."]
# #                 md += [""]
# #             if R.wakes_b or R.wakes_c:
# #                 md += ["**Wake regions (calm air at 1.5 m)**", ""]
# #                 rows = []
# #                 for w in R.wakes_b:
# #                     rows.append([f"{w['id']} (base)", fmt_area(w["area"]), f"{w['length_m']:.0f} × {w['width_m']:.0f} m", w["building"] or "-",
# #                                  f"{w['utci_mean']:.1f}", w["fate"] + (f" → {w['cand_match']} {w['cand_area']:.0f} m²" if w["cand_match"] else ""),
# #                                  f"{w.get('d_wind', float('nan')):+.2f}" if "d_wind" in w else "-",
# #                                  f"{w['d_u']:+.2f}" if "d_u" in w else "-"])
# #                 for w in R.wakes_c:
# #                     if w.get("is_new"):
# #                         rows.append([f"{w['id']} (layout)", fmt_area(w["area"]), f"{w['length_m']:.0f} × {w['width_m']:.0f} m", w["building"] or "-", f"{w['utci_mean']:.1f}", "new", "-", "-"])
# #                 md += md_table(["wake", "area", "length × width (along × across wind)", "trails building", "UTCI °C", "fate", "Δ wind m/s", "Δ UTCI °C"], rows) + [""]
# #                 md += ["**Zone-averaged inputs of the wakes (same footprint in both layouts)** — each cell: base → layout (change) → effect on UTCI in °C", ""]
# #                 md += md_zone_table(R.wakes_b + [w for w in R.wakes_c if w.get("is_new")], R.site_vals) + [""]
# #     # mitigation
# #     md += ["## Mitigation inference", ""]
# #     md += mitigation_text(results, cases.get(("base", seasons[0])), grids, a)
# #     md += ["", "### Rules behind these sentences (the template)", ""]
# #     md += md_table(["observation", "reading", "mitigation"], [list(r) for r in RULES_TABLE]) + [""]
# #     md += ["## Method and limits", "",
# #            f"- Sample points of `utci_points.csv` (z = 1.5 m) are interpolated linearly on a {', '.join(f'{grids[s].cell:.1f} m' for s in grids)} grid; cells inside buildings (of each layout), outside the site, "
# #            "or farther than 2.2 point spacings from a sample are left out. Changes are computed only where both layouts have open ground.",
# #            "- The driver split recomputes UTCI (pythermalcomfort, same call as `compute_utci.py`) for every combination of base / layout values of Tmrt, wind, Ta, RH "
# #            "(16 evaluations per cell) and averages over the orders: exact and additive. Recomputed UTCI of the base differs from the stored one by "
# #            + ", ".join(f"{np.mean([r.csv_err for r in results if r.season == s]):.2f} °C ({s})" for s in seasons if any(r.season == s for r in results)) + " (rounding / interpolation).",
# #            "- Zone values (Ta, Tmrt, RH, wind) are area means over the zone's cells, in the base and in the layout at the same footprint; the 'effect' next to each is the part of the "
# #            "UTCI change that this input alone caused (exact split), so a large change of an input with a small effect (or the reverse) shows how sensitive UTCI is to it in that zone.",
# #            "- The split of Tmrt is a linear regression on the sunlit share, the sky view and Ta; the remainder is attributed to surface temperatures / long-wave exchange.",
# #            "- One hour per season, steady flow field, one height. Wakes shown are pedestrian-level calm regions, not the full 3-D recirculation zones.",
# #            "- Layout context numbers (trees within 25 m, 70 m upwind corridor, ponds) show what changed near a region; they do not prove that the change caused the field change.", ""]
# #     md_text = "\n".join(md)
# #     hz = (cases[("base", seasons[0])].summary or {}).get("height_m") if ("base", seasons[0]) in cases else None
# #     if hz and abs(float(hz) - 1.5) > 1e-9:                    # the text says 1.5 m; the run used another height (compute_utci.py --height)
# #         md_text = md_text.replace("1.5 m", f"{float(hz):g} m")
# #     (outdir / "analysis_report.md").write_text(md_text)
# #     if not a.no_html:
# #         (outdir / "analysis_report.html").write_text(md_to_html(md_text, outdir))

# #     # ------------------------------------------------------------------ tables / json
# #     def drow(r):
# #         return dict(layout=r.name, season=r.season, utci_base=r.stats_b["UTCI_C"], utci_layout=r.stats_c["UTCI_C"], d_utci=r.dU_mean,
# #                     **{f"contrib_{d}": r.contrib_mean[d] for d in DRIVERS}, split_error=r.split_err,
# #                     wind_ratio_base=r.vent["wind_ratio_b"], wind_ratio_layout=r.vent["wind_ratio_c"], calm_area_base=r.vent["calm_area_b"], calm_area_layout=r.vent["calm_area_c"],
# #                     spot_threshold=r.thr, spot_area_base=r.spot_area_b, spot_area_layout=r.spot_area_c, sunlit_base=r.stats_b["sunlit"], sunlit_layout=r.stats_c["sunlit"],
# #                     sky_base=r.stats_b["sky_open"], sky_layout=r.stats_c["sky_open"], Tmrt_base=r.stats_b["Tmrt_C"], Tmrt_layout=r.stats_c["Tmrt_C"],
# #                     Ta_base=r.stats_b["Tair_C"], Ta_layout=r.stats_c["Tair_C"], RH_base=r.stats_b["RH_pct"], RH_layout=r.stats_c["RH_pct"],
# #                     **{f"{d}_change_site_mean": r.site_vals[d]["d"] for d in DRIVERS})

# #     def valcols(v):
# #         """flat columns: UTCI base / layout / change, and for Ta, Tmrt, RH, wind: base, layout, change, effect on UTCI"""
# #         if v is None:
# #             return {}
# #         o = dict(utci_base=v["UTCI"]["b"], utci_layout=v["UTCI"]["c"], utci_change=v["UTCI"]["d"])
# #         for d in ORDER:
# #             u = {"Ta": "C", "Tmrt": "C", "RH": "pct", "wind": "ms"}[d]
# #             o.update({f"{d}_base_{u}": v[d]["b"], f"{d}_layout_{u}": v[d]["c"], f"{d}_change_{u}": v[d]["d"], f"{d}_effect_on_UTCI_C": v[d]["eff"]})
# #         return o

# #     def write_csv(path, rows):
# #         if not rows:
# #             return
# #         keys = list(rows[0].keys())
# #         with open(path, "w", newline="") as fh:
# #             w = csv.DictWriter(fh, fieldnames=keys)
# #             w.writeheader()
# #             w.writerows(rows)
# #     write_csv(outdir / "tables" / "drivers.csv", [drow(r) for r in results])
# #     reg_rows, wake_rows = [], []
# #     for r in results:
# #         for g in r.regions:
# #             reg_rows.append(dict(layout=r.name, season=r.season, region=g["id"], area_m2=g["area"], x=g["centroid"][0], y=g["centroid"][1], status=g["status"], utci_base=g["u_base"],
# #                                  utci_layout=g["u_cand"], resolved_share=g["resolved_share"], dominant=g["dominant"],
# #                                  **{f"contrib_{d}": g["contrib"][d] for d in DRIVERS}, trees_added_near=g["context"]["trees_added_near"],
# #                                  trees_added_upwind=g["context"].get("trees_added_upwind", 0)))
# #         for w in r.wakes_b:
# #             wake_rows.append(dict(layout=r.name, season=r.season, wake=w["id"], which="base", area_m2=w["area"], length_m=w["length_m"], width_m=w["width_m"], building=w["building"],
# #                                   utci=w["utci_mean"], fate=w["fate"]))
# #         for w in r.wakes_c:
# #             wake_rows.append(dict(layout=r.name, season=r.season, wake=w["id"], which="layout", area_m2=w["area"], length_m=w["length_m"], width_m=w["width_m"], building=w["building"],
# #                                   utci=w["utci_mean"], fate="new" if w.get("is_new") else ""))
# #     zone_rows = []
# #     for r in results:
# #         base_ = dict(layout=r.name, season=r.season)
# #         zone_rows.append(dict(base_, kind="whole site", zone="site", status="", area_m2=r.site_vals["area"], **valcols(r.site_vals)))
# #         for g in r.regions:
# #             zone_rows.append(dict(base_, kind="spot of the base", zone=g["id"], status=g["status"], area_m2=g["area"], **valcols(g["vals"])))
# #         for g in r.new_regions:
# #             zone_rows.append(dict(base_, kind="new stressed region", zone=g["id"], status="new", area_m2=g["area"], **valcols(g["vals"])))
# #         for w in r.wakes_b:
# #             zone_rows.append(dict(base_, kind="wake of the base", zone=w["id"], status=w["fate"], area_m2=w["area"], **valcols(w["vals"])))
# #         for w in r.wakes_c:
# #             if w.get("is_new"):
# #                 zone_rows.append(dict(base_, kind="new wake", zone=w["id"], status="new", area_m2=w["area"], **valcols(w["vals"])))
# #     write_csv(outdir / "tables" / "zones.csv", zone_rows)
# #     write_csv(outdir / "tables" / "regions.csv", reg_rows)
# #     write_csv(outdir / "tables" / "wakes.csv", wake_rows)
# #     summ = dict(base=a.base, layouts=names, seasons=list(seasons), generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
# #                 results=[dict(drow(r), site_vals=r.site_vals, regions=[{k: v for k, v in g.items() if k not in ("mask", "context")} for g in r.regions]) for r in results])
# #     (outdir / "analysis_summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
# #     log(f"done in {time.time() - t0:.0f} s -> {outdir}/analysis_report.md" + ("" if a.no_html else " and .html"))
# #     return 0


# # if __name__ == "__main__":
# #     sys.exit(main())