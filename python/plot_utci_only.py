#!/usr/bin/env python3
"""
utci_plot.py  --  picture of the point-wise UTCI computed by compute_utci.py

    from utci_plot import save_utci_map
    save_utci_map("utci_map.png", xy, utci, site_poly=poly, buildings=btri, trees=ttri, summary=summary)

What the image shows
    * the UTCI at the pedestrian-height sample points (x, y in the mesh frame), as a smooth map
      (triangulated, with triangles that bridge a building or a gap left out) or as one dot per point
    * the site polygon, the building footprints and the tree crowns that the thermal model used
    * a colour bar with the UTCI stress categories, the site mean, the BASE case (individual_0) and the rural reference marked on it
    * a footer with the numbers of utci.json (mean, range, delta vs rural, Tmrt, sun / sky fractions)

It only needs numpy and matplotlib (scipy is used for the point spacing when it is installed) and it never
touches pyplot's global state, so it is safe to call from a batch job on a headless machine.

Stand-alone use on finished runs (nothing is recomputed, the results are only read):
    python3 utci_plot.py --case <case>                         # -> <case>/postProcessing/utci_map.png
    python3 utci_plot.py --case runs/*/case --common-range --out-dir maps/    # many candidates, one colour scale
    python3 utci_plot.py <case>/postProcessing/utci_points.csv map.png --json <case>/postProcessing/utci.json
                         [--layout layout.json] [--range 30 50] [--style points] [--cmap viridis]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.tri import LinearTriInterpolator, Triangulation

# upper limits of the UTCI stress categories (same table as compute_utci.CATEGORIES)
CATEGORIES = [(46.0, "extreme heat stress"), (38.0, "very strong heat stress"), (32.0, "strong heat stress"),
              (26.0, "moderate heat stress"), (9.0, "no thermal stress"), (0.0, "slight cold stress"),
              (-13.0, "moderate cold stress"), (-27.0, "strong cold stress"), (-40.0, "very strong cold stress")]
INK, MUTED = "#1c2830", "#6d7780"

# "Rainbow Desaturated" (the ParaView preset): blue -> cyan -> green -> yellow -> orange -> dark red, with the
# harsh saturated ends of the classic rainbow taken down.  Any matplotlib colormap name also works (--cmap).
RAINBOW_DESATURATED = LinearSegmentedColormap.from_list("rainbow_desaturated", [
    (0.000, (0.278431, 0.278431, 0.858824)),
    (0.143, (0.000000, 0.000000, 0.360784)),
    (0.285, (0.000000, 1.000000, 1.000000)),
    (0.429, (0.000000, 0.501961, 0.000000)),
    (0.571, (1.000000, 1.000000, 0.000000)),
    (0.714, (1.000000, 0.380392, 0.000000)),
    (0.857, (0.419608, 0.000000, 0.000000)),
    (1.000, (0.878431, 0.301961, 0.301961))])


def _cmap(name):
    return RAINBOW_DESATURATED if name in (None, "rainbow_desaturated") else name


# --------------------------------------------------------------------------- geometry helpers
def _footprints(obj):
    """accepts triangles (M,3,3) of an .obj mesh, or a list of (K,2/3) polygons -> list of (K,2) arrays.
    For a triangle mesh only the roughly horizontal triangles are kept (walls project to zero area)."""
    if obj is None:
        return []
    if isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[1:] == (3, 3):
        n = np.cross(obj[:, 1] - obj[:, 0], obj[:, 2] - obj[:, 0])
        nl = np.linalg.norm(n, axis=1)
        keep = (nl > 1e-12) & (np.abs(n[:, 2]) > 0.5 * nl)
        return [t[:, :2] for t in obj[keep]]
    return [np.asarray(p, float)[:, :2] for p in obj if len(p) >= 3]


def _inside(poly, x, y):
    """even-odd point-in-polygon test, vectorised over the points"""
    r = np.zeros(len(x), bool)
    px, py = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = (py[i] > y) != (py[j] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
        r ^= cond & (x < xint)
        j = i
    return r


def _spacing(xy):
    """mean distance between neighbouring sample points (~ sqrt(area / N)): 1.8 x the median nearest-neighbour
    distance, which holds for scattered as well as for evenly stratified points"""
    try:
        from scipy.spatial import cKDTree
        d = cKDTree(xy).query(xy, k=2)[0][:, 1]
        sp = float(1.8 * np.median(d))
        if sp > 0:
            return sp
        raise ValueError("zero spacing")
    except Exception:
        span = np.ptp(xy, axis=0)
        return float(np.sqrt(max(span[0] * span[1], 1e-9) / len(xy)))


def _bands(vmin, vmax):
    """(lo, hi, name) of the stress categories, clipped to the colour range"""
    edges = sorted(c[0] for c in CATEGORIES)
    names = {c[0]: c[1] for c in CATEGORIES}
    out, lo = [], -1e9
    for e in edges + [1e9]:
        name = "extreme cold stress" if lo == -1e9 else names[lo]
        hi = e
        a, b = max(lo, vmin), min(hi, vmax)
        if b > a:
            out.append((a, b, name))
        lo = e
    # category(v) = name of the highest lower limit below v, so band [lo_i, lo_{i+1}] is called names[lo_i]
    # (and the open band above the top limit is "extreme heat stress")
    return out


def _inside_any(poly, X, Y, offs, all_points=False):
    """point-in-polygon for a cell: its centre and four points towards its corners (any / all must be inside)"""
    r = None
    for ox, oy in offs:
        m = _inside(poly, X + ox, Y + oy)
        r = m if r is None else (r & m if all_points else r | m)
    return r


def _nbr_sum(A):
    """sum of the 8 neighbours of every cell"""
    P = np.pad(A, 1)
    return sum(P[1 + dy:P.shape[0] - 1 + dy, 1 + dx:P.shape[1] - 1 + dx]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))


def _gauss(A, sigma):
    """separable Gaussian filter (sigma in cells), zero padded"""
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    A = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, A)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, A)


def _upsample(A, f):
    """bilinear up-sampling of a cell-centred grid by an integer factor (edge cells are extended)"""
    ny, nx = A.shape
    tx = (np.arange(nx * f) + 0.5) / f - 0.5
    ty = (np.arange(ny * f) + 0.5) / f - 0.5
    B = np.array([np.interp(tx, np.arange(nx), row) for row in A])
    return np.array([np.interp(ty, np.arange(ny), col) for col in B.T]).T


def _grid_image(xy, u, h, cell, site_poly, bpolys, sm, smooth=1.0, reach=4, gap_alpha=1.0):
    """
    The UTCI on a uniform grid:
      1. every sample point goes into the square cell it lies in; a cell shows the mean of its points
      2. an empty cell of open ground takes the mean of its filled neighbours (8-neighbourhood); this is repeated
         `reach` times so that wider gaps close from their edges.  Cells inside a building are never filled.
      3. a Gaussian smoothing (sigma = `smooth` cells) that only uses filled cells, so the empty cells and the
         buildings do not pull the values down
    Returns (rgba image, extent, fraction of open-ground cells that were filled, cell size).
    """
    site = np.asarray(site_poly, float)[:, :2] if site_poly is not None else None
    lo, hi = (site.min(axis=0), site.max(axis=0)) if site is not None else (xy.min(axis=0), xy.max(axis=0))
    cell = float(cell) if cell else 1.6 * h
    nx, ny = max(int(np.ceil((hi[0] - lo[0]) / cell)), 1), max(int(np.ceil((hi[1] - lo[1]) / cell)), 1)
    ix = np.clip(((xy[:, 0] - lo[0]) // cell).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) // cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    cnt = np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
    tot = np.bincount(flat, weights=u, minlength=nx * ny).reshape(ny, nx)
    data = cnt > 0
    Z = np.where(data, tot / np.maximum(cnt, 1), 0.0)

    # which cells are open ground: touches the site, is not entirely inside a building
    CX, CY = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cell, lo[1] + (np.arange(ny) + 0.5) * cell)
    q = 0.4 * cell
    offs = [(0, 0), (q, q), (-q, q), (q, -q), (-q, -q)]
    free = np.ones((ny, nx), bool)
    if site is not None:
        free = _inside_any(site, CX.ravel(), CY.ravel(), offs).reshape(ny, nx)
    for bp in _merge_small(bpolys):
        free &= ~_inside_any(bp, CX.ravel(), CY.ravel(), offs, all_points=True).reshape(ny, nx)
    free |= data

    known, filled = data.copy(), np.zeros_like(data)
    for _ in range(max(int(reach), 0)):
        S, C = _nbr_sum(np.where(known, Z, 0.0)), _nbr_sum(known.astype(float))
        new = free & ~known & (C > 0)
        if not new.any():
            break
        Z[new] = S[new] / C[new]
        known |= new
        filled |= new

    if smooth and smooth > 0:
        w = np.where(data, 1.0, np.where(filled, 0.5, 0.0))      # measured cells count double
        num, den = _gauss(Z * w, smooth), _gauss(w, smooth)
        Z = np.where(known & (den > 1e-9), num / np.maximum(den, 1e-9), Z)

    # extend the field one cell beyond its edge (it ends up under a building or outside the site boundary, both
    # drawn over it) so that it fades out there and not half a cell before the building edge
    S, C = _nbr_sum(np.where(known, Z, 0.0)), _nbr_sum(known.astype(float))
    ext = ~known & (C > 0)
    Z[ext] = S[ext] / C[ext]
    known = known | ext

    # up-sample for display (normalised, so the empty cells do not darken the edges)
    f = int(np.clip(round(cell / (max(hi - lo) / 700.0)), 1, 24))
    Wk = known.astype(float)
    Wa = np.where(filled, gap_alpha, 1.0) * Wk
    Wu, Vu, Au = _upsample(Wk, f), _upsample(Z * Wk, f), _upsample(Wa, f)
    Zf = np.where(Wu > 1e-6, Vu / np.maximum(Wu, 1e-6), np.nan)
    rgba = sm.to_rgba(Zf)
    rgba[..., 3] = np.clip(Au, 0, 1)
    extent = (lo[0], lo[0] + nx * cell, lo[1], lo[1] + ny * cell)
    return rgba, extent, float(filled.sum() / max(free.sum(), 1)), cell


def _field_image(xy, u, h, site_poly, bpolys, sm, fill=True, fill_max=2.5, gap_alpha=0.55):
    """The UTCI field on a regular grid -> (rgba image, extent, fraction of the site that is a nearest-point fill).

    1. linear interpolation inside the triangulation of the sample points; a triangle is dropped when one of its
       edges is longer than 3 point spacings, when it lies in a building footprint, or outside the site
    2. fallback for the cells that leave blank: take the value of the NEAREST sample point, drawn faded
       (gap_alpha) so it cannot be mistaken for interpolated data.  Cells farther than fill_max spacings from every
       point (typically the inside of a building) stay blank on purpose.
    """
    tri = Triangulation(xy[:, 0], xy[:, 1])
    P = xy[tri.triangles]
    longest = np.max(np.linalg.norm(P - np.roll(P, 1, axis=1), axis=2), axis=1)
    mask = longest > 3.0 * h
    cen = P.mean(axis=1)
    for bp in _merge_small(bpolys):
        mask |= _inside(bp, cen[:, 0], cen[:, 1])
    site = np.asarray(site_poly, float)[:, :2] if site_poly is not None else None
    if site is not None:
        mask |= ~_inside(site, cen[:, 0], cen[:, 1])
    tri.set_mask(mask)

    lo, hi = (site.min(axis=0), site.max(axis=0)) if site is not None else (xy.min(axis=0), xy.max(axis=0))
    step = max(h / 4.0, float(np.max(hi - lo)) / 900.0)
    xs, ys = np.arange(lo[0], hi[0] + step, step), np.arange(lo[1], hi[1] + step, step)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full(X.shape, np.nan)
    if (~mask).any():
        Z = np.ma.filled(LinearTriInterpolator(tri, u)(X, Y).astype(float), np.nan)
    alpha = np.isfinite(Z).astype(float)
    in_site = _inside(site, X.ravel(), Y.ravel()).reshape(X.shape) if site is not None else np.ones(X.shape, bool)
    for bp in _merge_small(bpolys):                          # open ground only: no fill (and no statistics) under buildings
        in_site &= ~_inside(bp, X.ravel(), Y.ravel()).reshape(X.shape)
    filled = np.zeros(X.shape, bool)
    gap = np.flatnonzero((~np.isfinite(Z) & in_site).ravel())
    if fill and len(gap):
        q = np.column_stack([X.ravel()[gap], Y.ravel()[gap]])
        try:
            from scipy.spatial import cKDTree
            d, idx = cKDTree(xy).query(q)
            take = d <= fill_max * h
            val = u[idx]
        except Exception:                                   # no scipy: interpolate over the whole convex hull instead
            full = Triangulation(xy[:, 0], xy[:, 1])
            val = np.ma.filled(LinearTriInterpolator(full, u)(q[:, 0], q[:, 1]).astype(float), np.nan)
            take = np.isfinite(val)
        Zr, Ar, Fr = Z.reshape(-1), alpha.reshape(-1), filled.reshape(-1)    # views
        Zr[gap[take]], Ar[gap[take]], Fr[gap[take]] = val[take], gap_alpha, True
    rgba = sm.to_rgba(Z)
    rgba[..., 3] = alpha
    extent = (xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2)
    return rgba, extent, float(filled.sum() / max(in_site.sum(), 1))


# --------------------------------------------------------------------------- the picture
def save_utci_map(out, xy, utci, *, site_poly=None, buildings=None, trees=None, summary=None, title=None,
                  vrange=None, style="grid", cmap="rainbow_desaturated", dpi=150, height_m=None, fill_gaps=True,
                  cell=None, smooth=1.0, base_utci=None, base_name=None, is_base=False):
    """
    out        png path (parent folder is created)
    base_utci  site-mean UTCI of the base case (individual_0): marked on the colour bar and in the footer (optional)
    xy         (N,2) sample point coordinates, utci (N,) values in degC
    site_poly  (K,2) site outline, buildings / trees  triangles (M,3,3) or list of polygons (optional)
    summary    the dict written to utci.json (optional, adds the footer and the mean / rural marks)
    vrange     (lo, hi) fixed colour range: use the same one for all candidates to compare their pictures
    style      "grid"   uniform grid: cell means, empty cells from their neighbours, then smoothing (default)
               "smooth" triangulated, interpolated field
               "points" one disc per sample point
    cell       grid style: cell size in m (default 1.6 x the mean point spacing)
    smooth     grid style: Gaussian smoothing, sigma in cells (0 = off)
    fill_gaps  smooth style: fill the blank patches between sample points with the nearest point's value, faded
    """
    cmap = _cmap(cmap)
    xy = np.asarray(xy, float).reshape(-1, 2)
    u = np.asarray(utci, float).ravel()
    good = np.isfinite(u) & np.isfinite(xy).all(axis=1)
    xy, u = xy[good], u[good]
    mean_all = float(u.mean())                      # the repeats are area weights: this is the mean that compute_utci.py reports
    # compute_utci.py samples the mesh cells with probability proportional to their area, with replacement: on a coarse mesh the same cell
    # (same x, y, same values) is written many times.  For drawing, every distinct location is used once (the repeats only act as weights).
    uq, inv = np.unique(np.round(xy, 4), axis=0, return_inverse=True)
    if len(uq) < len(xy):
        inv = np.asarray(inv).ravel()
        u = np.bincount(inv, weights=u) / np.bincount(inv)
        xy = uq
    if len(u) < 3:
        raise ValueError(f"only {len(u)} distinct finite UTCI points - nothing to draw")
    vmin, vmax = vrange if vrange else (float(np.floor(u.min())), float(np.ceil(u.max())))
    if not vrange and base_utci is not None and np.isfinite(base_utci) and not is_base:     # keep the base mark on the colour bar
        vmin, vmax = min(vmin, float(np.floor(base_utci))), max(vmax, float(np.ceil(base_utci)))
    if vmax - vmin < 2.0:
        vmin, vmax = (vmin + vmax) / 2 - 1.0, (vmin + vmax) / 2 + 1.0
    spacing = _spacing(xy)
    bpolys, tpolys = _footprints(buildings), _footprints(trees)

    fig = Figure(figsize=(10.5, 8.0), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.065, 0.215, 0.57, 0.70])
    cax = fig.add_axes([0.715, 0.215, 0.02, 0.70])
    ax.set_aspect("equal")
    ax.set_facecolor("#f4f2ea")

    mappable, filled_frac, cell_used = None, None, None
    if style == "grid":
        try:
            mappable = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
            mappable.set_array(np.array([]))
            rgba, extent, filled_frac, cell_used = _grid_image(xy, u, spacing, cell, site_poly, bpolys, mappable,
                                                               smooth=smooth)
            im = ax.imshow(rgba, extent=extent, origin="lower", interpolation="bilinear", zorder=2)
            if site_poly is not None:
                im.set_clip_path(Polygon(np.asarray(site_poly, float)[:, :2], closed=True, transform=ax.transData))
        except Exception:                                            # fall back to the triangulated map
            mappable, filled_frac, style = None, None, "smooth"
    if style == "smooth" and mappable is None:
        try:
            mappable = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
            mappable.set_array(np.array([]))
            rgba, extent, filled_frac = _field_image(xy, u, spacing, site_poly, bpolys, mappable, fill=fill_gaps)
            im = ax.imshow(rgba, extent=extent, origin="lower", interpolation="bilinear", zorder=2)
            if site_poly is not None:
                im.set_clip_path(Polygon(np.asarray(site_poly, float)[:, :2], closed=True, transform=ax.transData))
        except Exception:                                            # e.g. collinear points: fall back to dots
            mappable, filled_frac = None, None
    if mappable is None:                                             # "points" style (or smooth failed)
        mappable = EllipseCollection(widths=spacing * 0.9, heights=spacing * 0.9, angles=0, units="xy",
                                     offsets=xy, offset_transform=ax.transData, array=u, cmap=cmap, zorder=2)
        mappable.set_clim(vmin, vmax)
        ax.add_collection(mappable)

    if tpolys:
        ax.add_collection(PolyCollection(tpolys, facecolors=(1, 1, 1, .16), edgecolors=(0, 0, 0, .55), linewidths=.7, zorder=3))
    if bpolys:
        ax.add_collection(PolyCollection(bpolys, facecolors="#8d9094", edgecolors="#8d9094", linewidths=.4,
                                         zorder=4))
    if site_poly is not None:
        ax.add_patch(Polygon(np.asarray(site_poly, float)[:, :2], closed=True, fc="none", ec=INK, lw=1.4, zorder=5))
        pts = np.asarray(site_poly, float)[:, :2]
    else:
        pts = xy
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.04 * float(np.max(hi - lo))
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_xlabel("x (m)", fontsize=9, color=MUTED)
    ax.set_ylabel("y (m)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#d5d0c3")

    # ---- colour bar (ticks on the left); stress categories and the site mean / rural reference on the right
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("UTCI (°C)", fontsize=9, color=INK)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cb.ax.tick_params(labelsize=8)
    cax.set_ylim(vmin, vmax)

    def yfig(val):                                   # data value on the colour bar -> figure fraction
        return fig.transFigure.inverted().transform((0, cax.transData.transform((0, val))[1]))[1]

    for a, b, name in _bands(vmin, vmax):
        if a > vmin + 1e-9:
            cax.axhline(a, color=INK, lw=.6, alpha=.6)
        if (b - a) > 0.035 * (vmax - vmin):
            fig.text(0.835, yfig((a + b) / 2), name, fontsize=7.5, color=MUTED, va="center", ha="left")
    mean = mean_all
    rural = (summary or {}).get("rural_utci_C")
    marks = [(mean, f"site mean {mean:.1f}" + (" (base)" if is_base else ""), "-")]
    if base_utci is not None and np.isfinite(base_utci) and not is_base:
        marks.append((float(base_utci), f"base {float(base_utci):.1f}", ":"))
    if rural is not None and np.isfinite(rural):
        marks.append((float(rural), f"rural {float(rural):.1f}", "--"))
    ys = [yfig(m[0]) for m in marks]
    order = sorted(range(len(ys)), key=lambda i: ys[i])          # labels at least 0.026 of the figure height apart (pushed apart symmetrically)
    for _ in range(50):
        moved = False
        for a_, b_ in zip(order[:-1], order[1:]):
            gap = ys[b_] - ys[a_]
            if gap < 0.026:
                d = 0.5 * (0.026 - gap) + 1e-4
                ys[a_] -= d
                ys[b_] += d
                moved = True
        if not moved:
            break
    for (val, lab, ls), yl in zip(marks, ys):
        if vmin <= val <= vmax:
            cax.plot([0, 1], [val, val], color=INK, lw=2, ls=ls, zorder=5, clip_on=False)
            fig.text(0.742, yl, "◂ " + lab, fontsize=7.5, color=INK, va="center", ha="left", fontweight="bold")

    # ---- title and footer
    h = height_m if height_m is not None else (summary or {}).get("height_m")
    ttl = title or "UTCI at pedestrian height" + (f" (z = {h:g} m)" if h is not None else "")
    fig.text(0.075, 0.955, ttl, fontsize=13, fontweight="bold", color=INK, ha="left", va="center")
    s = summary or {}
    sub = s.get("target")
    if isinstance(sub, dict):
        try:
            fig.text(0.075, 0.928, f"{sub['year']}-{int(sub['month']):02d}-{int(sub['day']):02d}  hour_end "
                                   f"{int(sub['hour_end']):02d}   ·   CFD time {s.get('time', '?')}   ·   "
                                   f"{len(u)} sample points", fontsize=8.5, color=MUTED, ha="left", va="center")
        except Exception:
            pass
    lines = [f"site mean UTCI {mean:.2f} °C" + (f" ({s['category']})" if "category" in s else "")
             + f"    range {u.min():.1f} .. {u.max():.1f} °C"]
    if rural is not None:
        lines[0] += f"    rural {float(rural):.2f} °C    Δ vs rural {mean - float(rural):+.2f} °C"
    if base_utci is not None and np.isfinite(base_utci) and not is_base:
        lines.append(f"base {base_name or ''} {float(base_utci):.2f} °C    Δ vs base {mean - float(base_utci):+.2f} °C".replace("base  ", "base "))
    bits = []
    for key, lab, fmt in (("Tair_C", "air T", "{:.1f} °C"), ("RH_pct", "RH", "{:.0f} %"),
                          ("wind_ms", "wind", "{:.2f} m/s"), ("Tmrt_C", "Tmrt", "{:.1f} °C")):
        if key in s:
            bits.append(f"{lab} {fmt.format(s[key])}")
    if bits:
        lines.append("mean   " + "   ".join(bits))
    bits = []
    for key, lab in (("sunlit_fraction", "sunlit"), ("sky_open", "sky open"), ("sky_tree", "sky tree"),
                     ("sky_building", "sky building")):
        if key in s:
            bits.append(f"{lab} {100 * s[key]:.0f} %")
    if bits:
        lines.append("   ".join(bits))
    fig.text(0.075, 0.095, "\n".join(lines), fontsize=8.5, family="DejaVu Sans Mono", color=INK, ha="left",
             va="center", linespacing=1.6)
    note = "grey = buildings   circles = tree crowns   black line = site boundary"
    if cell_used:
        note += (f"\ngrid {cell_used:.1f} m cells" + (f", Gaussian smoothing σ = {smooth * cell_used:.1f} m" if smooth else "")
                 + (f"   {100 * filled_frac:.0f} % of the open-ground cells had no sample point and were filled "
                    "from their neighbours" if filled_frac else ""))
    elif filled_frac:
        note += (f"   faded = no sample point close by, nearest value shown ({100 * filled_frac:.0f} % of the site)")
    fig.text(0.075, 0.03, note, fontsize=7.5, color=MUTED, ha="left", va="center", linespacing=1.5)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    return out


def _merge_small(polys):
    """roof triangles of one building are many small polygons; testing each against every triangle centroid
    is O(buildings x triangles), so for big meshes thin them out to the convex-ish outer ones by area"""
    return polys if len(polys) <= 400 else polys[:: max(1, len(polys) // 400)]


# --------------------------------------------------------------------------- command line
def _load_points(csv_path):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    return np.column_stack([d["x"], d["y"]]), d["UTCI_C"]


def _layout_geometry(path):
    """site polygon + building footprints (one per storey stack) + tree crowns from a scene json"""
    J = json.loads(Path(path).read_text())
    J = J.get("siteOnly_json", J)
    co = (J.get("site") or {}).get("coordinates")
    site = np.array([[p["x"], p["y"]] for p in co], float) if co else None
    b = {}
    for bd in J.get("buildings") or []:
        vs = (bd.get("dimensions") or {}).get("baseVertices")
        if vs and len(vs) >= 3:
            b[tuple(round(v["x"], 3) for v in vs) + tuple(round(v["y"], 3) for v in vs)] = \
                np.array([[v["x"], v["y"]] for v in vs], float)
    th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    crowns = []
    for t in J.get("trees") or []:
        r = (t.get("dimensions") or {}).get("crownRadius") or (t.get("crownDiameter") or 4) / 2
        crowns.append(np.column_stack([t["position"]["x"] + r * np.cos(th), t["position"]["y"] + r * np.sin(th)]))
    return site, list(b.values()), crowns


def _read_obj(path):
    """triangles (M,3,3) of a Wavefront .obj (the surfaces/buildings.obj and trees.obj the thermal model wrote, in the
    mesh frame); quads and larger faces are fanned into triangles"""
    V, T = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(V) + i for i in idx]
                T.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    return np.array(V, float)[np.array(T)] if T and V else None


def _case_inputs(case, layout=None):
    """everything utci_plot needs from a finished airTqFoam case folder, found the way compute_utci.py finds it"""
    case = Path(case).resolve()
    pp = case / "postProcessing"
    csv = pp / "utci_points.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} not found - is {case} a case folder compute_utci.py has run in?")
    summary = json.loads((pp / "utci.json").read_text()) if (pp / "utci.json").is_file() else None
    site = None
    lj = Path(layout) if layout else None
    if lj is None:                                       # same default as compute_utci.py: the only json in ../momentum
        c = sorted((case.parent / "momentum").glob("*.json"))
        lj = c[0] if len(c) == 1 else None
    if lj is not None:
        site = _layout_geometry(lj)[0]
        if site is not None and np.allclose(site[0], site[-1]):
            site = site[:-1]
    b = case / "surfaces" / "buildings.obj"
    t = case / "surfaces" / "trees.obj"
    return dict(csv=csv, summary=summary, site=site,
                buildings=_read_obj(b) if b.is_file() else None, trees=_read_obj(t) if t.is_file() else None)


def _find_base(case, base):
    """(site-mean UTCI of the base case, its name, is this the base) for run/<name>/<season>/temperature_humidity; (None, base, False) if not found"""
    case = Path(case).resolve()
    try:
        name, season, run = case.parent.parent.name, case.parent.name, case.parent.parent.parent
        if name == base:
            return None, base, True
        f = run / base / season / case.name / "postProcessing" / "utci.json"
        if f.is_file():
            v = json.loads(f.read_text()).get("utci_C")
            if v is not None and np.isfinite(float(v)):
                return float(v), base, False
        print(f"[utci_plot] base case not found: {f}", file=sys.stderr)
    except Exception as exc:
        print(f"[utci_plot] base lookup failed: {exc}", file=sys.stderr)
    return None, base, False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points_csv", nargs="?", type=Path, help="<case>/postProcessing/utci_points.csv (single-file mode)")
    ap.add_argument("out", nargs="?", type=Path, help="png to write (single-file mode)")
    ap.add_argument("--case", nargs="+", type=Path, metavar="CASE",
                    help="finished case folder(s): reads postProcessing/utci_points.csv + utci.json, surfaces/*.obj and "
                         "the site outline from ../momentum/*.json, writes <case>/postProcessing/utci_map.png")
    ap.add_argument("--out-dir", type=Path, help="with --case: collect all images here as <candidate>.png instead")
    ap.add_argument("--common-range", action="store_true",
                    help="with --case: one colour range for all the cases, so that the pictures can be compared")
    ap.add_argument("--json", type=Path, help="utci.json (single-file mode; adds the footer)")
    ap.add_argument("--layout", type=Path, help="layout json: site outline (and, in single-file mode, buildings and trees)")
    ap.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"), help="fixed colour range")
    ap.add_argument("--style", choices=("grid", "smooth", "points"), default="grid")
    ap.add_argument("--cmap", default="rainbow_desaturated", help="colour scale: rainbow_desaturated or any matplotlib name")
    ap.add_argument("--cell", type=float, help="grid style: cell size in m (default 1.6 x point spacing)")
    ap.add_argument("--smooth", type=float, default=1.0, help="grid style: Gaussian sigma in cells, 0 = off")
    ap.add_argument("--base", default="individual_0", help="with --case run/<name>/<season>/temperature_humidity: name of the base case whose site-mean UTCI is marked "
                    "on the colour bar (read from run/<base>/<season>/temperature_humidity/postProcessing/utci.json; default individual_0)")
    ap.add_argument("--base-utci", type=float, help="base-case site-mean UTCI in degC (single-file mode, or to override the lookup)")
    ap.add_argument("--no-base", action="store_true", help="do not mark the base case")
    ap.add_argument("--no-fill", action="store_true", help="smooth style: leave the gaps between sample points blank")
    a = ap.parse_args(argv)
    common = dict(style=a.style, cmap=a.cmap, cell=a.cell, smooth=a.smooth, fill_gaps=not a.no_fill)

    if a.case:
        jobs = []
        for c in a.case:
            try:
                jobs.append((c, _case_inputs(c, a.layout)))
            except Exception as exc:
                print(f"[utci_plot] {c}: {exc}", file=sys.stderr)
        vr = tuple(a.range) if a.range else None
        if a.common_range and jobs and vr is None:
            allu = np.concatenate([_load_points(j[1]["csv"])[1] for j in jobs])
            allu = allu[np.isfinite(allu)]
            vr = (float(np.floor(allu.min())), float(np.ceil(allu.max())))
            print(f"[utci_plot] common colour range {vr[0]:g} .. {vr[1]:g} degC")
        failed = len(a.case) - len(jobs)
        for c, inp in jobs:
            try:
                xy, u = _load_points(inp["csv"])
                c = Path(c).resolve()
                png = (a.out_dir / f"{c.parent.name}.png") if a.out_dir else (c / "postProcessing" / "utci_map.png")
                h = (inp["summary"] or {}).get("height_m")
                ttl = "UTCI at pedestrian height" + (f" (z = {h:g} m)" if h is not None else "") + f"  ·  {c.parent.name}"
                bu, bn, isb = (a.base_utci, a.base, False) if (a.base_utci is not None) else ((None, a.base, False) if a.no_base else _find_base(c, a.base))
                print(save_utci_map(png, xy, u, site_poly=inp["site"], buildings=inp["buildings"], trees=inp["trees"],
                                    summary=inp["summary"], vrange=vr, title=ttl, base_utci=bu, base_name=bn, is_base=isb, **common))
            except Exception as exc:
                failed += 1
                print(f"[utci_plot] {c}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1 if failed else 0

    if not (a.points_csv and a.out):
        ap.error("give either  points_csv out  or  --case CASE [CASE ...]")
    xy, u = _load_points(a.points_csv)
    site = b = t = None
    if a.layout:
        site, b, t = _layout_geometry(a.layout)
    summary = json.loads(a.json.read_text()) if a.json else None
    print(save_utci_map(a.out, xy, u, site_poly=site, buildings=b, trees=t, summary=summary,
                        vrange=tuple(a.range) if a.range else None, base_utci=None if a.no_base else a.base_utci, base_name=a.base, **common))
    return 0


if __name__ == "__main__":
    sys.exit(main())