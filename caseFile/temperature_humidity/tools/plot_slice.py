#!/usr/bin/env python3
"""
plot_slice.py  --  horizontal-slice images of |U|, T and RH at a fixed height above the ground (default 2 m)

Reads the LATEST time of a (reconstructed) airTqFoam run (T, q, and U from the same time or from 0/) and the
mesh, takes the cell that contains z = height in every vertical column (z0 <= height < z1), and paints it with
its value.  Style: whole domain, jet colour map from the minimum to the maximum of the slice, white where
there are no cells (buildings, tree crowns), colour bar titled with the quantity, "Max: ... / Min: ..." below it.

Writes   <out>/U_z2m.png   <out>/T_z2m.png   <out>/RH_z2m.png   (the "2m" follows --height)
and      <out>/slice_z2m_stats.json   (min / max / mean of every image, for the logs)

    python3 tools/plot_slice.py --case . --out images                 # 2 m
    python3 tools/plot_slice.py --case . --height 1.5 --dpi 200

Smooth contours
---------------
Default `--style contour`: the cell values (one per vertical column, at z = height) are interpolated linearly between the cell centres
(a triangulation in which no triangle bridges a building / tree void), smoothed over about half a local cell (`--smooth`, a smoothing that never mixes
values across a building) and drawn as filled contours with `--levels` levels (`--lines N` adds N thin iso-lines).  Buildings and tree crowns
stay white, and the "Max / Min" line, the statistics and the common colour limits use the REAL cell values, not the smoothed ones.
`--style pixels` gives the old picture (every cell painted with its own value).
    --style contour|pixels   --levels 64   --lines 0   --smooth 0.6 (x local cell size)   --contour-px 1000 (pixels along the longer side of the contour grid)

Common colour bars (same limits for every layout of an optimisation)
--------------------------------------------------------------------
The case of the reference layout (default `individual_0`, the base) writes the minimum / maximum of every field of its slice, widened
by 10 %, to  run/slice_limits_<season>_z<height>m.json.  Every other case (run/<name>/<season>/...) reads that file and uses the same
limits for its colour bar, so all images of a season are drawn on one scale.  If the file is not there yet, the case waits for it
(it is written by the reference case as soon as that case reaches this step).  Widening: 10 % of the value (|min| below, |max| above),
`--margin-mode span` takes 10 % of (max - min) instead; U is kept >= 0 and RH within 0 ... 100.  Values of a case that fall outside the
limits are painted in the end colour and the colour bar gets an arrow; the "Max / Min" line under the bar always shows the actual
values of that image.
    --limits-role auto|write|read|off     auto: write for the reference case, read for the others (found from the folder run/<name>/<season>/...)
    --limits-file PATH                    another place for the file (also $SLICE_LIMITS_FILE); --limits-role also from $SLICE_LIMITS_ROLE
    --reference individual_0              name of the reference layout
    --wait-timeout-min N                  stop waiting after N minutes (default 0 = wait until the file exists)

RH is computed from T, q and the pressure of weatherData/target.json (same formula as site_average.py).
The image is drawn on a pixel grid (nearest cell, kept only where the pixel lies inside the cell's own
footprint), so even meshes with millions of cells give an image in a few seconds.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foam_io  # noqa: E402
from site_average import cell_geometry, e_sat_pa  # noqa: E402


def cell_footprints(mesh):
    """x/y bounding box of every cell (from the points of its faces)."""
    pts, faces = mesh["points"], mesh["faces"]
    own, nei = mesh["owner"], mesh["neighbour"]
    ncell = int(max(own.max(), nei.max() if len(nei) else 0)) + 1
    nint = len(nei)
    fb = np.empty((len(faces), 4))
    nv = np.fromiter((len(f) for f in faces), dtype=int, count=len(faces))
    for L in np.unique(nv):                       # faces with the same number of vertices are handled together
        ids = np.flatnonzero(nv == L)
        idx = np.array([faces[i] for i in ids]).reshape(len(ids), L)
        px, py = pts[idx, 0], pts[idx, 1]
        fb[ids] = np.column_stack([px.min(1), px.max(1), py.min(1), py.max(1)])
    x0 = np.full(ncell, np.inf)
    x1 = np.full(ncell, -np.inf)
    y0 = np.full(ncell, np.inf)
    y1 = np.full(ncell, -np.inf)
    for ids, cells in ((np.arange(len(faces)), own), (np.arange(nint), nei)):
        np.minimum.at(x0, cells, fb[ids, 0])
        np.maximum.at(x1, cells, fb[ids, 1])
        np.minimum.at(y0, cells, fb[ids, 2])
        np.maximum.at(y1, cells, fb[ids, 3])
    return x0, x1, y0, y1


def rasterise(cen, box, values, extent, width_px, return_index=False):
    """Pixel image (ny, nx) of `values` for the selected cells; NaN where no cell covers the pixel."""
    from scipy.spatial import cKDTree
    xmin, xmax, ymin, ymax = extent
    w, h = xmax - xmin, ymax - ymin
    nx = int(width_px) if w >= h else max(int(round(width_px * w / h)), 2)
    ny = int(width_px) if h > w else max(int(round(width_px * h / w)), 2)
    xs = xmin + (np.arange(nx) + 0.5) * w / nx
    ys = ymin + (np.arange(ny) + 0.5) * h / ny
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    tree = cKDTree(cen[:, :2])
    k = min(6, len(cen))
    _, idx = tree.query(P, k=k)
    idx = idx.reshape(len(P), -1)
    x0, x1, y0, y1 = box
    img = np.full(len(P), np.nan)
    cidx = np.full(len(P), -1, dtype=np.int64)
    todo = np.ones(len(P), dtype=bool)
    tol = 1e-6 * max(w, h)
    for j in range(idx.shape[1]):
        c = idx[todo, j]
        p = P[todo]
        hit = (p[:, 0] >= x0[c] - tol) & (p[:, 0] <= x1[c] + tol) & (p[:, 1] >= y0[c] - tol) & (p[:, 1] <= y1[c] + tol)
        where = np.flatnonzero(todo)[hit]
        img[where] = values[c[hit]]
        cidx[where] = c[hit]
        todo[where] = False
        if not todo.any():
            break
    if return_index:
        return img.reshape(ny, nx), cidx.reshape(ny, nx)
    return img.reshape(ny, nx)


class SmoothField:
    """linear interpolation of the cell values on a pixel grid, without bridging voids (buildings, tree crowns)"""

    def __init__(self, cen, box, extent, width_px, smooth_px):
        import matplotlib.tri as mtri
        from scipy.spatial import cKDTree
        self.extent = extent
        self.box, self.cen = box, cen
        xmin, xmax, ymin, ymax = extent
        w, h = xmax - xmin, ymax - ymin
        self.nx = int(width_px) if w >= h else max(int(round(width_px * w / h)), 2)
        self.ny = int(width_px) if h > w else max(int(round(width_px * h / w)), 2)
        self.xs = xmin + (np.arange(self.nx) + 0.5) * w / self.nx
        self.ys = ymin + (np.arange(self.ny) + 0.5) * h / self.ny
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.smooth = smooth_px
        x0, x1, y0, y1 = box
        size = np.maximum(x1 - x0, y1 - y0)                                   # size of every cell
        tri = mtri.Triangulation(cen[:, 0], cen[:, 1])
        t = tri.triangles
        edge = np.zeros(len(t))
        for i, j in ((0, 1), (1, 2), (2, 0)):                                 # an edge much longer than the cells it joins crosses a void
            d = np.hypot(cen[t[:, i], 0] - cen[t[:, j], 0], cen[t[:, i], 1] - cen[t[:, j], 1])
            edge = np.maximum(edge, d / (0.5 * (size[t[:, i]] + size[t[:, j]])))
        tri.set_mask(edge > 2.6)
        self.tri = tri
        self.mtri = mtri
        # neighbours of every cell (not across a void) for a least-squares gradient: used where the triangulation does not reach a pixel
        # (the strip between the outermost cell centres and the domain / building wall), so that those strips are not painted flat
        k = min(9, len(cen))
        d, nb = cKDTree(cen).query(cen, k=k)
        d, nb = d[:, 1:], nb[:, 1:]
        ok = d <= 2.6 * 0.5 * (size[:, None] + size[nb])
        self.nb, self.ok = nb, ok
        dx, dy = cen[nb, 0] - cen[:, 0:1], cen[nb, 1] - cen[:, 1:2]
        w = np.where(ok, 1.0 / np.maximum(d, 1e-9) ** 2, 0.0)
        self.dx, self.dy, self.w = dx, dy, w
        sxx, sxy, syy = (w * dx * dx).sum(1), (w * dx * dy).sum(1), (w * dy * dy).sum(1)
        det = sxx * syy - sxy ** 2
        self.det_ok = det > 1e-12 * np.maximum(sxx * syy, 1e-30)
        self.inv = (syy, -sxy, sxx, np.where(self.det_ok, det, 1.0))
        self.size = size

    def _smooth(self, Z, cell_size):
        """Gaussian smoothing whose width follows the local cell size (sigma = `smooth` x cell size): coarse far-field cells are smoothed over their
        own size (no kinks in the contours), the fine cells next to the buildings keep their detail.  Normalised by the mask, so values never mix
        across a building or a tree crown."""
        from scipy.ndimage import gaussian_filter
        px = (self.extent[1] - self.extent[0]) / self.nx
        want = self.smooth * cell_size / px                                   # wanted sigma in pixels, per pixel
        sig = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        want = np.clip(np.where(np.isfinite(want), want, 0.0), 0.0, sig[-1])
        m = np.isfinite(Z)
        z0 = np.where(m, Z, 0.0)
        levels = [Z]
        for sg in sig[1:]:
            num, den = gaussian_filter(z0, sg, mode="nearest"), gaussian_filter(m.astype(float), sg, mode="nearest")
            levels.append(np.where(m, num / np.maximum(den, 1e-9), np.nan))
        k = np.clip(np.searchsorted(sig, want, side="right") - 1, 0, len(sig) - 2)
        t = np.clip((want - sig[k]) / (sig[k + 1] - sig[k]), 0.0, 1.0)
        L = np.stack(levels)
        ar = np.arange(Z.size)
        a_ = L.reshape(len(sig), -1)[k.ravel(), ar]
        b_ = L.reshape(len(sig), -1)[k.ravel() + 1, ar]
        return (a_ * (1 - t.ravel()) + b_ * t.ravel()).reshape(Z.shape)

    def gradient(self, values):
        dv = values[self.nb] - values[:, None]
        bx, by = (self.w * self.dx * dv).sum(1), (self.w * self.dy * dv).sum(1)
        syy, mxy, sxx, det = self.inv
        gx = np.where(self.det_ok, (syy * bx + mxy * by) / det, 0.0)
        gy = np.where(self.det_ok, (mxy * bx + sxx * by) / det, 0.0)
        return gx, gy

    def image(self, values):
        near, idx = rasterise(self.cen, self.box, values, self.extent, max(self.nx, self.ny), return_index=True)    # nearest cell inside its footprint
        if near.shape != self.X.shape:
            raise RuntimeError("grid mismatch")
        cover = idx >= 0
        lin = self.mtri.LinearTriInterpolator(self.tri, values)(self.X, self.Y)
        lin = np.ma.filled(lin.astype(float), np.nan)
        gx, gy = self.gradient(values)
        ii = np.where(cover, idx, 0)
        ext = values[ii] + gx[ii] * (self.X - self.cen[ii, 0]) + gy[ii] * (self.Y - self.cen[ii, 1])      # the cell's own value, continued with its gradient
        half = 0.5 * self.size[ii]
        ext = np.clip(ext, values[ii] - 2 * np.abs(gx[ii]) * half - 2 * np.abs(gy[ii]) * half, values[ii] + 2 * np.abs(gx[ii]) * half + 2 * np.abs(gy[ii]) * half)
        Z = np.where(np.isfinite(lin), lin, ext)
        Z = np.where(cover, Z, np.nan)
        if self.smooth and self.smooth > 0:
            Z = self._smooth(Z, np.where(cover, self.size[ii], np.nan))
        return np.clip(Z, np.nanmin(values), np.nanmax(values)), near


def draw(img, extent, title, label, path, dpi, fmt="{:.2f}", vmin=None, vmax=None, style="pixels", true_range=None, levels=64, lines=0):
    """paints img; vmin / vmax (both given) fix the colour range, otherwise the range of the image.  Returns the colour range used.
    style 'pixels': img is the cell picture;  style 'contour': img is the smoothed grid (filled contours), true_range = (min, max) of the real cell values"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = img[np.isfinite(img)]
    dmin, dmax = (float(true_range[0]), float(true_range[1])) if true_range is not None else (float(finite.min()), float(finite.max()))   # actual values
    lo = dmin if vmin is None else vmin
    hi = dmax if vmax is None else vmax
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    xmin, xmax, ymin, ymax = extent
    ratio = (ymax - ymin) / (xmax - xmin)
    fig_w = 10.0
    fig_h = min(max(fig_w * 0.86 * ratio + 2.1, 4.0), 16.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    below, above = dmin < lo - 1e-9, dmax > hi + 1e-9                     # values outside a fixed range: arrow on the colour bar
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    if style == "contour":
        ny, nx = img.shape
        xs = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
        ys = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
        Z = np.ma.masked_invalid(img)
        lv = np.linspace(lo, hi, int(levels) + 1)
        im = ax.contourf(xs, ys, Z, levels=lv, cmap=cmap, extend=extend, antialiased=False)
        if lines and lines > 0:
            ax.contour(xs, ys, Z, levels=np.linspace(lo, hi, int(lines) + 2)[1:-1], colors="k", linewidths=0.35, alpha=0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
    else:
        im = ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=(xmin, xmax, ymin, ymax), cmap=cmap,
                       vmin=lo, vmax=hi, interpolation="nearest", aspect="equal")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, aspect=40, shrink=0.7, extend=extend)
    if style == "contour":
        cb.set_ticks([t for t in matplotlib.ticker.MaxNLocator(8).tick_values(lo, hi) if lo - 1e-9 <= t <= hi + 1e-9])
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=8)
    cb.ax.text(0.5, -3.3, f"Max: {fmt.format(dmax)} / Min: {fmt.format(dmin)}", ha="center", va="top", fontsize=10,
               transform=cb.ax.transAxes)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return dmin, dmax, lo, hi


# ------------------------------------------------------------------------------------------------ common colour limits
def case_identity(case):
    """(run folder, layout name, season) when the case lies in  <run>/<name>/<season>/...  (run, run_full ...), else (None, None, None)"""
    parts = case.parts
    for i in range(len(parts) - 1, -1, -1):
        if re.fullmatch(r"run(_.+)?", parts[i]) and i + 2 < len(parts):
            return Path(*parts[:i + 1]), parts[i + 1], parts[i + 2]
    return None, None, None


def widen(lo, hi, name, pct, mode):
    """colour limits from the minimum / maximum of the reference image: |min| / |max| (mode 'value') or the span (mode 'span') times pct % outwards"""
    f = pct / 100.0
    if mode == "span":
        m = f * (hi - lo)
        vmin, vmax = lo - m, hi + m
    else:
        vmin, vmax = lo - f * abs(lo), hi + f * abs(hi)
    if name == "U":
        vmin = max(vmin, 0.0)
    if name == "RH":
        vmin, vmax = max(vmin, 0.0), min(vmax, 100.0)
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9
    return float(vmin), float(vmax)


def write_limits(path, doc):
    """atomic write: a reader never sees a half-written file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


def wait_for_limits(path, who, timeout_min, poll_s=10.0):
    """block until the reference case has written the limits file; returns its content"""
    t0 = time.time()
    last = -1e9
    while True:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except (ValueError, OSError):
                pass                                                        # being replaced: try again
        waited = time.time() - t0
        if timeout_min and waited > 60.0 * timeout_min:
            raise SystemExit(f"[plot_slice] gave up waiting for {path} after {timeout_min:g} min (the reference layout has not written it)")
        if waited - last >= 60.0:
            print(f"[plot_slice] {who}: waiting for the reference limits {path}  ({waited / 60:.0f} min)", flush=True)
            last = waited
        time.sleep(poll_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=Path("."))
    ap.add_argument("--time", default=None, help="time directory (default: latest)")
    ap.add_argument("--target", type=Path, default=None, help="weatherData/target.json (pressure for RH)")
    ap.add_argument("--height", type=float, default=2.0, help="height above the ground [m] (default 2)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default <case>/images)")
    ap.add_argument("--width-px", type=int, default=2000, help="pixels along the longer side (default 2000)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--style", choices=["contour", "pixels"], default="contour",
                    help="contour: smooth filled contours (default); pixels: every cell painted with its own value")
    ap.add_argument("--levels", type=int, default=64, help="contour style: number of filled-contour levels (default 64)")
    ap.add_argument("--lines", type=int, default=0, help="contour style: number of thin iso-lines drawn over the colours (default 0)")
    ap.add_argument("--smooth", type=float, default=0.6, help="contour style: Gaussian smoothing, sigma = this x the local cell size (default 0.6, 0 = off); never across a building")
    ap.add_argument("--contour-px", type=int, default=1000, help="contour style: pixels along the longer side of the interpolation grid (default 1000)")
    ap.add_argument("--fields", nargs="+", default=["U", "T", "RH"], choices=["U", "T", "RH"])
    ap.add_argument("--limits-role", choices=["auto", "write", "read", "off"], default=os.environ.get("SLICE_LIMITS_ROLE", "auto"),
                    help="common colour limits: write = this is the reference case, read = take the limits of the reference (wait for them), "
                         "off = colour range of the image itself, auto = from the folder run/<name>/<season>/ (default)")
    ap.add_argument("--limits-file", type=Path, default=os.environ.get("SLICE_LIMITS_FILE"),
                    help="file with the limits (default <run>/slice_limits_<season>_z<height>m.json)")
    ap.add_argument("--reference", default="individual_0", help="layout whose slices define the limits (default individual_0)")
    ap.add_argument("--margin-pct", type=float, default=10.0, help="widening of the reference min / max (default 10 %%)")
    ap.add_argument("--margin-mode", choices=["value", "span"], default="value",
                    help="value: 10 %% of |min| and |max|; span: 10 %% of (max - min) (default value)")
    ap.add_argument("--wait-timeout-min", type=float, default=0.0, help="give up waiting for the reference limits after N minutes (default 0 = never)")
    a = ap.parse_args(argv)
    case = a.case.resolve()
    out = (a.out or case / "images")
    out.mkdir(parents=True, exist_ok=True)
    tname = a.time or foam_io.latest_time(case)
    if tname is None:
        raise SystemExit("no time directory found")
    tdir = case / tname
    mesh = foam_io.read_mesh(case, with_cells=True)
    cen, vol, z0, z1, _ = cell_geometry(mesh)
    n = len(vol)
    h = a.height
    # one cell per vertical column: z0 <= h < z1 (a cell that ends exactly at h belongs to the cell above)
    sel = np.flatnonzero((z0 <= h) & (z1 > h))
    if len(sel) == 0:
        raise SystemExit(f"no cells contain z = {h} m")
    box_all = cell_footprints(mesh)
    box = tuple(b[sel] for b in box_all)
    pts = mesh["points"]
    extent = (float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max()))

    T = foam_io.read_internal_field(tdir / "T", n) - 273.15
    q = foam_io.read_internal_field(tdir / "q", n)
    upath = next((p for p in (tdir / "U", case / "0" / "U") if p.is_file()), None)
    if upath is None:
        raise SystemExit(f"no U found in {tdir} or {case / '0'}")
    spd = np.linalg.norm(foam_io.read_internal_field(upath, n, rank="vector"), axis=1)
    tpath = Path(a.target) if a.target else case / "weatherData/target.json"
    p_pa = json.loads(tpath.read_text())["boundary"]["pressure_Pa"] if tpath.is_file() else 101325.0
    e = q * p_pa / (0.622 + 0.378 * q)
    RH = 100.0 * e / e_sat_pa(T)

    hs = f"{h:g}".replace(".", "p")
    spec = {"U": (spd, "U [m/s]", "|U|", "{:.2f}"),
            "T": (T, "T [°C]", "T", "{:.2f}"),
            "RH": (RH, "RH [%]", "RH", "{:.1f}")}
    imgs = {name: rasterise(cen[sel], box, spec[name][0][sel], extent, a.width_px) for name in a.fields}     # the real cell values (limits, statistics)
    smooth_imgs = {}
    if a.style == "contour":
        sf = SmoothField(cen[sel], box, extent, a.contour_px, a.smooth)
        for name in a.fields:
            smooth_imgs[name], _ = sf.image(spec[name][0][sel])

    # ---- which colour limits: own range, written (reference case) or read (all other cases)
    run_dir, lname, season = case_identity(case)
    role = a.limits_role
    lfile = Path(a.limits_file) if a.limits_file else (run_dir / f"slice_limits_{season}_z{hs}m.json" if run_dir else None)
    if role == "auto":
        if lname is None:                                                    # not in run/<name>/<season>/ : cannot tell reference from the others
            role = "off"
            print("[plot_slice] case is not in run/<name>/<season>/ : own colour range (use --limits-role write | read with --limits-file for common limits)")
        else:
            role = "write" if lname == a.reference else "read"
    if role in ("write", "read") and lfile is None:
        raise SystemExit("[plot_slice] --limits-role write / read needs --limits-file (the case is not in run/<name>/<season>/)")
    limits = {}
    if role == "write":
        doc = {"reference": lname or a.reference, "season": season, "height_m": h, "time": tname, "margin_pct": a.margin_pct,
               "margin_mode": a.margin_mode, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "fields": {}}
        for name, img in imgs.items():
            fin = img[np.isfinite(img)]
            rmin, rmax = float(fin.min()), float(fin.max())
            vmin, vmax = widen(rmin, rmax, name, a.margin_pct, a.margin_mode)
            doc["fields"][name] = dict(min=rmin, max=rmax, vmin=vmin, vmax=vmax)
            limits[name] = (vmin, vmax)
        write_limits(lfile, doc)
        print(f"[plot_slice] wrote the common colour limits -> {lfile}")
    elif role == "read":
        doc = wait_for_limits(lfile, f"{lname or case.name}", a.wait_timeout_min)
        for name in imgs:
            f = doc.get("fields", {}).get(name)
            if f is None:
                print(f"[plot_slice] {lfile.name} has no limits for {name}: own colour range used for it")
                continue
            limits[name] = (float(f["vmin"]), float(f["vmax"]))
        print(f"[plot_slice] colour limits of {doc.get('reference', a.reference)} from {lfile}")

    stats = {"time": tname, "height_m": h, "cells": int(len(sel)), "limits": {"role": role, "file": str(lfile) if role in ("write", "read") else None},
             "fields": {}}
    for name in a.fields:
        vals, label, short, fmt = spec[name]
        img = imgs[name]
        path = out / f"{name}_z{hs}m.png"
        vmin, vmax = limits.get(name, (None, None))
        fin = img[np.isfinite(img)]
        if a.style == "contour":
            dmin, dmax, lo, hi = draw(smooth_imgs[name], extent, f"{short} at z = {h:g} m above ground   (time {tname})", label, path, a.dpi, fmt, vmin, vmax,
                                      style="contour", true_range=(float(fin.min()), float(fin.max())), levels=a.levels, lines=a.lines)
        else:
            dmin, dmax, lo, hi = draw(img, extent, f"{short} at z = {h:g} m above ground   (time {tname})", label, path, a.dpi, fmt, vmin, vmax)
        stats["fields"][name] = dict(file=path.name, min=dmin, max=dmax, colour_min=lo, colour_max=hi, mean=float(np.mean(vals[sel])),
                                     covered_pct=float(100 * np.isfinite(img).mean()))
        print(f"[plot_slice] {path}   min {dmin:.3f}  max {dmax:.3f}   colour bar {lo:.3f} ... {hi:.3f}")
    (out / f"slice_z{hs}m_stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())