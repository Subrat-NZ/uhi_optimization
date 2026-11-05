"""
Wind-model VTK (3D RectilinearGrid, 150x150x50)  ->  ONE 2D .npy : U, 512x512

Runs on the RunPod worker (worker/handler.py, task "wind") right after the wind model writes its VTK.

Usage:
    python3 vtk_to_2d.py <result.vtk> <out_U_2d.npy> [--meta geometry_meta.json]
                         [--res 512] [--z-index 1]

Output: sirf ek file, (H, W) float32
    value  = |U| velocity magnitude [m/s]  (VTK point-data `U_mag`)
    slice  = horizontal z-layer, --z-index 1 (second layer, pedestrian level;
             WindAI ki stateFileStream.py bhi yahi layer use karti hai)
    row 0  = y_max, last row = y_min ; col 0 = x_min, last col = x_max
             (rasterize_obj.py / field_to_2d_aligned.py jaisa)
    NaN    = building (solid) cells, aur VTK domain ke bahar ke pixels

--meta geometry_meta.json do to output geometry_labels.npy ke extent_xy pe resample
hota hai (pixel-to-pixel aligned). Bina --meta ke VTK ka apna x/y domain use hota hai.
"""
import os
import json
import argparse

import numpy as np

U_FIELD = "U_mag"


def _frac_index(coords, targets):
    n = len(coords)
    inside = (targets >= coords[0] - 1e-9) & (targets <= coords[-1] + 1e-9)
    t = np.clip(targets, coords[0], coords[-1])
    i1 = np.clip(np.searchsorted(coords, t, side="right"), 1, n - 1)
    i0 = i1 - 1
    span = coords[i1] - coords[i0]
    w1 = np.where(span > 0, (t - coords[i0]) / span, 0.0)
    return i0, i1, w1, inside


def resample(field2d, solid2d, xs, ys, tx, ty):
    """NaN-aware bilinear (sirf fluid corners); nearest cell solid -> NaN."""
    xi0, xi1, wx, in_x = _frac_index(xs, tx)
    yi0, yi1, wy, in_y = _frac_index(ys, ty)
    Y0, X0 = np.meshgrid(yi0, xi0, indexing="ij")
    Y1, X1 = np.meshgrid(yi1, xi1, indexing="ij")
    WY, WX = np.meshgrid(wy, wx, indexing="ij")

    num = np.zeros(WY.shape)
    den = np.zeros(WY.shape)
    for yy, xx, w in ((Y0, X0, (1 - WY) * (1 - WX)), (Y0, X1, (1 - WY) * WX),
                      (Y1, X0, WY * (1 - WX)), (Y1, X1, WY * WX)):
        v = field2d[yy, xx]
        ok = np.isfinite(v)
        num += np.where(ok, v * w, 0.0)
        den += np.where(ok, w, 0.0)
    out = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)

    Yn = np.where(WY >= 0.5, Y1, Y0)
    Xn = np.where(WX >= 0.5, X1, X0)
    out = np.where(solid2d[Yn, Xn] > 0.5, np.nan, out)
    out = np.where(in_y[:, None] & in_x[None, :], out, np.nan)
    return out.astype(np.float32)


def inlet_direction(uvec, solid, band=3, min_flux=0.05):
    """VTK domain ki boundary pe jahan hawa ANDAR aa rahi hai (inlet), wahan ka mean U vector.

    uvec  : (ny, nx, >=2)  U vector of the z-layer   (index 0 = y_min / x_min)
    solid : (ny, nx)       solid mask
    Poore domain ka mean nahi -- buildings ke recirculation se wo direction bigaad deta hai.
    """
    ux, uy = uvec[..., 0], uvec[..., 1]
    fluid = (solid <= 0.5) & np.isfinite(ux) & np.isfinite(uy)
    ny_, nx_ = fluid.shape
    b = max(1, min(band, ny_ // 4, nx_ // 4))
    regions = {  # edge: (slice, inward normal)
        "x_min": ((slice(None), slice(0, b)), (1.0, 0.0)),
        "x_max": ((slice(None), slice(nx_ - b, nx_)), (-1.0, 0.0)),
        "y_min": ((slice(0, b), slice(None)), (0.0, 1.0)),
        "y_max": ((slice(ny_ - b, ny_), slice(None)), (0.0, -1.0)),
    }
    flux = {}
    for name, (sl, (nxv, nyv)) in regions.items():
        f = fluid[sl]
        flux[name] = float((ux[sl][f] * nxv + uy[sl][f] * nyv).mean()) if f.any() else 0.0
    fmax = max(flux.values())
    edges = [k for k, v in flux.items() if fmax >= min_flux and v >= min_flux and v >= 0.3 * fmax]

    mask = np.zeros_like(fluid)
    for k in edges:
        m = np.zeros_like(fluid); m[regions[k][0]] = True
        mask |= m & fluid
    method = "inlet_boundary"
    if not mask.any():                       # koi clear inlet nahi -> domain mean
        mask, method, edges = fluid, "domain_mean", []
    mx, my = float(ux[mask].mean()), float(uy[mask].mean())
    return {
        "ux": mx, "uy": my,
        "direction_deg": float(np.degrees(np.arctan2(my, mx)) % 360.0),
        "speed_xy": float(np.hypot(mx, my)),
        "method": method, "inlet_edges_vtk": edges,
        "edge_inflow_flux": flux, "n_fluid": int(mask.sum()),
    }


def vtk_to_u_2d(vtk_path, out_npy, meta_path=None, res=512, z_index=1):
    import pyvista as pv  # lazy: --help bina pyvista ke bhi chale

    grid = pv.read(vtk_path)
    if not (hasattr(grid, "x") and hasattr(grid, "y") and hasattr(grid, "z")):
        raise ValueError(f"{vtk_path}: RectilinearGrid expected, got {type(grid).__name__}")
    nx, ny, nz = grid.dimensions
    xs = np.asarray(grid.x, dtype=np.float64)
    ys = np.asarray(grid.y, dtype=np.float64)
    if not 0 <= z_index < nz:
        raise ValueError(f"--z-index {z_index} out of range 0..{nz - 1}")

    def layer(name):
        if name not in grid.point_data:
            raise KeyError(f"'{name}' not in VTK point data: {list(grid.point_data.keys())}")
        a = np.asarray(grid.point_data[name], dtype=np.float64)
        if a.size != nx * ny * nz:
            raise ValueError(f"'{name}': size {a.size} != grid {nx}x{ny}x{nz}")
        return a.reshape((nz, ny, nx))[z_index]  # VTK order: x fastest

    u_layer, solid_layer = layer(U_FIELD), layer("solid")

    # ── flow direction from the VTK's U vector (mean over fluid points of this layer) ──
    # U_2d.npy sirf |U| hai (direction nahi), aur VTK baad me delete ho jati hai, isliye
    # direction yahin nikaal ke U_flow_direction.json me rakh rahe hain.
    flow_dir = None
    if "U" in grid.point_data:
        uvec = np.asarray(grid.point_data["U"], dtype=np.float64)
        if uvec.ndim == 2 and uvec.shape[1] >= 2 and uvec.shape[0] == nx * ny * nz:
            uvec = uvec.reshape((nz, ny, nx, uvec.shape[1]))[z_index]
            flow_dir = inlet_direction(uvec, solid_layer)
            flow_dir["z_index"] = int(z_index)
            flow_dir["convention"] = "direction the air moves TOWARD, deg CCW from +x (x=East, y=North)"
            # reference: poore domain ka mean (sirf info ke liye)
            fl = (solid_layer <= 0.5) & np.isfinite(uvec[..., 0]) & np.isfinite(uvec[..., 1])
            if fl.any():
                mx, my = float(uvec[..., 0][fl].mean()), float(uvec[..., 1][fl].mean())
                flow_dir["domain_mean_direction_deg"] = float(np.degrees(np.arctan2(my, mx)) % 360.0)

    if meta_path:
        with open(meta_path) as f:
            meta = json.load(f)
        x0, x1, y0, y1 = meta["extent_xy"]
        W, H = meta["resolution"]
    else:
        x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
        W = H = res
    tx = np.linspace(x0, x1, W)
    ty = np.linspace(y1, y0, H)  # row 0 = y_max

    # kitna VTK domain raster me aaya (100% = U cut nahi hua)
    ov_x = max(0.0, min(x1, xs[-1]) - max(x0, xs[0]))
    ov_y = max(0.0, min(y1, ys[-1]) - max(y0, ys[0]))
    used = (ov_x * ov_y) / max((xs[-1] - xs[0]) * (ys[-1] - ys[0]), 1e-12)
    print(f"[vtk2d] VTK domain x[{xs[0]:.2f},{xs[-1]:.2f}] y[{ys[0]:.2f},{ys[-1]:.2f}]  "
          f"raster extent x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]  -> {used:.1%} of VTK domain used")
    if used < 0.99:
        print(f"[vtk2d] WARNING: raster covers only {used:.1%} of the VTK domain -> outer part of U is CUT")

    arr = resample(u_layer, solid_layer, xs, ys, tx, ty)
    valid = float(np.isfinite(arr).mean())
    if not np.isfinite(arr).any():
        raise RuntimeError(
            "U 2D is all NaN: VTK domain does not overlap the geometry extent "
            f"(VTK x[{xs[0]:.2f},{xs[-1]:.2f}] y[{ys[0]:.2f},{ys[-1]:.2f}] vs "
            f"geometry x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]) -- coordinate frames differ?"
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_npy)), exist_ok=True)
    np.save(out_npy, arr)
    if flow_dir is not None:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(out_npy)), "U_flow_direction.json")
        with open(dir_path, "w") as f:
            json.dump(flow_dir, f, indent=2)
        print(f"[vtk2d] INLET flow direction {flow_dir['direction_deg']:.1f} deg via {flow_dir['method']} "
              f"edges={flow_dir['inlet_edges_vtk']} "
              f"(ux={flow_dir['ux']:.4f}, uy={flow_dir['uy']:.4f}) -> {dir_path}")
    else:
        print("[vtk2d] WARNING: no 'U' vector in VTK -> U_flow_direction.json not written")
    print(f"[vtk2d] U {arr.shape} {arr.dtype} valid={valid:.1%} "
          f"min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} -> {out_npy}")
    if valid < 0.5:
        print(f"[vtk2d] WARNING: only {valid:.1%} pixels valid -- VTK domain covers just part "
              "of the geometry extent (or coordinate frames differ). Check alignment.")
    return out_npy


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="wind VTK -> U 2D 512x512 npy")
    ap.add_argument("vtk", help="result.vtk (RectilinearGrid)")
    ap.add_argument("out_npy", help="output file, e.g. .../raster2d/U_2d.npy")
    ap.add_argument("--meta", default=None, help="geometry_meta.json -> align to geometry raster")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--z-index", type=int, default=1)
    a = ap.parse_args()
    vtk_to_u_2d(a.vtk, a.out_npy, a.meta, a.res, a.z_index)