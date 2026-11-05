#!/usr/bin/env python3
"""
Facade pressure coefficient

            p - p_ref
    Cp = ---------------      q_ref = 1/2 * rho * U_ref^2
            q_ref

for the buildings in an OBJ, using the pressure field of a rectilinear VTK.

How it works
------------
1. The OBJ (one `o` object per building) is refined (longest-edge bisection) until each triangle is
   no larger than a right triangle with legs --max-edge (default 0.8 m), giving vertical as
   well as horizontal resolution.
   Faces are classified by their outward normal: wall (facade), roof, ground.
2. Pressure is sampled at  centroid + offset * normal  (default offset 1 m, about one
   grid cell outside the wall) by trilinear interpolation.  Solid (NaN) nodes
   are replaced by their nearest fluid value first, and the share of each
   interpolation stencil that touched solid is reported (`solid_frac`).
3. p_ref  : "inlet" (default) = mean of the pressure field on the inlet nodes, found
           with inlet.identify_inlet (U.n < 0); or give a number in Pa (e.g. 101325).
   U_ref  : mean |U| over the inlet nodes at a reference height.
           --ref-mode global   one U_ref for all buildings, taken at --uref-height
                               (default: roof height of the tallest building)
           --ref-mode building each building uses U_ref at its own roof height
           or fix it directly with --uref <m/s>.

Outputs: a VTK PolyData (cell data Cp, p, area, building, surface, solid_frac, normal),
a per-face CSV, and a per-building summary printed to the screen.

    python facade_cp.py pressure_out.vtk --obj buildings_cylinders.obj --pressure-field p_calc
    python facade_cp.py fa44c42d-sample_315.vtk --obj buildings_cylinders.obj --pressure-field p
"""
import argparse
import csv
import re
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

from vtkio import read_rectilinear_vtk
from inlet import identify_inlet

RHO_AIR = 1.225


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def read_obj_objects(path):
    """Parse an OBJ -> {name: (vertices (n,3), triangles (m,3))}; polygons are fan-triangulated."""
    verts, objs, cur = [], {}, "object_1"
    for ln in open(path):
        t = ln.split()
        if not t:
            continue
        if t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] in ("o", "g") and len(t) > 1:
            cur = t[1]
            objs.setdefault(cur, [])
        elif t[0] == "f":
            ids = [int(s.split("/")[0]) for s in t[1:]]
            ids = [i - 1 if i > 0 else len(verts) + i for i in ids]
            tris = [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
            objs.setdefault(cur, []).extend(tris)
    V = np.array(verts)
    out = {}
    for name, tris in objs.items():
        if not tris:
            continue
        F = np.array(tris)
        used, inv = np.unique(F, return_inverse=True)
        v, f = V[used], inv.reshape(F.shape)
        vol = np.einsum("ij,ij->", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])) / 6.0
        if vol < 0:                       # make normals point outward
            f = f[:, ::-1]
        out[name] = (v, f)
    return out


def refine(v, f, max_edge):
    """Longest-edge bisection until every triangle has area <= 0.5*max_edge**2 (the area of a
    right triangle with legs max_edge).  An area rule, not an edge rule, so long thin triangles
    are not chopped into slivers.  Orientation is preserved; vertices are not welded
    (only cell data is used downstream)."""
    tri = v[f]                                           # (m, 3, 3) triangle corner coordinates
    done = []
    while len(tri):
        e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], 1)
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        big = area > 0.5 * max_edge**2
        done.append(tri[~big])
        tri, e = tri[big], e[big]
        if not len(tri):
            break
        k = e.argmax(1)                                  # longest edge = vertex k -> k+1
        idx = (np.arange(3)[None, :] + k[:, None]) % 3
        t = np.take_along_axis(tri, idx[:, :, None], axis=1)
        m = 0.5 * (t[:, 0] + t[:, 1])
        tri = np.concatenate([np.stack([t[:, 0], m, t[:, 2]], 1), np.stack([m, t[:, 1], t[:, 2]], 1)])
    T = np.concatenate(done)
    return T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3)


def face_geometry(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(2 * area[:, None], 1e-300)
    return (a + b + c) / 3.0, n, area


# ----------------------------------------------------------------------------
# fields
# ----------------------------------------------------------------------------
def make_sampler(x, y, z, field, fluid):
    """Trilinear sampler of `field` (nz,ny,nx); NaN solid nodes -> nearest fluid value.
    Returns f(points (N,3) in x,y,z) -> (values, solid_fraction_of_stencil)."""
    idx = ndi.distance_transform_edt(~fluid, return_indices=True)[1]
    filled = field[tuple(idx)]
    fi = RegularGridInterpolator((z, y, x), filled, bounds_error=False, fill_value=None)
    si = RegularGridInterpolator((z, y, x), (~fluid).astype(float), bounds_error=False, fill_value=None)

    def sample(P):
        q = P[:, ::-1]                    # (x,y,z) -> (z,y,x)
        return fi(q), si(q)
    return sample


def inlet_reference(x, y, z, U, fluid, inlet_mask, href):
    """Mean |U| over inlet nodes at height href (profile built level by level), and mean wind direction there."""
    mag = np.linalg.norm(np.where(fluid[..., None], U, 0.0), axis=-1)
    prof, dirs, zz = [], [], []
    for k in range(len(z)):
        m = inlet_mask[k]
        if m.any():
            prof.append(mag[k][m].mean())
            dirs.append(np.where(fluid[k][..., None], U[k], 0.0)[m].mean(0))
            zz.append(z[k])
    prof, dirs, zz = np.array(prof), np.array(dirs), np.array(zz)
    u = float(np.interp(href, zz, prof))
    d = np.array([np.interp(href, zz, dirs[:, i]) for i in range(3)])
    d[2] = 0.0
    d /= max(np.linalg.norm(d), 1e-12)
    return u, d


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------
def write_polydata(path, V, F, cell_scalars, cell_vectors):
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nfacade Cp\nASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(V)} float\n")
        np.savetxt(fh, V, fmt="%.5f")
        fh.write(f"POLYGONS {len(F)} {4 * len(F)}\n")
        np.savetxt(fh, np.c_[np.full(len(F), 3), F], fmt="%d")
        fh.write(f"CELL_DATA {len(F)}\n")
        for name, a in cell_vectors.items():
            fh.write(f"VECTORS {name} float\n")
            np.savetxt(fh, a, fmt="%.6g")
        for name, a in cell_scalars.items():
            fh.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(fh, np.asarray(a, float), fmt="%.6g")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vtk", help="VTK with the pressure and velocity fields")
    ap.add_argument("--obj", required=True, help="OBJ with one object per building")
    ap.add_argument("--building-pattern", default="(?i)building",
                    help="regular expression selecting OBJ object names to include")
    ap.add_argument("--pressure-field", default="p")
    ap.add_argument("--velocity", default="U")
    ap.add_argument("--rho", type=float, default=RHO_AIR)
    ap.add_argument("--p-ref", default="inlet", help="'inlet' (mean pressure on inlet nodes) or a number in Pa")
    ap.add_argument("--uref", type=float, default=None, help="fixed reference velocity [m/s] (overrides the inlet profile)")
    ap.add_argument("--uref-height", type=float, default=None, help="height for U_ref in global mode (default: tallest roof)")
    ap.add_argument("--ref-mode", choices=["global", "building"], default="global")
    ap.add_argument("--offset", type=float, default=1.0, help="sampling distance off the wall [m]")
    ap.add_argument("--max-edge", type=float, default=0.8, help="target face size [m]: triangles are split until area <= 0.5*max_edge^2")
    ap.add_argument("--exclude-faces", nargs="*", default=["zmin"], help="boundary faces ignored by the inlet search")
    ap.add_argument("-o", "--output", default="facade_cp.vtk")
    ap.add_argument("--csv", default="facade_cp.csv")
    ap.add_argument("--summary", default="facade_cp_summary.txt", help="per-building averaged Cp report")
    a = ap.parse_args()

    d = read_rectilinear_vtk(a.vtk)
    x, y, z = d["x"], d["y"], d["z"]
    U = d["fields"][a.velocity]
    P = d["fields"][a.pressure_field]
    fluid = np.isfinite(U).all(-1) & np.isfinite(P)
    P = np.where(fluid, P, np.nan)

    # --- inlet, p_ref, U_ref ------------------------------------------------------------------
    inl = identify_inlet(x, y, z, U, fluid, exclude_faces=tuple(a.exclude_faces))
    if not inl.mask.any():
        raise RuntimeError("No inlet nodes found in the VTK; cannot determine the Cp reference.")
    p_ref = float(np.nanmean(P[inl.mask])) if a.p_ref == "inlet" else float(a.p_ref)
    print(f"pressure field '{a.pressure_field}': inlet nodes {int(inl.mask.sum())}, "
          f"mean p on inlet {np.nanmean(P[inl.mask]):.2f} Pa  ->  p_ref = {p_ref:.2f} Pa")

    # --- buildings ----------------------------------------------------------------------------
    try:
        building_pattern = re.compile(a.building_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --building-pattern: {exc}") from exc
    objs = {
        name: obj for name, obj in read_obj_objects(a.obj).items()
        if building_pattern.search(name)
    }
    if not objs:
        raise RuntimeError(
            f"No OBJ objects matched --building-pattern {a.building_pattern!r} in {a.obj}"
        )
    roof_h = {n: float(v[:, 2].max()) for n, (v, f) in objs.items()}
    href_global = a.uref_height if a.uref_height is not None else max(roof_h.values())

    def uref_for(href):
        if a.uref is not None:
            _, dd = inlet_reference(x, y, z, U, fluid, inl.mask, href)
            return a.uref, dd
        return inlet_reference(x, y, z, U, fluid, inl.mask, href)

    sample = make_sampler(x, y, z, P, fluid)

    allV, allF, rows = [], [], []
    cs = {k: [] for k in ["Cp", "p", "area", "building", "surface", "solid_frac", "q_ref"]}
    normals = []
    summary = []
    for bi, (name, (v, f)) in enumerate(objs.items(), 1):
        v, f = refine(v, f, a.max_edge)
        c, n, area = face_geometry(v, f)
        surf = np.where(n[:, 2] > 0.7, 1, np.where(n[:, 2] < -0.7, 2, 0))   # 0 wall, 1 roof, 2 ground
        href = href_global if a.ref_mode == "global" else roof_h[name]
        uref, wind = uref_for(href)
        q = 0.5 * a.rho * uref**2
        if not np.isfinite(q) or q <= 0.0:
            raise RuntimeError(f"Invalid reference dynamic pressure for building {name!r}: U_ref={uref}")
        keep = surf != 2
        pts = c[keep] + a.offset * n[keep]
        p_s, sfrac = sample(pts)
        cp = (p_s - p_ref) / q
        off = sum(len(q) for q in allV)
        allV.append(v); allF.append(f[keep] + off)
        for key, val in [("Cp", cp), ("p", p_s), ("area", area[keep]), ("building", np.full(keep.sum(), bi)),
                         ("surface", surf[keep]), ("solid_frac", sfrac), ("q_ref", np.full(keep.sum(), q))]:
            cs[key].append(val)
        normals.append(n[keep])
        for i in range(keep.sum()):
            rows.append([name, *c[keep][i], *n[keep][i], area[keep][i], surf[keep][i], p_s[i], cp[i], sfrac[i]])

        w = surf[keep] == 0
        A = area[keep]
        wind_dot = n[keep] @ wind
        wm = lambda m: float(np.sum(cp[m] * A[m]) / np.sum(A[m])) if m.any() else np.nan
        summary.append(dict(name=name, H=roof_h[name], href=href, uref=uref, q=q,
                            wall_mean=wm(w), wall_min=float(cp[w].min()) if w.any() else np.nan,
                            wall_max=float(cp[w].max()) if w.any() else np.nan,
                            wall_area=float(A[w].sum()),
                            windward=wm(w & (wind_dot < 0)), leeward=wm(w & (wind_dot >= 0)),
                            roof=wm(surf[keep] == 1), n_faces=int(keep.sum()),
                            bad=float(np.mean(sfrac[w] > 0.02)), wind=wind))

    # --- report -------------------------------------------------------------------------------
    print(f"\nCp = (p - {p_ref:.2f}) / (0.5*{a.rho}*U_ref^2),  offset {a.offset} m, ref-mode {a.ref_mode}\n")
    print(f"{'building':11s}{'H[m]':>6s}{'Uref':>6s}{'q[Pa]':>7s}{'Cp wall':>9s}{'windward':>10s}{'leeward':>9s}"
          f"{'Cp max':>8s}{'Cp min':>8s}{'roof':>7s}{'faces':>7s}{'stencil in solid':>18s}")
    for s in summary:
        print(f"{s['name']:11s}{s['H']:6.1f}{s['uref']:6.2f}{s['q']:7.2f}{s['wall_mean']:9.3f}{s['windward']:10.3f}"
              f"{s['leeward']:9.3f}{s['wall_max']:8.2f}{s['wall_min']:8.2f}{s['roof']:7.2f}{s['n_faces']:7d}"
              f"{100 * s['bad']:17.1f}%")
    print(f"\nmean wind direction (inlet, horizontal) at reference height: "
          f"({summary[0]['wind'][0]:+.2f}, {summary[0]['wind'][1]:+.2f})")
    print("windward/leeward = wall faces with n.u_wind < 0 / >= 0;  Cp columns are area-weighted wall means")
    print("'stencil in solid' = share of wall samples whose interpolation touched solid nodes (raise --offset if large)")

    # --- write --------------------------------------------------------------------------------
    V = np.concatenate(allV); F = np.concatenate(allF)
    write_polydata(a.output, V, F, {k: np.concatenate(v) for k, v in cs.items()}, {"normal": np.concatenate(normals)})
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["building", "x", "y", "z", "nx", "ny", "nz", "area_m2", "surface(0wall,1roof)", "p_Pa", "Cp", "solid_frac"])
        w.writerows(rows)
    total_wall_area = sum(s["wall_area"] for s in summary)
    overall_wall_cp = (
        sum(s["wall_mean"] * s["wall_area"] for s in summary if np.isfinite(s["wall_mean"]))
        / sum(s["wall_area"] for s in summary if np.isfinite(s["wall_mean"]))
        if any(np.isfinite(s["wall_mean"]) and s["wall_area"] > 0 for s in summary)
        else np.nan
    )
    with open(a.summary, "w") as fh:
        fh.write(f"Pressure reference: {p_ref:.6f} Pa\n")
        fh.write(f"Density: {a.rho:.6f} kg/m^3\n")
        fh.write(f"Reference mode: {a.ref_mode}\n")
        fh.write(f"Overall area-weighted wall Cp: {overall_wall_cp:.8g}\n")
        fh.write(f"Total wall area: {total_wall_area:.6f} m^2\n\n")
        fh.write("building\theight_m\turef_m_s\tdynamic_pressure_Pa\twall_mean_Cp\t"
                 "windward_mean_Cp\tleeward_mean_Cp\twall_min_Cp\twall_max_Cp\t"
                 "roof_mean_Cp\twall_area_m2\twall_faces\n")
        for s in summary:
            fh.write(
                f"{s['name']}\t{s['H']:.6g}\t{s['uref']:.6g}\t{s['q']:.6g}\t"
                f"{s['wall_mean']:.8g}\t{s['windward']:.8g}\t{s['leeward']:.8g}\t"
                f"{s['wall_min']:.8g}\t{s['wall_max']:.8g}\t{s['roof']:.8g}\t"
                f"{s['wall_area']:.8g}\t{s['n_faces']}\n"
            )
    print(f"\nwrote {a.output} and {a.csv}")
    print(f"wrote {a.summary}")


if __name__ == "__main__":
    main()
