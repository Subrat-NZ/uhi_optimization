#!/usr/bin/env python3
"""
map_surface_temperature.py  --  thermal-model surfaces  ->  boundary conditions on the momentum mesh

Reads
  <case>/constant/polyMesh                    the momentum mesh (patches: inlet outlet top ground buildings)
  <case>/surfaces/thermal_surfaces.npz        written by run_thermal.py
  <case>/weatherData/target.json              boundary values of the target hour (Ta, q, p, ...)
  <case>/config/surface_types.json

Writes
  <case>/0/T          inlet fixedValue Ta | outlet inletOutlet | top symmetry |
                      ground, buildings: fixedValue, one value per patch face  (mapped surface temperature)
  <case>/0/q          inlet fixedValue q_amb | outlet inletOutlet | top symmetry |
                      ground, buildings: `mixed` per face, zero value fraction (zero gradient plus the flux below)
  <case>/0/qFlux      kg/(m2 s) of vapour entering the air from every ground / buildings face, ANALYTIC
                      (tools/humidity_sources.py): ponds and lawns as permanently wet surfaces (surface energy balance),
                      Penman-Monteith tree transpiration. airTqFoam turns it into the `mixed` refGradient
                      with the effective diffusivity at the wall, so the flux is imposed exactly.
  <case>/constant/transportProperties        nu Pr Prt Sc Sct rho  pRef qRef + saturation switches
  <case>/surfaces/mapping_report.json        distances, unmapped samples, area per class
  <case>/surfaces/mapped_patches.vtk         per-face T and q-flux on the two patches (for ParaView)

Mapping rules
  * buildings patch  : every mesh face is sampled with points spread over its area (about one per
    half a source cell); each sample takes the temperature of the nearest source cell whose normal
    points the same way (|n.n| > 0.3 for crowns). The face value is the mean of its samples. Source
    cells are building triangles (thermal-model surface temperature) and tree crown triangles
    (ambient temperature, trees are not thermally solved).
  * ground patch     : the ground triangles were created from these very faces, so the mapping is exact;
    a face value is the area-weighted mean of its triangles, per ground class, blended with the class
    fractions of the face (pond / pathway / lawn / bare ground). Faces whose triangles all lie under a
    building (removed in the thermal model) take the nearest valid ground face.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foam_io  # noqa: E402
import humidity_sources as hs  # noqa: E402
from foam_io import foam_header, list_scalar  # noqa: E402

CLASSES = ["default", "pathway", "lawn", "pond"]
LATENT_HEAT = 2.45e6          # J/kg
SIGMA = 5.670374419e-8
RHO_WATER_MM = 1.0            # 1 mm of water = 1 kg/m2


def e_sat_pa(t_c):
    return 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def q_sat(t_c, p):
    e = e_sat_pa(t_c)
    return 0.622 * e / (p - 0.378 * e)


LATENT_SCALE = 1.0


def mm_day_to_flux(x):
    return float(x) * RHO_WATER_MM / 86400.0 * LATENT_SCALE


def face_samples(pts, faces, area, n_per_face):
    """Low-discrepancy sample points in every face. Returns (points, face_index)."""
    g1, g2 = 0.7548776662466927, 0.5698402909980532
    P_out, F_out = [], []
    for fi, f in enumerate(faces):
        P = pts[f]
        c = P.mean(axis=0)
        k = len(P)
        n = int(n_per_face[fi])
        s = np.arange(n)
        u = (0.5 + s * g1) % 1.0
        v = (0.5 + s * g2) % 1.0
        flip = (u + v) > 1.0
        u, v = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - v, v)
        i = s % k
        a, b = P[i], P[(i + 1) % k]
        P_out.append(c + u[:, None] * (a - c) + v[:, None] * (b - c))
        F_out.append(np.full(n, fi))
    return np.vstack(P_out), np.concatenate(F_out)


def sample_triangles(tri, spacing, cap=3_000_000):
    """Points spread over triangles so that the nearest sample is within ~spacing of any surface point.
    Returns (points (m,3), source triangle index (m,))."""
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = np.maximum(1, np.ceil(a / spacing ** 2)).astype(int)
    if n.sum() > cap:
        n = np.maximum(1, (n * cap / n.sum()).astype(int))
    pts = [tri.mean(axis=1)[n == 1]]
    ids = [np.flatnonzero(n == 1)]
    g1, g2 = 0.7548776662466927, 0.5698402909980532
    for i in np.flatnonzero(n > 1):
        s_ = np.arange(n[i])
        u = (0.5 + s_ * g1) % 1.0
        v = (0.5 + s_ * g2) % 1.0
        f = (u + v) > 1.0
        u, v = np.where(f, 1 - u, u), np.where(f, 1 - v, v)
        t = tri[i]
        pts.append(t[0] + u[:, None] * (t[1] - t[0]) + v[:, None] * (t[2] - t[0]))
        ids.append(np.full(n[i], i))
    return np.vstack(pts), np.concatenate(ids)


def point_tri_distance(P, T):
    """Exact distance from points P (m,3) to triangles T (m,3,3), pairwise."""
    A, B, C = T[:, 0], T[:, 1], T[:, 2]
    n = np.cross(B - A, C - A)
    nn = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nn, 1e-300)[:, None]
    dn = np.einsum("mj,mj->m", P - A, n)
    Pp = P - dn[:, None] * n
    c0 = np.einsum("mj,mj->m", np.cross(B - A, Pp - A), n)
    c1 = np.einsum("mj,mj->m", np.cross(C - B, Pp - B), n)
    c2 = np.einsum("mj,mj->m", np.cross(A - C, Pp - C), n)
    inside = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (nn > 1e-12)

    def seg(a, b):
        ab = b - a
        t = np.clip(np.einsum("mj,mj->m", P - a, ab) / np.maximum(np.einsum("mj,mj->m", ab, ab), 1e-300), 0, 1)
        return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)

    de = np.minimum(np.minimum(seg(A, B), seg(B, C)), seg(C, A))
    return np.where(inside, np.abs(dn), de)


def tri_area(tri):
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def write_vtk(path, points, faces_list, scalars):
    """Legacy ASCII POLYDATA of mesh faces with CELL_DATA scalars."""
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nmapped patch values\nASCII\nDATASET POLYDATA\n")
        used = np.unique(np.concatenate(faces_list))
        remap = {int(u): i for i, u in enumerate(used)}
        f.write(f"POINTS {len(used)} float\n")
        for p in points[used]:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        total = sum(len(x) + 1 for x in faces_list)
        f.write(f"POLYGONS {len(faces_list)} {total}\n")
        for fc in faces_list:
            f.write(str(len(fc)) + " " + " ".join(str(remap[int(i)]) for i in fc) + "\n")
        f.write(f"CELL_DATA {len(faces_list)}\n")
        for name, vals in scalars.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            f.write("\n".join(f"{v:.6g}" for v in vals) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=Path("."))
    ap.add_argument("--surfaces", type=Path, default=None)
    ap.add_argument("--target", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=400, help="cap on samples per mesh face")
    ap.add_argument("--k-neighbours", type=int, default=24)
    ap.add_argument("--Pr", type=float, default=0.71)
    ap.add_argument("--Prt", type=float, default=0.85)
    ap.add_argument("--Sc", type=float, default=0.60)
    ap.add_argument("--Sct", type=float, default=0.70)
    ap.add_argument("--nu", type=float, default=1.5e-5)
    ap.add_argument("--latent-scale", type=float, default=1.0,
                    help="multiplies every analytic evaporation / transpiration flux "
                         "(1 = as computed, 0 = switch all humidity sources off)")
    ap.add_argument("--no-evap-limit", action="store_true",
                    help="keep the evaporation / transpiration flux fixed (no reduction when the air next to the surface saturates)")
    ap.add_argument("--no-condensation", action="store_true",
                    help="do not remove vapour above RHmax of saturation (condensation sink in airTqFoam)")
    ap.add_argument("--RHmax", type=float, default=1.0, help="largest relative humidity of the air, as a fraction (default 1.0)")
    ap.add_argument("--condensation-rate", type=float, default=100.0, help="rate [1/s] of the implicit condensation sink (default 100)")
    ap.add_argument("--no-vtk", action="store_true")
    a = ap.parse_args(argv)

    from scipy.spatial import cKDTree

    global LATENT_SCALE
    LATENT_SCALE = a.latent_scale
    case = a.case.resolve()
    surf = a.surfaces or case / "surfaces"
    tgt = json.loads(Path(a.target or case / "weatherData/target.json").read_text())
    cfg = json.loads(Path(a.config or case / "config/surface_types.json").read_text())
    S = np.load(surf / "thermal_surfaces.npz")
    b = tgt["boundary"]
    Ta, qa, pa, rho = b["air_temperature_C"], b["specific_humidity_kg_per_kg"], b["pressure_Pa"], b["air_density_kg_m3"]
    mesh = foam_io.read_mesh(case)
    pts = mesh["points"]
    roles = foam_io.classify_patches(mesh, cfg.get("patches"))
    bld_patches = roles["by_role"].get("buildings", [])
    tree_patches = roles["by_role"].get("trees", [])
    trees_separate = bool(tree_patches)
    print("[map] patches by role: " + "; ".join(f"{r}: {', '.join(n)}" for r, n in roles["by_role"].items()))
    report = {"patches": roles["by_role"]}
    wind = b["wind_speed_10m_ms"]
    hc = hs.h_conv(wind)
    sun_el = b.get("sun_elevation_deg", 0.0)
    tcfg = dict(cfg.get("trees", {}).get("transpiration", {}))
    enabled = bool(tcfg.pop("enabled", True))
    daytime_only = bool(tcfg.pop("daytime_only", True))
    tree_info = hs.tree_transpiration(Ta, qa, pa, hc, b.get("ghi_Wm2", 0.0), b.get("horizontal_ir_Wm2"),
                                      sun_el if daytime_only else 90.0, rho=rho, **tcfg)
    if not enabled:
        tree_info["flux_surface"] = 0.0
    tree_info["flux_surface"] *= LATENT_SCALE
    report["trees"] = dict(tree_info, flux_surface_mm_per_h=tree_info["flux_surface"] * 3600.0)

    # ------------------------------------------------------------------ building / tree patches
    tree_flux = tree_info["flux_surface"]
    bld_flux = mm_day_to_flux(cfg.get("buildings", {}).get("latent_flux_mm_per_day", 0.0))
    patch_T, patch_flux, patch_faces_ = {}, {}, {}
    report["buildings"] = {}
    # source surfaces: building cells (+ the tree crowns when they are part of the building patch itself)
    use_trees = (len(S["t_tri"]) > 0) and not trees_separate
    src_tri = np.concatenate([S["b_tri"], S["t_tri"]]) if use_trees else S["b_tri"]
    src_T = np.concatenate([S["b_T"], S["t_T"]]) if use_trees else S["b_T"]
    src_is_tree = np.concatenate([np.zeros(len(S["b_tri"]), bool), np.ones(len(S["t_tri"]), bool)]) if use_trees \
        else np.zeros(len(S["b_tri"]), bool)
    nb = len(S["b_tri"])
    if len(src_tri):
        nrm = np.cross(src_tri[:, 1] - src_tri[:, 0], src_tri[:, 2] - src_tri[:, 0])
        sarea = 0.5 * np.linalg.norm(nrm, axis=1)
        nrm = nrm / np.maximum(2 * sarea, 1e-300)[:, None]
        src_n = np.concatenate([S["b_normal"], nrm[nb:]]) if use_trees else S["b_normal"]
        spacing = max(0.25, min(2.0, float(np.sqrt(max(sarea.sum(), 1.0) / 1.5e6))))
        src_pts, src_id = sample_triangles(src_tri, spacing)
        tree = cKDTree(src_pts)

    for pname in bld_patches:
        bid, bfaces = foam_io.patch_faces(mesh, pname)
        bC, bS, bA = foam_io.face_geometry(pts, bfaces)
        bN = bS / np.maximum(bA, 1e-300)[:, None]
        patch_faces_[pname] = bfaces
        if len(src_tri) == 0:
            patch_T[pname] = np.full(len(bfaces), Ta)
            patch_flux[pname] = np.zeros(len(bfaces))
            report["buildings"][pname] = dict(faces=len(bfaces), note="no source surfaces; ambient temperature used")
            continue
        n_per = np.clip(np.ceil(bA / max(spacing ** 2 * 4.0, 1e-3)), 5, a.max_samples).astype(int)
        sp, sf = face_samples(pts, bfaces, bA, n_per)
        k = min(a.k_neighbours, len(src_pts))
        _, idx = tree.query(sp, k=k)
        if k == 1:
            idx = idx[:, None]
        cand = src_id[idx]                                # (samples, k) candidate source triangles
        d_ex = np.empty(cand.shape)
        step = 100000
        for i0 in range(0, len(sp), step):
            c = cand[i0:i0 + step]
            pp = np.repeat(sp[i0:i0 + step][:, None, :], k, axis=1).reshape(-1, 3)
            d_ex[i0:i0 + step] = point_tri_distance(pp, src_tri[c.reshape(-1)]).reshape(c.shape)
        # OpenFOAM boundary faces point OUT of the fluid domain, i.e. into the building; the source cells' normals
        # (thermal model) point out of the building into the air -> compare with the reversed patch normal
        sn = -bN[sf]
        dots = np.einsum("sj,skj->sk", sn, src_n[cand])
        dots = np.where(src_is_tree[cand], np.abs(dots), dots)
        ok = dots > 0.3
        matched = ok.any(axis=1)
        pick = np.where(matched, np.argmin(np.where(ok, d_ex, np.inf), axis=1), np.argmin(d_ex, axis=1))
        chosen = cand[np.arange(len(sp)), pick]
        cdist = d_ex[np.arange(len(sp)), pick]
        sT = src_T[chosen]
        sTree = src_is_tree[chosen]
        cnt = np.bincount(sf, minlength=len(bfaces)).astype(float)
        bT = np.bincount(sf, weights=sT, minlength=len(bfaces)) / cnt
        ftree = np.bincount(sf, weights=sTree.astype(float), minlength=len(bfaces)) / cnt
        bflux = ftree * tree_flux + (1.0 - ftree) * bld_flux
        tol = spacing + 0.25 * np.sqrt(bA[sf])
        far = cdist > tol
        patch_T[pname], patch_flux[pname] = bT, bflux
        report["buildings"][pname] = dict(
            faces=len(bfaces), samples=int(len(sp)), source_cells=int(len(src_tri)), source_sample_spacing_m=spacing,
            sample_distance_to_source_m=dict(median=float(np.median(cdist)), p95=float(np.percentile(cdist, 95)),
                                             max=float(cdist.max())),
            samples_without_normal_match=int((~matched).sum()),
            samples_farther_than_tolerance=int(far.sum()),
            faces_with_far_samples=int(len(np.unique(sf[far]))),
            faces_mostly_tree_pct=float(100 * np.mean(ftree > 0.5)),
            T_min=float(bT.min()), T_max=float(bT.max()), T_mean_area_weighted=float(np.sum(bT * bA) / bA.sum()))
        if far.mean() > 0.05 or (~matched).mean() > 0.05:
            print(f"[map] WARNING: patch {pname}: {100*far.mean():.1f} % of the samples are far from any source cell and "
                  f"{100*(~matched).mean():.1f} % have no normal match: the mesh was probably not built from "
                  "this layout json (check geometry_check in thermal_report.json).", file=sys.stderr)
    # tree crowns that are patches of their own: ambient temperature, transpiration flux
    for pname in tree_patches:
        fcs = foam_io.patch_faces(mesh, pname)[1]
        patch_faces_[pname] = fcs
        patch_T[pname] = np.full(len(fcs), Ta)
        patch_flux[pname] = np.full(len(fcs), tree_flux)
        report["buildings"][pname] = dict(faces=len(fcs), role="trees", T=Ta, flux_kg_m2_s=tree_flux)

    # ------------------------------------------------------------------ ground patch
    gnames = [str(x) for x in S["g_patch_names"]]
    gpatch = S["g_patch_of_face"]
    gfaces = []
    for n in gnames:
        gfaces += foam_io.patch_faces(mesh, n)[1]
    gC, gS, gA = foam_io.face_geometry(pts, gfaces)
    tri_face = S["g_tri_face"]
    # ground triangles in the same order as run_thermal.py created them
    tris = []
    for fi, f in enumerate(gfaces):
        for k_ in range(1, len(f) - 1):
            tris.append((f[0], f[k_], f[k_ + 1]))
    tris = np.array(tris)
    tarea = tri_area(pts[tris])
    frac = S["g_frac"]
    lawn_cfg = cfg["ground"]["lawn"]
    pond_cfg = cfg["ground"]["pond"]
    wet_bulb = b["wet_bulb_C"]
    pond_method = str(pond_cfg.get("method", "energy_balance")).lower()   # energy_balance | wet_bulb | fixed
    hyd_mm = float(lawn_cfg.get("hydration_depth_mm", 5.0))

    def face_mean(values):
        ok_ = np.isfinite(values)
        w_ = np.where(ok_, tarea, 0.0)
        num_ = np.bincount(tri_face, weights=np.where(ok_, values, 0.0) * w_, minlength=len(gfaces))
        den_ = np.bincount(tri_face, weights=w_, minlength=len(gfaces))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den_ > 0, num_ / den_, np.nan), den_ > 0

    Tcls = np.full((len(gfaces), len(CLASSES)), np.nan)
    Fcls = np.zeros((len(gfaces), len(CLASSES)))
    valid_face = np.zeros(len(gfaces), bool)
    lawn_info = None
    pond_info = None
    for ci, cname in enumerate(CLASSES):
        key = f"g_T_{cname}"
        if key not in S.files:
            continue
        T = S[key].copy()
        Fm = np.full(len(T), mm_day_to_flux(cfg["ground"][cname].get("latent_flux_mm_per_day", 0.0)))
        wet_energy = cname == "lawn" or (cname == "pond" and pond_method == "energy_balance")
        if wet_energy:
            # permanently wet surface (lawn: kept hydrated, pond: open water): surface temperature and
            # evaporation flux from the surface energy balance
            hg = float(S[f"g_k_{cname}"]) / float(S[f"g_depth_{cname}"])
            ok_ = np.isfinite(T)
            Ts_, F_, res_ = hs.wet_surface(np.where(ok_, S[f"g_Rni_{cname}"], 0.0), Ta, qa, pa, hc, hg,
                                           float(S[f"g_Tdeep_{cname}"]), eps=float(S[f"g_eps_{cname}"]))
            T = np.where(ok_, Ts_, np.nan)
            Fm = np.where(ok_, F_ * LATENT_SCALE, np.nan)
            wl = np.where(ok_, tarea, 0.0)
            F_mean = float(np.nansum(Fm * wl) / max(wl.sum(), 1e-12))
            info = dict(method="energy_balance", mean_flux_kg_m2_s=F_mean, mean_flux_mm_per_h=F_mean * 3600.0,
                        T_surface_C_min=float(np.nanmin(T)), T_surface_C_max=float(np.nanmax(T)),
                        T_surface_C_mean=float(np.nansum(T * wl) / max(wl.sum(), 1e-12)),
                        max_energy_balance_residual_Wm2=float(np.abs(res_).max()),
                        dry_model_T_surface_C_mean=float(np.nanmean(S[key])))
            if cname == "lawn":
                info["irrigation_needed_to_stay_wet_mm_per_h"] = F_mean * 3600.0
                lawn_info = info
            else:
                pond_info = info
        elif cname == "pond":
            t_p = wet_bulb if pond_method == "wet_bulb" else float(pond_cfg["temperature_C"])
            T = np.full(len(T), t_p)
            Fp = float(hs.pond_flux(t_p, Ta, qa, pa, hc)) * LATENT_SCALE
            Fm = np.where(np.isfinite(S[key]), Fp, np.nan)
            pond_info = dict(method=pond_method, T_surface_C=float(t_p), mean_flux_kg_m2_s=Fp,
                             mean_flux_mm_per_h=Fp * 3600.0)
        Tcls[:, ci], v1 = face_mean(T)
        Fcls[:, ci], _ = face_mean(Fm)
        valid_face |= v1
    # faces with no valid triangle (all under buildings): borrow from the nearest valid face
    n_borrowed = 0
    if (~valid_face).any() and valid_face.any():
        kd = cKDTree(gC[valid_face][:, :2])
        _, nn = kd.query(gC[~valid_face][:, :2])
        src = np.flatnonzero(valid_face)[nn]
        Tcls[~valid_face] = Tcls[src]
        Fcls[~valid_face] = Fcls[src]
        n_borrowed = int((~valid_face).sum())
    pi_ = CLASSES.index("pond")
    Fcls = np.nan_to_num(Fcls)
    # blend by class fraction; where a class has no value on a face fall back to the face's other classes
    Tgr = np.zeros(len(gfaces))
    gflux = np.zeros(len(gfaces))
    for fi in range(len(gfaces)):
        w = frac[fi].copy()
        v = Tcls[fi]
        bad = ~np.isfinite(v)
        if bad.any():                       # e.g. lawn fraction but all lawn triangles removed
            w[bad] = 0.0
            if w.sum() <= 0:
                w[~bad] = 1.0
        Tgr[fi] = np.sum(w * np.where(bad, 0, v)) / w.sum()
        gflux[fi] = np.sum(w * Fcls[fi]) / w.sum()
    f_pond = np.zeros(len(gfaces))
    q_pond = qa                       # unused (valueFraction 0): sources enter through qFlux
    report["ground"] = dict(
        faces=len(gfaces), faces_borrowed_from_neighbour=n_borrowed,
        T_min=float(Tgr.min()), T_max=float(Tgr.max()), T_mean_area_weighted=float(np.sum(Tgr * gA) / gA.sum()),
        pond=pond_info,
        lawn=lawn_info, h_conv_Wm2K=float(hc),
        area_by_class_m2={c: float((frac[:, i] * gA).sum()) for i, c in enumerate(CLASSES)},
        faces_with_pond=int((frac[:, pi_] > 0).sum()), faces_with_flux=int((gflux > 0).sum()),
        total_vapour_source_kg_s=float(np.sum(gflux * gA)))

    # ------------------------------------------------------------------ write fields
    case0 = case / "0"
    case0.mkdir(exist_ok=True)
    Tin_K = Ta + 273.15

    def write_field(name, cls, dims, internal, patches):
        txt = foam_header(cls, name, None) + f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
        for pname, body in patches:
            txt += f"    {pname}\n    {{\n" + "\n".join("        " + l if l.strip() else l for l in body.splitlines()) + "\n    }\n"
        txt += "}\n\n// ************************************************************************* //\n"
        (case0 / name).write_text(txt)

    # per-patch arrays of every wall patch: ground-like patches are slices of the concatenated ground arrays
    off = 0
    for i, n in enumerate(gnames):
        nf_ = int((gpatch == i).sum())
        patch_T[n] = Tgr[off:off + nf_]
        patch_flux[n] = gflux[off:off + nf_]
        patch_faces_[n] = gfaces[off:off + nf_]
        off += nf_
    zero = np.zeros

    def flow_bc(pt, field):
        ty, low = pt["type"], pt["name"].lower()
        if ty in ("symmetry", "symmetryPlane", "empty", "wedge", "cyclic"):
            return f"type            {ty};"
        val = {"T": Tin_K, "q": qa, "qFlux": 0.0}[field]
        if field == "qFlux":
            return "type            calculated;\nvalue           uniform 0;"
        if "outlet" in low:
            return f"type            inletOutlet;\ninletValue      uniform {val:.8g};\nvalue           uniform {val:.8g};"
        return f"type            fixedValue;\nvalue           uniform {val:.8g};"

    ent = {"T": [], "q": [], "qFlux": []}
    for pt in mesh["patches"]:
        n = pt["name"]
        if roles["role"][n] == "flow":
            for f_ in ent:
                ent[f_].append((n, flow_bc(pt, f_)))
            continue
        nf_ = pt["nFaces"]
        ent["T"].append((n, "type            fixedValue;\nvalue           " + list_scalar(patch_T[n] + 273.15) + ";"))
        ent["q"].append((n, "type            mixed;\nrefValue        " + list_scalar(np.full(nf_, qa)) + ";\n"
                         "refGradient     " + list_scalar(zero(nf_)) + ";\nvalueFraction   " + list_scalar(zero(nf_)) + ";\n"
                         "value           " + list_scalar(np.full(nf_, qa)) + ";"))
        ent["qFlux"].append((n, "type            calculated;\nvalue           " + list_scalar(patch_flux[n]) + ";"))
    write_field("T", "volScalarField", "[0 0 0 1 0 0 0]", f"uniform {Tin_K:.6g}", ent["T"])
    write_field("q", "volScalarField", "[0 0 0 0 0 0 0]", f"uniform {qa:.8g}", ent["q"])
    write_field("qFlux", "volScalarField", "[1 -2 -1 0 0 0 0]", "uniform 0", ent["qFlux"])

    prop = (foam_header("dictionary", "transportProperties", "constant") +
            "// Effective diffusivities used by airTqFoam:\n"
            "//   alphaEff = nu/Pr + nut/Prt     (temperature)\n"
            "//   DEff     = nu/Sc + nut/Sct     (water vapour, specific humidity)\n"
            f"nu      [0 2 -1 0 0 0 0] {a.nu:g};\n"
            f"Pr      [0 0 0 0 0 0 0] {a.Pr:g};\n"
            f"Prt     [0 0 0 0 0 0 0] {a.Prt:g};\n"
            f"Sc      [0 0 0 0 0 0 0] {a.Sc:g};\n"
            f"Sct     [0 0 0 0 0 0 0] {a.Sct:g};\n"
            f"rho     [1 -3 0 0 0 0 0] {rho:.5g};      // air density at the target hour (p/(R T))\n\n"
            "// Saturation (airTqFoam): the fluxes qFlux were computed for ambient air (qRef, pRef).\n"
            "//   limitEvaporation      flux falls to 0 as the air at the surface reaches saturation (Robin condition)\n"
            "//   limitSupersaturation  vapour above RHmax*qsat(T) is removed as liquid (implicit sink, condensationRate [1/s])\n"
            f"pRef                {pa:.6g};\n"
            f"qRef                {qa:.8g};\n"
            f"limitEvaporation    {'no' if a.no_evap_limit else 'yes'};\n"
            f"limitSupersaturation {'no' if a.no_condensation else 'yes'};\n"
            f"RHmax               {a.RHmax:g};\n"
            f"condensationRate    {a.condensation_rate:g};\n\n"
            "// ************************************************************************* //\n")
    (case / "constant").mkdir(exist_ok=True)
    (case / "constant/transportProperties").write_text(prop)

    report["boundary"] = dict(Ta_C=Ta, q_amb=qa, p_Pa=pa, rho=rho)
    (surf / "mapping_report.json").write_text(json.dumps(report, indent=2))
    if not a.no_vtk:
        wn = [n for n in patch_T]
        write_vtk(surf / "mapped_patches.vtk", pts, [np.asarray(f) for n in wn for f in patch_faces_[n]],
                  {"T_C": np.concatenate([patch_T[n] for n in wn]), "q_flux": np.concatenate([patch_flux[n] for n in wn]),
                   "patch_id": np.concatenate([np.full(len(patch_T[n]), i) for i, n in enumerate(wn)])})
        (surf / "mapped_patches_ids.txt").write_text("\n".join(f"{i} {n} ({roles['role'][n]})" for i, n in enumerate(wn)) + "\n")
    for n in patch_T:
        print(f"[map] {n:28s} {roles['role'][n]:9s} {len(patch_T[n]):7d} faces, T {patch_T[n].min():.1f}..{patch_T[n].max():.1f} C, "
              f"vapour flux {patch_flux[n].min()*3600:.3f}..{patch_flux[n].max()*3600:.3f} mm/h")
    print(f"[map] wrote 0/T 0/q 0/qFlux constant/transportProperties (Ta {Ta:.1f} C, q {qa*1000:.2f} g/kg, "
          f"pond {pond_info['T_surface_C_mean'] if pond_info and 'T_surface_C_mean' in pond_info else float('nan'):.1f} C)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
