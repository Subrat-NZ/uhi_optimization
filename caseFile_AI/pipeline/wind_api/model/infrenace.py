# ── Edit these lines ──────────────────────────────────────────────────────────
from pathlib import Path  # moved up: was previously imported AFTER first use, causing NameError


# BASE_DIR = folder this script lives in, e.g.
#   ...\Desktop\end_point\   (contains infrenace.py, model\, turbdiff\)
BASE_DIR = Path(__file__).resolve().parent

# EXACT model folder — same layout as your `model` folder next to this script
# (contains best.ckpt + stats.pickle). No Drive search, no zip — just this
# fixed local path, so there is no ambiguity about which checkpoint loads.
MODEL_DIR = BASE_DIR / "model"
CKPT      = MODEL_DIR / "best.ckpt"
STATS     = MODEL_DIR / "stats.pickle"

# turbdiff package sits directly next to this script (BASE_DIR/turbdiff/...),
# NOT inside a zip — so it's importable as soon as BASE_DIR is on sys.path.
ROOT = BASE_DIR
# ─────────────────────────────────────────────────────────────────────────────

import sys
import threading

# Local output folder (was: DRIVE_ROOT / "turbdiff_outputs" on Google Drive).
OUT = BASE_DIR / "turbdiff_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Verify exact checkpoint / stats paths exist ───────────────────────────────
if not CKPT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at exact path: {CKPT}\n"
        f"Make sure the 'model' folder (with best.ckpt + stats.pickle) sits "
        f"next to this script, or update MODEL_DIR above."
    )
if not STATS.exists():
    raise FileNotFoundError(f"stats.pickle not found at exact path: {STATS}")

print(f"Checkpoint : {CKPT}")
print(f"Stats      : {STATS}")

# ── make turbdiff importable ──────────────────────────────────────────────────
# FIX: Python caches imported modules in sys.modules. If `turbdiff` was ever
# imported earlier in this SAME process (e.g. from a previous run in an
# interactive session), a plain `import turbdiff` here would silently return
# that STALE cached module instead of re-scanning sys.path. We force-clear
# any turbdiff-* modules from the cache first so this always (re-)imports
# fresh from ROOT, regardless of what ran earlier in this session.
for _mod_name in list(sys.modules):
    if _mod_name == "turbdiff" or _mod_name.startswith("turbdiff."):
        del sys.modules[_mod_name]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wind_api.model.turbdiff as turbdiff
print(f"[Diagnostic] turbdiff imported from: {turbdiff.__file__}")
if not str(turbdiff.__file__).startswith(str(ROOT)):
    print(f"  WARNING: turbdiff is NOT loading from the local bundled copy "
          f"({ROOT})! Some other turbdiff install is being picked up first — "
          f"this is almost certainly why load_model() built the wrong "
          f"architecture. Check for a pip-installed 'turbdiff' package "
          f"(pip show turbdiff).")

# ── Windows compatibility patch (NEW) ─────────────────────────────────────────
# turbdiff.models.metrics imports the `deadpool` package, which is a POSIX
# (Linux/Colab) process-pool library. One of its class bodies references
# `signal.SIGKILL` as a default argument value at class-definition time.
# SIGKILL is a Unix-only signal and simply does not exist in Python's
# `signal` module on Windows, so the import crashes immediately with:
#   AttributeError: module 'signal' has no attribute 'SIGKILL'
# Fix: on Windows only, alias SIGKILL to SIGTERM before turbdiff is imported.
# This doesn't change any real behavior for us — inference here runs single-
# process/single-thread (no deadpool worker pool is ever actually spawned),
# so the attribute is only needed to let the module import cleanly; the
# process-killing codepath that would use it never executes in this server.
import signal as _signal
if not hasattr(_signal, "SIGKILL"):
    _signal.SIGKILL = _signal.SIGTERM
    print("[Compat] Patched missing signal.SIGKILL -> SIGTERM (Windows only, "
          "needed for turbdiff's optional 'deadpool' dependency to import).")

# ═══════════════════════════════════════════════════════════════════════════
# BEGIN: embedded inference_geom__3_.py  (verbatim — every function unchanged)
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import math
import numpy as np
import pyvista as pv
import torch
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree

from wind_api.model.turbdiff.data.ofles import (
    BoundaryCondition,
    ChannelHole,
    OpenFOAMBatch,
    OpenFOAMData,
    OpenFOAMMetadata,
    OpenFOAMStats,
    Variable,
)
from wind_api.model.turbdiff.models.diffusion import DiffusionTraining
from wind_api.model.turbdiff.models.utils import select_cells

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NX, NY, NZ = 150, 150, 50   # must match training grid resolution
P_ATM      = 101_325.0       # Pa

# Solid-mask voxelization constants (mirrors FIXED_unStruct2struct.py exactly)
N_VOTE_ORIENTATIONS = 3      # must be odd
MIN_SOLID_CLUSTER   = 50     # voxels

# Face-normal dot-product tolerance for inlet/outlet/parallel classification.
# NOTE: this must be RELATIVE to wind speed, not an absolute number. Real
# wind vectors (even for a "pure" axis-aligned case like 270 deg) carry tiny
# non-zero residual components from floating point / solver noise, e.g.
# U = [0.000564, -7.1786, 0.0025] for a case that is really "pure -Y wind".
# 0.000564 is ~0.008% of the total speed (7.18 m/s) -- physically zero --
# but is 600x larger than a naive absolute epsilon like 1e-6, so an absolute
# threshold wrongly treats it as a real direction and creates a spurious
# second inlet/outlet face pair on a case that should only have ONE inlet
# face. DOT_EPS_REL is a FRACTION of the total wind speed |U|: only when a
# face's dot product exceeds this fraction of |U| is it treated as a real
# inlet/outlet face; otherwise the face is "parallel" (folded into top).
DOT_EPS_REL = 0.05   # 5% of |U| -- comfortably separates ~0.01%-noise
                     # components from genuine ~50-100%-of-|U| diagonal
                     # components (45/135/225/315 deg cases)


# =============================================================================
# SOLID-MASK VOXELIZATION  (identical algorithm to FIXED_unStruct2struct.py)
# =============================================================================
def _prepare_surface(obj_file):
    """
    Load the OBJ and return a clean, closed, triangulated surface.

    Same fix as FIXED_unStruct2struct.py: multi-object OBJ files can carry
    stale 'GroupIds'/'vtkOriginalCellIds' cell arrays whose length goes out
    of sync after .clean(), which corrupts .triangulate() and can collapse
    the surface to almost nothing. clear_data() removes them first.
    """
    raw     = pv.read(str(obj_file))
    try:
        surface = raw.extract_surface(algorithm='dataset_surface')
    except TypeError:
        surface = raw.extract_surface()
    surface.clear_data()
    surface = (surface
               .clean()
               .triangulate()
               .fill_holes(hole_size=1e10)
               .compute_normals(consistent_normals=True, auto_orient_normals=True))
    return surface


def _rotate_z(pts, angle_deg):
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return pts @ R.T


def _single_enclosed_test(struct_pts, surface, tol=1e-6):
    result = pv.PolyData(struct_pts).select_enclosed_points(
        surface, tolerance=tol, check_surface=False
    )
    return result['SelectedPoints'].astype(bool)


def _voting_enclosed_test(struct_pts, surface, n_orientations):
    """Multi-orientation inside/outside test with majority vote (suppresses
    ray-leakage through residual holes in the surface mesh)."""
    angles = np.linspace(0.0, 180.0, n_orientations, endpoint=False)
    votes  = np.zeros(len(struct_pts), dtype=np.int32)
    needed = (n_orientations + 1) // 2

    for angle in angles:
        if angle == 0.0:
            s_rot, pts_rot = surface, struct_pts
        else:
            s_rot   = surface.rotate_z(angle)
            pts_rot = _rotate_z(struct_pts, angle)
        votes += _single_enclosed_test(pts_rot, s_rot).astype(np.int32)

    return votes >= needed


def _remove_stray_voxels(mask, nx, ny, nz, min_cluster):
    """Remove isolated solid clusters smaller than min_cluster voxels
    (residual ray-casting noise near edges/thin features)."""
    mask_3d = mask.reshape((nx, ny, nz), order='F')
    labeled, n_feat = nd_label(mask_3d, structure=np.ones((3, 3, 3), dtype=int))
    if n_feat == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = sizes >= min_cluster
    keep[0] = False
    return keep[labeled].ravel(order='F')


def compute_solid_mask(struct_pts, obj_file, x, y, z):
    """
    Identify structured grid points that lie INSIDE building solids.
    Same 3-stage algorithm used to build the training data:
      1. fill_holes -> watertight surface
      2. N_VOTE_ORIENTATIONS-orientation majority-vote ray casting
      3. connected-component cleanup (drop stray sub-cluster voxels)

    NOTE: unlike training (FIXED_unStruct2struct.py), there is no paired
    CFD volume mesh for brand-new inference geometry, so the ProbeFilter
    void-supplement step is skipped -- ray-casting + cleanup is enough for
    a manufactured/CAD building mesh (it's usually much cleaner than the
    raw OpenFOAM triSurface exports used for training).
    """
    print('  [Solid mask] Preparing building surface (fill holes, triangulate, normals) ...')
    surface = _prepare_surface(obj_file)
    print(f'    Surface ready: {surface.n_points:,} pts, {surface.n_cells:,} triangles')

    print(f'  [Solid mask] Ray-casting with {N_VOTE_ORIENTATIONS}-orientation majority vote ...')
    mask = _voting_enclosed_test(struct_pts, surface, N_VOTE_ORIENTATIONS)
    print(f'    After voting: solid={mask.sum():,} / {len(mask):,}')

    print(f'  [Solid mask] Connected-component cleanup (min cluster = {MIN_SOLID_CLUSTER} voxels) ...')
    nx, ny, nz = len(x), len(y), len(z)
    mask = _remove_stray_voxels(mask, nx, ny, nz, MIN_SOLID_CLUSTER)
    print(f'    After cleanup: solid={mask.sum():,} / {len(mask):,}')

    return mask


# =============================================================================
# DYNAMIC PATCH LABELING  (face-normal · U_inlet dot-product test)
# =============================================================================
# Label codes (same scheme as FIXED_unStruct2struct.py / VTK_to_h5.py):
#   0 = fluid interior
#   1 = solid (inside building)
#   2 = building surface (fluid cells adjacent to solid -- informational only,
#       NOT used to build ghost cells; ghosts for "buildings" come straight
#       from solid_mask, same as the training pipeline)
#   3 = ground  (z_min face -- always)
#   4 = inlet   (side face(s) the wind blows INTO the domain through)
#   5 = outlet  (side face(s) the wind blows OUT of the domain through)
#   6 = top     (z_max face -- always, AND any side face exactly parallel
#                to the wind, since neither training data nor this scheme
#                has a separate "side wall" category)
LBL_SOLID, LBL_BLDG_SURF = 1, 2
LBL_GROUND, LBL_INLET, LBL_OUTLET, LBL_TOP = 3, 4, 5, 6

# Outward unit normal of each of the 4 vertical domain faces
FACE_NORMALS = {
    "x_max": np.array([+1.0, 0.0, 0.0]),   # "east"
    "x_min": np.array([-1.0, 0.0, 0.0]),   # "west"
    "y_max": np.array([0.0, +1.0, 0.0]),   # "north"
    "y_min": np.array([0.0, -1.0, 0.0]),   # "south"
}
FACE_ALIAS = {
    "x_max": "N_hat_east", "x_min": "N_hat_west",
    "y_max": "N_hat_north", "y_min": "N_hat_south",
}


def convert_wind_direction_270_minus(ux_in: float, uy_in: float, uz_in: float):
    """
    Frontend se raw ux, uy, uz aata hai. Model tak jaane se PEHLE, sabse
    pehle in teenon ko is tarah process karte hain:

      1. ux, uy  ->  compass/direction degrees me convert (standard math
                     convention: 0 deg = +X axis, counter-clockwise se badhta hai)
      2. new_direction = 270 - direction   (required transform)
      3. new_direction (degrees) -> wapas ux, uy me convert, SAME horizontal
                     speed rakh ke (magnitude nahi badalta, sirf direction rotate hota hai)

    uz (vertical component) ko as-is pass kar diya jaata hai -- direction ka
    concept sirf horizontal (X-Y) plane me hota hai, uz kabhi bhi direction
    calculation me involve nahi hota isliye ise touch nahi karte.

    Ye function ek hi jagah se call hota hai (predict() endpoint me, request
    receive hote hi) -- taaki poora downstream pipeline (voxelization,
    inlet/outlet classification, boundary conditions, model input) hamesha
    isi converted vector par kaam kare.
    """
    speed_xy = float(np.hypot(ux_in, uy_in))
    if speed_xy < 1e-12:
        # Horizontal wind hi nahi hai (pure vertical/zero) -- rotate karne ko
        # kuch nahi hai, as-is return kar do.
        print("  [Wind convert] horizontal speed ~0, skipping direction rotation")
        return float(ux_in), float(uy_in), float(uz_in)

    direction_deg     = float(np.degrees(np.arctan2(uy_in, ux_in))) % 360.0
    new_direction_deg = (270.0 - direction_deg) % 360.0
    new_rad           = np.radians(new_direction_deg)

    ux_out = speed_xy * float(np.cos(new_rad))
    uy_out = speed_xy * float(np.sin(new_rad))
    uz_out = float(uz_in)

    print(f"  [Wind convert] raw=({ux_in:.4f}, {uy_in:.4f}, {uz_in:.4f})  "
          f"|U_xy|={speed_xy:.4f}  direction={direction_deg:.2f}deg  "
          f"-> (270-direction)={new_direction_deg:.2f}deg  "
          f"-> converted=({ux_out:.4f}, {uy_out:.4f}, {uz_out:.4f})")

    return ux_out, uy_out, uz_out


def classify_side_faces(ux: float, uy: float, uz: float):
    """
    Dot each of the 4 vertical face outward-normals with U_inlet=(ux,uy,uz)
    to decide, per face, whether wind flows IN (inlet), OUT (outlet), or
    parallel (neither -- folded into 'top', see module docstring).

      dot < 0  -> wind flows INTO the domain through this face  -> inlet
      dot > 0  -> wind flows OUT of the domain through this face -> outlet
      dot ~ 0  -> face is parallel to the wind -> not inlet/outlet

    Returns
    -------
    dict  face_key -> "inlet" | "outlet" | "parallel"
    """
    u_vec = np.array([ux, uy, uz])
    u_speed = float(np.linalg.norm(u_vec))
    if u_speed < 1e-9:
        raise RuntimeError("Wind vector has ~zero speed; cannot classify inlet/outlet faces.")
    eps = DOT_EPS_REL * u_speed   # <-- relative threshold, not a fixed absolute number

    result = {}
    print("  [Wind classification] U_inlet =", u_vec, f" |U|={u_speed:.4f}  eps={eps:.4f}")
    for face_key, n_hat in FACE_NORMALS.items():
        dot = float(np.dot(n_hat, u_vec))
        if dot < -eps:
            kind = "inlet"
        elif dot > eps:
            kind = "outlet"
        else:
            kind = "parallel"
        result[face_key] = kind
        print(f"    {FACE_ALIAS[face_key]:14s} ({face_key}): "
              f"dot={dot:+.4f}  -> {kind}")
    n_inlet  = sum(1 for k in result.values() if k == "inlet")
    n_outlet = sum(1 for k in result.values() if k == "outlet")
    if n_inlet == 0 or n_outlet == 0:
        raise RuntimeError(
            f"Degenerate wind vector {u_vec.tolist()}: found {n_inlet} inlet "
            f"face(s) and {n_outlet} outlet face(s). Need at least one of "
            f"each -- check ux/uy/uz aren't both ~0."
        )
    return result


def _derive_patch_label(solid_mask_3d: np.ndarray, ux: float, uy: float, uz: float) -> np.ndarray:
    """
    Derive integer patch_label from a (NX, NY, NZ) solid_mask AND the wind
    direction (ux, uy, uz) -- replaces the old hardcoded NE-wind-only
    version. Works for any wind angle (0-360 degrees), matching how the
    multi-direction training data was labeled.
    """
    sm = solid_mask_3d
    pl = np.zeros_like(sm, dtype=np.int32)

    # 1 -- solid interior
    pl[sm] = LBL_SOLID

    # 2 -- building surface: fluid cells adjacent to solid (informational only)
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(sm, structure=np.ones((3, 3, 3), dtype=bool))
    pl[dilated & ~sm] = LBL_BLDG_SURF

    # 3 -- ground (z_min face) -- always, regardless of wind
    pl[:, :, 0][~sm[:, :, 0]] = LBL_GROUND

    # 6 -- top (z_max face) -- always, regardless of wind
    pl[:, :, -1][~sm[:, :, -1]] = LBL_TOP

    # 4/5/6 -- side faces, decided dynamically from wind direction
    face_kind = classify_side_faces(ux, uy, uz)
    face_slice = {
        "x_max": (-1, slice(None), slice(None)),
        "x_min": (0,  slice(None), slice(None)),
        "y_max": (slice(None), -1, slice(None)),
        "y_min": (slice(None), 0,  slice(None)),
    }
    for face_key, kind in face_kind.items():
        sl = face_slice[face_key]
        fluid_here = ~sm[sl]
        if kind == "inlet":
            pl[sl][fluid_here] = LBL_INLET
        elif kind == "outlet":
            pl[sl][fluid_here] = LBL_OUTLET
        else:  # "parallel" -- fold into top/symmetry category (see docstring)
            pl[sl][fluid_here] = LBL_TOP

    return pl


def _print_label_stats(patch_label_3d: np.ndarray):
    names = {0: "fluid", LBL_SOLID: "solid", LBL_BLDG_SURF: "bldg_surf",
             LBL_GROUND: "ground", LBL_INLET: "inlet",
             LBL_OUTLET: "outlet", LBL_TOP: "top"}
    flat = patch_label_3d.reshape(-1, order="F")
    for lbl, name in names.items():
        print(f"    label {lbl} ({name:12s}): {(flat == lbl).sum():>10,}")


# ──────────────────────────────────────────────────────────────────────────────
# Path A: load from pre-voxelized RectilinearGrid VTK
# ──────────────────────────────────────────────────────────────────────────────

def load_from_vtk(vtk_path: Path, ux: float, uy: float, uz: float):
    """
    Read a RectilinearGrid VTK produced by obj_to_vtk.py.
    Expects a 'solid' (or 'solid_mask') point-data field (1=building, 0=fluid).
    patch_label is re-derived from solid_mask + wind direction.
    """
    print(f"\n[Geometry] Loading pre-voxelized VTK: {vtk_path}")
    rg = pv.read(str(vtk_path))

    if not isinstance(rg, pv.RectilinearGrid):
        raise TypeError(
            f"Expected a RectilinearGrid VTK (output of obj_to_vtk.py), "
            f"got {type(rg).__name__}."
        )

    x_coords = np.array(rg.x, dtype=np.float32)
    y_coords = np.array(rg.y, dtype=np.float32)
    z_coords = np.array(rg.z, dtype=np.float32)
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

    print(f"  Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} pts")
    print(f"  Bounds: X=[{x_coords[0]:.2f}, {x_coords[-1]:.2f}]  "
          f"Y=[{y_coords[0]:.2f}, {y_coords[-1]:.2f}]  "
          f"Z=[{z_coords[0]:.2f}, {z_coords[-1]:.2f}]")

    solid_arr = rg.point_data.get("solid")
    if solid_arr is None:
        solid_arr = rg.point_data.get("solid_mask")
    if solid_arr is None:
        raise KeyError(
            "VTK file does not contain 'solid' or 'solid_mask' point-data field."
        )

    solid_mask = solid_arr.astype(bool)          # (NX*NY*NZ,) Fortran order
    sm3d       = solid_mask.reshape((nx, ny, nz), order="F")
    pl3d       = _derive_patch_label(sm3d, ux, uy, uz)

    print(f"  Solid: {solid_mask.sum():,}  Fluid: {(~solid_mask).sum():,}")
    _print_label_stats(pl3d)

    patch_label = pl3d.reshape(-1, order="F").astype(np.int32)
    return solid_mask, patch_label, x_coords, y_coords, z_coords


# ──────────────────────────────────────────────────────────────────────────────
# Path B: voxelize from .obj surface mesh
# ──────────────────────────────────────────────────────────────────────────────

def _compute_domain(obj_path: Path):
    """Expand bounding box of OBJ geometry into a computational domain."""
    mesh = pv.read(str(obj_path))
    xmin, xmax = mesh.bounds[0], mesh.bounds[1]
    ymin, ymax = mesh.bounds[2], mesh.bounds[3]
    zmin, zmax = mesh.bounds[4], mesh.bounds[5]
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin

    dom_xmin = xmin - 0.10 * dx
    dom_xmax = xmax + 0.10 * dx
    dom_ymin = ymin - 0.10 * dy
    dom_ymax = ymax + 0.10 * dy
    dom_zmin = zmin
    dom_zmax = 2.0 * dz    # double the building height

    print(f"  OBJ bbox:  dx={dx:.2f}  dy={dy:.2f}  dz={dz:.2f}")
    print(f"  Domain: X=[{dom_xmin:.2f}, {dom_xmax:.2f}]  "
          f"Y=[{dom_ymin:.2f}, {dom_ymax:.2f}]  "
          f"Z=[{dom_zmin:.2f}, {dom_zmax:.2f}]")
    return dom_xmin, dom_xmax, dom_ymin, dom_ymax, dom_zmin, dom_zmax


def load_from_obj(obj_path: Path, ux: float, uy: float, uz: float):
    """
    Voxelize a .obj surface mesh into a 150x150x50 structured grid.

    Pipeline (matches training data generation exactly):
      1. Auto-compute domain from OBJ bounding box (_compute_domain)
      2. Solid-mask via 3-orientation ray-cast + connected-component
         cleanup (compute_solid_mask -- same algorithm as
         FIXED_unStruct2struct.py)
      3. Domain-face clear (boundary faces are always fluid)
      4. Derive patch_label from solid_mask + wind direction
         (_derive_patch_label -- dynamic inlet/outlet, any angle)

    Returns
    -------
    solid_mask  : (NX*NY*NZ,) bool, Fortran order
    patch_label : (NX*NY*NZ,) int32, Fortran order
    x_coords, y_coords, z_coords : float32 1D arrays
    """
    print(f"\n[Geometry] Voxelizing OBJ: {obj_path}")

    dom_xmin, dom_xmax, dom_ymin, dom_ymax, dom_zmin, dom_zmax = _compute_domain(obj_path)

    x_coords = np.linspace(dom_xmin, dom_xmax, NX, dtype=np.float32)
    y_coords = np.linspace(dom_ymin, dom_ymax, NY, dtype=np.float32)
    z_coords = np.linspace(dom_zmin, dom_zmax, NZ, dtype=np.float32)

    grid       = pv.RectilinearGrid(x_coords, y_coords, z_coords)
    struct_pts = grid.points   # (NX*NY*NZ, 3), Fortran-ordered like training

    # ── Solid mask (same algorithm as training data) ─────────────────────────
    solid_mask = compute_solid_mask(struct_pts, obj_path, x_coords, y_coords, z_coords)
    sm3d       = solid_mask.reshape((NX, NY, NZ), order="F")

    # ── Domain boundary faces must always be fluid ──────────────────────────
    sm3d[0,  :, :] = sm3d[-1, :, :] = False
    sm3d[:,  0, :] = sm3d[:, -1, :] = False
    sm3d[:, :,  0] = sm3d[:, :, -1] = False
    solid_mask = sm3d.reshape(-1, order="F")
    print(f"  Solid after domain-face clear: {solid_mask.sum():,} / {len(solid_mask):,}")

    # ── Patch labels (dynamic, wind-direction-aware) ─────────────────────────
    pl3d        = _derive_patch_label(sm3d, ux, uy, uz)
    _print_label_stats(pl3d)
    patch_label = pl3d.reshape(-1, order="F").astype(np.int32)

    return solid_mask, patch_label, x_coords, y_coords, z_coords


# =============================================================================
# GRID EMBEDDING  (mirrors the FIXED grid_embedding.py -- axis-restricted
# ghost cells, no cross-contamination between caps and side walls)
# =============================================================================

def _cell_idx(fluid_flat: np.ndarray, N_X: int, N_Y: int, N_Z: int) -> np.ndarray:
    """Fluid cells (F-order in unpadded grid) -> C-order raveled index in padded grid."""
    ix, iy, iz = np.unravel_index(fluid_flat, (N_X, N_Y, N_Z), order="F")
    return np.ravel_multi_index(np.stack([ix + 1, iy + 1, iz + 1]),
                                (N_X + 2, N_Y + 2, N_Z + 2))


def _valid_face_configs_for(bname, N_X, N_Y, N_Z):
    """
    The PHYSICALLY valid ghost-cell directions for each patch type.

    FIX (same as grid_embedding.py): ground/top are horizontal caps and only
    ever get a ghost cell in the z-direction. inlet/outlet are vertical side
    walls (may span 1 or 2 adjacent faces for diagonal wind) and only ever
    get ghost cells in x/y -- NEVER z, even though a side wall physically
    touches z=0 and z=z_max at its top/bottom edge. Without this restriction,
    inlet/outlet ghosts leak into the ground/top ghost layer at every shared
    edge (verified on real training data: ~5% of the cap plane, every case).
    """
    if bname == "ground":
        return [(2, 1, -1)]
    if bname == "top":
        return [(2, N_Z, +1)]
    if bname in ("inlet", "outlet"):
        return [
            (0, 1,   -1), (0, N_X, +1),
            (1, 1,   -1), (1, N_Y, +1),
        ]
    return [  # unknown patch name: safe fallback, all 6 faces
        (0, 1,   -1), (0, N_X, +1),
        (1, 1,   -1), (1, N_Y, +1),
        (2, 1,   -1), (2, N_Z, +1),
    ]


def _ghost_idx(solid_mask: np.ndarray, patch_label: np.ndarray,
               N_X: int, N_Y: int, N_Z: int) -> dict:
    """Ghost-cell indices for each BC patch in the padded grid (axis-restricted)."""
    padded = (N_X + 2, N_Y + 2, N_Z + 2)
    fluid  = ~solid_mask

    named = [("ground", LBL_GROUND), ("inlet", LBL_INLET),
             ("outlet", LBL_OUTLET), ("top", LBL_TOP)]

    result = {}
    for bname, label in named:
        mask = (patch_label == label) & fluid
        if mask.sum() == 0:
            result[bname] = np.empty(0, dtype=np.int64)
            continue
        flat       = np.where(mask)[0]
        ix, iy, iz = np.unravel_index(flat, (N_X, N_Y, N_Z), order="F")
        cp         = np.stack([ix + 1, iy + 1, iz + 1])

        all_ghost = []
        for ax, ext, off in _valid_face_configs_for(bname, N_X, N_Y, N_Z):
            at = cp[ax] == ext
            if at.sum() == 0:
                continue
            gc = cp[:, at].copy()
            gc[ax] += off
            inb = np.ones(gc.shape[1], dtype=bool)
            for d in range(3):
                inb &= (gc[d] >= 0) & (gc[d] < padded[d])
            gc = gc[:, inb]
            if gc.shape[1] > 0:
                all_ghost.append(np.ravel_multi_index(gc, padded))
        result[bname] = (np.unique(np.concatenate(all_ghost)).astype(np.int64)
                          if all_ghost else np.empty(0, dtype=np.int64))

    # buildings: solid cells occupy their own padded positions (unaffected by
    # the label-2 "building surface" informational label above)
    sf = np.where(solid_mask)[0]
    if len(sf) > 0:
        six, siy, siz = np.unravel_index(sf, (N_X, N_Y, N_Z), order="F")
        result["buildings"] = np.unique(
            np.ravel_multi_index(np.stack([six + 1, siy + 1, siz + 1]), padded)
        ).astype(np.int64)
    else:
        result["buildings"] = np.empty(0, dtype=np.int64)

    return result


def _compute_holes(solid_mask_3d, x_c, y_c, z_c):
    """Connected building clusters -> (position_center, size) bounding boxes.
    Mirrors grid_embedding.py's compute_holes exactly."""
    labeled, n = nd_label(solid_mask_3d, structure=np.ones((3, 3, 3), dtype=int))
    holes = []
    for i in range(1, n + 1):
        cluster = (labeled == i)
        xs = np.where(cluster.any(axis=(1, 2)))[0]
        ys = np.where(cluster.any(axis=(0, 2)))[0]
        zs = np.where(cluster.any(axis=(0, 1)))[0]
        if len(xs) == 0:
            continue
        xmin, xmax = float(x_c[xs[0]]), float(x_c[xs[-1]])
        ymin, ymax = float(y_c[ys[0]]), float(y_c[ys[-1]])
        zmin, zmax = float(z_c[zs[0]]), float(z_c[zs[-1]])
        holes.append({
            "position": [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2],
            "size":     [xmax - xmin, ymax - ymin, zmax - zmin],
        })
    return holes


# =============================================================================
# BUILD IN-MEMORY OpenFOAMBatch  (for model.sample())
# =============================================================================

def _boundary_condition_scheme(ux, uy, uz):
    """
    Single source of truth for per-patch, per-variable BC types+values --
    used by BOTH build_batch() (in-memory, for the model) and
    save_inference_h5() (on-disk data.h5), so the two never drift apart.
    """
    U_INLET = [float(ux), float(uy), float(uz)]
    return {
        "u": {
            "buildings": ("fixed-value",   [0.0, 0.0, 0.0]),  # no-slip wall
            "ground":    ("fixed-value",   [0.0, 0.0, 0.0]),  # no-slip wall
            "inlet":     ("fixed-value",   U_INLET),           # prescribed wind
            "outlet":    ("inlet-outlet",  None),
            "top":       ("zero-gradient", None),
        },
        "p": {
            "buildings": ("zero-gradient", None),
            "ground":    ("zero-gradient", None),
            "inlet":     ("zero-gradient", None),
            "outlet":    ("fixed-value",   [0.0]),             # gauge p = 0
            "top":       ("zero-gradient", None),
        },
        "k": {
            "buildings": ("zero-gradient", None),
            "ground":    ("zero-gradient", None),
            "inlet":     ("fixed-value",   [0.15]),            # matches training TI
            "outlet":    ("inlet-outlet",  None),
            "top":       ("zero-gradient", None),
        },
        "nut": {
            "buildings": ("zero-gradient", None),
            "ground":    ("zero-gradient", None),
            "inlet":     ("zero-gradient", None),
            "outlet":    ("zero-gradient", None),
            "top":       ("zero-gradient", None),
        },
    }


def build_batch(solid_mask: np.ndarray, patch_label: np.ndarray,
                x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray,
                ux: float, uy: float, uz: float,
                stats: OpenFOAMStats, device: torch.device) -> OpenFOAMBatch:
    """
    Build an OpenFOAMBatch from geometry arrays -- no H5 file needed.

    The batch contains:
      - geometry metadata  (grid indices, bounding box, cell counts)
      - boundary conditions  (inlet U = [ux, uy, uz], outlet p = 0 gauge, etc.)
      - ZERO placeholder fields  (model generates U/p from noise -- not these)
    """
    fluid_flat   = np.where(~solid_mask)[0]
    n_fluid      = len(fluid_flat)
    padded_shape = (NX + 2, NY + 2, NZ + 2)

    print(f"\n[Batch] Fluid cells: {n_fluid:,}")

    cell_idx_np = _cell_idx(fluid_flat, NX, NY, NZ)
    bound_idx   = _ghost_idx(solid_mask, patch_label, NX, NY, NZ)
    for bname, bidx in bound_idx.items():
        print(f"  boundary/{bname}: {len(bidx):,} ghost cells")

    bb        = torch.tensor([x_coords[-1] - x_coords[0],
                               y_coords[-1] - y_coords[0],
                               z_coords[-1] - z_coords[0]], dtype=torch.float32)
    h_spacing = bb / torch.tensor([NX, NY, NZ], dtype=torch.float32)

    boundaries = {
        bname: {
            "type": "wall" if bname in ("buildings", "ground", "top") else bname,
            "idx":  torch.tensor(bidx, dtype=torch.long).to(device),
        }
        for bname, bidx in bound_idx.items()
    }

    scheme = _boundary_condition_scheme(ux, uy, uz)

    def to_bc(type_str, value):
        if type_str == "fixed-value":
            return BoundaryCondition(BoundaryCondition.Type.FIXED_VALUE,
                                      torch.tensor(value, dtype=torch.float32).to(device))
        elif type_str == "zero-gradient":
            return BoundaryCondition(BoundaryCondition.Type.ZERO_GRADIENT)
        elif type_str == "inlet-outlet":
            return BoundaryCondition(BoundaryCondition.Type.INLET_OUTLET)
        raise ValueError(type_str)

    boundary_conditions = {
        Variable.U: {p: to_bc(t, v) for p, (t, v) in scheme["u"].items()},
        Variable.P: {p: to_bc(t, v) for p, (t, v) in scheme["p"].items()},
    }

    metadata = OpenFOAMMetadata(
        file=Path("geom_inference"),
        nu=1.5e-5,
        h=h_spacing,
        cell_counts=np.array(padded_shape, dtype=np.int64),
        cell_idx=torch.tensor(cell_idx_np, dtype=torch.long).to(device),
        boundaries=boundaries,
        boundary_conditions=boundary_conditions,
        holes=[],   # model does not need explicit hole objects for inference
    )

    samples = {
        Variable.U: torch.zeros((1, n_fluid, 3), dtype=torch.float32, device=device),
        Variable.P: torch.zeros((1, n_fluid, 1), dtype=torch.float32, device=device),
    }

    data = OpenFOAMData(
        metadata=metadata,
        t=torch.tensor([0.0], dtype=torch.float32, device=device),
        samples=samples,
    )
    return OpenFOAMBatch(data=data, stats=stats)


# =============================================================================
# WRITE A TRAINING-SCHEMA-COMPATIBLE data.h5  (single timestep, zero fields)
# =============================================================================

def save_inference_h5(solid_mask: np.ndarray, patch_label: np.ndarray,
                       x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray,
                       ux: float, uy: float, uz: float, out_path: Path):
    """
    Write a data.h5 with the EXACT same group/dataset layout as the training
    pipeline (VTK_to_h5.py + grid_embedding.py combined), for one geometry /
    one wind direction, single timestep:

      domain/       patch_label, solid_mask, x_coords, y_coords, z_coords
      geometry/     bounding_box, cell_counts (unpadded), holes/positions, holes/sizes
      grid/         cell_counts (padded), cell_idx, boundaries/{name}
      boundary-conditions/{u,p,k,nut}/{patch}   type attr (+ value dataset if fixed-value)
      data/         u, p, k, nut  -- all ZERO for every fluid cell (model
                    generates these from noise; this file is for inspection
                    /compatibility, not for feeding real field data)
      physical      attrs: nu

    Only boundary-conditions/u/inlet has a non-zero "value" -- the computed
    wind vector (ux, uy, uz). Every other fixed-value BC that is physically
    zero (no-slip walls, gauge pressure) stays zero, matching training data.
    """
    import h5py as h5
    import json

    print(f"\n[data.h5] Building training-schema HDF5 -> {out_path}")

    fluid_flat = np.where(~solid_mask)[0]
    n_fluid    = len(fluid_flat)
    N_X, N_Y, N_Z = NX, NY, NZ
    padded = (N_X + 2, N_Y + 2, N_Z + 2)

    cell_idx  = _cell_idx(fluid_flat, N_X, N_Y, N_Z)
    bound_idx = _ghost_idx(solid_mask, patch_label, N_X, N_Y, N_Z)

    sm3d  = solid_mask.reshape((N_X, N_Y, N_Z), order="F")
    holes = _compute_holes(sm3d, x_coords, y_coords, z_coords)

    bounding_box = np.array([x_coords[-1] - x_coords[0],
                              y_coords[-1] - y_coords[0],
                              z_coords[-1] - z_coords[0]], dtype=np.float32)
    unpadded_counts = np.array([N_X, N_Y, N_Z], dtype=np.int64)
    padded_counts   = unpadded_counts + 2

    scheme = _boundary_condition_scheme(ux, uy, uz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5.File(str(out_path), "w") as f:
        # ---- domain/ ----
        dom = f.create_group("domain")
        dom["patch_label"] = patch_label
        dom["solid_mask"]  = solid_mask.astype(np.int8)
        dom["x_coords"]    = x_coords
        dom["y_coords"]    = y_coords
        dom["z_coords"]    = z_coords
        dom.attrs["patch_label_map"] = json.dumps({
            0: "internal", 1: "solid", 2: "buildings",
            3: "ground", 4: "inlet", 5: "outlet", 6: "top",
        })
        dom.attrs["boundaries"] = json.dumps({
            "ground":  {"type": "wall"},
            "inlet":   {"type": "inlet"},
            "outlet":  {"type": "outlet"},
            "top":     {"type": "wall"},
        })

        # ---- geometry/ ----
        geo = f.create_group("geometry")
        geo["bounding_box"] = bounding_box
        geo["cell_counts"]  = unpadded_counts
        hole_pos  = (np.array([h["position"] for h in holes], dtype=np.float32)
                     if holes else np.zeros((0, 3), dtype=np.float32))
        hole_size = (np.array([h["size"] for h in holes], dtype=np.float32)
                     if holes else np.zeros((0, 3), dtype=np.float32))
        geo["holes/positions"] = hole_pos
        geo["holes/sizes"]     = hole_size

        # ---- grid/ ----
        grd = f.create_group("grid")
        grd["cell_counts"] = padded_counts
        grd["cell_idx"]    = cell_idx.astype(np.int64)
        bnd_grp = grd.create_group("boundaries")
        for bname, bidx in bound_idx.items():
            ds = bnd_grp.create_dataset(bname, data=bidx)
            ds.attrs["type"]  = "wall" if bname in ("buildings", "ground", "top") else bname
            ds.attrs["start"] = 0
            ds.attrs["n"]     = int(len(bidx))

        # ---- boundary-conditions/ ----
        bc_grp = f.create_group("boundary-conditions")
        for var_name, patches in scheme.items():
            var_grp = bc_grp.create_group(var_name)
            for patch_name, (bc_type, value) in patches.items():
                p_grp = var_grp.create_group(patch_name)
                p_grp.attrs["type"] = bc_type
                if bc_type == "fixed-value":
                    p_grp["value"] = np.asarray(value, dtype=np.float64)

        # ---- data/ (single timestep, all zero) ----
        data_grp = f.create_group("data")
        data_grp["u"]     = np.zeros((1, n_fluid, 3), dtype=np.float32)
        data_grp["p"]     = np.zeros((1, n_fluid),    dtype=np.float32)
        data_grp["k"]     = np.zeros((1, n_fluid),    dtype=np.float32)
        data_grp["nut"]   = np.zeros((1, n_fluid),    dtype=np.float32)
        data_grp["times"] = np.array([0.0], dtype=np.float32)

        # ---- physical ----
        phys = f.create_group("physical")
        phys.attrs["nu"] = 1.5e-5

    print(f"  Saved: {out_path}")
    print(f"    fluid cells   : {n_fluid:,}")
    print(f"    U_inlet value : [{ux:.4f}, {uy:.4f}, {uz:.5f}]")
    print(f"    holes/buildings clusters : {len(holes)}")


# =============================================================================
# LOAD MODEL FROM CHECKPOINT
# =============================================================================

def load_model(ckpt_path: Path, out_dir: Path, device: torch.device) -> DiffusionTraining:
    """Reconstruct the DiffusionTraining model exactly as during training.
    All architecture hyper-parameters are read from the checkpoint config."""
    print(f"\n[Model] Loading: {ckpt_path}", flush=True)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    print(f"[Model] torch.load() finished, building architecture...", flush=True)
    cfg  = ckpt["config"]["model"]

    def b(v):
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")

    model = DiffusionTraining(
        data_dir=out_dir,
        samples_root=out_dir,
        dim                       = int(cfg["dim"]),
        u_net_levels              = int(cfg.get("u_net_levels", 4)),
        with_geometry_embedding   = b(cfg["with_geometry_embedding"]),
        timesteps                 = int(cfg["timesteps"]),
        normalization_mode        = cfg["normalization_mode"],
        beta_schedule             = cfg["beta_schedule"],
        loss                      = cfg["loss"],
        learned_variances         = b(cfg["learned_variances"]),
        elbo_weight               = float(cfg["elbo_weight"]),
        detach_elbo_mean          = b(cfg["detach_elbo_mean"]),
        clip_denoised             = b(cfg["clip_denoised"]),
        noise_bcs                 = b(cfg["noise_bcs"]),
        time_embedding            = cfg["time_embedding"],
        actfn                     = cfg["actfn"],
        optimizer                 = cfg["optimizer"],
        norm_type                 = cfg["norm_type"],
        cell_type_features        = b(cfg["cell_type_features"]),
        cell_type_embedding_type  = cfg["cell_type_embedding_type"],
        cell_type_embedding_dim   = int(cfg["cell_type_embedding_dim"]),
        cell_pos_features         = b(cfg["cell_pos_features"]),
        variables                 = (Variable.U, Variable.P),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {n_params:,}")
    print(f"  dim         : {cfg['dim']}")
    print(f"  u_net_levels: {cfg.get('u_net_levels', 4)}")
    print(f"  timesteps   : {cfg['timesteps']}")
    print(f"  norm_mode   : {cfg['normalization_mode']}")
    print(f"  clip_denoised: {model.model.clip_denoised}")
    return model


# =============================================================================
# VTK OUTPUT
# =============================================================================

def save_vtk(u_field: np.ndarray, p_field: np.ndarray,
             x_c: np.ndarray, y_c: np.ndarray, z_c: np.ndarray,
             solid_mask: np.ndarray, out_prefix: Path):
    """Save predicted fields to a VTK RectilinearGrid file."""
    fluid_idx = np.where(~solid_mask)[0]
    assert u_field.shape[0] == len(fluid_idx), (
        f"Shape mismatch: u_field has {u_field.shape[0]} rows, "
        f"but {len(fluid_idx)} fluid cells exist."
    )
    n_total = len(solid_mask)

    arr_u     = np.full((n_total, 3), np.nan, dtype=np.float32)
    arr_p     = np.full(n_total,      np.nan, dtype=np.float32)
    arr_umag  = np.full(n_total,      np.nan, dtype=np.float32)
    arr_solid = solid_mask.astype(np.float32)

    arr_u[fluid_idx]    = u_field
    arr_p[fluid_idx]    = p_field
    arr_umag[fluid_idx] = np.linalg.norm(u_field, axis=1)

    grid = pv.RectilinearGrid(x_c, y_c, z_c)
    grid.point_data["U"]     = arr_u
    grid.point_data["U_mag"] = arr_umag
    grid.point_data["p"]     = arr_p
    grid.point_data["solid"] = arr_solid

    out_file = str(out_prefix) + ".vtk"
    grid.save(out_file)
    print(f"  Saved: {out_file}  [U, U_mag, p (abs), solid]")
    print(f"    U:  min={u_field.min():.4f}  max={u_field.max():.4f}")
    print(f"    p:  min={p_field.min():.1f}  max={p_field.max():.1f}  [Pa abs]")

# ═══════════════════════════════════════════════════════════════════════════
# ADDED (not part of original inference_geom__3_.py): inlet-speed
# renormalization. Rescales the WHOLE predicted velocity field (every fluid
# cell -- interior AND all boundary patches: inlet/outlet/ground/top/
# building-surface; solid cells are never part of u_pred to begin with) by a
# single scalar factor so the mean predicted speed on the inlet patch
# matches the requested |U_inlet| exactly.
#
#   factor = mean(|U_predicted| on inlet-patch fluid cells) / |U_inlet|
#   U_corrected = U_predicted / factor
#
# Called in _run(), right after u_pred comes out of the model and BEFORE
# save_vtk() / anything is written to Drive or returned to the frontend --
# so both the VTK file and whatever gets served downstream already reflect
# the corrected field. Nothing else in the pipeline is touched.
# ═══════════════════════════════════════════════════════════════════════════

def renormalize_to_inlet_speed(u_pred: np.ndarray, solid_mask: np.ndarray,
                                patch_label: np.ndarray,
                                ux: float, uy: float, uz: float) -> np.ndarray:
    """
    u_pred      : (N_fluid, 3) predicted velocity, rows in the SAME order as
                  np.where(~solid_mask)[0] (how build_batch/_cell_idx ordered
                  fluid cells going into the model, and how save_vtk scatters
                  them back out).
    solid_mask  : (NX*NY*NZ,) bool, F-order flat.
    patch_label : (NX*NY*NZ,) int32, F-order flat (0=interior fluid,
                  1=solid, 2=bldg-surf, 3=ground, 4=inlet, 5=outlet, 6=top).
    """
    fluid_idx = np.where(~solid_mask)[0]
    assert u_pred.shape[0] == len(fluid_idx), (
        f"u_pred has {u_pred.shape[0]} rows but {len(fluid_idx)} fluid cells "
        f"exist -- ordering mismatch, cannot renormalize safely."
    )
    patch_label_fluid = patch_label[fluid_idx]

    inlet_mask = patch_label_fluid == LBL_INLET
    n_inlet = int(inlet_mask.sum())
    if n_inlet == 0:
        print("  [Renorm] WARNING: no inlet-labeled fluid cells found -- "
              "skipping renormalization.")
        return u_pred

    u_inlet_speed = float(np.sqrt(ux**2 + uy**2 + uz**2))
    if u_inlet_speed < 1e-9:
        print("  [Renorm] WARNING: requested inlet speed ~0 -- skipping "
              "renormalization.")
        return u_pred

    mag_at_inlet = np.linalg.norm(u_pred[inlet_mask], axis=1)
    mean_mag_at_inlet = float(np.mean(mag_at_inlet))
    factor = mean_mag_at_inlet / u_inlet_speed

    print(f"  [Renorm] inlet fluid cells        : {n_inlet:,}")
    print(f"  [Renorm] mean|U_pred| near inlet  : {mean_mag_at_inlet:.4f} m/s")
    print(f"  [Renorm] requested |U_inlet|      : {u_inlet_speed:.4f} m/s")
    print(f"  [Renorm] correction factor        : {factor:.5f}")

    return u_pred / factor

# ═══════════════════════════════════════════════════════════════════════════
# END: embedded inference_geom__3_.py
# (argparse/CLI `main()` block from the original file is intentionally not
#  invoked here; inference is exposed through run_inference() below.)
# ═══════════════════════════════════════════════════════════════════════════

# ── Load model ONCE ───────────────────────────────────────────────────────────
print("[Startup] Checking for GPU (torch.cuda.is_available)...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Startup] Using device: {device}", flush=True)

print(f"[Startup] Loading stats.pickle from: {STATS}", flush=True)
stats  = OpenFOAMStats.from_file(STATS)
stats.stats = {
    k: {kk: vv.to(device) for kk, vv in v.items()}
    for k, v in stats.stats.items()
}
print("[Startup] stats.pickle loaded OK.", flush=True)

print(f"[Startup] Loading checkpoint (this can take a while for a large "
      f".ckpt, especially if it's inside a OneDrive-synced folder that "
      f"hasn't fully downloaded locally): {CKPT}", flush=True)
model = load_model(CKPT, OUT, device)
print(f"Model ready on {device}", flush=True)

# ── GPU inference lock — ek waqt mein sirf ek sample() ───────────────────────
_gpu_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# OBJ PRE-PROCESSING PIPELINE — runs on EVERY uploaded .obj, before it is
# ever handed to load_from_obj(). Two steps, in order:
#
#   Step 1 (test.py logic)         : drop whole `o <name>` groups matching
#                                     known non-building patterns
#                                     (site_boundary, vegetation_zone_N,
#                                     rainwater_pond_N).
#   Step 2 (remove_internal_faces) : generically detect ANY set of `o`
#                                     objects in the file that share the same
#                                     XY footprint and stack contiguously in
#                                     Z (i.e. a building modeled floor-by-
#                                     floor), and strip the internal
#                                     floor-separator faces between them,
#                                     welding coincident seam vertices so the
#                                     result stays manifold/watertight.
#
# Both steps only ever remove faces / merge duplicate points — no vertex
# is moved and no surface that survives is altered, so any building's true
# geometry is preserved exactly. Step 2 is NOT hardcoded to any particular
# object name or floor count: it works on whatever multi-floor buildings
# happen to be present in a given upload, or does nothing if there aren't any.
# ═══════════════════════════════════════════════════════════════════════════

import re as _re
from collections import defaultdict as _defaultdict

# ---- Step 1: remove named non-building zones -------------------------------
_REMOVE_NAME_PATTERNS = [
    r"site_boundary",
    r"vegetation_zone_\d+",
    r"rainwater_pond_\d+",
]


def _should_remove_obj_name(name: str) -> bool:
    return any(_re.match(p, name) for p in _REMOVE_NAME_PATTERNS)


def remove_named_zones(input_path: Path, output_path: Path) -> None:
    """Strip whole `o <name>` groups (and their faces) matching known
    non-building patterns (site boundary, vegetation zones, ponds) from an
    OBJ file. Vertex lines are always kept (harmless if left unreferenced).
    Same logic as test.py, just parameterised on input/output paths."""
    skip = False
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if line.startswith("o "):
                name = line.split()[1]
                skip = _should_remove_obj_name(name)
                if not skip:
                    fout.write(line)
                continue
            if line.startswith("v "):
                fout.write(line)
                continue
            if line.startswith("f ") and skip:
                continue
            if not skip:
                fout.write(line)


# ---- Step 2: remove internal (floor-separator) faces from ANY building ----
_EPS = 1e-6


def _parse_obj_for_internal_pass(path):
    verts, objects, cur = [], [], None

    def _ensure_cur():
        # Some OBJ exporters emit v/f/other lines before ANY "o " (or "g ")
        # declaration -- e.g. all building faces sitting at the top of the
        # file with only "usemtl" lines, and the first real "o " appearing
        # much later (or never). Rather than crashing on cur["faces"] when
        # cur is still None, we lazily create an implicit default object to
        # hold that ungrouped geometry so it still passes through untouched.
        nonlocal cur
        if cur is None:
            cur = {"name": "default", "header": "o default\n", "lines": [], "faces": []}
            objects.append(cur)
        return cur

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    line_ending = "\r\n" if raw_lines and raw_lines[0].endswith("\r\n") else "\n"
    for raw in raw_lines:
        line = raw.rstrip("\r\n")
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if cur is not None:
                cur["lines"].append(("v_passthrough", raw))
        elif line.startswith("o ") or line.startswith("g "):
            cur = {"name": line[2:].strip(), "header": raw, "lines": [], "faces": []}
            objects.append(cur)
        elif line.startswith("f "):
            idxs = tuple(int(p.split("/")[0]) for p in line.split()[1:])
            cur = _ensure_cur()
            cur["faces"].append(idxs)
            cur["lines"].append(("f", idxs, raw))
        else:
            if cur is not None:
                cur["lines"].append(("other", raw))
    return verts, objects, line_ending


def _footprint_signature(faces, verts, decimals=2):
    pts = set()
    for face in faces:
        for vi in face:
            x, y, _z = verts[vi - 1]
            pts.add((round(x, decimals), round(y, decimals)))
    return frozenset(pts)


def _z_range(faces, verts):
    zs = [verts[vi - 1][2] for face in faces for vi in face]
    return min(zs), max(zs)


def _group_floors(objects, verts):
    """Group `o` objects sharing an identical XY footprint and stacking
    contiguously in Z into ordered floor-groups. Works for any count of
    such groups, anywhere in the file, any number of floors each."""
    by_footprint = _defaultdict(list)
    for obj in objects:
        if not obj["faces"]:
            continue
        sig = _footprint_signature(obj["faces"], verts)
        by_footprint[sig].append(obj)

    groups = []
    for _sig, objs in by_footprint.items():
        if len(objs) < 2:
            continue
        objs_sorted = sorted(objs, key=lambda o: _z_range(o["faces"], verts)[0])
        contiguous = True
        for a, b in zip(objs_sorted, objs_sorted[1:]):
            _, a_zmax = _z_range(a["faces"], verts)
            b_zmin, _ = _z_range(b["faces"], verts)
            if abs(a_zmax - b_zmin) > 1e-3:
                contiguous = False
                break
        if contiguous:
            groups.append(objs_sorted)
    return groups


def _is_horizontal_face(face, verts):
    zs = [verts[vi - 1][2] for vi in face]
    return (max(zs) - min(zs)) <= _EPS


def _strip_internal_faces(group, verts):
    all_faces = [f for obj in group for f in obj["faces"]]
    zmin, zmax = _z_range(all_faces, verts)
    drop = _defaultdict(set)
    for obj in group:
        for i, face in enumerate(obj["faces"]):
            if _is_horizontal_face(face, verts):
                z = verts[face[0] - 1][2]
                is_boundary = (abs(z - zmin) <= 1e-3) or (abs(z - zmax) <= 1e-3)
                if not is_boundary:
                    drop[obj["name"]].add(i)
    return drop


def _weld_vertices(verts, decimals=5):
    """Merge geometrically-coincident vertices (e.g. duplicated points at
    floor seams that were never index-shared) into single vertex indices.
    Pure topology cleanup — no coordinate is changed or moved."""
    key_to_new, new_verts, old_to_new = {}, [], {}
    for old_i, (x, y, z) in enumerate(verts, start=1):
        key = (round(x, decimals), round(y, decimals), round(z, decimals))
        if key not in key_to_new:
            key_to_new[key] = len(new_verts) + 1
            new_verts.append((x, y, z))
        old_to_new[old_i] = key_to_new[key]
    return new_verts, old_to_new


def remove_internal_faces(input_path: Path, output_path: Path) -> dict:
    """
    Generic internal-face remover — handles ANY .obj, any number of
    multi-floor buildings, any floor count:
      - Auto-detects `o` groups that share the same XY footprint and stack
        contiguously in Z (floor-by-floor buildings).
      - Drops horizontal (constant-Z) faces strictly inside each group's
        Z-range (internal floor separators), keeping only the outer
        top/bottom caps.
      - Welds duplicate coincident vertices at floor seams so the surface
        stays manifold / watertight.
      - Every other object (single-mass buildings, trees, etc.) passes
        through byte-for-byte unchanged (aside from vertex re-indexing).
    Returns a stats dict for logging.
    """
    verts, objects, line_ending = _parse_obj_for_internal_pass(input_path)
    groups = _group_floors(objects, verts)

    drop_map = {}
    stats = {"multi_floor_groups": len(groups), "faces_removed": 0}
    for g in groups:
        drop = _strip_internal_faces(g, verts)
        stats["faces_removed"] += sum(len(v) for v in drop.values())
        drop_map.update(drop)

    new_verts, old_to_new = _weld_vertices(verts, decimals=5)

    with open(output_path, "w", encoding="utf-8") as out:
        for x, y, z in new_verts:
            out.write(f"v {x:.6f} {y:.6f} {z:.6f}{line_ending}")
        for obj in objects:
            out.write(obj["header"])
            face_counter = 0
            drops = drop_map.get(obj["name"], set())
            for item in obj["lines"]:
                if item[0] == "f":
                    _, idxs, raw = item
                    if face_counter not in drops:
                        remapped = tuple(old_to_new[vi] for vi in idxs)
                        out.write("f " + " ".join(str(vi) for vi in remapped) + line_ending)
                    face_counter += 1
                elif item[0] == "v_passthrough":
                    pass
                else:
                    out.write(item[1])

    stats["vertices_before"] = len(verts)
    stats["vertices_after"] = len(new_verts)
    return stats


def normalize_tree_trunks(input_path: Path, output_path: Path,
                           target_diameter: float = 1.0) -> dict:
    """
    Force every `o tree_*` object in an OBJ to a fixed trunk diameter
    (default 1 metre), regardless of how wide it was modeled.

    Each tree object is a vertical prism (its faces form a closed ring of
    verticals extruded in Z — a cylinder/octagon "trunk"). For every such
    object this:
      1. Finds all vertices referenced by that object's faces.
      2. Computes their XY centroid (the trunk's central axis).
      3. Finds the current XY radius (max distance from centroid).
      4. Rescales each vertex's X/Y around that centroid so the new radius
         is target_diameter / 2. Z (height) is left untouched.

    Non-tree objects (buildings, site_boundary, etc.) and their vertices
    are passed through byte-for-byte unchanged. Vertices are only rewritten
    if they belong exclusively to tree objects; any vertex shared with a
    non-tree object is left alone (safety net — trees aren't expected to
    share verts with buildings, but this avoids corrupting them if they do).
    Returns a stats dict for logging.
    """
    verts = []          # 1-indexed list of [x, y, z] (mutable)
    obj_headers = []     # raw "o name\n" lines, in file order
    obj_name_at = {}     # object name -> index into obj_headers/order
    face_vidx_by_obj = {}  # object name -> set of vertex indices it uses
    order = []           # list of (kind, payload) to replay when writing
    line_ending = "\n"
    cur_obj = None

    with open(input_path, "rb") as f:
        raw = f.read()
    if b"\r\n" in raw:
        line_ending = "\r\n"
    text = raw.decode("utf-8")

    for line in text.splitlines(keepends=False):
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            order.append(("v_ref", len(verts) - 1))
        elif line.startswith("o "):
            cur_obj = line.split(None, 1)[1].strip()
            face_vidx_by_obj.setdefault(cur_obj, set())
            order.append(("raw", line))
        elif line.startswith("f ") and cur_obj is not None:
            idxs = [int(p.split("/")[0]) for p in line.split()[1:]]
            face_vidx_by_obj[cur_obj].update(idxs)
            order.append(("raw", line))
        else:
            order.append(("raw", line))

    # vertices that belong to more than one object (or a non-tree object)
    # are left untouched for safety.
    vert_owner_count = {}
    for name, idxs in face_vidx_by_obj.items():
        for vi in idxs:
            vert_owner_count[vi] = vert_owner_count.get(vi, 0) + 1

    trees_scaled = 0
    for name, idxs in face_vidx_by_obj.items():
        if not name.startswith("tree"):
            continue
        safe_idxs = [vi for vi in idxs if vert_owner_count.get(vi) == 1]
        if not safe_idxs:
            continue
        xs = [verts[vi - 1][0] for vi in safe_idxs]
        ys = [verts[vi - 1][1] for vi in safe_idxs]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        r = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
        if r == 0:
            continue
        scale = (target_diameter / 2.0) / r
        for vi in safe_idxs:
            x, y, z = verts[vi - 1]
            verts[vi - 1] = [cx + (x - cx) * scale, cy + (y - cy) * scale, z]
        trees_scaled += 1

    with open(output_path, "w", encoding="utf-8") as out:
        for kind, payload in order:
            if kind == "v_ref":
                x, y, z = verts[payload]
                out.write(f"v {x:.6f} {y:.6f} {z:.6f}{line_ending}")
            else:
                out.write(payload + line_ending)

    return {"trees_scaled": trees_scaled, "target_diameter": target_diameter}


def preprocess_obj(raw_path: Path, jid: str) -> Path:
    """Full OBJ pre-processing pipeline, run on every uploaded .obj BEFORE
    it reaches load_from_obj():
        1. remove_named_zones    -> strip site_boundary / vegetation_zone_* / rainwater_pond_*
        2. remove_internal_faces -> strip internal floor-separator faces from
                                     any stacked/floor-by-floor building found
        3. normalize_tree_trunks -> force every tree_* object to a 1 m trunk
                                     diameter, regardless of source width
    Returns the path to the final cleaned OBJ."""
    step1 = raw_path.with_name(f"{jid}_step1_zones_removed.obj")
    step2 = raw_path.with_name(f"{jid}_step2_no_internal.obj")
    step3 = raw_path.with_name(f"{jid}_step3_trees_normalized.obj")

    remove_named_zones(raw_path, step1)
    stats = remove_internal_faces(step1, step2)
    tree_stats = normalize_tree_trunks(step2, step3, target_diameter=1.0)
    print(
        f"[{jid}] OBJ preprocess: multi-floor groups={stats['multi_floor_groups']}, "
        f"internal faces removed={stats['faces_removed']}, "
        f"vertices {stats['vertices_before']} -> {stats['vertices_after']}, "
        f"trees normalized to {tree_stats['target_diameter']}m trunk = "
        f"{tree_stats['trees_scaled']}"
    )
    return step3


def run_inference(
    obj_path: Path,
    ux: float,
    uy: float,
    uz: float,
    output_vtk: Path,
) -> Path:
    """Run one local OBJ inference and write the resulting VTK file."""
    obj_path = Path(obj_path).resolve()
    output_vtk = Path(output_vtk).resolve()
    if obj_path.suffix.lower() != ".obj":
        raise ValueError("Only .obj geometry is supported")
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    ux, uy, uz = convert_wind_direction_270_minus(ux, uy, uz)
    job_id = output_vtk.parent.parent.name or output_vtk.stem
    cleanup_paths = [
        obj_path.with_name(f"{job_id}_step1_zones_removed.obj"),
        obj_path.with_name(f"{job_id}_step2_no_internal.obj"),
        obj_path.with_name(f"{job_id}_step3_trees_normalized.obj"),
    ]

    try:
        geometry_path = preprocess_obj(obj_path, job_id)
        solid_mask, patch_label, x_c, y_c, z_c = load_from_obj(
            geometry_path, ux, uy, uz
        )
        batch = build_batch(
            solid_mask, patch_label, x_c, y_c, z_c,
            ux, uy, uz, stats, device,
        )

        with _gpu_lock, torch.inference_mode():
            prediction = model.sample(batch, start_from=None)

        fields = select_cells(prediction, batch.data.metadata.cell_idx)
        u_pred = fields[0, :3, :].T.cpu().numpy()
        p_pred = fields[0, 3, :].cpu().numpy() + P_ATM
        u_pred = renormalize_to_inlet_speed(
            u_pred, solid_mask, patch_label, ux, uy, uz
        )

        output_vtk.parent.mkdir(parents=True, exist_ok=True)
        save_vtk(
            u_pred,
            p_pred,
            x_c,
            y_c,
            z_c,
            solid_mask,
            output_vtk.with_suffix(""),
        )
        if not output_vtk.is_file() or output_vtk.stat().st_size == 0:
            raise RuntimeError(f"Inference did not create a valid VTK: {output_vtk}")
        return output_vtk
    finally:
        for path in cleanup_paths:
            path.unlink(missing_ok=True)
