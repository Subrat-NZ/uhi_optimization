#!/usr/bin/env python3
"""
json_to_obj.py
Extract buildings, trees, ponds, lawns (grassAreas) and pathways (roads)
from a MassingOBJ JSON file and write surface-meshed OBJ files.

Usage:
    python json_to_obj.py [--coarsen FACTOR]
                          [--building-mesh-size M] [--road-mesh-size M]
                          [--lawn-mesh-size M] [--pond-mesh-size M]
                          [--tree-divisions N]
                          [input1.json [input2.json ...]] [output_dir]

Mesh coarseness:
    --coarsen FACTOR   multiplies every baseline size (default 3).
                       1 = the previous fine mesh, larger = coarser.
                       Baselines: buildings 1 m, roads 1 m, lawns 2.5 m,
                       ponds 1 m, trees crown_radius/10.
    The other options override one surface type directly (metres; for trees,
    N = crown radius / size, so a smaller N is coarser).

Defaults:
    input  : MassingOBJ_reference.json
    output : constant/triSurface/

All JSON files are merged; the output dir is the last argument if it is not
an existing file (or ends with /).  Supports both the wrapped format
(data['siteOnly_json']) and the flat format (data directly contains the keys).

Surface mesh size = building_domain_size * 0.01
Output files:
    buildings_refined.obj  -- quad faces subdivided into regular triangle grid
    trees.obj              -- cylinder trunk + switchable sphere/cone/spheroid crown

Tree crown selection:
    --tree-crown sphere      True sphere (radius = crownRadius) resting on
                             top of the trunk (default)
    --tree-crown cone        Conical crown
    --tree-crown spheroid    Crown fits between trunk top and total tree height
    --tree-crown ellipsoid   crownRadius = horizontal semi-minor axis;
                             crownHeight = vertical semi-major axis

The TREE_CROWN_SHAPE environment variable can also be set to sphere, cone,
spheroid, or ellipsoid.
The command-line option takes precedence.
    ponds.obj              -- closed watertight mesh, all faces subdivided
    lawns.obj              -- flat meshed polygon (grassAreas)
    pathways.obj           -- flat meshed polygon (roads)
"""

import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------



GROUND_Z_TOL = 1e-1        # a floor whose lowest z is above this is "elevated"
COINCIDENT_Z_TOL = 1e-1    # roof/floor at the same height within this tolerance


def _point_on_or_in_polygon_2d(px, py, polygon, tol=1e-6):
    """True if the point is inside the polygon or within `tol` of its boundary."""
    if _point_in_polygon_2d(px, py, polygon):
        return True
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-18:
            continue
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg2))
        qx, qy = x0 + t * dx, y0 + t * dy
        if (px - qx) ** 2 + (py - qy) ** 2 <= tol * tol:
            return True
    return False


def _polygon_within_polygon_2d(inner, outer, tol=1e-6):
    """True if every vertex of `inner` lies inside or on the boundary of `outer`."""
    return all(_point_on_or_in_polygon_2d(x, y, outer, tol) for x, y in inner)


def _interior_point_2d(polygon):
    """A point guaranteed (in practice) to lie inside `polygon` (hole seed)."""
    n = len(polygon)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    if _point_in_polygon_2d(cx, cy, polygon):
        return cx, cy
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for div in (4, 8, 16, 32, 64):
        for j in range(1, div):
            for i in range(1, div):
                px = x0 + (x1 - x0) * i / div
                py = y0 + (y1 - y0) * j / div
                if _point_in_polygon_2d(px, py, polygon):
                    return px, py
    return cx, cy


def _drop_faces_inside_holes(verts, faces, holes):
    """Fallback hole handling: discard triangles whose centroid is in a hole."""
    if not holes:
        return verts, faces
    kept = []
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        gx = (a[0] + b[0] + c[0]) / 3.0
        gy = (a[1] + b[1] + c[1]) / 3.0
        if not any(_point_in_polygon_2d(gx, gy, h) for h in holes):
            kept.append(f)
    used = sorted({i for f in kept for i in f})
    remap = {old: new for new, old in enumerate(used)}
    return ([verts[i] for i in used],
            [tuple(remap[i] for i in f) for f in kept])



def fan_triangulate(n_verts):
    return [(0, i, i + 1) for i in range(1, n_verts - 1)]


def write_obj(path, vertices, faces, comment=""):
    if not vertices or not faces:
        with open(path, 'w') as fh:
            if comment:
                fh.write(f"# {comment}\n")
        print(f"  Written: {path}  (empty - no {comment} in input)")
        return

    # Faces are constructed independently throughout this module, so adjoining
    # faces often contain separate vertex records at identical coordinates.
    # OpenFOAM treats those as disconnected/open edges. Weld at the same
    # precision used below when writing coordinates, then remap every face.
    unique_vertices = []
    vertex_ids = {}
    old_to_new = []
    for vertex in vertices:
        key = tuple(round(float(coordinate), 6) for coordinate in vertex[:3])
        if key not in vertex_ids:
            vertex_ids[key] = len(unique_vertices)
            unique_vertices.append(key)
        old_to_new.append(vertex_ids[key])

    welded_faces = []
    for face in faces:
        welded = tuple(old_to_new[index] for index in face)
        if len(set(welded)) == 3:
            welded_faces.append(welded)

    with open(path, 'w') as fh:
        if comment:
            fh.write(f"# {comment}\n")
        for v in unique_vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in welded_faces:
            fh.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(
        f"  Written: {path}  ({len(unique_vertices)} verts, "
        f"{len(welded_faces)} tris; welded from {len(vertices)} verts)"
    )


def _remove_closing_dup(verts):
    if len(verts) > 1 and verts[0] == verts[-1]:
        return verts[:-1]
    return verts





# ---------------------------------------------------------------------------
# mesh-size computation
# ---------------------------------------------------------------------------

# ONE place decides how coarse every surface is.
#
# Baseline sizes (metres) are the fine values this script used before.  Each
# size is  baseline * COARSEN  unless it is overridden on the command line.
# --coarsen 1 reproduces the old fine mesh exactly.
#
# Why coarsening cannot break the surfaces:
#  * Buildings use ONE global size.  Walls, roofs and floors all split a
#    shared edge with the same rule  max(1, round(edge_len / size)), so
#    neighbouring faces always get identical boundary vertices (watertight, no
#    cracks) for ANY size - even one larger than the building itself, which
#    simply gives 1 segment per edge (2 triangles per face).
#  * Roads / lawns / ponds keep every original boundary vertex; only the
#    interior triangle density changes, so outlines never move or vanish.
#  * Trees use a fixed number of divisions relative to their own crown radius,
#    so a tree is never under-resolved however small or large it is.
#  * Every size is validated and clamped; degenerate polygons are skipped; the
#    triangulator has a fallback chain; a per-polygon triangle budget stops
#    runaway meshes.

DEFAULT_COARSEN = 3.0

BASE_BUILDING_SIZE = 1.0      # m   walls / roofs / floors
BASE_ROAD_SIZE = 1.0          # m   pathways
BASE_LAWN_SIZE = 2.5          # m   grass areas
BASE_POND_SIZE = 1.0          # m   ponds
BASE_TREE_DIVISIONS = 10.0    # crown radius / this = tree surface size

MIN_TREE_DIVISIONS = 2.5      # tree never coarser than this (crown radius / 2.5)
CROWN_MIN_SEGMENTS = 8        # circumferential segments, however coarse
MIN_MESH_SIZE = 0.05          # m   absolute lower clamp for any size
MAX_MESH_SIZE = 1.0e4         # m   absolute upper clamp (guards typos)
MAX_TRIS_PER_POLYGON = 50000  # interior-density budget for one flat polygon


def _valid_positive(value, default, name, lo, hi):
    """Return ``value`` clamped to [lo, hi]; ``default`` if unusable."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float('nan')
    if not math.isfinite(value) or value <= 0:
        print(f"  Warning: invalid {name}={value!r}; using {default:g}",
              file=sys.stderr)
        return default
    return min(max(value, lo), hi)


class MeshPolicy:
    """Resolved mesh sizes for every surface type (see block comment above)."""

    def __init__(self, coarsen=DEFAULT_COARSEN, building=None, road=None,
                 lawn=None, pond=None, tree_divisions=None):
        self.coarsen = _valid_positive(coarsen, DEFAULT_COARSEN, 'coarsen',
                                       0.1, 100.0)

        def size(override, base, name):
            if override is None:
                return min(max(base * self.coarsen, MIN_MESH_SIZE), MAX_MESH_SIZE)
            return _valid_positive(override, base * self.coarsen, name,
                                   MIN_MESH_SIZE, MAX_MESH_SIZE)

        self.building = size(building, BASE_BUILDING_SIZE, 'building mesh size')
        self.road = size(road, BASE_ROAD_SIZE, 'road mesh size')
        self.lawn = size(lawn, BASE_LAWN_SIZE, 'lawn mesh size')
        self.pond = size(pond, BASE_POND_SIZE, 'pond mesh size')

        default_div = max(MIN_TREE_DIVISIONS, BASE_TREE_DIVISIONS / self.coarsen)
        if tree_divisions is None:
            self.tree_divisions = default_div
        else:
            self.tree_divisions = _valid_positive(tree_divisions, default_div,
                                                  'tree divisions', 1.0, 100.0)


POLICY = MeshPolicy()


def _min_edge_length_2d(poly, floor=1e-6):
    """Smallest edge length of a closed 2D polygon ([(x, y), ...]).

    Degenerate edges (length <= floor, e.g. a duplicated vertex) are
    ignored. Returns None if the polygon has fewer than 2 valid edges.
    """
    n = len(poly)
    if n < 2:
        return None
    min_edge = float('inf')
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        length = math.hypot(x1 - x0, y1 - y0)
        if length > floor:
            min_edge = min(min_edge, length)
    return min_edge if min_edge != float('inf') else None


# ---------------------------------------------------------------------------
# quad subdivision
# ---------------------------------------------------------------------------

def subdivide_quad(v0, v1, v2, v3, mesh_size):
    """
    Subdivide a planar quad (v0->v1->v2->v3) into a regular triangle grid
    using bilinear interpolation.
    Returns (local_verts, local_faces) with 0-based indices.
    """
    e01 = math.sqrt(sum((v1[k] - v0[k]) ** 2 for k in range(3)))
    e03 = math.sqrt(sum((v3[k] - v0[k]) ** 2 for k in range(3)))
    nu = max(1, round(e01 / mesh_size))
    nv = max(1, round(e03 / mesh_size))

    verts = []
    for j in range(nv + 1):
        t = j / nv
        for i in range(nu + 1):
            s = i / nu
            p = tuple(
                (1 - s) * (1 - t) * v0[k] + s * (1 - t) * v1[k]
                + s * t * v2[k] + (1 - s) * t * v3[k]
                for k in range(3)
            )
            verts.append(p)

    faces = []
    for j in range(nv):
        for i in range(nu):
            p00 = j * (nu + 1) + i
            p10 = j * (nu + 1) + i + 1
            p11 = (j + 1) * (nu + 1) + i + 1
            p01 = (j + 1) * (nu + 1) + i
            faces.append((p00, p10, p11))
            faces.append((p00, p11, p01))
    return verts, faces


# ---------------------------------------------------------------------------
# flat polygon mesh
# ---------------------------------------------------------------------------

def _point_in_polygon_2d(px, py, polygon):
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        denom = (yj - yi) or 1e-12
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / denom + xi):
            inside = not inside
        j = i
    return inside


def mesh_flat_polygon(polygon_2d, mesh_size, z=0.0):
    """
    Tessellate a flat 2-D polygon with a regular grid.
    Grid points that fall inside the polygon become vertices.
    The effective grid spacing is clamped to at most 1/4 of the polygon's
    shorter bounding-box side so thin shapes (roads, lawns) are always meshed.
    Falls back to fan triangulation if no interior grid points are found.
    Returns (verts, faces) with 0-based indices.
    """
    xs = [p[0] for p in polygon_2d]
    ys = [p[1] for p in polygon_2d]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    # Clamp mesh_size so we always have at least 4 divisions across the narrow side
    short_side = min(x1 - x0, y1 - y0)
    if short_side > 0:
        effective_ms = min(mesh_size, short_side / 4.0)
    else:
        effective_ms = mesh_size
    mesh_size = effective_ms

    nx = max(1, math.ceil((x1 - x0) / mesh_size))
    ny = max(1, math.ceil((y1 - y0) / mesh_size))
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    grid = {}
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            px = x0 + i * dx
            py = y0 + j * dy
            if _point_in_polygon_2d(px, py, polygon_2d):
                grid[(i, j)] = len(verts)
                verts.append((px, py, z))

    faces = []
    for j in range(ny):
        for i in range(nx):
            p00 = grid.get((i,     j    ))
            p10 = grid.get((i + 1, j    ))
            p11 = grid.get((i + 1, j + 1))
            p01 = grid.get((i,     j + 1))
            if p00 is not None and p10 is not None and p11 is not None:
                faces.append((p00, p10, p11))
            if p00 is not None and p11 is not None and p01 is not None:
                faces.append((p00, p11, p01))

    # Fallback: fan-triangulate the boundary polygon if grid gave nothing
    if not faces:
        verts = [(p[0], p[1], z) for p in polygon_2d]
        faces = fan_triangulate(len(verts))

    return verts, faces


def tessellate_polygon_delaunay(polygon_2d, mesh_size, z=0.0):
    """
    Triangulate a flat polygon using Delaunay triangulation (scipy).

    All polygon boundary vertices are included as input points, so the
    roof edge aligns exactly with the wall-top edge — no gap.  Interior
    Cartesian grid points are added at 'mesh_size' spacing to control
    triangle density.  Only triangles whose centroid lies inside the
    polygon are kept, which cleanly handles non-convex shapes without
    needing a per-triangle normal correction.

    Falls back to tessellate_ring_polygon if scipy is unavailable.
    """
    try:
        from scipy.spatial import Delaunay
        import numpy as np
    except ImportError:
        return tessellate_ring_polygon(polygon_2d, mesh_size, z)

    xs = [p[0] for p in polygon_2d]
    ys = [p[1] for p in polygon_2d]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    # Polygon boundary vertices + intermediate points along each edge.
    # Using round(edge_len / mesh_size) subdivisions — the same formula as
    # subdivide_quad — ensures intermediate positions coincide exactly with
    # the wall-top-face vertices, keeping the surface geometrically closed.
    # Dense boundary points prevent unconstrained Delaunay from creating long
    # "shortcut" triangles that span concave corners; such triangles would
    # fail the inside-polygon test and leave a visible gap.
    pts = []
    n_poly = len(polygon_2d)
    for i in range(n_poly):
        p0 = polygon_2d[i]
        p1 = polygon_2d[(i + 1) % n_poly]
        pts.append(p0)                    # corner vertex (always included)
        edge_len = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        n_edge = max(1, round(edge_len / mesh_size))
        for k in range(1, n_edge):        # skip k=0 (corner) and k=n_edge (next corner)
            t = k / n_edge
            pts.append((p0[0] + t * (p1[0] - p0[0]),
                        p0[1] + t * (p1[1] - p0[1])))

    # Interior Cartesian grid points
    nx = max(1, math.ceil((x1 - x0) / mesh_size))
    ny = max(1, math.ceil((y1 - y0) / mesh_size))
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    for j in range(ny + 1):
        for i in range(nx + 1):
            px = x0 + i * dx
            py = y0 + j * dy
            if _point_in_polygon_2d(px, py, polygon_2d):
                pts.append((px, py))

    pts_arr = np.array(pts)
    try:
        tri = Delaunay(pts_arr)
    except Exception:
        # Collinear / degenerate input: last-resort ring tessellation.
        return tessellate_ring_polygon(polygon_2d, mesh_size, z)

    verts = [(p[0], p[1], z) for p in pts]
    faces = []
    for simplex in tri.simplices:
        a = pts_arr[simplex[0]]
        b = pts_arr[simplex[1]]
        c = pts_arr[simplex[2]]
        # Test 4 points (barycentric interior samples) to decide if the
        # triangle lies inside the polygon.  Testing only the centroid
        # fails for thin triangles at convex boundary corners: floating-
        # point arithmetic can place the centroid just outside the polygon
        # edge, falsely rejecting a valid triangle and leaving a gap.
        # Using 4 samples (centroid + one point biased 50 % toward each
        # vertex) and requiring ≥ 3 of 4 to be inside is robust enough to
        # accept corner triangles while still excluding triangles that span
        # a concave notch (whose sample points straddle the notch boundary).
        samples = (
            ((a[0] + b[0] + c[0]) / 3,       (a[1] + b[1] + c[1]) / 3),
            ((2*a[0] + b[0] + c[0]) / 4,     (2*a[1] + b[1] + c[1]) / 4),
            ((a[0] + 2*b[0] + c[0]) / 4,     (a[1] + 2*b[1] + c[1]) / 4),
            ((a[0] + b[0] + 2*c[0]) / 4,     (a[1] + b[1] + 2*c[1]) / 4),
        )
        if sum(_point_in_polygon_2d(sx, sy, polygon_2d) for sx, sy in samples) >= 3:
            faces.append(tuple(int(v) for v in simplex))

    if not faces:
        return tessellate_ring_polygon(polygon_2d, mesh_size, z)

    return verts, faces

def tessellate_polygon_constrained(polygon_2d, mesh_size, z=0.0, holes=None):
    """
    Constrained Delaunay triangulation of a flat polygon (Shewchuk's Triangle).

    Polygon edges (and the edges of every polygon in `holes`) are hard
    segments, so no triangle can cross them. Density is controlled by a maximum
    triangle area of mesh_size**2 / 2, and each edge is split at ~mesh_size
    spacing (same rule as subdivide_quad, so boundaries match adjacent walls).

    `holes` is an optional list of 2-D polygons cut out of the surface.
    Without the 'triangle' package the function falls back to the Delaunay
    tessellation and simply discards triangles inside the holes (approximate).
    """
    holes = [h for h in (holes or []) if len(h) >= 3]
    try:
        import triangle as tr
        import numpy as np
    except ImportError:
        vs, fs = tessellate_polygon_delaunay(polygon_2d, mesh_size, z)
        return _drop_faces_inside_holes(vs, fs, holes)

    pts = []      # [x, y]
    index = {}    # de-duplicates shared vertices (e.g. hole touching outline)
    segs = []

    def _pid(x, y):
        key = (round(x, 9), round(y, 9))
        if key not in index:
            index[key] = len(pts)
            pts.append([x, y])
        return index[key]

    def _add_ring(ring):
        ids = []
        n = len(ring)
        for i in range(n):
            p0 = ring[i]
            p1 = ring[(i + 1) % n]
            ids.append(_pid(p0[0], p0[1]))
            edge_len = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
            n_edge = max(1, round(edge_len / mesh_size))
            for k in range(1, n_edge):
                t = k / n_edge
                ids.append(_pid(p0[0] + t * (p1[0] - p0[0]),
                                p0[1] + t * (p1[1] - p0[1])))
        m = len(ids)
        for i in range(m):
            a, b = ids[i], ids[(i + 1) % m]
            if a != b:
                segs.append([a, b])

    _add_ring(polygon_2d)
    for h in holes:
        _add_ring(h)

    data = {
        'vertices': np.array(pts, dtype=np.float64),
        'segments': np.array(segs, dtype=np.int32),
    }
    if holes:
        data['holes'] = np.array([_interior_point_2d(h) for h in holes],
                                 dtype=np.float64)

    # Interior density: area of a triangle of side mesh_size, but never so
    # small that the polygon would need more than MAX_TRIS_PER_POLYGON
    # triangles.  The budget only touches the interior; the boundary split
    # above stays tied to mesh_size so it still matches neighbouring faces.
    _outer_area = 0.5 * abs(sum(
        polygon_2d[i][0] * polygon_2d[(i + 1) % len(polygon_2d)][1]
        - polygon_2d[(i + 1) % len(polygon_2d)][0] * polygon_2d[i][1]
        for i in range(len(polygon_2d))))
    max_area = max(0.5 * mesh_size * mesh_size,
                   _outer_area / MAX_TRIS_PER_POLYGON, 1e-12)
    # p = PSLG (enforce segments), a = area constraint. No 'q': quality
    # refinement would insert Steiner points on the boundary edges.
    try:
        result = tr.triangulate(data, f'pa{max_area:.10f}')
    except Exception as exc:
        print(f"  Warning: constrained triangulation failed ({exc}); "
              "falling back to Delaunay tessellation.")
        vs, fs = tessellate_polygon_delaunay(polygon_2d, mesh_size, z)
        return _drop_faces_inside_holes(vs, fs, holes)

    verts = [(float(p[0]), float(p[1]), z) for p in result['vertices']]
    faces = [tuple(int(v) for v in tri) for tri in result.get('triangles', [])]

    if not faces:
        vs, fs = tessellate_polygon_delaunay(polygon_2d, mesh_size, z)
        return _drop_faces_inside_holes(vs, fs, holes)

    return verts, faces


def tessellate_ring_polygon(polygon_2d, mesh_size, z=0.0):
    """
    Tessellate a flat polygon using concentric rings from the centroid outward.
    The outermost ring IS the polygon boundary - no gap between mesh and perimeter.
    Suitable for circular or any convex/mildly concave N-gon face.
    """
    n = len(polygon_2d)
    cx = sum(p[0] for p in polygon_2d) / n
    cy = sum(p[1] for p in polygon_2d) / n

    avg_r = sum(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                for p in polygon_2d) / n
    n_rings = max(1, round(avg_r / mesh_size))

    verts = []
    faces = []

    # Index 0: centroid
    verts.append((cx, cy, z))
    ring_base = {}  # ring_base[r] = start index in verts for ring r (1-indexed)

    for r in range(1, n_rings + 1):
        t = r / n_rings          # 0 = centroid, 1 = boundary
        start = len(verts)
        ring_base[r] = start
        for px, py in polygon_2d:
            verts.append((cx + t * (px - cx), cy + t * (py - cy), z))

        if r == 1:
            # Centroid (idx 0) to first ring: N triangles
            for i in range(n):
                j = (i + 1) % n
                faces.append((0, start + i, start + j))
        else:
            # Ring r-1 to ring r: N quads split into 2 triangles each
            prev = ring_base[r - 1]
            for i in range(n):
                j = (i + 1) % n
                p0, p1 = prev + i,  prev + j
                q0, q1 = start + i, start + j
                faces.append((p0, q0, q1))
                faces.append((p0, q1, p1))

    return verts, faces


def tessellate_planar_face_constrained(fv, mesh_size):
    """
    Constrained-Delaunay mesh of any planar 3-D polygon (e.g. a notched wall
    with reflex corners, where a fan triangulation would cross the notch).

    Builds a local 2-D frame in the face's own plane (Newell normal, with the
    first in-plane axis horizontal so vertical wall edges stay vertical),
    projects the polygon into it, meshes it with `tessellate_polygon_constrained`
    (the same routine used for roofs/floors, so boundary points land at the
    same round(edge_len / mesh_size) spacing as the neighbouring faces), then
    maps the result back to 3-D.

    Returns (verts3d, faces) or None if the polygon is degenerate (zero-area
    or fewer than 3 usable vertices after removing a duplicated closing point).
    """
    pts = [(v['x'], v['y'], v['z']) for v in fv]
    pts = _remove_closing_dup(pts)
    n = len(pts)
    if n < 3:
        return None

    # Newell's method: robust normal for a possibly non-planar-by-a-hair,
    # possibly concave polygon.
    nx = ny = nz = 0.0
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl < 1e-9:
        return None
    nx, ny, nz = nx / nl, ny / nl, nz / nl

    # In-plane basis: e1 = n x z_up (horizontal, matches vertical wall edges),
    # e2 = n x e1 (completes a right-handed frame so the mapping back to 3-D
    # preserves winding). Falls back to an arbitrary basis when n is parallel
    # to z_up; that case is a horizontal face, which this branch is not meant
    # to handle anyway.
    e1x, e1y, e1z = ny, -nx, 0.0
    l1 = math.sqrt(e1x * e1x + e1y * e1y)
    if l1 < 1e-9:
        e1x, e1y, e1z = 1.0, 0.0, 0.0
    else:
        e1x, e1y, e1z = e1x / l1, e1y / l1, 0.0
    e2x = ny * e1z - nz * e1y
    e2y = nz * e1x - nx * e1z
    e2z = nx * e1y - ny * e1x

    ox, oy, oz = pts[0]
    poly2d = []
    for (px, py, pz) in pts:
        dx, dy, dz = px - ox, py - oy, pz - oz
        u = dx * e1x + dy * e1y + dz * e1z
        v = dx * e2x + dy * e2y + dz * e2z
        poly2d.append((u, v))

    # Degenerate-area guard (near-collinear points collapsed by rounding).
    area2 = sum(poly2d[i][0] * poly2d[(i + 1) % n][1]
                - poly2d[(i + 1) % n][0] * poly2d[i][1] for i in range(n))
    if abs(area2) < 1e-9:
        return None

    lv, lf = tessellate_polygon_constrained(poly2d, mesh_size, z=0.0)
    verts3d = [(ox + u * e1x + v * e2x,
                oy + u * e1y + v * e2y,
                oz + u * e1z + v * e2z) for (u, v, _) in lv]
    return verts3d, lf


# ---------------------------------------------------------------------------
# buildings
# ---------------------------------------------------------------------------

def buildings_to_obj(buildings, path, mesh_size=None):
    """
    Write the exterior building surface mesh.

    The JSON encodes multi-storey buildings as a ground-floor outer shell
    ('Building N') plus per-storey slices ('building_N_internal_floor_K').
    Each slice has its own wall faces spanning one storey height.

    Face-type selection:
      - 'wall'   -> always included.
      - 'floor'  -> included ONLY when the building is elevated (its floor is
                    above the ground plane, z > GROUND_Z_TOL) and is not an
                    '_internal_floor' slice. Ground-level base slabs and
                    inter-storey slabs are excluded.
                    If the elevated floor rests on another building's roof at
                    the same height, the overlap is removed (see module
                    docstring): the supporting roof is dropped and its
                    footprint becomes a hole in the floor.
      - 'roof'   -> included only from the TOPMOST '_internal_floor' entry of a
                    multi-storey building, or from a single-storey main entry
                    that has no '_internal_floor' siblings. Any other roof is
                    an internal slab and is excluded.
      - other    -> excluded.
    """
    # Main building IDs that have _internal_floor siblings: their own 'roof'
    # is just the ground-floor ceiling (internal slab).
    main_ids_with_internal: set = set()
    for b in buildings:
        bid = b.get('id', '')
        if '_internal_floor' in bid:
            main_ids_with_internal.add(bid.split('_internal_floor')[0].lower())

    # Only the topmost storey's roof of a multi-storey building is exterior.
    topmost_internal_id: dict = {}   # prefix -> (bldg_id, roof_z)
    for b in buildings:
        bid = b.get('id', '')
        if '_internal_floor' not in bid:
            continue
        roof_face = next((f for f in b.get('faces', [])
                          if f.get('surfaceType') == 'roof'), None)
        if roof_face is None or not roof_face.get('vertices'):
            continue
        zr = max(v['z'] for v in roof_face['vertices'])
        prefix = bid.split('_internal_floor')[0]
        if prefix not in topmost_internal_id or zr > topmost_internal_id[prefix][1]:
            topmost_internal_id[prefix] = (bid, zr)
    topmost_internal_ids = {v[0] for v in topmost_internal_id.values()}

    def _belongs_to_exterior(bldg, face):
        """Face-type rule: does this face belong in the exterior surface?"""
        st = face['surfaceType']
        fv = face['vertices']
        bid = bldg.get('id', '')
        is_internal = '_internal_floor' in bid
        if st == 'wall':
            return True
        if st == 'roof':
            if is_internal:
                return bid in topmost_internal_ids
            return bid.lower().replace(' ', '_') not in main_ids_with_internal
        if st == 'floor':
            # NEW: keep the underside of buildings that do not touch the ground.
            if is_internal or not fv:
                return False
            return min(v['z'] for v in fv) > GROUND_Z_TOL
        return False   # ceiling / foundation / other

    # ---- pre-pass: resolve elevated floors against supporting roofs --------
    exterior_roofs = []     # ((bldg_idx, face_idx), z, polygon)
    elevated_floors = []
    for bi, bldg in enumerate(buildings):
        for fi, face in enumerate(bldg.get('faces', [])):
            st = face['surfaceType']
            fv = face['vertices']
            if st not in ('roof', 'floor') or len(fv) < 3:
                continue
            if not _belongs_to_exterior(bldg, face):
                continue
            entry = ((bi, fi),
                     sum(v['z'] for v in fv) / len(fv),
                     [(v['x'], v['y']) for v in fv])
            (exterior_roofs if st == 'roof' else elevated_floors).append(entry)

    skip_faces = set()      # (bldg_idx, face_idx) removed as internal
    floor_holes = {}        # floor key -> [hole polygons]
    for fkey, fz, fpoly in elevated_floors:
        holes = []
        covered = False
        for rkey, rz, rpoly in exterior_roofs:
            if abs(rz - fz) > COINCIDENT_Z_TOL:
                continue
            roof_in_floor = _polygon_within_polygon_2d(rpoly, fpoly)
            floor_in_roof = _polygon_within_polygon_2d(fpoly, rpoly)
            if roof_in_floor and floor_in_roof:      # identical footprints
                skip_faces.add(rkey)
                covered = True
            elif roof_in_floor:                      # roof supports the floor
                skip_faces.add(rkey)
                holes.append(rpoly)
            elif floor_in_roof:                      # floor fully covered
                covered = True
        if covered:
            skip_faces.add(fkey)
        elif holes:
            floor_holes[fkey] = holes

    verts = []
    faces = []
    for bi, bldg in enumerate(buildings):
        # XY centroid: outward-normal reference for wall faces (fallback when
        # no 'normal' is present, and post-subdivision sanity check).
        _all_fv = [v for f in bldg.get('faces', []) for v in f['vertices']]
        if _all_fv:
            bldg_cx = sum(v['x'] for v in _all_fv) / len(_all_fv)
            bldg_cy = sum(v['y'] for v in _all_fv) / len(_all_fv)
        else:
            bldg_cx = bldg_cy = 0.0

        for fi, face in enumerate(bldg.get('faces', [])):
            key = (bi, fi)
            if key in skip_faces:
                continue
            st = face['surfaceType']
            fv = face['vertices']
            if not _belongs_to_exterior(bldg, face):
                continue
            holes = floor_holes.get(key)

            # Horizontal faces: wind so the normal points OUTWARD
            # (roof -> +z, floor -> -z).
            if st in ('roof', 'floor') and len(fv) >= 3:
                signed_area = sum(
                    fv[i]['x'] * fv[(i + 1) % len(fv)]['y']
                    - fv[(i + 1) % len(fv)]['x'] * fv[i]['y']
                    for i in range(len(fv))
                )
                needs_reversed = (
                    (st == 'roof' and signed_area < 0) or
                    (st == 'floor' and signed_area > 0)
                )
                if needs_reversed:
                    fv = list(reversed(fv))

            # Walls: flip winding if it disagrees with the stated normal.
            elif st == 'wall' and len(fv) >= 3:
                given_n = face.get('normal')
                if given_n:
                    a = (fv[0]['x'], fv[0]['y'], fv[0]['z'])
                    b_ = (fv[1]['x'], fv[1]['y'], fv[1]['z'])
                    c = (fv[2]['x'], fv[2]['y'], fv[2]['z'])
                    ax, ay, az = b_[0]-a[0], b_[1]-a[1], b_[2]-a[2]
                    bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
                    nx = ay*bz - az*by
                    ny = az*bx - ax*bz
                    nz = ax*by - ay*bx
                    dot = nx*given_n['x'] + ny*given_n['y'] + nz*given_n['z']
                    if dot < 0:
                        fv = list(reversed(fv))

            base = len(verts)
            face_start = len(faces)
            if mesh_size is not None and len(fv) == 4 and not holes:
                # Rectangular face without holes -> bilinear quad subdivision
                v0 = (fv[0]['x'], fv[0]['y'], fv[0]['z'])
                v1 = (fv[1]['x'], fv[1]['y'], fv[1]['z'])
                v2 = (fv[2]['x'], fv[2]['y'], fv[2]['z'])
                v3 = (fv[3]['x'], fv[3]['y'], fv[3]['z'])
                lv, lf = subdivide_quad(v0, v1, v2, v3, mesh_size)
                verts.extend(lv)
                for tri in lf:
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))
            elif mesh_size is not None and st in ('floor', 'roof'):
                # Polygonal horizontal face (or a face with holes) ->
                # constrained Delaunay; boundary vertices coincide with the
                # wall edges so no gaps are left.
                z_val = fv[0]['z']
                poly = [(v['x'], v['y']) for v in fv]
                lv, lf = tessellate_polygon_constrained(
                    poly, mesh_size, z=z_val, holes=holes)
                verts.extend(lv)
                for tri in lf:
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))
            elif mesh_size is not None and st == 'wall' and len(fv) > 4:
                # Non-quad wall (e.g. a notched wall around a bridge opening,
                # with reflex corners) -> mesh it the same way as roofs/floors
                # instead of fan-triangulating, which only gives a correct
                # result for convex polygons.
                res = tessellate_planar_face_constrained(fv, mesh_size)
                if res is not None:
                    lv, lf = res
                    verts.extend(lv)
                    for tri in lf:
                        faces.append((base + tri[0], base + tri[1], base + tri[2]))
                else:
                    for v in fv:
                        verts.append((v['x'], v['y'], v['z']))
                    for tri in fan_triangulate(len(fv)):
                        faces.append((base + tri[0], base + tri[1], base + tri[2]))
            else:
                # Non-quad wall or no mesh_size -> fan triangulate
                for v in fv:
                    verts.append((v['x'], v['y'], v['z']))
                for tri in fan_triangulate(len(fv)):
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))

            # Drop zero-area slivers (they crash surfaceFeatures) and enforce
            # outward normals for roof (+z), floor (-z) and walls.
            corrected = []
            for k in range(face_start, len(faces)):
                tri = faces[k]
                a = verts[tri[0]]; b = verts[tri[1]]; c = verts[tri[2]]
                ax, ay, az = b[0]-a[0], b[1]-a[1], b[2]-a[2]
                bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
                nx = ay*bz - az*by
                ny = az*bx - ax*bz
                nz = ax*by - ay*bx
                if (nx*nx + ny*ny + nz*nz) < 1e-8:
                    continue
                if st == 'roof' and nz < 0:
                    tri = (tri[0], tri[2], tri[1])
                elif st == 'floor' and nz > 0:
                    tri = (tri[0], tri[2], tri[1])
                elif st == 'wall':
                    fcx = (a[0] + b[0] + c[0]) / 3
                    fcy = (a[1] + b[1] + c[1]) / 3
                    if nx * (fcx - bldg_cx) + ny * (fcy - bldg_cy) < 0:
                        tri = (tri[0], tri[2], tri[1])
                corrected.append(tri)
            del faces[face_start:]
            faces.extend(corrected)
    write_obj(path, verts, faces, comment="buildings")



# ---------------------------------------------------------------------------
# OSM buildings -> synthetic face format
# ---------------------------------------------------------------------------

def convert_osm_building(bldg):
    """
    Convert an 'osmBuildings' / 'osm_buildings' entry into the same
    {'id', 'faces': [{'surfaceType', 'vertices':[{x,y,z}, ...]}]} shape that
    buildings_to_obj() expects for the main 'buildings' entries.

    OSM entries have an empty 'faces' list and instead store the footprint in
    dimensions.baseVertices.  Some exporters use x/y for the ground plane and
    z for elevation, while older variants use x/z for the ground plane.  The
    varying coordinate is detected below.  Extrusion is from OBJ z=0 to the
    building's top-level ``height``.
    """
    base_verts = bldg.get('dimensions', {}).get('baseVertices', [])
    y_span = (max((v.get('y', 0.0) for v in base_verts), default=0.0)
              - min((v.get('y', 0.0) for v in base_verts), default=0.0))
    z_span = (max((v.get('z', 0.0) for v in base_verts), default=0.0)
              - min((v.get('z', 0.0) for v in base_verts), default=0.0))
    ground_key = 'y' if y_span >= z_span else 'z'
    poly = [(v['x'], v.get(ground_key, 0.0)) for v in base_verts]
    poly = _remove_closing_dup(poly)
    if len(poly) < 3:
        return None

    height = bldg.get('height', 0)
    if height <= 0:
        return None

    # Ensure CCW winding (viewed from above) so wall normals point outward.
    # OSM/GIS sources are inconsistent about footprint winding direction; a
    # CW polygon here would make every wall quad for this building face
    # inward instead of outward.
    n_poly = len(poly)
    signed_area = sum(
        poly[i][0] * poly[(i + 1) % n_poly][1]
        - poly[(i + 1) % n_poly][0] * poly[i][1]
        for i in range(n_poly)
    )
    if signed_area < 0:
        poly = list(reversed(poly))

    n = len(poly)
    faces = []

    # Wall faces: one quad per footprint edge, from ground (z=0) to roof (z=height)
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = poly[i]
        x1, y1 = poly[j]
        faces.append({
            'surfaceType': 'wall',
            'vertices': [
                {'x': x0, 'y': y0, 'z': 0.0},
                {'x': x1, 'y': y1, 'z': 0.0},
                {'x': x1, 'y': y1, 'z': height},
                {'x': x0, 'y': y0, 'z': height},
            ],
        })

    # Roof face: footprint polygon at z=height (winding fixed up automatically
    # in buildings_to_obj)
    faces.append({
        'surfaceType': 'roof',
        'vertices': [{'x': x, 'y': y, 'z': height} for (x, y) in poly],
    })

    return {'id': bldg.get('id', 'osm_building'), 'faces': faces}


def collect_osm_buildings(site):
    """
    Return converted OSM buildings from a site dict, checking both the
    'osmBuildings' and 'osm_buildings' key spellings and de-duplicating by id
    (some exports include both keys pointing at the same data).
    """
    raw = site.get('osmBuildings', []) or []
    raw_alt = site.get('osm_buildings', []) or []

    seen_ids = set()
    merged = []
    for b in list(raw) + list(raw_alt):
        bid = b.get('id')
        if bid is not None and bid in seen_ids:
            continue
        if bid is not None:
            seen_ids.add(bid)
        merged.append(b)

    converted = []
    for b in merged:
        c = convert_osm_building(b)
        if c is not None:
            converted.append(c)
    return converted


# ---------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------

def _add_cylinder(cx, cy, z_bot, z_top, radius, n_seg, n_height, verts, faces,
                   phase=0.0, cap_bottom=True, cap_top=True):
    """Closed (or partially open) cylinder with n_height height bands and
    n_seg circumferential segments.

    ``phase`` rotates the cross-section (radians). For n_seg=4 (a square
    cross-section), phase=pi/4 aligns the flat faces with the x/y axes
    instead of leaving the corners pointing along x/y.

    ``cap_bottom``/``cap_top`` control whether a disk/polygon cap is added
    at each end. Set cap_top=False to leave the top ring open (e.g. so a
    tree crown can attach there instead of resting on a closed cap) —
    the returned info lets the caller pick up the exact same ring.
    """
    base = len(verts)
    h = z_top - z_bot
    for hi in range(n_height + 1):
        z = z_bot + hi * h / n_height
        for si in range(n_seg):
            a = 2 * math.pi * si / n_seg + phase
            verts.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z))

    bot_ring = base
    top_ring = base + n_height * n_seg

    for hi in range(n_height):
        for si in range(n_seg):
            sj = (si + 1) % n_seg
            b0 = base + hi * n_seg + si
            b1 = base + hi * n_seg + sj
            t0 = base + (hi + 1) * n_seg + si
            t1 = base + (hi + 1) * n_seg + sj
            faces.append((b0, b1, t1))
            faces.append((b0, t1, t0))

    if cap_bottom:
        bc = len(verts)
        verts.append((cx, cy, z_bot))
        for si in range(n_seg):
            sj = (si + 1) % n_seg
            faces.append((bc, bot_ring + sj, bot_ring + si))   # bottom cap

    if cap_top:
        tc = len(verts)
        verts.append((cx, cy, z_top))
        for si in range(n_seg):
            sj = (si + 1) % n_seg
            faces.append((tc, top_ring + si, top_ring + sj))   # top cap

    return {"top_ring_start": top_ring, "n_top_ring": n_seg}


def _add_cone(cx, cy, z_base, z_tip, radius, n_seg, n_height, verts, faces):
    """Closed cone with n_height rings tapering from base radius to apex."""
    base = len(verts)
    h = z_tip - z_base
    for hi in range(n_height):
        r_hi = radius * (1.0 - hi / n_height)
        z_hi = z_base + hi * h / n_height
        for si in range(n_seg):
            a = 2 * math.pi * si / n_seg
            verts.append((cx + r_hi * math.cos(a), cy + r_hi * math.sin(a), z_hi))
    verts.append((cx, cy, z_tip))   # apex:        base + n_height*n_seg
    verts.append((cx, cy, z_base))  # base centre: base + n_height*n_seg + 1

    for hi in range(n_height - 1):
        for si in range(n_seg):
            sj = (si + 1) % n_seg
            b0 = base + hi * n_seg + si;       b1 = base + hi * n_seg + sj
            t0 = base + (hi + 1) * n_seg + si; t1 = base + (hi + 1) * n_seg + sj
            faces.append((b0, b1, t1));  faces.append((b0, t1, t0))

    apex     = base + n_height * n_seg
    top_ring = base + (n_height - 1) * n_seg
    for si in range(n_seg):
        sj = (si + 1) % n_seg
        faces.append((top_ring + si, top_ring + sj, apex))

    bc = base + n_height * n_seg + 1
    for si in range(n_seg):
        sj = (si + 1) % n_seg
        faces.append((bc, base + sj, base + si))   # base cap


def _add_spheroid(cx, cy, z_bottom, z_top, radius, n_seg, n_height, verts, faces):
    """
    Add a closed prolate/oblate spheroid crown.

    ``radius`` is the horizontal semi-axis, while half of ``z_top-z_bottom``
    is the vertical semi-axis.  The lower pole is placed at the trunk top and
    the upper pole at the total tree height.
    """
    crown_h = z_top - z_bottom
    if crown_h <= 0 or radius <= 0:
        return

    # At least two intermediate latitude rings are needed for a recognisable
    # spheroid.  More rings are selected by trees_to_obj when mesh_size is set.
    n_rings = max(2, n_height)
    base = len(verts)

    bottom_pole = len(verts)
    verts.append((cx, cy, z_bottom))

    z_mid = 0.5 * (z_bottom + z_top)
    vertical_radius = 0.5 * crown_h
    ring_starts = []

    # Latitude angle runs from -pi/2 (bottom pole) to +pi/2 (top pole).
    # The poles themselves are separate vertices, so only interior rings are
    # generated here.
    for ri in range(1, n_rings + 1):
        latitude = -0.5 * math.pi + math.pi * ri / (n_rings + 1)
        ring_radius = radius * math.cos(latitude)
        z = z_mid + vertical_radius * math.sin(latitude)
        ring_starts.append(len(verts))
        for si in range(n_seg):
            a = 2.0 * math.pi * si / n_seg
            verts.append((cx + ring_radius * math.cos(a),
                          cy + ring_radius * math.sin(a),
                          z))

    top_pole = len(verts)
    verts.append((cx, cy, z_top))

    # Bottom pole to first ring: reverse circumferential order so normals point
    # outward/downward near the lower pole.
    first_ring = ring_starts[0]
    for si in range(n_seg):
        sj = (si + 1) % n_seg
        faces.append((bottom_pole, first_ring + sj, first_ring + si))

    # Connect consecutive latitude rings.
    for ri in range(len(ring_starts) - 1):
        lower = ring_starts[ri]
        upper = ring_starts[ri + 1]
        for si in range(n_seg):
            sj = (si + 1) % n_seg
            b0, b1 = lower + si, lower + sj
            t0, t1 = upper + si, upper + sj
            faces.append((b0, b1, t1))
            faces.append((b0, t1, t0))

    # Last ring to top pole.
    last_ring = ring_starts[-1]
    for si in range(n_seg):
        sj = (si + 1) % n_seg
        faces.append((top_pole, last_ring + si, last_ring + sj))


def _bridge_rings(lower_start, n_lower, upper_start, n_upper, faces):
    """Triangulate the band between two co-axial rings that may have a
    different number of vertices (e.g. a 4-vertex square hole widening into
    an n_seg-vertex circular ring). Both rings must be ordered by increasing
    angle around the same center/axis.

    Winding matches the convention used elsewhere in this file for ascending
    bands (cylinder side walls, spheroid latitude bands): outward normals.
    Reduces to the standard per-quad split when n_lower == n_upper.
    """
    i = j = 0
    steps = n_lower + n_upper
    for _ in range(steps):
        advance_lower = (j >= n_upper) or (
            i < n_lower and (i + 1) / n_lower <= (j + 1) / n_upper)
        if advance_lower:
            i_next = (i + 1) % n_lower
            faces.append((lower_start + i, lower_start + i_next,
                          upper_start + (j % n_upper)))
            i += 1
        else:
            j_next = (j + 1) % n_upper
            faces.append((lower_start + (i % n_lower),
                          upper_start + j_next, upper_start + j))
            j += 1


def _add_sphere_crown_with_hole(cx, cy, z_hole, hole_r, hole_phase,
                                 crown_r, n_seg, n_height, verts, faces):
    """Spherical crown with a small square hole at the bottom instead of a
    closed pole, so it seats flush on an open-topped trunk of matching
    cross-section.

    The hole ring (4 vertices, radius=hole_r, centered at (cx,cy,z_hole),
    rotated by hole_phase) is placed at EXACTLY the coordinates the trunk's
    open top ring uses, so the two surfaces share a common boundary loop —
    no cap on either side, no touching-at-a-point pinch, just one
    continuous open surface through the join.

    Returns the apex z (top pole height), since slicing the bottom means
    the crown is very slightly shorter than a full 2*crown_r resting sphere.
    """
    if crown_r <= 0 or hole_r <= 0 or hole_r >= crown_r:
        # Degenerate (trunk as wide as the crown): fall back to the old
        # closed sphere resting on top rather than producing a bad hole.
        z_top = z_hole + 2.0 * crown_r
        _add_spheroid(cx, cy, z_hole, z_top, crown_r, n_seg, n_height, verts, faces)
        return z_top

    hole_seg = 4
    # Latitude (below the equator) at which a true sphere of radius crown_r
    # has ring-radius == hole_r; the hole ring sits there instead of at the
    # pole, effectively slicing off a small polar cap the trunk occupies.
    latitude0 = -math.acos(hole_r / crown_r)
    z_center  = z_hole - crown_r * math.sin(latitude0)
    z_top     = z_center + crown_r

    hole_start = len(verts)
    for si in range(hole_seg):
        a = 2.0 * math.pi * si / hole_seg + hole_phase
        verts.append((cx + hole_r * math.cos(a), cy + hole_r * math.sin(a), z_hole))

    n_rings = max(2, n_height)
    ring_starts = []
    for ri in range(1, n_rings + 1):
        latitude = latitude0 + (0.5 * math.pi - latitude0) * ri / (n_rings + 1)
        ring_radius = crown_r * math.cos(latitude)
        z = z_center + crown_r * math.sin(latitude)
        ring_starts.append(len(verts))
        for si in range(n_seg):
            a = 2.0 * math.pi * si / n_seg
            verts.append((cx + ring_radius * math.cos(a),
                          cy + ring_radius * math.sin(a),
                          z))

    top_pole = len(verts)
    verts.append((cx, cy, z_top))

    # Hole ring -> first latitude ring (different vertex counts: bridge it).
    _bridge_rings(hole_start, hole_seg, ring_starts[0], n_seg, faces)

    # Interior latitude rings (equal counts throughout, same as _add_spheroid).
    for ri in range(len(ring_starts) - 1):
        lower = ring_starts[ri]
        upper = ring_starts[ri + 1]
        for si in range(n_seg):
            sj = (si + 1) % n_seg
            b0, b1 = lower + si, lower + sj
            t0, t1 = upper + si, upper + sj
            faces.append((b0, b1, t1))
            faces.append((b0, t1, t0))

    # Last ring to top pole (closed apex, unchanged from _add_spheroid).
    last_ring = ring_starts[-1]
    for si in range(n_seg):
        sj = (si + 1) % n_seg
        faces.append((top_pole, last_ring + si, last_ring + sj))

    return z_top


def _add_ellipsoid(cx, cy, z_bottom, semi_minor, semi_major,
                   n_seg, n_height, verts, faces):
    """
    Add a closed, vertically oriented prolate ellipsoid crown.

    ``semi_minor`` is the horizontal semi-axis in both x and y and is taken
    directly from ``crownRadius``. ``semi_major`` is the vertical semi-axis
    and is taken directly from the crown height
    (``totalHeight - trunkHeight``).

    The lower pole is attached to the trunk top at ``z_bottom``. Therefore,
    the crown centre is ``z_bottom + semi_major`` and its upper pole is
    ``z_bottom + 2*semi_major``.
    """
    if semi_major <= 0 or semi_minor <= 0:
        return

    z_top = z_bottom + 2.0 * semi_major
    _add_spheroid(cx, cy, z_bottom, z_top, semi_minor,
                  n_seg, n_height, verts, faces)


def trees_to_obj(trees, path, mesh_size=None, crown_shape="sphere"):
    verts = []
    faces = []
    for tree in trees:
        pos     = tree['position']
        cx, cy  = pos['x'], pos['y']
        pz      = pos.get('z', 0.0)
        d       = tree['dimensions']
        trunk_h = d['trunkHeight']
        crown_r = d['crownRadius']
        crown_r = crown_r*0.95
        trunk_r = 0.5 * crown_r
        total_h = d['totalHeight']

        z_ground    = pz
        z_trunk_top = pz + trunk_h
        z_apex      = pz + total_h

        TRUNK_PHASE = math.pi / 4   # axis-aligned square (flat faces on x/y)

        if not (crown_r > 1e-6 and math.isfinite(crown_r)):
            continue   # degenerate tree record: nothing sensible to mesh

        # Tree mesh density is tied to this tree's own crown radius:
        # local_ms = crown_r / POLICY.tree_divisions.  Because it is relative,
        # every tree gets the same number of segments/rings regardless of its
        # size, so no tree can be under-resolved by a global metre value.
        # Counts are clamped so tall thin trees stay light and no tree falls
        # below a recognisable minimum shape.
        local_ms    = crown_r / POLICY.tree_divisions
        n_seg_crown = max(CROWN_MIN_SEGMENTS,
                          round(2 * math.pi * crown_r / local_ms))
        n_h_trunk   = min(4, max(1, round(trunk_h / local_ms)))
        n_h_crown   = min(16, max(2, round((total_h - trunk_h) / local_ms)))

        if crown_shape == "sphere":
            # Square trunk, top left open; the sphere crown's bottom hole
            # (same 4-vertex square, same coordinates) fills the opening,
            # so trunk and crown form one continuous surface instead of a
            # capped trunk touching a closed sphere at a single point.
            _add_cylinder(cx, cy, z_ground, z_trunk_top, trunk_r,
                          4, n_h_trunk, verts, faces,
                          phase=TRUNK_PHASE, cap_top=False)
            _add_sphere_crown_with_hole(cx, cy, z_trunk_top, trunk_r, TRUNK_PHASE,
                                         crown_r, n_seg_crown, n_h_crown, verts, faces)
            continue

        _add_cylinder(cx, cy, z_ground, z_trunk_top, trunk_r,
                      4, n_h_trunk, verts, faces, phase=TRUNK_PHASE)
        if crown_shape == "cone":
            _add_cone(cx, cy, z_trunk_top, z_apex, crown_r,
                      n_seg_crown, n_h_crown, verts, faces)
        elif crown_shape == "spheroid":
            _add_spheroid(cx, cy, z_trunk_top, z_apex, crown_r,
                           n_seg_crown, n_h_crown, verts, faces)
        elif crown_shape == "ellipsoid":
            crown_h = total_h - trunk_h
            _add_ellipsoid(cx, cy, z_trunk_top, crown_r, crown_h,
                           n_seg_crown, max(2, 2 * n_h_crown), verts, faces)
        else:
            raise ValueError(
                f"Unsupported crown_shape '{crown_shape}'. "
                "Choose 'cone', 'sphere', 'spheroid', or 'ellipsoid'."
            )

    write_obj(path, verts, faces, comment="trees")


# ---------------------------------------------------------------------------
# ponds
# ---------------------------------------------------------------------------

def ponds_to_obj(ponds, path, mesh_size=None):
    """
    Closed watertight pond: meshed top (z=0), meshed bottom (z=-depth),
    subdivided side quads for each boundary edge.
    """
    verts = []
    faces = []
    for pond in ponds:
        coords = pond['coordinates']
        depth  = 0.0
        poly   = _remove_closing_dup([(c['x'], c['y']) for c in coords])
        # Ensure CCW winding so top face has +z normal
        _n = len(poly)
        _sa = sum(poly[i][0]*poly[(_i1:=(i+1)%_n)][1] - poly[_i1][0]*poly[i][1]
                  for i in range(_n))
        if _sa < 0:
            poly = list(reversed(poly))
        n      = len(poly)

        if n < 3:
            continue   # degenerate pond outline

        # Pond density comes from the global policy (coarsen * base size).
        # Boundary vertices are always kept, so the outline is unchanged.
        mesh_size = POLICY.pond

        if depth == 0.0:
            # Flat pond (no volume) – emit top face only
            if mesh_size is not None:
                tv, tf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
                base = len(verts); verts.extend(tv)
                for tri in tf:
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))
            else:
                base = len(verts)
                verts.extend([(p[0], p[1], 0.0) for p in poly])
                for tri in fan_triangulate(n):
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))
        elif mesh_size is not None:
            tv, tf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
            base = len(verts); verts.extend(tv)
            for tri in tf:
                faces.append((base + tri[0], base + tri[1], base + tri[2]))

            bv, bf = tessellate_polygon_constrained(poly, mesh_size, z=-depth)
            base = len(verts); verts.extend(bv)
            for tri in bf:                          # reverse winding for bottom
                faces.append((base + tri[0], base + tri[2], base + tri[1]))

            for i in range(n):
                j = (i + 1) % n
                sv0 = (poly[i][0], poly[i][1],  0.0)
                sv1 = (poly[j][0], poly[j][1],  0.0)
                sv2 = (poly[j][0], poly[j][1], -depth)
                sv3 = (poly[i][0], poly[i][1], -depth)
                lv, lf = subdivide_quad(sv0, sv1, sv2, sv3, mesh_size)
                base = len(verts); verts.extend(lv)
                for tri in lf:
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))
        else:
            top_base = len(verts)
            verts.extend([(p[0], p[1],  0.0)   for p in poly])
            bot_base = len(verts)
            verts.extend([(p[0], p[1], -depth) for p in poly])
            for tri in fan_triangulate(n):
                faces.append((top_base + tri[0], top_base + tri[1], top_base + tri[2]))
            for i in range(n):
                j = (i + 1) % n
                t0, t1 = top_base + i, top_base + j
                b0, b1 = bot_base + i, bot_base + j
                faces.append((t0, b0, b1));  faces.append((t0, b1, t1))
            for tri in fan_triangulate(n):
                faces.append((bot_base + tri[0], bot_base + tri[2], bot_base + tri[1]))

    write_obj(path, verts, faces, comment="ponds")


# ---------------------------------------------------------------------------
# lawns / grassAreas
# ---------------------------------------------------------------------------

def lawns_to_obj(grass_areas, path, mesh_size=None):
    verts = []
    faces = []
    for area in grass_areas:
        poly = _remove_closing_dup([(c['x'], c['y']) for c in area['coordinates']])
        # Ensure CCW winding so tessellated faces have +z (upward) normals
        _n = len(poly)
        _sa = sum(poly[i][0]*poly[(_i1:=(i+1)%_n)][1] - poly[_i1][0]*poly[i][1]
                  for i in range(_n))
        if _sa < 0:
            poly = list(reversed(poly))

        if len(poly) < 3:
            continue   # degenerate lawn outline

        # Lawn density comes from the global policy (coarsen * base size).
        mesh_size = POLICY.lawn

        if mesh_size is not None:
            # CDT enforces polygon edges as hard constraints — no triangle can
            # cross the boundary so curved shapes have no outward spikes.
            lv, lf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
            base = len(verts); verts.extend(lv)
            for tri in lf:
                faces.append((base + tri[0], base + tri[1], base + tri[2]))
        else:
            base = len(verts)
            verts.extend([(p[0], p[1], 0.0) for p in poly])
            for tri in fan_triangulate(len(poly)):
                faces.append((base + tri[0], base + tri[1], base + tri[2]))
    write_obj(path, verts, faces, comment="lawns")


# ---------------------------------------------------------------------------
# pathways / roads
# ---------------------------------------------------------------------------

def pathways_to_obj(roads, path, mesh_size=None):
    verts = []
    faces = []
    for road in roads:
        coords = road.get('coordinates', [])
        if not coords:
            continue
        poly = _remove_closing_dup([(c['x'], c['y']) for c in coords])
        if len(poly) < 3:
            continue   # degenerate road outline
        # Ensure CCW winding so tessellated faces have +z (upward) normals
        _n = len(poly)
        _sa = sum(poly[i][0]*poly[(_i1:=(i+1)%_n)][1] - poly[_i1][0]*poly[i][1]
                  for i in range(_n))
        if _sa < 0:
            poly = list(reversed(poly))
        # Road density comes from the global policy unless a size is passed.
        if mesh_size is None:
            mesh_size = POLICY.road
        if mesh_size is not None:
            lv, lf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
            base = len(verts); verts.extend(lv)
            for tri in lf:
                faces.append((base + tri[0], base + tri[1], base + tri[2]))
        else:
            base = len(verts)
            verts.extend([(p[0], p[1], 0.0) for p in poly])
            for tri in fan_triangulate(len(poly)):
                faces.append((base + tri[0], base + tri[1], base + tri[2]))
    write_obj(path, verts, faces, comment="pathways")


# ---------------------------------------------------------------------------
# JSON structure helper
# ---------------------------------------------------------------------------

def _extract_site(data):
    """
    Return the site dict regardless of whether the JSON uses the wrapped format
    (data['siteOnly_json']) or the flat format (data contains the keys directly).
    """
    if 'siteOnly_json' in data:
        return data['siteOnly_json']
    if 'buildings' in data or 'metadata' in data:
        return data
    raise KeyError("Cannot find site data in JSON: expected 'siteOnly_json' key "
                   "or flat structure with 'buildings'/'metadata'.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    # Mesh-size options.  Each is removed from `args` before the positional
    # input/output parsing, so old command lines keep working unchanged.
    # Values must be finite and > 0 so a typo cannot cause a runaway mesh.
    def pop_positive_float(flag, what):
        if flag not in args:
            return None
        option_index = args.index(flag)
        if option_index + 1 >= len(args):
            print(f"Error: {flag} requires a positive value ({what}).",
                  file=sys.stderr)
            sys.exit(2)
        try:
            value = float(args[option_index + 1])
        except ValueError:
            print(f"Error: {flag} must be a number ({what}).", file=sys.stderr)
            sys.exit(2)
        if not math.isfinite(value) or value <= 0:
            print(f"Error: {flag} must be finite and greater than zero.",
                  file=sys.stderr)
            sys.exit(2)
        del args[option_index:option_index + 2]
        return value

    coarsen = pop_positive_float("--coarsen", "multiplier; 1 = old fine mesh")
    building_mesh_size = pop_positive_float("--building-mesh-size", "metres")
    road_mesh_size = pop_positive_float("--road-mesh-size", "metres")
    lawn_mesh_size = pop_positive_float("--lawn-mesh-size", "metres")
    pond_mesh_size = pop_positive_float("--pond-mesh-size", "metres")
    tree_divisions = pop_positive_float("--tree-divisions",
                                        "crown radius / size; smaller = coarser")

    global POLICY
    POLICY = MeshPolicy(
        coarsen=DEFAULT_COARSEN if coarsen is None else coarsen,
        building=building_mesh_size, road=road_mesh_size,
        lawn=lawn_mesh_size, pond=pond_mesh_size,
        tree_divisions=tree_divisions)

    # Optional tree-crown switch.  It is removed before the existing positional
    # input/output parsing so all old invocations continue to work unchanged.
    crown_shape = os.environ.get("TREE_CROWN_SHAPE", "sphere").strip().lower()
    if "--tree-crown" in args:
        option_index = args.index("--tree-crown")
        if option_index + 1 >= len(args):
            print("Error: --tree-crown requires 'sphere', 'cone', 'spheroid', or 'ellipsoid'.",
                  file=sys.stderr)
            sys.exit(2)
        crown_shape = args[option_index + 1].strip().lower()
        del args[option_index:option_index + 2]

    crown_aliases = {
        "sphere": "sphere",
        "spherical": "sphere",
        "cone": "cone",
        "conical": "cone",
        "spheroid": "spheroid",
        "spheroidal": "spheroid",
        "ellipsoid": "ellipsoid",
        "ellipsoidal": "ellipsoid",
    }
    crown_shape = crown_aliases.get(crown_shape, crown_shape)
    if crown_shape not in ("sphere", "cone", "spheroid", "ellipsoid"):
        print(f"Error: invalid tree crown shape '{crown_shape}'. "
              "Use 'sphere', 'cone', 'spheroid', or 'ellipsoid'.", file=sys.stderr)
        sys.exit(2)

    # Last arg is out_dir if it does not end with .json
    if args and not args[-1].lower().endswith('.json'):
        out_dir    = args[-1]
        json_paths = args[:-1]
    else:
        out_dir    = "constant/triSurface"
        json_paths = args

    if not json_paths:
        json_paths = ["MassingOBJ_reference.json"]

    for p in json_paths:
        if not os.path.isfile(p):
            print(f"Error: JSON file not found: {p}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    all_buildings = []
    all_trees     = []
    all_ponds     = []
    all_grass     = []
    all_roads     = []

    for json_path in json_paths:
        print(f"Reading {json_path} ...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        try:
            site = _extract_site(data)
        except KeyError as e:
            print(f"  Warning: skipping '{json_path}': {e}", file=sys.stderr)
            continue

        meta = site.get('metadata', {}).get('sceneInfo', {})
        roads = site.get('roads', []) + site.get('pathways', [])
        osm_buildings = collect_osm_buildings(site)
        print(f"  Scene: {meta.get('totalBuildings', len(site.get('buildings', [])))}"
              f" buildings, {len(osm_buildings)} osm buildings,"
              f" {meta.get('totalTrees', len(site.get('trees', [])))}"
              f" trees, {meta.get('totalPonds', len(site.get('ponds', [])))}"
              f" ponds, {meta.get('totalGrassAreas', len(site.get('grassAreas', [])))}"
              f" grass areas, {meta.get('totalRoads', len(roads))}"
              f" roads")

        all_buildings.extend(site.get('buildings',  []))
        all_buildings.extend(osm_buildings)
        all_trees    .extend(site.get('trees',       []))
        all_ponds    .extend(site.get('ponds',       []))
        all_grass    .extend(site.get('grassAreas',  []))
        all_roads    .extend(roads)

    print(f"\nTotal across {len(json_paths)} file(s): "
          f"{len(all_buildings)} buildings, {len(all_trees)} trees, "
          f"{len(all_ponds)} ponds, {len(all_grass)} lawns, "
          f"{len(all_roads)} roads")

    print(f"\nMesh coarsening factor: {POLICY.coarsen:g}  (1 = original fine mesh)")
    print(f"Mesh size (buildings): {POLICY.building:.3f} m")
    print(f"Mesh size (roads):     {POLICY.road:.3f} m")
    print(f"Mesh size (lawns):     {POLICY.lawn:.3f} m")
    print(f"Mesh size (ponds):     {POLICY.pond:.3f} m")
    print(f"Mesh size (trees):     per-tree, crown radius / {POLICY.tree_divisions:.2f}\n")

    print("Extracting buildings  -> buildings_refined.obj")
    buildings_to_obj(all_buildings,
                     os.path.join(out_dir, 'buildings_refined.obj'),
                     POLICY.building)

    print(f"Extracting trees      -> trees.obj  (crown: {crown_shape})")
    trees_to_obj(all_trees, os.path.join(out_dir, 'trees.obj'),
                 crown_shape=crown_shape)

    print("Extracting ponds      -> ponds.obj")
    ponds_to_obj(all_ponds, os.path.join(out_dir, 'ponds.obj'))

    print("Extracting lawns      -> lawns.obj")
    lawns_to_obj(all_grass, os.path.join(out_dir, 'lawns.obj'))

    print("Extracting pathways   -> pathways.obj")
    pathways_to_obj(all_roads, os.path.join(out_dir, 'pathways.obj'),
                    POLICY.road)

    print("\nDone.")

    print("\n \nEnd\n")


if __name__ == "__main__":
    main()






#Subrat: OBJ coarseness added in the above code to the existing below code.

# #!/usr/bin/env python3
# """
# json_to_obj.py
# Extract buildings, trees, ponds, lawns (grassAreas) and pathways (roads)
# from a MassingOBJ JSON file and write surface-meshed OBJ files.

# Usage:
#     python json_to_obj.py [--building-mesh-size METRES]
#                           [input1.json [input2.json ...]] [output_dir]

# Defaults:
#     input  : MassingOBJ_reference.json
#     output : constant/triSurface/

# All JSON files are merged; the output dir is the last argument if it is not
# an existing file (or ends with /).  Supports both the wrapped format
# (data['siteOnly_json']) and the flat format (data directly contains the keys).

# Surface mesh size = building_domain_size * 0.01
# Output files:
#     buildings_refined.obj  -- quad faces subdivided into regular triangle grid
#     trees.obj              -- cylinder trunk + switchable sphere/cone/spheroid crown

# Tree crown selection:
#     --tree-crown sphere      True sphere (radius = crownRadius) resting on
#                              top of the trunk (default)
#     --tree-crown cone        Conical crown
#     --tree-crown spheroid    Crown fits between trunk top and total tree height
#     --tree-crown ellipsoid   crownRadius = horizontal semi-minor axis;
#                              crownHeight = vertical semi-major axis

# The TREE_CROWN_SHAPE environment variable can also be set to sphere, cone,
# spheroid, or ellipsoid.
# The command-line option takes precedence.
#     ponds.obj              -- closed watertight mesh, all faces subdivided
#     lawns.obj              -- flat meshed polygon (grassAreas)
#     pathways.obj           -- flat meshed polygon (roads)
# """

# import json
# import math
# import os
# import sys

# # ---------------------------------------------------------------------------
# # basic helpers
# # ---------------------------------------------------------------------------



# GROUND_Z_TOL = 1e-1        # a floor whose lowest z is above this is "elevated"
# COINCIDENT_Z_TOL = 1e-1    # roof/floor at the same height within this tolerance


# def _point_on_or_in_polygon_2d(px, py, polygon, tol=1e-6):
#     """True if the point is inside the polygon or within `tol` of its boundary."""
#     if _point_in_polygon_2d(px, py, polygon):
#         return True
#     n = len(polygon)
#     for i in range(n):
#         x0, y0 = polygon[i]
#         x1, y1 = polygon[(i + 1) % n]
#         dx, dy = x1 - x0, y1 - y0
#         seg2 = dx * dx + dy * dy
#         if seg2 < 1e-18:
#             continue
#         t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg2))
#         qx, qy = x0 + t * dx, y0 + t * dy
#         if (px - qx) ** 2 + (py - qy) ** 2 <= tol * tol:
#             return True
#     return False


# def _polygon_within_polygon_2d(inner, outer, tol=1e-6):
#     """True if every vertex of `inner` lies inside or on the boundary of `outer`."""
#     return all(_point_on_or_in_polygon_2d(x, y, outer, tol) for x, y in inner)


# def _interior_point_2d(polygon):
#     """A point guaranteed (in practice) to lie inside `polygon` (hole seed)."""
#     n = len(polygon)
#     cx = sum(p[0] for p in polygon) / n
#     cy = sum(p[1] for p in polygon) / n
#     if _point_in_polygon_2d(cx, cy, polygon):
#         return cx, cy
#     xs = [p[0] for p in polygon]
#     ys = [p[1] for p in polygon]
#     x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
#     for div in (4, 8, 16, 32, 64):
#         for j in range(1, div):
#             for i in range(1, div):
#                 px = x0 + (x1 - x0) * i / div
#                 py = y0 + (y1 - y0) * j / div
#                 if _point_in_polygon_2d(px, py, polygon):
#                     return px, py
#     return cx, cy


# def _drop_faces_inside_holes(verts, faces, holes):
#     """Fallback hole handling: discard triangles whose centroid is in a hole."""
#     if not holes:
#         return verts, faces
#     kept = []
#     for f in faces:
#         a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
#         gx = (a[0] + b[0] + c[0]) / 3.0
#         gy = (a[1] + b[1] + c[1]) / 3.0
#         if not any(_point_in_polygon_2d(gx, gy, h) for h in holes):
#             kept.append(f)
#     used = sorted({i for f in kept for i in f})
#     remap = {old: new for new, old in enumerate(used)}
#     return ([verts[i] for i in used],
#             [tuple(remap[i] for i in f) for f in kept])



# def fan_triangulate(n_verts):
#     return [(0, i, i + 1) for i in range(1, n_verts - 1)]


# def write_obj(path, vertices, faces, comment=""):
#     if not vertices or not faces:
#         with open(path, 'w') as fh:
#             if comment:
#                 fh.write(f"# {comment}\n")
#         print(f"  Written: {path}  (empty - no {comment} in input)")
#         return

#     # Faces are constructed independently throughout this module, so adjoining
#     # faces often contain separate vertex records at identical coordinates.
#     # OpenFOAM treats those as disconnected/open edges. Weld at the same
#     # precision used below when writing coordinates, then remap every face.
#     unique_vertices = []
#     vertex_ids = {}
#     old_to_new = []
#     for vertex in vertices:
#         key = tuple(round(float(coordinate), 6) for coordinate in vertex[:3])
#         if key not in vertex_ids:
#             vertex_ids[key] = len(unique_vertices)
#             unique_vertices.append(key)
#         old_to_new.append(vertex_ids[key])

#     welded_faces = []
#     for face in faces:
#         welded = tuple(old_to_new[index] for index in face)
#         if len(set(welded)) == 3:
#             welded_faces.append(welded)

#     with open(path, 'w') as fh:
#         if comment:
#             fh.write(f"# {comment}\n")
#         for v in unique_vertices:
#             fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
#         for face in welded_faces:
#             fh.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
#     print(
#         f"  Written: {path}  ({len(unique_vertices)} verts, "
#         f"{len(welded_faces)} tris; welded from {len(vertices)} verts)"
#     )


# def _remove_closing_dup(verts):
#     if len(verts) > 1 and verts[0] == verts[-1]:
#         return verts[:-1]
#     return verts





# # ---------------------------------------------------------------------------
# # mesh-size computation
# # ---------------------------------------------------------------------------

# MIN_BUILDING_MESH_SIZE = 1.0
# MESH_SIZE_CAP = 1.0  # no automatically generated mesh size may exceed this


# def compute_mesh_size(buildings, factor=1.0,
#                       minimum=MIN_BUILDING_MESH_SIZE):
#     """Return a safe global building mesh size.

#     The candidate size is the smallest valid building-edge length times
#     ``factor``, but it is clamped to ``minimum``.  This lower bound is
#     essential for city-scale data: tiny OSM edges must not force every large
#     roof and wall to be tessellated at centimetre resolution.

#     Degenerate edges (near-duplicate vertices, length below 1e-6 m) are
#     skipped so one bad vertex pair in the source JSON can't collapse
#     mesh_size to ~0 for the whole run.
#     """
#     min_edge = float('inf')
#     for bldg in buildings:
#         for face in bldg.get('faces', []):
#             vs = face.get('vertices', [])
#             n = len(vs)
#             if n < 2:
#                 continue
#             for i in range(n):
#                 v0, v1 = vs[i], vs[(i + 1) % n]
#                 dx = v1['x'] - v0['x']
#                 dy = v1['y'] - v0['y']
#                 dz = v1['z'] - v0['z']
#                 length = math.sqrt(dx * dx + dy * dy + dz * dz)
#                 if length > 1e-6:
#                     min_edge = min(min_edge, length)
#     if min_edge == float('inf'):
#         return minimum
#     return min(max(min_edge * factor, minimum), MESH_SIZE_CAP)


# def _min_edge_length_2d(poly, floor=1e-6):
#     """Smallest edge length of a closed 2D polygon ([(x, y), ...]).

#     Degenerate edges (length <= floor, e.g. a duplicated vertex) are
#     ignored. Returns None if the polygon has fewer than 2 valid edges.
#     """
#     n = len(poly)
#     if n < 2:
#         return None
#     min_edge = float('inf')
#     for i in range(n):
#         x0, y0 = poly[i]
#         x1, y1 = poly[(i + 1) % n]
#         length = math.hypot(x1 - x0, y1 - y0)
#         if length > floor:
#             min_edge = min(min_edge, length)
#     return min_edge if min_edge != float('inf') else None


# # ---------------------------------------------------------------------------
# # quad subdivision
# # ---------------------------------------------------------------------------

# def subdivide_quad(v0, v1, v2, v3, mesh_size):
#     """
#     Subdivide a planar quad (v0->v1->v2->v3) into a regular triangle grid
#     using bilinear interpolation.
#     Returns (local_verts, local_faces) with 0-based indices.
#     """
#     e01 = math.sqrt(sum((v1[k] - v0[k]) ** 2 for k in range(3)))
#     e03 = math.sqrt(sum((v3[k] - v0[k]) ** 2 for k in range(3)))
#     nu = max(1, round(e01 / mesh_size))
#     nv = max(1, round(e03 / mesh_size))

#     verts = []
#     for j in range(nv + 1):
#         t = j / nv
#         for i in range(nu + 1):
#             s = i / nu
#             p = tuple(
#                 (1 - s) * (1 - t) * v0[k] + s * (1 - t) * v1[k]
#                 + s * t * v2[k] + (1 - s) * t * v3[k]
#                 for k in range(3)
#             )
#             verts.append(p)

#     faces = []
#     for j in range(nv):
#         for i in range(nu):
#             p00 = j * (nu + 1) + i
#             p10 = j * (nu + 1) + i + 1
#             p11 = (j + 1) * (nu + 1) + i + 1
#             p01 = (j + 1) * (nu + 1) + i
#             faces.append((p00, p10, p11))
#             faces.append((p00, p11, p01))
#     return verts, faces


# # ---------------------------------------------------------------------------
# # flat polygon mesh
# # ---------------------------------------------------------------------------

# def _point_in_polygon_2d(px, py, polygon):
#     """Ray-casting point-in-polygon test."""
#     n = len(polygon)
#     inside = False
#     j = n - 1
#     for i in range(n):
#         xi, yi = polygon[i]
#         xj, yj = polygon[j]
#         denom = (yj - yi) or 1e-12
#         if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / denom + xi):
#             inside = not inside
#         j = i
#     return inside


# def mesh_flat_polygon(polygon_2d, mesh_size, z=0.0):
#     """
#     Tessellate a flat 2-D polygon with a regular grid.
#     Grid points that fall inside the polygon become vertices.
#     The effective grid spacing is clamped to at most 1/4 of the polygon's
#     shorter bounding-box side so thin shapes (roads, lawns) are always meshed.
#     Falls back to fan triangulation if no interior grid points are found.
#     Returns (verts, faces) with 0-based indices.
#     """
#     xs = [p[0] for p in polygon_2d]
#     ys = [p[1] for p in polygon_2d]
#     x0, x1 = min(xs), max(xs)
#     y0, y1 = min(ys), max(ys)

#     # Clamp mesh_size so we always have at least 4 divisions across the narrow side
#     short_side = min(x1 - x0, y1 - y0)
#     if short_side > 0:
#         effective_ms = min(mesh_size, short_side / 4.0)
#     else:
#         effective_ms = mesh_size
#     mesh_size = effective_ms

#     nx = max(1, math.ceil((x1 - x0) / mesh_size))
#     ny = max(1, math.ceil((y1 - y0) / mesh_size))
#     dx = (x1 - x0) / nx
#     dy = (y1 - y0) / ny

#     grid = {}
#     verts = []
#     for j in range(ny + 1):
#         for i in range(nx + 1):
#             px = x0 + i * dx
#             py = y0 + j * dy
#             if _point_in_polygon_2d(px, py, polygon_2d):
#                 grid[(i, j)] = len(verts)
#                 verts.append((px, py, z))

#     faces = []
#     for j in range(ny):
#         for i in range(nx):
#             p00 = grid.get((i,     j    ))
#             p10 = grid.get((i + 1, j    ))
#             p11 = grid.get((i + 1, j + 1))
#             p01 = grid.get((i,     j + 1))
#             if p00 is not None and p10 is not None and p11 is not None:
#                 faces.append((p00, p10, p11))
#             if p00 is not None and p11 is not None and p01 is not None:
#                 faces.append((p00, p11, p01))

#     # Fallback: fan-triangulate the boundary polygon if grid gave nothing
#     if not faces:
#         verts = [(p[0], p[1], z) for p in polygon_2d]
#         faces = fan_triangulate(len(verts))

#     return verts, faces


# def tessellate_polygon_delaunay(polygon_2d, mesh_size, z=0.0):
#     """
#     Triangulate a flat polygon using Delaunay triangulation (scipy).

#     All polygon boundary vertices are included as input points, so the
#     roof edge aligns exactly with the wall-top edge — no gap.  Interior
#     Cartesian grid points are added at 'mesh_size' spacing to control
#     triangle density.  Only triangles whose centroid lies inside the
#     polygon are kept, which cleanly handles non-convex shapes without
#     needing a per-triangle normal correction.

#     Falls back to tessellate_ring_polygon if scipy is unavailable.
#     """
#     try:
#         from scipy.spatial import Delaunay
#         import numpy as np
#     except ImportError:
#         return tessellate_ring_polygon(polygon_2d, mesh_size, z)

#     xs = [p[0] for p in polygon_2d]
#     ys = [p[1] for p in polygon_2d]
#     x0, x1 = min(xs), max(xs)
#     y0, y1 = min(ys), max(ys)

#     # Polygon boundary vertices + intermediate points along each edge.
#     # Using round(edge_len / mesh_size) subdivisions — the same formula as
#     # subdivide_quad — ensures intermediate positions coincide exactly with
#     # the wall-top-face vertices, keeping the surface geometrically closed.
#     # Dense boundary points prevent unconstrained Delaunay from creating long
#     # "shortcut" triangles that span concave corners; such triangles would
#     # fail the inside-polygon test and leave a visible gap.
#     pts = []
#     n_poly = len(polygon_2d)
#     for i in range(n_poly):
#         p0 = polygon_2d[i]
#         p1 = polygon_2d[(i + 1) % n_poly]
#         pts.append(p0)                    # corner vertex (always included)
#         edge_len = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
#         n_edge = max(1, round(edge_len / mesh_size))
#         for k in range(1, n_edge):        # skip k=0 (corner) and k=n_edge (next corner)
#             t = k / n_edge
#             pts.append((p0[0] + t * (p1[0] - p0[0]),
#                         p0[1] + t * (p1[1] - p0[1])))

#     # Interior Cartesian grid points
#     nx = max(1, math.ceil((x1 - x0) / mesh_size))
#     ny = max(1, math.ceil((y1 - y0) / mesh_size))
#     dx = (x1 - x0) / nx
#     dy = (y1 - y0) / ny
#     for j in range(ny + 1):
#         for i in range(nx + 1):
#             px = x0 + i * dx
#             py = y0 + j * dy
#             if _point_in_polygon_2d(px, py, polygon_2d):
#                 pts.append((px, py))

#     pts_arr = np.array(pts)
#     tri = Delaunay(pts_arr)

#     verts = [(p[0], p[1], z) for p in pts]
#     faces = []
#     for simplex in tri.simplices:
#         a = pts_arr[simplex[0]]
#         b = pts_arr[simplex[1]]
#         c = pts_arr[simplex[2]]
#         # Test 4 points (barycentric interior samples) to decide if the
#         # triangle lies inside the polygon.  Testing only the centroid
#         # fails for thin triangles at convex boundary corners: floating-
#         # point arithmetic can place the centroid just outside the polygon
#         # edge, falsely rejecting a valid triangle and leaving a gap.
#         # Using 4 samples (centroid + one point biased 50 % toward each
#         # vertex) and requiring ≥ 3 of 4 to be inside is robust enough to
#         # accept corner triangles while still excluding triangles that span
#         # a concave notch (whose sample points straddle the notch boundary).
#         samples = (
#             ((a[0] + b[0] + c[0]) / 3,       (a[1] + b[1] + c[1]) / 3),
#             ((2*a[0] + b[0] + c[0]) / 4,     (2*a[1] + b[1] + c[1]) / 4),
#             ((a[0] + 2*b[0] + c[0]) / 4,     (a[1] + 2*b[1] + c[1]) / 4),
#             ((a[0] + b[0] + 2*c[0]) / 4,     (a[1] + b[1] + 2*c[1]) / 4),
#         )
#         if sum(_point_in_polygon_2d(sx, sy, polygon_2d) for sx, sy in samples) >= 3:
#             faces.append(tuple(int(v) for v in simplex))

#     if not faces:
#         return tessellate_ring_polygon(polygon_2d, mesh_size, z)

#     return verts, faces

# def tessellate_polygon_constrained(polygon_2d, mesh_size, z=0.0, holes=None):
#     """
#     Constrained Delaunay triangulation of a flat polygon (Shewchuk's Triangle).

#     Polygon edges (and the edges of every polygon in `holes`) are hard
#     segments, so no triangle can cross them. Density is controlled by a maximum
#     triangle area of mesh_size**2 / 2, and each edge is split at ~mesh_size
#     spacing (same rule as subdivide_quad, so boundaries match adjacent walls).

#     `holes` is an optional list of 2-D polygons cut out of the surface.
#     Without the 'triangle' package the function falls back to the Delaunay
#     tessellation and simply discards triangles inside the holes (approximate).
#     """
#     holes = [h for h in (holes or []) if len(h) >= 3]
#     try:
#         import triangle as tr
#         import numpy as np
#     except ImportError:
#         vs, fs = tessellate_polygon_delaunay(polygon_2d, mesh_size, z)
#         return _drop_faces_inside_holes(vs, fs, holes)

#     pts = []      # [x, y]
#     index = {}    # de-duplicates shared vertices (e.g. hole touching outline)
#     segs = []

#     def _pid(x, y):
#         key = (round(x, 9), round(y, 9))
#         if key not in index:
#             index[key] = len(pts)
#             pts.append([x, y])
#         return index[key]

#     def _add_ring(ring):
#         ids = []
#         n = len(ring)
#         for i in range(n):
#             p0 = ring[i]
#             p1 = ring[(i + 1) % n]
#             ids.append(_pid(p0[0], p0[1]))
#             edge_len = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
#             n_edge = max(1, round(edge_len / mesh_size))
#             for k in range(1, n_edge):
#                 t = k / n_edge
#                 ids.append(_pid(p0[0] + t * (p1[0] - p0[0]),
#                                 p0[1] + t * (p1[1] - p0[1])))
#         m = len(ids)
#         for i in range(m):
#             a, b = ids[i], ids[(i + 1) % m]
#             if a != b:
#                 segs.append([a, b])

#     _add_ring(polygon_2d)
#     for h in holes:
#         _add_ring(h)

#     data = {
#         'vertices': np.array(pts, dtype=np.float64),
#         'segments': np.array(segs, dtype=np.int32),
#     }
#     if holes:
#         data['holes'] = np.array([_interior_point_2d(h) for h in holes],
#                                  dtype=np.float64)

#     max_area = 0.5 * mesh_size * mesh_size
#     # p = PSLG (enforce segments), a = area constraint. No 'q': quality
#     # refinement would insert Steiner points on the boundary edges.
#     try:
#         result = tr.triangulate(data, f'pa{max_area:.10f}')
#     except Exception as exc:
#         if not holes:
#             raise
#         print(f"  Warning: constrained triangulation with holes failed ({exc}); "
#               "falling back to approximate hole removal.")
#         vs, fs = tessellate_polygon_delaunay(polygon_2d, mesh_size, z)
#         return _drop_faces_inside_holes(vs, fs, holes)

#     verts = [(float(p[0]), float(p[1]), z) for p in result['vertices']]
#     faces = [tuple(int(v) for v in tri) for tri in result['triangles']]

#     if not faces:
#         vs, fs = tessellate_polygon_delaunay(polygon_2d, mesh_size, z)
#         return _drop_faces_inside_holes(vs, fs, holes)

#     return verts, faces


# def tessellate_ring_polygon(polygon_2d, mesh_size, z=0.0):
#     """
#     Tessellate a flat polygon using concentric rings from the centroid outward.
#     The outermost ring IS the polygon boundary - no gap between mesh and perimeter.
#     Suitable for circular or any convex/mildly concave N-gon face.
#     """
#     n = len(polygon_2d)
#     cx = sum(p[0] for p in polygon_2d) / n
#     cy = sum(p[1] for p in polygon_2d) / n

#     avg_r = sum(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
#                 for p in polygon_2d) / n
#     n_rings = max(1, round(avg_r / mesh_size))

#     verts = []
#     faces = []

#     # Index 0: centroid
#     verts.append((cx, cy, z))
#     ring_base = {}  # ring_base[r] = start index in verts for ring r (1-indexed)

#     for r in range(1, n_rings + 1):
#         t = r / n_rings          # 0 = centroid, 1 = boundary
#         start = len(verts)
#         ring_base[r] = start
#         for px, py in polygon_2d:
#             verts.append((cx + t * (px - cx), cy + t * (py - cy), z))

#         if r == 1:
#             # Centroid (idx 0) to first ring: N triangles
#             for i in range(n):
#                 j = (i + 1) % n
#                 faces.append((0, start + i, start + j))
#         else:
#             # Ring r-1 to ring r: N quads split into 2 triangles each
#             prev = ring_base[r - 1]
#             for i in range(n):
#                 j = (i + 1) % n
#                 p0, p1 = prev + i,  prev + j
#                 q0, q1 = start + i, start + j
#                 faces.append((p0, q0, q1))
#                 faces.append((p0, q1, p1))

#     return verts, faces


# def tessellate_planar_face_constrained(fv, mesh_size):
#     """
#     Constrained-Delaunay mesh of any planar 3-D polygon (e.g. a notched wall
#     with reflex corners, where a fan triangulation would cross the notch).

#     Builds a local 2-D frame in the face's own plane (Newell normal, with the
#     first in-plane axis horizontal so vertical wall edges stay vertical),
#     projects the polygon into it, meshes it with `tessellate_polygon_constrained`
#     (the same routine used for roofs/floors, so boundary points land at the
#     same round(edge_len / mesh_size) spacing as the neighbouring faces), then
#     maps the result back to 3-D.

#     Returns (verts3d, faces) or None if the polygon is degenerate (zero-area
#     or fewer than 3 usable vertices after removing a duplicated closing point).
#     """
#     pts = [(v['x'], v['y'], v['z']) for v in fv]
#     pts = _remove_closing_dup(pts)
#     n = len(pts)
#     if n < 3:
#         return None

#     # Newell's method: robust normal for a possibly non-planar-by-a-hair,
#     # possibly concave polygon.
#     nx = ny = nz = 0.0
#     for i in range(n):
#         x0, y0, z0 = pts[i]
#         x1, y1, z1 = pts[(i + 1) % n]
#         nx += (y0 - y1) * (z0 + z1)
#         ny += (z0 - z1) * (x0 + x1)
#         nz += (x0 - x1) * (y0 + y1)
#     nl = math.sqrt(nx * nx + ny * ny + nz * nz)
#     if nl < 1e-9:
#         return None
#     nx, ny, nz = nx / nl, ny / nl, nz / nl

#     # In-plane basis: e1 = n x z_up (horizontal, matches vertical wall edges),
#     # e2 = n x e1 (completes a right-handed frame so the mapping back to 3-D
#     # preserves winding). Falls back to an arbitrary basis when n is parallel
#     # to z_up; that case is a horizontal face, which this branch is not meant
#     # to handle anyway.
#     e1x, e1y, e1z = ny, -nx, 0.0
#     l1 = math.sqrt(e1x * e1x + e1y * e1y)
#     if l1 < 1e-9:
#         e1x, e1y, e1z = 1.0, 0.0, 0.0
#     else:
#         e1x, e1y, e1z = e1x / l1, e1y / l1, 0.0
#     e2x = ny * e1z - nz * e1y
#     e2y = nz * e1x - nx * e1z
#     e2z = nx * e1y - ny * e1x

#     ox, oy, oz = pts[0]
#     poly2d = []
#     for (px, py, pz) in pts:
#         dx, dy, dz = px - ox, py - oy, pz - oz
#         u = dx * e1x + dy * e1y + dz * e1z
#         v = dx * e2x + dy * e2y + dz * e2z
#         poly2d.append((u, v))

#     # Degenerate-area guard (near-collinear points collapsed by rounding).
#     area2 = sum(poly2d[i][0] * poly2d[(i + 1) % n][1]
#                 - poly2d[(i + 1) % n][0] * poly2d[i][1] for i in range(n))
#     if abs(area2) < 1e-9:
#         return None

#     lv, lf = tessellate_polygon_constrained(poly2d, mesh_size, z=0.0)
#     verts3d = [(ox + u * e1x + v * e2x,
#                 oy + u * e1y + v * e2y,
#                 oz + u * e1z + v * e2z) for (u, v, _) in lv]
#     return verts3d, lf


# # ---------------------------------------------------------------------------
# # buildings
# # ---------------------------------------------------------------------------

# def buildings_to_obj(buildings, path, mesh_size=None):
#     """
#     Write the exterior building surface mesh.

#     The JSON encodes multi-storey buildings as a ground-floor outer shell
#     ('Building N') plus per-storey slices ('building_N_internal_floor_K').
#     Each slice has its own wall faces spanning one storey height.

#     Face-type selection:
#       - 'wall'   -> always included.
#       - 'floor'  -> included ONLY when the building is elevated (its floor is
#                     above the ground plane, z > GROUND_Z_TOL) and is not an
#                     '_internal_floor' slice. Ground-level base slabs and
#                     inter-storey slabs are excluded.
#                     If the elevated floor rests on another building's roof at
#                     the same height, the overlap is removed (see module
#                     docstring): the supporting roof is dropped and its
#                     footprint becomes a hole in the floor.
#       - 'roof'   -> included only from the TOPMOST '_internal_floor' entry of a
#                     multi-storey building, or from a single-storey main entry
#                     that has no '_internal_floor' siblings. Any other roof is
#                     an internal slab and is excluded.
#       - other    -> excluded.
#     """
#     # Main building IDs that have _internal_floor siblings: their own 'roof'
#     # is just the ground-floor ceiling (internal slab).
#     main_ids_with_internal: set = set()
#     for b in buildings:
#         bid = b.get('id', '')
#         if '_internal_floor' in bid:
#             main_ids_with_internal.add(bid.split('_internal_floor')[0].lower())

#     # Only the topmost storey's roof of a multi-storey building is exterior.
#     topmost_internal_id: dict = {}   # prefix -> (bldg_id, roof_z)
#     for b in buildings:
#         bid = b.get('id', '')
#         if '_internal_floor' not in bid:
#             continue
#         roof_face = next((f for f in b.get('faces', [])
#                           if f.get('surfaceType') == 'roof'), None)
#         if roof_face is None or not roof_face.get('vertices'):
#             continue
#         zr = max(v['z'] for v in roof_face['vertices'])
#         prefix = bid.split('_internal_floor')[0]
#         if prefix not in topmost_internal_id or zr > topmost_internal_id[prefix][1]:
#             topmost_internal_id[prefix] = (bid, zr)
#     topmost_internal_ids = {v[0] for v in topmost_internal_id.values()}

#     def _belongs_to_exterior(bldg, face):
#         """Face-type rule: does this face belong in the exterior surface?"""
#         st = face['surfaceType']
#         fv = face['vertices']
#         bid = bldg.get('id', '')
#         is_internal = '_internal_floor' in bid
#         if st == 'wall':
#             return True
#         if st == 'roof':
#             if is_internal:
#                 return bid in topmost_internal_ids
#             return bid.lower().replace(' ', '_') not in main_ids_with_internal
#         if st == 'floor':
#             # NEW: keep the underside of buildings that do not touch the ground.
#             if is_internal or not fv:
#                 return False
#             return min(v['z'] for v in fv) > GROUND_Z_TOL
#         return False   # ceiling / foundation / other

#     # ---- pre-pass: resolve elevated floors against supporting roofs --------
#     exterior_roofs = []     # ((bldg_idx, face_idx), z, polygon)
#     elevated_floors = []
#     for bi, bldg in enumerate(buildings):
#         for fi, face in enumerate(bldg.get('faces', [])):
#             st = face['surfaceType']
#             fv = face['vertices']
#             if st not in ('roof', 'floor') or len(fv) < 3:
#                 continue
#             if not _belongs_to_exterior(bldg, face):
#                 continue
#             entry = ((bi, fi),
#                      sum(v['z'] for v in fv) / len(fv),
#                      [(v['x'], v['y']) for v in fv])
#             (exterior_roofs if st == 'roof' else elevated_floors).append(entry)

#     skip_faces = set()      # (bldg_idx, face_idx) removed as internal
#     floor_holes = {}        # floor key -> [hole polygons]
#     for fkey, fz, fpoly in elevated_floors:
#         holes = []
#         covered = False
#         for rkey, rz, rpoly in exterior_roofs:
#             if abs(rz - fz) > COINCIDENT_Z_TOL:
#                 continue
#             roof_in_floor = _polygon_within_polygon_2d(rpoly, fpoly)
#             floor_in_roof = _polygon_within_polygon_2d(fpoly, rpoly)
#             if roof_in_floor and floor_in_roof:      # identical footprints
#                 skip_faces.add(rkey)
#                 covered = True
#             elif roof_in_floor:                      # roof supports the floor
#                 skip_faces.add(rkey)
#                 holes.append(rpoly)
#             elif floor_in_roof:                      # floor fully covered
#                 covered = True
#         if covered:
#             skip_faces.add(fkey)
#         elif holes:
#             floor_holes[fkey] = holes

#     verts = []
#     faces = []
#     for bi, bldg in enumerate(buildings):
#         # XY centroid: outward-normal reference for wall faces (fallback when
#         # no 'normal' is present, and post-subdivision sanity check).
#         _all_fv = [v for f in bldg.get('faces', []) for v in f['vertices']]
#         if _all_fv:
#             bldg_cx = sum(v['x'] for v in _all_fv) / len(_all_fv)
#             bldg_cy = sum(v['y'] for v in _all_fv) / len(_all_fv)
#         else:
#             bldg_cx = bldg_cy = 0.0

#         for fi, face in enumerate(bldg.get('faces', [])):
#             key = (bi, fi)
#             if key in skip_faces:
#                 continue
#             st = face['surfaceType']
#             fv = face['vertices']
#             if not _belongs_to_exterior(bldg, face):
#                 continue
#             holes = floor_holes.get(key)

#             # Horizontal faces: wind so the normal points OUTWARD
#             # (roof -> +z, floor -> -z).
#             if st in ('roof', 'floor') and len(fv) >= 3:
#                 signed_area = sum(
#                     fv[i]['x'] * fv[(i + 1) % len(fv)]['y']
#                     - fv[(i + 1) % len(fv)]['x'] * fv[i]['y']
#                     for i in range(len(fv))
#                 )
#                 needs_reversed = (
#                     (st == 'roof' and signed_area < 0) or
#                     (st == 'floor' and signed_area > 0)
#                 )
#                 if needs_reversed:
#                     fv = list(reversed(fv))

#             # Walls: flip winding if it disagrees with the stated normal.
#             elif st == 'wall' and len(fv) >= 3:
#                 given_n = face.get('normal')
#                 if given_n:
#                     a = (fv[0]['x'], fv[0]['y'], fv[0]['z'])
#                     b_ = (fv[1]['x'], fv[1]['y'], fv[1]['z'])
#                     c = (fv[2]['x'], fv[2]['y'], fv[2]['z'])
#                     ax, ay, az = b_[0]-a[0], b_[1]-a[1], b_[2]-a[2]
#                     bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
#                     nx = ay*bz - az*by
#                     ny = az*bx - ax*bz
#                     nz = ax*by - ay*bx
#                     dot = nx*given_n['x'] + ny*given_n['y'] + nz*given_n['z']
#                     if dot < 0:
#                         fv = list(reversed(fv))

#             base = len(verts)
#             face_start = len(faces)
#             if mesh_size is not None and len(fv) == 4 and not holes:
#                 # Rectangular face without holes -> bilinear quad subdivision
#                 v0 = (fv[0]['x'], fv[0]['y'], fv[0]['z'])
#                 v1 = (fv[1]['x'], fv[1]['y'], fv[1]['z'])
#                 v2 = (fv[2]['x'], fv[2]['y'], fv[2]['z'])
#                 v3 = (fv[3]['x'], fv[3]['y'], fv[3]['z'])
#                 lv, lf = subdivide_quad(v0, v1, v2, v3, mesh_size)
#                 verts.extend(lv)
#                 for tri in lf:
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))
#             elif mesh_size is not None and st in ('floor', 'roof'):
#                 # Polygonal horizontal face (or a face with holes) ->
#                 # constrained Delaunay; boundary vertices coincide with the
#                 # wall edges so no gaps are left.
#                 z_val = fv[0]['z']
#                 poly = [(v['x'], v['y']) for v in fv]
#                 lv, lf = tessellate_polygon_constrained(
#                     poly, mesh_size, z=z_val, holes=holes)
#                 verts.extend(lv)
#                 for tri in lf:
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))
#             elif mesh_size is not None and st == 'wall' and len(fv) > 4:
#                 # Non-quad wall (e.g. a notched wall around a bridge opening,
#                 # with reflex corners) -> mesh it the same way as roofs/floors
#                 # instead of fan-triangulating, which only gives a correct
#                 # result for convex polygons.
#                 res = tessellate_planar_face_constrained(fv, mesh_size)
#                 if res is not None:
#                     lv, lf = res
#                     verts.extend(lv)
#                     for tri in lf:
#                         faces.append((base + tri[0], base + tri[1], base + tri[2]))
#                 else:
#                     for v in fv:
#                         verts.append((v['x'], v['y'], v['z']))
#                     for tri in fan_triangulate(len(fv)):
#                         faces.append((base + tri[0], base + tri[1], base + tri[2]))
#             else:
#                 # Non-quad wall or no mesh_size -> fan triangulate
#                 for v in fv:
#                     verts.append((v['x'], v['y'], v['z']))
#                 for tri in fan_triangulate(len(fv)):
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))

#             # Drop zero-area slivers (they crash surfaceFeatures) and enforce
#             # outward normals for roof (+z), floor (-z) and walls.
#             corrected = []
#             for k in range(face_start, len(faces)):
#                 tri = faces[k]
#                 a = verts[tri[0]]; b = verts[tri[1]]; c = verts[tri[2]]
#                 ax, ay, az = b[0]-a[0], b[1]-a[1], b[2]-a[2]
#                 bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
#                 nx = ay*bz - az*by
#                 ny = az*bx - ax*bz
#                 nz = ax*by - ay*bx
#                 if (nx*nx + ny*ny + nz*nz) < 1e-8:
#                     continue
#                 if st == 'roof' and nz < 0:
#                     tri = (tri[0], tri[2], tri[1])
#                 elif st == 'floor' and nz > 0:
#                     tri = (tri[0], tri[2], tri[1])
#                 elif st == 'wall':
#                     fcx = (a[0] + b[0] + c[0]) / 3
#                     fcy = (a[1] + b[1] + c[1]) / 3
#                     if nx * (fcx - bldg_cx) + ny * (fcy - bldg_cy) < 0:
#                         tri = (tri[0], tri[2], tri[1])
#                 corrected.append(tri)
#             del faces[face_start:]
#             faces.extend(corrected)
#     write_obj(path, verts, faces, comment="buildings")



# # ---------------------------------------------------------------------------
# # OSM buildings -> synthetic face format
# # ---------------------------------------------------------------------------

# def convert_osm_building(bldg):
#     """
#     Convert an 'osmBuildings' / 'osm_buildings' entry into the same
#     {'id', 'faces': [{'surfaceType', 'vertices':[{x,y,z}, ...]}]} shape that
#     buildings_to_obj() expects for the main 'buildings' entries.

#     OSM entries have an empty 'faces' list and instead store the footprint in
#     dimensions.baseVertices.  Some exporters use x/y for the ground plane and
#     z for elevation, while older variants use x/z for the ground plane.  The
#     varying coordinate is detected below.  Extrusion is from OBJ z=0 to the
#     building's top-level ``height``.
#     """
#     base_verts = bldg.get('dimensions', {}).get('baseVertices', [])
#     y_span = (max((v.get('y', 0.0) for v in base_verts), default=0.0)
#               - min((v.get('y', 0.0) for v in base_verts), default=0.0))
#     z_span = (max((v.get('z', 0.0) for v in base_verts), default=0.0)
#               - min((v.get('z', 0.0) for v in base_verts), default=0.0))
#     ground_key = 'y' if y_span >= z_span else 'z'
#     poly = [(v['x'], v.get(ground_key, 0.0)) for v in base_verts]
#     poly = _remove_closing_dup(poly)
#     if len(poly) < 3:
#         return None

#     height = bldg.get('height', 0)
#     if height <= 0:
#         return None

#     # Ensure CCW winding (viewed from above) so wall normals point outward.
#     # OSM/GIS sources are inconsistent about footprint winding direction; a
#     # CW polygon here would make every wall quad for this building face
#     # inward instead of outward.
#     n_poly = len(poly)
#     signed_area = sum(
#         poly[i][0] * poly[(i + 1) % n_poly][1]
#         - poly[(i + 1) % n_poly][0] * poly[i][1]
#         for i in range(n_poly)
#     )
#     if signed_area < 0:
#         poly = list(reversed(poly))

#     n = len(poly)
#     faces = []

#     # Wall faces: one quad per footprint edge, from ground (z=0) to roof (z=height)
#     for i in range(n):
#         j = (i + 1) % n
#         x0, y0 = poly[i]
#         x1, y1 = poly[j]
#         faces.append({
#             'surfaceType': 'wall',
#             'vertices': [
#                 {'x': x0, 'y': y0, 'z': 0.0},
#                 {'x': x1, 'y': y1, 'z': 0.0},
#                 {'x': x1, 'y': y1, 'z': height},
#                 {'x': x0, 'y': y0, 'z': height},
#             ],
#         })

#     # Roof face: footprint polygon at z=height (winding fixed up automatically
#     # in buildings_to_obj)
#     faces.append({
#         'surfaceType': 'roof',
#         'vertices': [{'x': x, 'y': y, 'z': height} for (x, y) in poly],
#     })

#     return {'id': bldg.get('id', 'osm_building'), 'faces': faces}


# def collect_osm_buildings(site):
#     """
#     Return converted OSM buildings from a site dict, checking both the
#     'osmBuildings' and 'osm_buildings' key spellings and de-duplicating by id
#     (some exports include both keys pointing at the same data).
#     """
#     raw = site.get('osmBuildings', []) or []
#     raw_alt = site.get('osm_buildings', []) or []

#     seen_ids = set()
#     merged = []
#     for b in list(raw) + list(raw_alt):
#         bid = b.get('id')
#         if bid is not None and bid in seen_ids:
#             continue
#         if bid is not None:
#             seen_ids.add(bid)
#         merged.append(b)

#     converted = []
#     for b in merged:
#         c = convert_osm_building(b)
#         if c is not None:
#             converted.append(c)
#     return converted


# # ---------------------------------------------------------------------------
# # trees
# # ---------------------------------------------------------------------------

# def _add_cylinder(cx, cy, z_bot, z_top, radius, n_seg, n_height, verts, faces,
#                    phase=0.0, cap_bottom=True, cap_top=True):
#     """Closed (or partially open) cylinder with n_height height bands and
#     n_seg circumferential segments.

#     ``phase`` rotates the cross-section (radians). For n_seg=4 (a square
#     cross-section), phase=pi/4 aligns the flat faces with the x/y axes
#     instead of leaving the corners pointing along x/y.

#     ``cap_bottom``/``cap_top`` control whether a disk/polygon cap is added
#     at each end. Set cap_top=False to leave the top ring open (e.g. so a
#     tree crown can attach there instead of resting on a closed cap) —
#     the returned info lets the caller pick up the exact same ring.
#     """
#     base = len(verts)
#     h = z_top - z_bot
#     for hi in range(n_height + 1):
#         z = z_bot + hi * h / n_height
#         for si in range(n_seg):
#             a = 2 * math.pi * si / n_seg + phase
#             verts.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z))

#     bot_ring = base
#     top_ring = base + n_height * n_seg

#     for hi in range(n_height):
#         for si in range(n_seg):
#             sj = (si + 1) % n_seg
#             b0 = base + hi * n_seg + si
#             b1 = base + hi * n_seg + sj
#             t0 = base + (hi + 1) * n_seg + si
#             t1 = base + (hi + 1) * n_seg + sj
#             faces.append((b0, b1, t1))
#             faces.append((b0, t1, t0))

#     if cap_bottom:
#         bc = len(verts)
#         verts.append((cx, cy, z_bot))
#         for si in range(n_seg):
#             sj = (si + 1) % n_seg
#             faces.append((bc, bot_ring + sj, bot_ring + si))   # bottom cap

#     if cap_top:
#         tc = len(verts)
#         verts.append((cx, cy, z_top))
#         for si in range(n_seg):
#             sj = (si + 1) % n_seg
#             faces.append((tc, top_ring + si, top_ring + sj))   # top cap

#     return {"top_ring_start": top_ring, "n_top_ring": n_seg}


# def _add_cone(cx, cy, z_base, z_tip, radius, n_seg, n_height, verts, faces):
#     """Closed cone with n_height rings tapering from base radius to apex."""
#     base = len(verts)
#     h = z_tip - z_base
#     for hi in range(n_height):
#         r_hi = radius * (1.0 - hi / n_height)
#         z_hi = z_base + hi * h / n_height
#         for si in range(n_seg):
#             a = 2 * math.pi * si / n_seg
#             verts.append((cx + r_hi * math.cos(a), cy + r_hi * math.sin(a), z_hi))
#     verts.append((cx, cy, z_tip))   # apex:        base + n_height*n_seg
#     verts.append((cx, cy, z_base))  # base centre: base + n_height*n_seg + 1

#     for hi in range(n_height - 1):
#         for si in range(n_seg):
#             sj = (si + 1) % n_seg
#             b0 = base + hi * n_seg + si;       b1 = base + hi * n_seg + sj
#             t0 = base + (hi + 1) * n_seg + si; t1 = base + (hi + 1) * n_seg + sj
#             faces.append((b0, b1, t1));  faces.append((b0, t1, t0))

#     apex     = base + n_height * n_seg
#     top_ring = base + (n_height - 1) * n_seg
#     for si in range(n_seg):
#         sj = (si + 1) % n_seg
#         faces.append((top_ring + si, top_ring + sj, apex))

#     bc = base + n_height * n_seg + 1
#     for si in range(n_seg):
#         sj = (si + 1) % n_seg
#         faces.append((bc, base + sj, base + si))   # base cap


# def _add_spheroid(cx, cy, z_bottom, z_top, radius, n_seg, n_height, verts, faces):
#     """
#     Add a closed prolate/oblate spheroid crown.

#     ``radius`` is the horizontal semi-axis, while half of ``z_top-z_bottom``
#     is the vertical semi-axis.  The lower pole is placed at the trunk top and
#     the upper pole at the total tree height.
#     """
#     crown_h = z_top - z_bottom
#     if crown_h <= 0 or radius <= 0:
#         return

#     # At least two intermediate latitude rings are needed for a recognisable
#     # spheroid.  More rings are selected by trees_to_obj when mesh_size is set.
#     n_rings = max(2, n_height)
#     base = len(verts)

#     bottom_pole = len(verts)
#     verts.append((cx, cy, z_bottom))

#     z_mid = 0.5 * (z_bottom + z_top)
#     vertical_radius = 0.5 * crown_h
#     ring_starts = []

#     # Latitude angle runs from -pi/2 (bottom pole) to +pi/2 (top pole).
#     # The poles themselves are separate vertices, so only interior rings are
#     # generated here.
#     for ri in range(1, n_rings + 1):
#         latitude = -0.5 * math.pi + math.pi * ri / (n_rings + 1)
#         ring_radius = radius * math.cos(latitude)
#         z = z_mid + vertical_radius * math.sin(latitude)
#         ring_starts.append(len(verts))
#         for si in range(n_seg):
#             a = 2.0 * math.pi * si / n_seg
#             verts.append((cx + ring_radius * math.cos(a),
#                           cy + ring_radius * math.sin(a),
#                           z))

#     top_pole = len(verts)
#     verts.append((cx, cy, z_top))

#     # Bottom pole to first ring: reverse circumferential order so normals point
#     # outward/downward near the lower pole.
#     first_ring = ring_starts[0]
#     for si in range(n_seg):
#         sj = (si + 1) % n_seg
#         faces.append((bottom_pole, first_ring + sj, first_ring + si))

#     # Connect consecutive latitude rings.
#     for ri in range(len(ring_starts) - 1):
#         lower = ring_starts[ri]
#         upper = ring_starts[ri + 1]
#         for si in range(n_seg):
#             sj = (si + 1) % n_seg
#             b0, b1 = lower + si, lower + sj
#             t0, t1 = upper + si, upper + sj
#             faces.append((b0, b1, t1))
#             faces.append((b0, t1, t0))

#     # Last ring to top pole.
#     last_ring = ring_starts[-1]
#     for si in range(n_seg):
#         sj = (si + 1) % n_seg
#         faces.append((top_pole, last_ring + si, last_ring + sj))


# def _bridge_rings(lower_start, n_lower, upper_start, n_upper, faces):
#     """Triangulate the band between two co-axial rings that may have a
#     different number of vertices (e.g. a 4-vertex square hole widening into
#     an n_seg-vertex circular ring). Both rings must be ordered by increasing
#     angle around the same center/axis.

#     Winding matches the convention used elsewhere in this file for ascending
#     bands (cylinder side walls, spheroid latitude bands): outward normals.
#     Reduces to the standard per-quad split when n_lower == n_upper.
#     """
#     i = j = 0
#     steps = n_lower + n_upper
#     for _ in range(steps):
#         advance_lower = (j >= n_upper) or (
#             i < n_lower and (i + 1) / n_lower <= (j + 1) / n_upper)
#         if advance_lower:
#             i_next = (i + 1) % n_lower
#             faces.append((lower_start + i, lower_start + i_next,
#                           upper_start + (j % n_upper)))
#             i += 1
#         else:
#             j_next = (j + 1) % n_upper
#             faces.append((lower_start + (i % n_lower),
#                           upper_start + j_next, upper_start + j))
#             j += 1


# def _add_sphere_crown_with_hole(cx, cy, z_hole, hole_r, hole_phase,
#                                  crown_r, n_seg, n_height, verts, faces):
#     """Spherical crown with a small square hole at the bottom instead of a
#     closed pole, so it seats flush on an open-topped trunk of matching
#     cross-section.

#     The hole ring (4 vertices, radius=hole_r, centered at (cx,cy,z_hole),
#     rotated by hole_phase) is placed at EXACTLY the coordinates the trunk's
#     open top ring uses, so the two surfaces share a common boundary loop —
#     no cap on either side, no touching-at-a-point pinch, just one
#     continuous open surface through the join.

#     Returns the apex z (top pole height), since slicing the bottom means
#     the crown is very slightly shorter than a full 2*crown_r resting sphere.
#     """
#     if crown_r <= 0 or hole_r <= 0 or hole_r >= crown_r:
#         # Degenerate (trunk as wide as the crown): fall back to the old
#         # closed sphere resting on top rather than producing a bad hole.
#         z_top = z_hole + 2.0 * crown_r
#         _add_spheroid(cx, cy, z_hole, z_top, crown_r, n_seg, n_height, verts, faces)
#         return z_top

#     hole_seg = 4
#     # Latitude (below the equator) at which a true sphere of radius crown_r
#     # has ring-radius == hole_r; the hole ring sits there instead of at the
#     # pole, effectively slicing off a small polar cap the trunk occupies.
#     latitude0 = -math.acos(hole_r / crown_r)
#     z_center  = z_hole - crown_r * math.sin(latitude0)
#     z_top     = z_center + crown_r

#     hole_start = len(verts)
#     for si in range(hole_seg):
#         a = 2.0 * math.pi * si / hole_seg + hole_phase
#         verts.append((cx + hole_r * math.cos(a), cy + hole_r * math.sin(a), z_hole))

#     n_rings = max(2, n_height)
#     ring_starts = []
#     for ri in range(1, n_rings + 1):
#         latitude = latitude0 + (0.5 * math.pi - latitude0) * ri / (n_rings + 1)
#         ring_radius = crown_r * math.cos(latitude)
#         z = z_center + crown_r * math.sin(latitude)
#         ring_starts.append(len(verts))
#         for si in range(n_seg):
#             a = 2.0 * math.pi * si / n_seg
#             verts.append((cx + ring_radius * math.cos(a),
#                           cy + ring_radius * math.sin(a),
#                           z))

#     top_pole = len(verts)
#     verts.append((cx, cy, z_top))

#     # Hole ring -> first latitude ring (different vertex counts: bridge it).
#     _bridge_rings(hole_start, hole_seg, ring_starts[0], n_seg, faces)

#     # Interior latitude rings (equal counts throughout, same as _add_spheroid).
#     for ri in range(len(ring_starts) - 1):
#         lower = ring_starts[ri]
#         upper = ring_starts[ri + 1]
#         for si in range(n_seg):
#             sj = (si + 1) % n_seg
#             b0, b1 = lower + si, lower + sj
#             t0, t1 = upper + si, upper + sj
#             faces.append((b0, b1, t1))
#             faces.append((b0, t1, t0))

#     # Last ring to top pole (closed apex, unchanged from _add_spheroid).
#     last_ring = ring_starts[-1]
#     for si in range(n_seg):
#         sj = (si + 1) % n_seg
#         faces.append((top_pole, last_ring + si, last_ring + sj))

#     return z_top


# def _add_ellipsoid(cx, cy, z_bottom, semi_minor, semi_major,
#                    n_seg, n_height, verts, faces):
#     """
#     Add a closed, vertically oriented prolate ellipsoid crown.

#     ``semi_minor`` is the horizontal semi-axis in both x and y and is taken
#     directly from ``crownRadius``. ``semi_major`` is the vertical semi-axis
#     and is taken directly from the crown height
#     (``totalHeight - trunkHeight``).

#     The lower pole is attached to the trunk top at ``z_bottom``. Therefore,
#     the crown centre is ``z_bottom + semi_major`` and its upper pole is
#     ``z_bottom + 2*semi_major``.
#     """
#     if semi_major <= 0 or semi_minor <= 0:
#         return

#     z_top = z_bottom + 2.0 * semi_major
#     _add_spheroid(cx, cy, z_bottom, z_top, semi_minor,
#                   n_seg, n_height, verts, faces)


# def trees_to_obj(trees, path, mesh_size=None, crown_shape="sphere"):
#     verts = []
#     faces = []
#     for tree in trees:
#         pos     = tree['position']
#         cx, cy  = pos['x'], pos['y']
#         pz      = pos.get('z', 0.0)
#         d       = tree['dimensions']
#         trunk_h = d['trunkHeight']
#         crown_r = d['crownRadius']
#         crown_r = crown_r*0.95
#         trunk_r = 0.5 * crown_r
#         total_h = d['totalHeight']

#         z_ground    = pz
#         z_trunk_top = pz + trunk_h
#         z_apex      = pz + total_h

#         TRUNK_PHASE = math.pi / 4   # axis-aligned square (flat faces on x/y)

#         # Tree mesh density is tied to this tree's own crown radius, not an
#         # externally supplied mesh_size: local_ms = crown_r / 10.
#         local_ms    = min(max(crown_r / 10.0, 1e-4), MESH_SIZE_CAP*0.5)
#         n_seg_crown = max(6, round(2 * math.pi * crown_r / local_ms))
#         n_h_trunk   = max(1, round(trunk_h / local_ms))
#         n_h_crown   = max(2, round((total_h - trunk_h) / local_ms))

#         if crown_shape == "sphere":
#             # Square trunk, top left open; the sphere crown's bottom hole
#             # (same 4-vertex square, same coordinates) fills the opening,
#             # so trunk and crown form one continuous surface instead of a
#             # capped trunk touching a closed sphere at a single point.
#             _add_cylinder(cx, cy, z_ground, z_trunk_top, trunk_r,
#                           4, n_h_trunk, verts, faces,
#                           phase=TRUNK_PHASE, cap_top=False)
#             _add_sphere_crown_with_hole(cx, cy, z_trunk_top, trunk_r, TRUNK_PHASE,
#                                          crown_r, n_seg_crown, n_h_crown, verts, faces)
#             continue

#         _add_cylinder(cx, cy, z_ground, z_trunk_top, trunk_r,
#                       4, n_h_trunk, verts, faces, phase=TRUNK_PHASE)
#         if crown_shape == "cone":
#             _add_cone(cx, cy, z_trunk_top, z_apex, crown_r,
#                       n_seg_crown, n_h_crown, verts, faces)
#         elif crown_shape == "spheroid":
#             _add_spheroid(cx, cy, z_trunk_top, z_apex, crown_r,
#                            n_seg_crown, n_h_crown, verts, faces)
#         elif crown_shape == "ellipsoid":
#             crown_h = total_h - trunk_h
#             _add_ellipsoid(cx, cy, z_trunk_top, crown_r, crown_h,
#                            n_seg_crown, max(2, 2 * n_h_crown), verts, faces)
#         else:
#             raise ValueError(
#                 f"Unsupported crown_shape '{crown_shape}'. "
#                 "Choose 'cone', 'sphere', 'spheroid', or 'ellipsoid'."
#             )

#     write_obj(path, verts, faces, comment="trees")


# # ---------------------------------------------------------------------------
# # ponds
# # ---------------------------------------------------------------------------

# def ponds_to_obj(ponds, path, mesh_size=None):
#     """
#     Closed watertight pond: meshed top (z=0), meshed bottom (z=-depth),
#     subdivided side quads for each boundary edge.
#     """
#     verts = []
#     faces = []
#     for pond in ponds:
#         coords = pond['coordinates']
#         depth  = 0.0
#         poly   = _remove_closing_dup([(c['x'], c['y']) for c in coords])
#         # Ensure CCW winding so top face has +z normal
#         _n = len(poly)
#         _sa = sum(poly[i][0]*poly[(_i1:=(i+1)%_n)][1] - poly[_i1][0]*poly[i][1]
#                   for i in range(_n))
#         if _sa < 0:
#             poly = list(reversed(poly))
#         n      = len(poly)

#         # This pond's own mesh density: its own minimum edge length / 2.
#         _min_edge = _min_edge_length_2d(poly)
#         mesh_size = min(max(_min_edge, 0.01), MESH_SIZE_CAP) if _min_edge is not None else None

#         if depth == 0.0:
#             # Flat pond (no volume) – emit top face only
#             if mesh_size is not None:
#                 tv, tf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
#                 base = len(verts); verts.extend(tv)
#                 for tri in tf:
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))
#             else:
#                 base = len(verts)
#                 verts.extend([(p[0], p[1], 0.0) for p in poly])
#                 for tri in fan_triangulate(n):
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))
#         elif mesh_size is not None:
#             tv, tf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
#             base = len(verts); verts.extend(tv)
#             for tri in tf:
#                 faces.append((base + tri[0], base + tri[1], base + tri[2]))

#             bv, bf = tessellate_polygon_constrained(poly, mesh_size, z=-depth)
#             base = len(verts); verts.extend(bv)
#             for tri in bf:                          # reverse winding for bottom
#                 faces.append((base + tri[0], base + tri[2], base + tri[1]))

#             for i in range(n):
#                 j = (i + 1) % n
#                 sv0 = (poly[i][0], poly[i][1],  0.0)
#                 sv1 = (poly[j][0], poly[j][1],  0.0)
#                 sv2 = (poly[j][0], poly[j][1], -depth)
#                 sv3 = (poly[i][0], poly[i][1], -depth)
#                 lv, lf = subdivide_quad(sv0, sv1, sv2, sv3, mesh_size)
#                 base = len(verts); verts.extend(lv)
#                 for tri in lf:
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))
#         else:
#             top_base = len(verts)
#             verts.extend([(p[0], p[1],  0.0)   for p in poly])
#             bot_base = len(verts)
#             verts.extend([(p[0], p[1], -depth) for p in poly])
#             for tri in fan_triangulate(n):
#                 faces.append((top_base + tri[0], top_base + tri[1], top_base + tri[2]))
#             for i in range(n):
#                 j = (i + 1) % n
#                 t0, t1 = top_base + i, top_base + j
#                 b0, b1 = bot_base + i, bot_base + j
#                 faces.append((t0, b0, b1));  faces.append((t0, b1, t1))
#             for tri in fan_triangulate(n):
#                 faces.append((bot_base + tri[0], bot_base + tri[2], bot_base + tri[1]))

#     write_obj(path, verts, faces, comment="ponds")


# # ---------------------------------------------------------------------------
# # lawns / grassAreas
# # ---------------------------------------------------------------------------

# def lawns_to_obj(grass_areas, path, mesh_size=None):
#     verts = []
#     faces = []
#     for area in grass_areas:
#         poly = _remove_closing_dup([(c['x'], c['y']) for c in area['coordinates']])
#         # Ensure CCW winding so tessellated faces have +z (upward) normals
#         _n = len(poly)
#         _sa = sum(poly[i][0]*poly[(_i1:=(i+1)%_n)][1] - poly[_i1][0]*poly[i][1]
#                   for i in range(_n))
#         if _sa < 0:
#             poly = list(reversed(poly))

#         # This lawn's own mesh density: its own minimum edge length / 2.
#         _min_edge = _min_edge_length_2d(poly)
#         mesh_size = min(max(_min_edge, 10.0),2.5 * MESH_SIZE_CAP) if _min_edge is not None else None

#         if mesh_size is not None:
#             # CDT enforces polygon edges as hard constraints — no triangle can
#             # cross the boundary so curved shapes have no outward spikes.
#             lv, lf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
#             base = len(verts); verts.extend(lv)
#             for tri in lf:
#                 faces.append((base + tri[0], base + tri[1], base + tri[2]))
#         else:
#             base = len(verts)
#             verts.extend([(p[0], p[1], 0.0) for p in poly])
#             for tri in fan_triangulate(len(poly)):
#                 faces.append((base + tri[0], base + tri[1], base + tri[2]))
#     write_obj(path, verts, faces, comment="lawns")


# # ---------------------------------------------------------------------------
# # pathways / roads
# # ---------------------------------------------------------------------------

# def pathways_to_obj(roads, path, mesh_size=None):
#     verts = []
#     faces = []
#     for road in roads:
#         coords = road.get('coordinates', [])
#         if not coords:
#             continue
#         poly = _remove_closing_dup([(c['x'], c['y']) for c in coords])
#         # Ensure CCW winding so tessellated faces have +z (upward) normals
#         _n = len(poly)
#         _sa = sum(poly[i][0]*poly[(_i1:=(i+1)%_n)][1] - poly[_i1][0]*poly[i][1]
#                   for i in range(_n))
#         if _sa < 0:
#             poly = list(reversed(poly))
#         if mesh_size is not None:
#             lv, lf = tessellate_polygon_constrained(poly, mesh_size, z=0.0)
#             base = len(verts); verts.extend(lv)
#             for tri in lf:
#                 faces.append((base + tri[0], base + tri[1], base + tri[2]))
#         else:
#             base = len(verts)
#             verts.extend([(p[0], p[1], 0.0) for p in poly])
#             for tri in fan_triangulate(len(poly)):
#                 faces.append((base + tri[0], base + tri[1], base + tri[2]))
#     write_obj(path, verts, faces, comment="pathways")


# # ---------------------------------------------------------------------------
# # JSON structure helper
# # ---------------------------------------------------------------------------

# def _extract_site(data):
#     """
#     Return the site dict regardless of whether the JSON uses the wrapped format
#     (data['siteOnly_json']) or the flat format (data contains the keys directly).
#     """
#     if 'siteOnly_json' in data:
#         return data['siteOnly_json']
#     if 'buildings' in data or 'metadata' in data:
#         return data
#     raise KeyError("Cannot find site data in JSON: expected 'siteOnly_json' key "
#                    "or flat structure with 'buildings'/'metadata'.")


# # ---------------------------------------------------------------------------
# # main
# # ---------------------------------------------------------------------------

# def main():
#     args = sys.argv[1:]

#     # Optional explicit building resolution.  The safe default is 1 m.  An
#     # explicit value bypasses automatic sizing but remains validated here so
#     # accidental zero/negative values cannot trigger runaway subdivisions.
#     building_mesh_size = None
#     if "--building-mesh-size" in args:
#         option_index = args.index("--building-mesh-size")
#         if option_index + 1 >= len(args):
#             print("Error: --building-mesh-size requires a positive value in metres.",
#                   file=sys.stderr)
#             sys.exit(2)
#         try:
#             building_mesh_size = float(args[option_index + 1])
#         except ValueError:
#             print("Error: --building-mesh-size must be a number in metres.",
#                   file=sys.stderr)
#             sys.exit(2)
#         if not math.isfinite(building_mesh_size) or building_mesh_size <= 0:
#             print("Error: --building-mesh-size must be finite and greater than zero.",
#                   file=sys.stderr)
#             sys.exit(2)
#         del args[option_index:option_index + 2]

#     # Optional tree-crown switch.  It is removed before the existing positional
#     # input/output parsing so all old invocations continue to work unchanged.
#     crown_shape = os.environ.get("TREE_CROWN_SHAPE", "sphere").strip().lower()
#     if "--tree-crown" in args:
#         option_index = args.index("--tree-crown")
#         if option_index + 1 >= len(args):
#             print("Error: --tree-crown requires 'sphere', 'cone', 'spheroid', or 'ellipsoid'.",
#                   file=sys.stderr)
#             sys.exit(2)
#         crown_shape = args[option_index + 1].strip().lower()
#         del args[option_index:option_index + 2]

#     crown_aliases = {
#         "sphere": "sphere",
#         "spherical": "sphere",
#         "cone": "cone",
#         "conical": "cone",
#         "spheroid": "spheroid",
#         "spheroidal": "spheroid",
#         "ellipsoid": "ellipsoid",
#         "ellipsoidal": "ellipsoid",
#     }
#     crown_shape = crown_aliases.get(crown_shape, crown_shape)
#     if crown_shape not in ("sphere", "cone", "spheroid", "ellipsoid"):
#         print(f"Error: invalid tree crown shape '{crown_shape}'. "
#               "Use 'sphere', 'cone', 'spheroid', or 'ellipsoid'.", file=sys.stderr)
#         sys.exit(2)

#     # Last arg is out_dir if it does not end with .json
#     if args and not args[-1].lower().endswith('.json'):
#         out_dir    = args[-1]
#         json_paths = args[:-1]
#     else:
#         out_dir    = "constant/triSurface"
#         json_paths = args

#     if not json_paths:
#         json_paths = ["MassingOBJ_reference.json"]

#     for p in json_paths:
#         if not os.path.isfile(p):
#             print(f"Error: JSON file not found: {p}", file=sys.stderr)
#             sys.exit(1)

#     os.makedirs(out_dir, exist_ok=True)

#     all_buildings = []
#     all_trees     = []
#     all_ponds     = []
#     all_grass     = []
#     all_roads     = []

#     for json_path in json_paths:
#         print(f"Reading {json_path} ...")
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         try:
#             site = _extract_site(data)
#         except KeyError as e:
#             print(f"  Warning: skipping '{json_path}': {e}", file=sys.stderr)
#             continue

#         meta = site.get('metadata', {}).get('sceneInfo', {})
#         roads = site.get('roads', []) + site.get('pathways', [])
#         osm_buildings = collect_osm_buildings(site)
#         print(f"  Scene: {meta.get('totalBuildings', len(site.get('buildings', [])))}"
#               f" buildings, {len(osm_buildings)} osm buildings,"
#               f" {meta.get('totalTrees', len(site.get('trees', [])))}"
#               f" trees, {meta.get('totalPonds', len(site.get('ponds', [])))}"
#               f" ponds, {meta.get('totalGrassAreas', len(site.get('grassAreas', [])))}"
#               f" grass areas, {meta.get('totalRoads', len(roads))}"
#               f" roads")

#         all_buildings.extend(site.get('buildings',  []))
#         all_buildings.extend(osm_buildings)
#         all_trees    .extend(site.get('trees',       []))
#         all_ponds    .extend(site.get('ponds',       []))
#         all_grass    .extend(site.get('grassAreas',  []))
#         all_roads    .extend(roads)

#     print(f"\nTotal across {len(json_paths)} file(s): "
#           f"{len(all_buildings)} buildings, {len(all_trees)} trees, "
#           f"{len(all_ponds)} ponds, {len(all_grass)} lawns, "
#           f"{len(all_roads)} roads")

#     if building_mesh_size is None:
#         mesh_size = compute_mesh_size(all_buildings)
#         mesh_description = f"automatic, minimum {MIN_BUILDING_MESH_SIZE:g} m"
#     else:
#         mesh_size = building_mesh_size
#         mesh_description = "command-line override"
#     print(f"Mesh size (buildings): {mesh_size:.4f} m  ({mesh_description})")
#     print("Mesh size (trees):  per-tree, crown radius / 10")
#     print("Mesh size (ponds):  per-pond, that pond's own minimum edge / 2")
#     print("Mesh size (lawns):  per-lawn, that lawn's own minimum edge / 2\n")

#     print("Extracting buildings  -> buildings_refined.obj")
#     buildings_to_obj(all_buildings,
#                      os.path.join(out_dir, 'buildings_refined.obj'), mesh_size)

#     print(f"Extracting trees      -> trees.obj  (crown: {crown_shape})")
#     trees_to_obj(all_trees, os.path.join(out_dir, 'trees.obj'),
#                  crown_shape=crown_shape)

#     print("Extracting ponds      -> ponds.obj")
#     ponds_to_obj(all_ponds, os.path.join(out_dir, 'ponds.obj'))

#     print("Extracting lawns      -> lawns.obj")
#     lawns_to_obj(all_grass, os.path.join(out_dir, 'lawns.obj'))

#     print("Extracting pathways   -> pathways.obj")
#     pathways_to_obj(all_roads, os.path.join(out_dir, 'pathways.obj'), mesh_size)

#     print("\nDone.")

#     print("\n \nEnd\n")


# if __name__ == "__main__":
#     main()
