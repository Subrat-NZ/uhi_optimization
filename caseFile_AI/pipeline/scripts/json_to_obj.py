#!/usr/bin/env python3
"""
json_to_obj.py
Extract buildings, trees, ponds, lawns (grassAreas) and pathways (roads)
from a MassingOBJ JSON file and write surface-meshed OBJ files.

Usage:
    python json_to_obj.py [--building-mesh-size METRES]
                          [input1.json [input2.json ...]] [output_dir]

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

def fan_triangulate(n_verts):
    return [(0, i, i + 1) for i in range(1, n_verts - 1)]


def write_obj(path, vertices, faces, comment=""):
    if not vertices or not faces:
        with open(path, 'w') as fh:
            if comment:
                fh.write(f"# {comment}\n")
        print(f"  Written: {path}  (empty - no {comment} in input)")
        return
    with open(path, 'w') as fh:
        if comment:
            fh.write(f"# {comment}\n")
        for v in vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            fh.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Written: {path}  ({len(vertices)} verts, {len(faces)} tris)")


def _remove_closing_dup(verts):
    if len(verts) > 1 and verts[0] == verts[-1]:
        return verts[:-1]
    return verts


def _polygon_signed_area_2d(poly):
    """Signed area of an open 2-D polygon vertex list."""
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def _orient_2d(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect_2d(a, b, c, d, tolerance=1e-9):
    """Return True when the two closed line segments intersect."""
    def sign(value):
        if value > tolerance:
            return 1
        if value < -tolerance:
            return -1
        return 0

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - tolerance <= q[0]
            <= max(p[0], r[0]) + tolerance
            and min(p[1], r[1]) - tolerance <= q[1]
            <= max(p[1], r[1]) + tolerance
        )

    o1 = _orient_2d(a, b, c)
    o2 = _orient_2d(a, b, d)
    o3 = _orient_2d(c, d, a)
    o4 = _orient_2d(c, d, b)
    s1, s2, s3, s4 = map(sign, (o1, o2, o3, o4))
    if s1 != s2 and s3 != s4:
        return True
    return (
        (s1 == 0 and on_segment(a, c, b))
        or (s2 == 0 and on_segment(a, d, b))
        or (s3 == 0 and on_segment(c, a, d))
        or (s4 == 0 and on_segment(c, b, d))
    )


def _polygon_is_simple_2d(poly):
    """Check that non-neighbouring polygon edges do not intersect."""
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        i_next = (i + 1) % n
        for j in range(i + 1, n):
            j_next = (j + 1) % n
            if i == j or i_next == j or j_next == i:
                continue
            if _segments_intersect_2d(
                    poly[i], poly[i_next], poly[j], poly[j_next]):
                return False
    return True


def remove_sharp_footprint_vertices(poly, building_id="osm_building",
                                    minimum_angle=30.0,
                                    maximum_angle=330.0,
                                    maximum_area_change=0.05):
    """Remove validated acute/reflex spikes from a building footprint.

    Angles are winding-aware interior angles in [0, 360). A candidate is
    removed only when the result remains simple, keeps its winding, has
    non-zero area, and changes the current area by no more than
    ``maximum_area_change``. One point is removed per pass so neighbouring
    angles are recalculated against the updated polygon.
    """
    cleaned = list(poly)
    removed = []
    while len(cleaned) > 3:
        signed_area = _polygon_signed_area_2d(cleaned)
        if abs(signed_area) <= 1e-12:
            break
        winding = 1.0 if signed_area > 0 else -1.0
        changed = False

        for i, current in enumerate(cleaned):
            previous = cleaned[i - 1]
            following = cleaned[(i + 1) % len(cleaned)]
            incoming = (current[0] - previous[0],
                        current[1] - previous[1])
            outgoing = (following[0] - current[0],
                        following[1] - current[1])
            if (math.hypot(*incoming) <= 1e-9
                    or math.hypot(*outgoing) <= 1e-9):
                continue

            turn = math.degrees(math.atan2(
                incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                incoming[0] * outgoing[0] + incoming[1] * outgoing[1],
            ))
            interior_angle = (180.0 - winding * turn) % 360.0
            if not (interior_angle < minimum_angle
                    or interior_angle > maximum_angle):
                continue

            candidate = cleaned[:i] + cleaned[i + 1:]
            candidate_area = _polygon_signed_area_2d(candidate)
            relative_area_change = (
                abs(candidate_area - signed_area) / abs(signed_area)
            )
            if (candidate_area * signed_area <= 0
                    or abs(candidate_area) <= 1e-12
                    or relative_area_change > maximum_area_change
                    or not _polygon_is_simple_2d(candidate)):
                continue

            removed.append((current, interior_angle, relative_area_change))
            cleaned = candidate
            changed = True
            break

        if not changed:
            break

    for point, angle, area_change in removed:
        print(
            f"  Removed sharp footprint vertex from {building_id}: "
            f"({point[0]:.9g}, {point[1]:.9g}), "
            f"interior angle={angle:.3f} deg, "
            f"area change={100.0 * area_change:.3f}%"
        )
    return cleaned


# ---------------------------------------------------------------------------
# mesh-size computation
# ---------------------------------------------------------------------------

MIN_BUILDING_MESH_SIZE = 1.0
MESH_SIZE_CAP = 1.0  # no automatically generated mesh size may exceed this


def compute_mesh_size(buildings, factor=1.0,
                      minimum=MIN_BUILDING_MESH_SIZE):
    """Return a safe global building mesh size.

    The candidate size is the smallest valid building-edge length times
    ``factor``, but it is clamped to ``minimum``.  This lower bound is
    essential for city-scale data: tiny OSM edges must not force every large
    roof and wall to be tessellated at centimetre resolution.

    Degenerate edges (near-duplicate vertices, length below 1e-6 m) are
    skipped so one bad vertex pair in the source JSON can't collapse
    mesh_size to ~0 for the whole run.
    """
    min_edge = float('inf')
    for bldg in buildings:
        for face in bldg.get('faces', []):
            vs = face.get('vertices', [])
            n = len(vs)
            if n < 2:
                continue
            for i in range(n):
                v0, v1 = vs[i], vs[(i + 1) % n]
                dx = v1['x'] - v0['x']
                dy = v1['y'] - v0['y']
                dz = v1['z'] - v0['z']
                length = math.sqrt(dx * dx + dy * dy + dz * dz)
                if length > 1e-6:
                    min_edge = min(min_edge, length)
    if min_edge == float('inf'):
        return minimum
    return min(max(min_edge * factor, minimum), MESH_SIZE_CAP)


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
    tri = Delaunay(pts_arr)

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


def tessellate_polygon_constrained(polygon_2d, mesh_size, z=0.0):
    """
    Triangulate a flat polygon using a Constrained Delaunay Triangulation (CDT)
    via the 'triangle' library (Shewchuk's Triangle).

    Polygon edges are registered as hard segments so the triangulator can NEVER
    create a triangle that crosses the boundary.  This eliminates the outward
    spikes produced by unconstrained Delaunay on concave polygons.

    Density is controlled by a maximum triangle area = mesh_size² / 2.
    Intermediate points are inserted along each edge at mesh_size spacing so
    the boundary matches the original curved polygon faithfully.

    Falls back to tessellate_polygon_delaunay when 'triangle' is not installed.
    """
    try:
        import triangle as tr
        import numpy as np
    except ImportError:
        return tessellate_polygon_delaunay(polygon_2d, mesh_size, z)

    n_poly = len(polygon_2d)
    pts = []   # [x, y] list
    segs = []  # [i, j] boundary segments

    # Walk every polygon edge: add corner vertex + intermediate subdivisions.
    # Segments are generated AFTER all points are collected so cross-edge
    # connections (last intermediate of edge i → corner of edge i+1) are
    # never accidentally omitted.
    for i in range(n_poly):
        p0 = polygon_2d[i]
        p1 = polygon_2d[(i + 1) % n_poly]
        pts.append([p0[0], p0[1]])
        edge_len = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        n_edge = max(1, round(edge_len / mesh_size))
        for k in range(1, n_edge):
            t = k / n_edge
            pts.append([p0[0] + t * (p1[0] - p0[0]),
                        p0[1] + t * (p1[1] - p0[1])])

    # One segment per consecutive pair, wrapping back to index 0
    n_pts = len(pts)
    segs = [[i, (i + 1) % n_pts] for i in range(n_pts)]

    # Maximum triangle area for density control (right-isoceles at mesh_size)
    max_area = 0.5 * mesh_size * mesh_size

    data = {
        'vertices': np.array(pts,  dtype=np.float64),
        'segments': np.array(segs, dtype=np.int32),
    }
    # Flags: p = PSLG (enforce segments), a = area constraint for density.
    # Do NOT use 'q' (quality refinement) — it inserts Steiner points on
    # constrained boundary edges, changing the boundary vertex positions and
    # breaking the exact match with adjacent patches.
    result = tr.triangulate(data, f'pa{max_area:.10f}')

    out_pts = result['vertices']
    out_tris = result['triangles']

    verts = [(float(p[0]), float(p[1]), z) for p in out_pts]
    faces = [tuple(int(v) for v in tri) for tri in out_tris]

    if not faces:
        return tessellate_polygon_delaunay(polygon_2d, mesh_size, z)

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
      - 'wall'        → always included (all storeys, forms complete exterior wall)
      - 'floor'       → excluded (ground-level base slab and every inter-storey
                        slab, none needed for CFD surface)
      - 'roof'        → included only from the TOPMOST _internal_floor entry
                        of each multi-storey building (its 'roof' is the true
                        exterior building top), or from single-storey main
                        entries that have no _internal_floor siblings
                        (single-storey / FreeCAD format).
                        Every other _internal_floor entry's 'roof' is an
                        internal slab (shared with the floor above it) and is
                        excluded, as is a multi-storey main entry's 'roof'
                        (the ground-floor ceiling).
      - 'ceiling' / 'foundation' / other → excluded (internal horizontal slabs)
    """
    # Pre-compute which main building IDs have _internal_floor siblings.
    # In the multi-storey JSON format the main entry ('Building N') carries a
    # 'roof' face that is the ground-floor *ceiling*, not the exterior top.
    # The true exterior roof is on the topmost _internal_floor entry.  We must
    # therefore suppress 'roof' on any main entry that has internal-floor peers.
    main_ids_with_internal: set = set()
    for b in buildings:
        bid = b.get('id', '')
        if '_internal_floor' in bid:
            # 'building_2_internal_floor_1' → prefix 'building_2'
            prefix = bid.split('_internal_floor')[0]
            main_ids_with_internal.add(prefix.lower())

    # A multi-storey building's _internal_floor entries each carry their own
    # 'roof' face, but only the TOPMOST storey's roof is the true exterior
    # roof — every other storey's 'roof' is an internal slab shared with the
    # floor above it (e.g. floor_1's roof at z=8.4 == floor_2's floor at
    # z=8.4). Without this, all intermediate-storey slabs leak into the
    # exterior mesh as internal faces. Find the highest-roof-z entry per
    # building prefix so only that one's roof is kept.
    topmost_internal_id: dict = {}   # prefix -> (bldg_id, roof_z)
    for b in buildings:
        bid = b.get('id', '')
        if '_internal_floor' not in bid:
            continue
        roof_face = next((f for f in b.get('faces', [])
                           if f.get('surfaceType') == 'roof'), None)
        if roof_face is None or not roof_face.get('vertices'):
            continue
        z = max(v['z'] for v in roof_face['vertices'])
        prefix = bid.split('_internal_floor')[0]
        if prefix not in topmost_internal_id or z > topmost_internal_id[prefix][1]:
            topmost_internal_id[prefix] = (bid, z)
    topmost_internal_ids = {v[0] for v in topmost_internal_id.values()}

    verts = []
    faces = []
    for bldg in buildings:
        is_internal = '_internal_floor' in bldg.get('id', '')
        bldg_id_key = bldg.get('id', '').lower().replace(' ', '_')

        # Compute XY centroid for this building to use as the outward-normal
        # reference for wall faces (fallback when no 'normal' field is present,
        # and as a post-subdivision sanity check for all walls).
        _all_fv = [v for f in bldg.get('faces', []) for v in f['vertices']]
        if _all_fv:
            bldg_cx = sum(v['x'] for v in _all_fv) / len(_all_fv)
            bldg_cy = sum(v['y'] for v in _all_fv) / len(_all_fv)
        else:
            bldg_cx = bldg_cy = 0.0

        for face in bldg.get('faces', []):
            st = face['surfaceType']
            fv = face['vertices']

            if st == 'wall':
                include = True
            elif st == 'roof':
                # Include only the TRUE exterior roof:
                #   - for an _internal_floor entry, only if it is the topmost
                #     storey (every other storey's 'roof' is an internal slab
                #     shared with the floor above it);
                #   - for a single-storey main entry with no _internal_floor
                #     siblings, its own roof.
                # A multi-storey main entry's 'roof' is the ground-floor
                # ceiling (internal slab) – always excluded.
                if is_internal:
                    include = bldg.get('id', '') in topmost_internal_ids
                else:
                    include = bldg_id_key not in main_ids_with_internal
            elif st == 'floor' and not is_internal:
                # ground-level base slab only
                include = False
            else:
                include = False   # ceiling / foundation / interior floor slabs

            if not include:
                continue

            # For horizontal faces (roof/floor), ensure the polygon winds the
            # correct way so its normal points OUTWARD from the building:
            #   roof  → normal must point +z (up, out the top of that storey)
            #   floor → normal must point -z (down, out the bottom of that storey)
            # Some JSON sources wind these polygons inconsistently, which
            # would otherwise cause extrudeMesh to extrude in the wrong
            # direction, generating degenerate cells with garbage centroids.
            if st in ('roof', 'floor') and len(fv) >= 3:
                signed_area = sum(
                    fv[i]['x'] * fv[(i + 1) % len(fv)]['y']
                    - fv[(i + 1) % len(fv)]['x'] * fv[i]['y']
                    for i in range(len(fv))
                )
                # roof wants CCW-from-above (signed_area > 0, i.e. +z normal);
                # floor wants CW-from-above (signed_area < 0, i.e. -z normal).
                needs_reversed = (
                    (st == 'roof' and signed_area < 0) or
                    (st == 'floor' and signed_area > 0)
                )
                if needs_reversed:
                    fv = list(reversed(fv))

            # For wall faces, the JSON itself supplies a ground-truth 'normal'
            # field, but the vertex winding it gives can point the opposite
            # way from that stated normal (seen consistently across this
            # source's wall faces). Walls aren't confined to a single global
            # plane like roof/floor, so we can't use a signed-area shortcut
            # here - instead, compute the normal implied by the first three
            # vertices and flip the winding if it disagrees with face['normal'].
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
            if mesh_size is not None and len(fv) == 4:
                # Rectangular face → bilinear quad subdivision
                v0 = (fv[0]['x'], fv[0]['y'], fv[0]['z'])
                v1 = (fv[1]['x'], fv[1]['y'], fv[1]['z'])
                v2 = (fv[2]['x'], fv[2]['y'], fv[2]['z'])
                v3 = (fv[3]['x'], fv[3]['y'], fv[3]['z'])
                lv, lf = subdivide_quad(v0, v1, v2, v3, mesh_size)
                verts.extend(lv)
                for tri in lf:
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))
            elif mesh_size is not None and st in ('floor', 'roof'):
                # Polygonal horizontal face → Constrained Delaunay Triangulation.
                # Boundary vertices are included so the roof edge meets the
                # wall-top edge with no gap. Registering the polygon edges as
                # hard segments (rather than the old centroid/sample-based
                # inside-test) guarantees the triangulation never crosses a
                # concave notch, so it can't leave holes the way the plain
                # Delaunay + heuristic filter could on non-convex footprints.
                z_val = fv[0]['z']
                poly  = [(v['x'], v['y']) for v in fv]
                lv, lf = tessellate_polygon_constrained(poly, mesh_size, z=z_val)
                verts.extend(lv)
                for tri in lf:
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))
            else:
                # Non-quad wall face or no mesh_size → fan triangulate
                for v in fv:
                    verts.append((v['x'], v['y'], v['z']))
                for tri in fan_triangulate(len(fv)):
                    faces.append((base + tri[0], base + tri[1], base + tri[2]))

            # Filter zero-area triangles for ALL face types (wall, roof, floor).
            # Fan-triangulation of degenerate JSON polygons (coincident or
            # collinear vertices) produces triangles with a zero cross-product
            # magnitude.  surfaceFeatures normalises edge normals and divides
            # by zero when it encounters such triangles → SIGFPE crash.
            # For roof/floor faces, also enforce the correct outward normal
            # direction (roof: +z, floor: -z): subdivision near concave
            # corners can produce sliver triangles with nz flipped the wrong
            # way, which cause extrudeMesh to extrude in the wrong direction,
            # placing micro-cells outside the building bounding box →
            # "not all cells assigned".
            corrected = []
            for k in range(face_start, len(faces)):
                tri = faces[k]
                a = verts[tri[0]]; b = verts[tri[1]]; c = verts[tri[2]]
                ax, ay, az = b[0]-a[0], b[1]-a[1], b[2]-a[2]
                bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
                nx = ay*bz - az*by
                ny = az*bx - ax*bz
                nz = ax*by - ay*bx
                if (nx*nx + ny*ny + nz*nz) < 1e-8:  # zero-area sliver – discard
                    continue
                if st == 'roof' and nz < 0:
                    tri = (tri[0], tri[2], tri[1])
                elif st == 'floor' and nz > 0:
                    tri = (tri[0], tri[2], tri[1])
                elif st == 'wall':
                    # Centroid-based outward check: catches cases where
                    # face['normal'] was absent OR where subdivision produced
                    # a triangle whose winding disagrees with the polygon.
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
    poly = remove_sharp_footprint_vertices(
        poly, building_id=bldg.get('id', 'osm_building')
    )
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

        # Tree mesh density is tied to this tree's own crown radius, not an
        # externally supplied mesh_size: local_ms = crown_r / 10.
        local_ms    = min(max(crown_r / 10.0, 1e-4), MESH_SIZE_CAP*0.5)
        n_seg_crown = max(6, round(2 * math.pi * crown_r / local_ms))
        n_h_trunk   = max(1, round(trunk_h / local_ms))
        n_h_crown   = max(2, round((total_h - trunk_h) / local_ms))

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

        # This pond's own mesh density: its own minimum edge length / 2.
        _min_edge = _min_edge_length_2d(poly)
        mesh_size = min(max(_min_edge, 0.01), MESH_SIZE_CAP) if _min_edge is not None else None

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

        # This lawn's own mesh density: its own minimum edge length / 2.
        _min_edge = _min_edge_length_2d(poly)
        mesh_size = min(max(_min_edge, 10.0),2.5 * MESH_SIZE_CAP) if _min_edge is not None else None

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
        # Ensure CCW winding so tessellated faces have +z (upward) normals
        _n = len(poly)
        _sa = sum(poly[i][0]*poly[(_i1:=(i+1)%_n)][1] - poly[_i1][0]*poly[i][1]
                  for i in range(_n))
        if _sa < 0:
            poly = list(reversed(poly))
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

    # Optional explicit building resolution.  The safe default is 1 m.  An
    # explicit value bypasses automatic sizing but remains validated here so
    # accidental zero/negative values cannot trigger runaway subdivisions.
    building_mesh_size = None
    if "--building-mesh-size" in args:
        option_index = args.index("--building-mesh-size")
        if option_index + 1 >= len(args):
            print("Error: --building-mesh-size requires a positive value in metres.",
                  file=sys.stderr)
            sys.exit(2)
        try:
            building_mesh_size = float(args[option_index + 1])
        except ValueError:
            print("Error: --building-mesh-size must be a number in metres.",
                  file=sys.stderr)
            sys.exit(2)
        if not math.isfinite(building_mesh_size) or building_mesh_size <= 0:
            print("Error: --building-mesh-size must be finite and greater than zero.",
                  file=sys.stderr)
            sys.exit(2)
        del args[option_index:option_index + 2]

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

    if building_mesh_size is None:
        mesh_size = compute_mesh_size(all_buildings)
        mesh_description = f"automatic, minimum {MIN_BUILDING_MESH_SIZE:g} m"
    else:
        mesh_size = building_mesh_size
        mesh_description = "command-line override"
    print(f"Mesh size (buildings): {mesh_size:.4f} m  ({mesh_description})")
    print("Mesh size (trees):  per-tree, crown radius / 10")
    print("Mesh size (ponds):  per-pond, that pond's own minimum edge / 2")
    print("Mesh size (lawns):  per-lawn, that lawn's own minimum edge / 2\n")

    print("Extracting buildings  -> buildings_refined.obj")
    buildings_to_obj(all_buildings,
                     os.path.join(out_dir, 'buildings_refined.obj'), mesh_size)

    print(f"Extracting trees      -> trees.obj  (crown: {crown_shape})")
    trees_to_obj(all_trees, os.path.join(out_dir, 'trees.obj'),
                 crown_shape=crown_shape)

    print("Extracting ponds      -> ponds.obj")
    ponds_to_obj(all_ponds, os.path.join(out_dir, 'ponds.obj'))

    print("Extracting lawns      -> lawns.obj")
    lawns_to_obj(all_grass, os.path.join(out_dir, 'lawns.obj'))

    print("Extracting pathways   -> pathways.obj")
    pathways_to_obj(all_roads, os.path.join(out_dir, 'pathways.obj'), mesh_size)

    print("\nDone.")

    print("\n \nEnd\n")


if __name__ == "__main__":
    main()








# #!/usr/bin/env python3
# """
# json_to_obj.py
# Extract buildings, trees, ponds, lawns (grassAreas) and pathways (roads)
# from a MassingOBJ JSON file and write surface-meshed OBJ files.

# Usage:
#     python json_to_obj.py [input1.json [input2.json ...]] [output_dir]

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

# def fan_triangulate(n_verts):
#     return [(0, i, i + 1) for i in range(1, n_verts - 1)]


# def write_obj(path, vertices, faces, comment=""):
#     if not vertices or not faces:
#         with open(path, 'w') as fh:
#             if comment:
#                 fh.write(f"# {comment}\n")
#         print(f"  Written: {path}  (empty - no {comment} in input)")
#         return
#     with open(path, 'w') as fh:
#         if comment:
#             fh.write(f"# {comment}\n")
#         for v in vertices:
#             fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
#         for face in faces:
#             fh.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
#     print(f"  Written: {path}  ({len(vertices)} verts, {len(faces)} tris)")


# def _remove_closing_dup(verts):
#     if len(verts) > 1 and verts[0] == verts[-1]:
#         return verts[:-1]
#     return verts


# def _simplify_polygon(poly, tol):
#     """
#     Remove near-collinear vertices from a 2-D polygon.

#     A vertex B is removed when its perpendicular distance from the chord
#     A→C (connecting its two neighbours) is less than *tol*.  The pass is
#     repeated until no more vertices are removed.  Vertices closer than
#     *tol* / 10 to their predecessor are also collapsed.

#     This cleans up OSM footprints where smooth curves or straight walls are
#     approximated with many closely-spaced vertices.  Such redundant vertices
#     cause the Delaunay roof tessellation to produce tiny sliver triangles
#     whose sample points fall just outside the polygon boundary (floating-point
#     rounding), so the triangles are incorrectly rejected and leave open edges
#     along the roof perimeter.
#     """
#     if tol <= 0:
#         return poly
#     changed = True
#     while changed and len(poly) > 3:
#         changed = False
#         new_poly = []
#         n = len(poly)
#         skip = [False] * n
#         for i in range(n):
#             if skip[i]:
#                 continue
#             A = poly[(i - 1) % n]
#             B = poly[i]
#             C = poly[(i + 1) % n]
#             # Duplicate / too-close-to-predecessor check
#             dx_ab = B[0] - A[0]; dy_ab = B[1] - A[1]
#             if math.sqrt(dx_ab*dx_ab + dy_ab*dy_ab) < tol / 10.0:
#                 skip[i] = True; changed = True; continue
#             # Perpendicular distance from B to chord A→C
#             dx_ac = C[0] - A[0]; dy_ac = C[1] - A[1]
#             len_ac = math.sqrt(dx_ac*dx_ac + dy_ac*dy_ac)
#             if len_ac < 1e-12:
#                 skip[i] = True; changed = True; continue
#             cross = abs(dx_ab * dy_ac - dy_ab * dx_ac)
#             if cross / len_ac < tol:
#                 skip[i] = True; changed = True
#         poly = [poly[i] for i in range(n) if not skip[i]]
#     return poly if len(poly) >= 3 else poly


# # ---------------------------------------------------------------------------
# # mesh-size computation
# # ---------------------------------------------------------------------------

# def compute_mesh_size(buildings, factor=0.01):
#     """Return max building-domain-extent * factor."""
#     xmin = ymin = zmin = float('inf')
#     xmax = ymax = zmax = float('-inf')
#     for bldg in buildings:
#         for face in bldg.get('faces', []):
#             for v in face['vertices']:
#                 xmin = min(xmin, v['x']); xmax = max(xmax, v['x'])
#                 ymin = min(ymin, v['y']); ymax = max(ymax, v['y'])
#                 zmin = min(zmin, v['z']); zmax = max(zmax, v['z'])
#     if xmin == float('inf'):
#         return 1.0
#     domain = max(xmax - xmin, ymax - ymin, zmax - zmin)
#     return max(domain * factor, 0.01)


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


# def tessellate_polygon_constrained(polygon_2d, mesh_size, z=0.0):
#     """
#     Triangulate a flat polygon using a Constrained Delaunay Triangulation (CDT)
#     via the 'triangle' library (Shewchuk's Triangle).

#     Polygon edges are registered as hard segments so the triangulator can NEVER
#     create a triangle that crosses the boundary.  This eliminates the outward
#     spikes produced by unconstrained Delaunay on concave polygons.

#     Density is controlled by a maximum triangle area = mesh_size² / 2.
#     Intermediate points are inserted along each edge at mesh_size spacing so
#     the boundary matches the original curved polygon faithfully.

#     Falls back to tessellate_polygon_delaunay when 'triangle' is not installed.
#     """
#     try:
#         import triangle as tr
#         import numpy as np
#     except ImportError:
#         return tessellate_polygon_delaunay(polygon_2d, mesh_size, z)

#     n_poly = len(polygon_2d)
#     pts = []   # [x, y] list
#     segs = []  # [i, j] boundary segments

#     # Walk every polygon edge: add corner vertex + intermediate subdivisions.
#     # Segments are generated AFTER all points are collected so cross-edge
#     # connections (last intermediate of edge i → corner of edge i+1) are
#     # never accidentally omitted.
#     for i in range(n_poly):
#         p0 = polygon_2d[i]
#         p1 = polygon_2d[(i + 1) % n_poly]
#         pts.append([p0[0], p0[1]])
#         edge_len = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
#         n_edge = max(1, round(edge_len / mesh_size))
#         for k in range(1, n_edge):
#             t = k / n_edge
#             pts.append([p0[0] + t * (p1[0] - p0[0]),
#                         p0[1] + t * (p1[1] - p0[1])])

#     # One segment per consecutive pair, wrapping back to index 0
#     n_pts = len(pts)
#     segs = [[i, (i + 1) % n_pts] for i in range(n_pts)]

#     # Maximum triangle area for density control (right-isoceles at mesh_size)
#     max_area = 0.5 * mesh_size * mesh_size

#     data = {
#         'vertices': np.array(pts,  dtype=np.float64),
#         'segments': np.array(segs, dtype=np.int32),
#     }
#     # Flags: p = PSLG (enforce segments), a = area constraint for density.
#     # Do NOT use 'q' (quality refinement) — it inserts Steiner points on
#     # constrained boundary edges, changing the boundary vertex positions and
#     # breaking the exact match with adjacent patches.
#     result = tr.triangulate(data, f'pa{max_area:.10f}')

#     out_pts = result['vertices']
#     out_tris = result['triangles']

#     verts = [(float(p[0]), float(p[1]), z) for p in out_pts]
#     faces = [tuple(int(v) for v in tri) for tri in out_tris]

#     if not faces:
#         return tessellate_polygon_delaunay(polygon_2d, mesh_size, z)

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
#       - 'wall'        → always included (all storeys, forms complete exterior wall)
#       - 'floor'       → excluded (ground-level base slab, not needed for CFD surface)
#       - 'roof'        → included only from _internal_floor entries (multi-storey format,
#                         where the topmost floor's 'roof' is the exterior building top)
#                         or from single-storey main entries that have no _internal_floor
#                         siblings (single-storey / FreeCAD format).
#                         A main entry's 'roof' in a multi-storey building is the
#                         ground-floor ceiling – an internal slab – and is excluded.
#       - 'ceiling' / 'foundation' / other → excluded (internal horizontal slabs)
#     """
#     # Pre-compute which main building IDs have _internal_floor siblings.
#     # In the multi-storey JSON format the main entry ('Building N') carries a
#     # 'roof' face that is the ground-floor *ceiling*, not the exterior top.
#     # The true exterior roof is on the topmost _internal_floor entry.  We must
#     # therefore suppress 'roof' on any main entry that has internal-floor peers.
#     main_ids_with_internal: set = set()
#     for b in buildings:
#         bid = b.get('id', '')
#         if '_internal_floor' in bid:
#             # 'building_2_internal_floor_1' → prefix 'building_2'
#             prefix = bid.split('_internal_floor')[0]
#             main_ids_with_internal.add(prefix.lower())

#     verts = []
#     faces = []
#     for bldg in buildings:
#         is_internal = '_internal_floor' in bldg.get('id', '')
#         bldg_id_key = bldg.get('id', '').lower().replace(' ', '_')
#         for face in bldg.get('faces', []):
#             st = face['surfaceType']
#             fv = face['vertices']

#             if st == 'wall':
#                 include = True
#             elif st == 'roof':
#                 # Include if this is an _internal_floor entry (the topmost floor
#                 # carries the exterior roof) or a single-storey main entry that
#                 # has no _internal_floor siblings.  A multi-storey main entry's
#                 # 'roof' is the ground-floor ceiling (internal slab) – exclude it.
#                 include = is_internal or (bldg_id_key not in main_ids_with_internal)
#             elif st == 'floor' and not is_internal:
#                 # ground-level base slab only
#                 include = False
#             else:
#                 include = False   # ceiling / foundation / interior floor slabs

#             if not include:
#                 continue

#             # For horizontal faces (roof/floor), ensure the polygon is wound
#             # CCW when viewed from above so the normal points +z (outward).
#             # Some JSON sources wind roof polygons CW, which would produce a
#             # downward-pointing normal and cause extrudeMesh to extrude in the
#             # wrong direction (outward from the building surface), generating
#             # degenerate cells with garbage centroids.
#             if st in ('roof', 'floor') and len(fv) >= 3:
#                 signed_area = sum(
#                     fv[i]['x'] * fv[(i + 1) % len(fv)]['y']
#                     - fv[(i + 1) % len(fv)]['x'] * fv[i]['y']
#                     for i in range(len(fv))
#                 )
#                 if signed_area < 0:
#                     fv = list(reversed(fv))

#             base = len(verts)
#             face_start = len(faces)
#             if mesh_size is not None and len(fv) == 4:
#                 # Rectangular face → bilinear quad subdivision
#                 v0 = (fv[0]['x'], fv[0]['y'], fv[0]['z'])
#                 v1 = (fv[1]['x'], fv[1]['y'], fv[1]['z'])
#                 v2 = (fv[2]['x'], fv[2]['y'], fv[2]['z'])
#                 v3 = (fv[3]['x'], fv[3]['y'], fv[3]['z'])
#                 lv, lf = subdivide_quad(v0, v1, v2, v3, mesh_size)
#                 verts.extend(lv)
#                 for tri in lf:
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))
#             elif mesh_size is not None and st in ('floor', 'roof'):
#                 # Polygonal horizontal face → Delaunay triangulation using the
#                 # polygon boundary vertices + interior Cartesian grid points.
#                 # Boundary vertices are included so the roof edge meets the
#                 # wall-top edge with no gap.  The centroid filter removes any
#                 # triangles outside the (possibly non-convex) polygon.
#                 z_val = fv[0]['z']
#                 poly  = [(v['x'], v['y']) for v in fv]
#                 lv, lf = tessellate_polygon_delaunay(poly, mesh_size, z=z_val)
#                 verts.extend(lv)
#                 for tri in lf:
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))
#             else:
#                 # Non-quad wall face or no mesh_size → fan triangulate
#                 for v in fv:
#                     verts.append((v['x'], v['y'], v['z']))
#                 for tri in fan_triangulate(len(fv)):
#                     faces.append((base + tri[0], base + tri[1], base + tri[2]))

#             # Filter zero-area triangles for ALL face types (wall, roof, floor).
#             # Fan-triangulation of degenerate JSON polygons (coincident or
#             # collinear vertices) produces triangles with a zero cross-product
#             # magnitude.  surfaceFeatures normalises edge normals and divides
#             # by zero when it encounters such triangles → SIGFPE crash.
#             # For roof faces, also enforce upward-pointing (+z) normals:
#             # subdivision near concave corners can produce sliver triangles
#             # with nz slightly negative that cause extrudeMesh to extrude in
#             # the wrong direction (outward), placing micro-cells outside the
#             # building bounding box → "not all cells assigned".
#             corrected = []
#             for k in range(face_start, len(faces)):
#                 tri = faces[k]
#                 a = verts[tri[0]]; b = verts[tri[1]]; c = verts[tri[2]]
#                 ax, ay, az = b[0]-a[0], b[1]-a[1], b[2]-a[2]
#                 bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
#                 nx = ay*bz - az*by
#                 ny = az*bx - ax*bz
#                 nz = ax*by - ay*bx
#                 if (nx*nx + ny*ny + nz*nz) < 1e-8:  # zero-area sliver – discard
#                     continue
#                 if st == 'roof' and nz < 0:
#                     tri = (tri[0], tri[2], tri[1])
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
#     dimensions.baseVertices, using 'x' and 'z' keys for the *ground-plane*
#     coordinates (their 'z' is the horizontal axis here, not height) with the
#     building's real height taken from the top-level 'height' field. Extrusion
#     is done from z=0 (ground) up to z=height in the OBJ's z-up convention.
#     """
#     base_verts = bldg.get('dimensions', {}).get('baseVertices', [])
#     poly = [(v['x'], v['z']) for v in base_verts]
#     poly = _remove_closing_dup(poly)
#     if len(poly) < 3:
#         return None

#     height = bldg.get('height', 0)
#     if height <= 0:
#         return None

#     # Ensure CCW winding so wall normals point outward.
#     # CW polygons produce inward-facing wall normals which snappyHexMesh
#     # misclassifies and which appear as missing/open faces in visualisers.
#     n_poly = len(poly)
#     signed_area = sum(
#         poly[i][0] * poly[(i + 1) % n_poly][1]
#         - poly[(i + 1) % n_poly][0] * poly[i][1]
#         for i in range(n_poly)
#     )
#     if signed_area < 0:
#         poly = list(reversed(poly))

#     # Simplify: remove near-collinear vertices that cause sliver Delaunay
#     # triangles at the roof boundary (these slivers fail the inside-polygon
#     # test and leave open edges along the roof perimeter).
#     # Tolerance = 0.5 m removes sub-metre mapping noise without altering
#     # real architectural features.
#     poly = _simplify_polygon(poly, tol=0.5)
#     if len(poly) < 3:
#         return None
#     n_poly = len(poly)
#     faces = []

#     # Wall faces: one quad per footprint edge, from ground (z=0) to roof (z=height)
#     for i in range(n_poly):
#         j = (i + 1) % n_poly
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

# def _add_cylinder(cx, cy, z_bot, z_top, radius, n_seg, n_height, verts, faces):
#     """Closed cylinder with n_height height bands and n_seg circumferential segments."""
#     base = len(verts)
#     h = z_top - z_bot
#     for hi in range(n_height + 1):
#         z = z_bot + hi * h / n_height
#         for si in range(n_seg):
#             a = 2 * math.pi * si / n_seg
#             verts.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z))
#     verts.append((cx, cy, z_bot))   # base centre: base + n_seg*(n_height+1)
#     verts.append((cx, cy, z_top))   # top  centre: base + n_seg*(n_height+1) + 1

#     for hi in range(n_height):
#         for si in range(n_seg):
#             sj = (si + 1) % n_seg
#             b0 = base + hi * n_seg + si
#             b1 = base + hi * n_seg + sj
#             t0 = base + (hi + 1) * n_seg + si
#             t1 = base + (hi + 1) * n_seg + sj
#             faces.append((b0, b1, t1))
#             faces.append((b0, t1, t0))

#     bot_ring = base
#     top_ring = base + n_height * n_seg
#     bc = base + n_seg * (n_height + 1)
#     tc = base + n_seg * (n_height + 1) + 1
#     for si in range(n_seg):
#         sj = (si + 1) % n_seg
#         faces.append((bc, bot_ring + sj, bot_ring + si))   # bottom cap
#         faces.append((tc, top_ring + si, top_ring + sj))   # top cap


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
#         trunk_r = d['trunkRadius']
#         trunk_h = d['trunkHeight']
#         crown_r = d['crownRadius']
#         crown_r = crown_r*0.95
#         total_h = d['totalHeight']

#         z_ground    = pz
#         z_trunk_top = pz + trunk_h
#         z_apex      = pz + total_h

#         if mesh_size and mesh_size > 0:
#             n_seg_trunk = max(6, round(2 * math.pi * trunk_r / mesh_size))
#             n_seg_crown = max(6, round(2 * math.pi * crown_r / mesh_size))
#             n_h_trunk   = max(1, round(trunk_h / mesh_size))
#             n_h_crown   = max(2, round((total_h - trunk_h) / mesh_size))
#         else:
#             n_seg_trunk = n_seg_crown = 12
#             n_h_trunk = n_h_crown = 1

#         _add_cylinder(cx, cy, z_ground, z_trunk_top, trunk_r,
#                       n_seg_trunk, n_h_trunk, verts, faces)
#         if crown_shape == "cone":
#             _add_cone(cx, cy, z_trunk_top, z_apex, crown_r,
#                       n_seg_crown, n_h_crown, verts, faces)
#         elif crown_shape == "sphere":
#             # True sphere (equal horizontal/vertical radius) resting on top
#             # of the trunk: bottom pole at trunk top, so it sits on the trunk
#             # rather than stretching to reach total_h like 'spheroid' does.
#             z_sphere_top = z_trunk_top + 2.0 * crown_r
#             _add_spheroid(cx, cy, z_trunk_top, z_sphere_top, crown_r,
#                            n_seg_crown, n_h_crown, verts, faces)
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
#         n      = len(poly)

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

#     mesh_size = compute_mesh_size(all_buildings)
#     print(f"Mesh size: {mesh_size:.4f} m  (building domain x 0.01)\n")

#     print("Extracting buildings  -> buildings_refined.obj")
#     buildings_to_obj(all_buildings,
#                      os.path.join(out_dir, 'buildings_refined.obj'), mesh_size)

#     print(f"Extracting trees      -> trees.obj  (crown: {crown_shape})")
#     trees_to_obj(all_trees, os.path.join(out_dir, 'trees.obj'), mesh_size,
#                  crown_shape=crown_shape)

#     print("Extracting ponds      -> ponds.obj")
#     ponds_to_obj(all_ponds, os.path.join(out_dir, 'ponds.obj'), mesh_size)

#     print("Extracting lawns      -> lawns.obj")
#     lawns_to_obj(all_grass, os.path.join(out_dir, 'lawns.obj'), mesh_size)

#     print("Extracting pathways   -> pathways.obj")
#     pathways_to_obj(all_roads, os.path.join(out_dir, 'pathways.obj'), mesh_size)

#     print("\nDone.")

#     print("\n \nEnd\n")


# if __name__ == "__main__":
#     main()

