#!/usr/bin/env python3
"""
gen_ground_obj.py
-----------------
Generate a surface-meshed bare-ground plane OBJ (ground.obj) for use with
OpenFOAM's extrudeMesh (constructFrom surface).

The ground plane is a flat rectangle at z = zMin whose horizontal extents
are read from a blockParameters file.  Polygon boundaries for lawns, ponds
and pathways are read from the site JSON so the ground CDT aligns exactly
with the adjacent patches.

Usage:
    python3 gen_ground_obj.py [blockParameters [output.obj]] [--json site.json]

Defaults:
    blockParameters  →  ./blockParameters
    output.obj       →  constant/triSurface/ground.obj

The mesh spacing defaults to 1% of the larger horizontal extent (same rule
as building faces in json_to_obj.py).  Override with --mesh-size VALUE.

Options:
    --mesh-size FLOAT    Override automatic mesh spacing (metres)
    --json FILE          Site JSON file; polygon boundaries are read directly
                         from it for exact CDT alignment (recommended).
    --exclude-surface OBJ
                         Fallback: remove ground triangles inside this OBJ.
                         Only used when --json is not given.
    --help               Show this message
"""

import json
import math
import multiprocessing
import os
import sys


# ---------------------------------------------------------------------------
# Quad subdivision  (same algorithm as json_to_obj.py)
# ---------------------------------------------------------------------------

def subdivide_quad(v0, v1, v2, v3, mesh_size):
    """
    Subdivide a planar quad (v0→v1→v2→v3) into a regular triangle grid
    using bilinear interpolation.  Returns (verts, faces) with 0-based indices.
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
# blockParameters parser
# ---------------------------------------------------------------------------

def read_block_parameters(path):
    """
    Parse key   value;  lines from a blockParameters file.
    Returns a dict of {key: float}.
    """
    params = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("/*"):
                continue
            # strip trailing semicolon and optional inline comment
            line = line.split("//")[0].rstrip("; \t")
            parts = line.split()
            if len(parts) == 2:
                try:
                    params[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return params


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------

def write_obj(path, verts, faces):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# ground.obj — flat ground plane for OpenFOAM extrudeMesh\n")
        fh.write(f"# {len(verts)} vertices, {len(faces)} triangles\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    print(f"Written: {path}  ({len(verts)} verts, {len(faces)} tris)")


# ---------------------------------------------------------------------------
# Exclusion-surface helpers
# ---------------------------------------------------------------------------

def read_obj_triangles(path):
    verts = []
    triangles = []
    if not path or not os.path.exists(path):
        return triangles

    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idxs = []
                for token in line.split()[1:]:
                    idxs.append(int(token.split("/")[0]) - 1)
                if len(idxs) == 3:
                    triangles.append((verts[idxs[0]], verts[idxs[1]], verts[idxs[2]]))
                elif len(idxs) > 3:
                    for i in range(1, len(idxs) - 1):
                        triangles.append((verts[idxs[0]], verts[idxs[i]], verts[idxs[i + 1]]))
    return triangles


def point_in_triangle_2d(px, py, tri, eps=1e-9):
    (ax, ay, _), (bx, by, _), (cx, cy, _) = tri
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def remove_excluded_faces(verts, faces, exclusion_triangles, label):
    if not exclusion_triangles:
        return faces

    kept = []
    removed = 0
    for face in faces:
        cx = sum(verts[i][0] for i in face) / 3.0
        cy = sum(verts[i][1] for i in face) / 3.0
        if any(point_in_triangle_2d(cx, cy, tri) for tri in exclusion_triangles):
            removed += 1
        else:
            kept.append(face)

    print(f"Excluded {label} from ground: {removed} triangles removed")
    return kept


# ---------------------------------------------------------------------------
# Parallel worker (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _check_faces_chunk(args):
    """
    Process a chunk of faces and return (kept_faces, removed_count).
    args = (faces_chunk, verts, exclusion_triangles, tri_bounds)
    """
    faces_chunk, verts, exclusion_triangles, tri_bounds = args
    kept = []
    removed = 0
    for face in faces_chunk:
        cx = sum(verts[i][0] for i in face) / 3.0
        cy = sum(verts[i][1] for i in face) / 3.0
        inside = False
        for tri, (xmin, xmax, ymin, ymax) in zip(exclusion_triangles, tri_bounds):
            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                if point_in_triangle_2d(cx, cy, tri):
                    inside = True
                    break
        if inside:
            removed += 1
        else:
            kept.append(face)
    return kept, removed


#Subrat: Added fast face removal
def remove_excluded_faces_fast(verts, faces, exclusion_triangles, label, n_workers=1):
    if not exclusion_triangles:
        return faces
    
    # Build bounding boxes for exclusion triangles
    tri_bounds = []
    for tri in exclusion_triangles:
        xs = [p[0] for p in tri]
        ys = [p[1] for p in tri]
        tri_bounds.append((min(xs), max(xs), min(ys), max(ys)))
    
    if n_workers > 1:
        # Split faces into equal chunks and process in parallel
        chunk_size = max(1, len(faces) // n_workers)
        chunks = [faces[i:i + chunk_size] for i in range(0, len(faces), chunk_size)]
        task_args = [(chunk, verts, exclusion_triangles, tri_bounds) for chunk in chunks]
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_check_faces_chunk, task_args)
        kept = [face for chunk_kept, _ in results for face in chunk_kept]
        removed = sum(r for _, r in results)
    else:
        kept = []
        removed = 0
        for face in faces:
            cx = sum(verts[i][0] for i in face) / 3.0
            cy = sum(verts[i][1] for i in face) / 3.0

            inside = False
            for tri, (xmin, xmax, ymin, ymax) in zip(exclusion_triangles, tri_bounds):
                # Fast bounding-box check first
                if xmin <= cx <= xmax and ymin <= cy <= ymax:
                    # Only do expensive point-in-triangle if bbox overlaps
                    if point_in_triangle_2d(cx, cy, tri):
                        inside = True
                        break

            if inside:
                removed += 1
            else:
                kept.append(face)

    print(f"Excluded {label} from ground: {removed} triangles removed")
    return kept
#Subrat: End of fast face removal


# ---------------------------------------------------------------------------
# Delaunay-based ground mesh that snaps to exclusion-surface boundaries
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Site JSON polygon reader
# ---------------------------------------------------------------------------

def _remove_closing_dup(verts):
    if len(verts) > 1 and verts[0] == verts[-1]:
        return verts[:-1]
    return verts


def read_site_json_polygons(json_path):
    """
    Read all flat ground-patch polygons (grassAreas, ponds, roads/pathways)
    from the site JSON file.  Returns a list of polygons, each a list of
    (x, y) tuples.  These are used as hole-boundary constraints in the CDT
    so the ground mesh aligns exactly with lawns / ponds / pathways.
    """
    with open(json_path, encoding='utf-8') as fh:
        data = json.load(fh)

    # Support both wrapped ('siteOnly_json') and flat JSON formats
    if 'siteOnly_json' in data:
        site = data['siteOnly_json']
    elif 'buildings' in data or 'metadata' in data:
        site = data
    else:
        return []

    polygons = []
    for area in site.get('grassAreas', []):
        coords = area.get('coordinates', [])
        poly = _remove_closing_dup([(c['x'], c['y']) for c in coords])
        if len(poly) >= 3:
            polygons.append(poly)

    for pond in site.get('ponds', []):
        coords = pond.get('coordinates', [])
        poly = _remove_closing_dup([(c['x'], c['y']) for c in coords])
        if len(poly) >= 3:
            polygons.append(poly)

    for road in site.get('roads', []) + site.get('pathways', []):
        coords = road.get('coordinates', [])
        poly = _remove_closing_dup([(c['x'], c['y']) for c in coords])
        if len(poly) >= 3:
            polygons.append(poly)

    return _dedupe_polygons(polygons)


def _dedupe_polygons(polygons, tol=1e-6):
    """
    Remove exact/near-duplicate polygons (same ring, possibly rotated start
    point or reversed winding) that would otherwise place two fully
    coincident hole boundaries into the PSLG and crash the `triangle` C
    library. Source site JSONs have been observed to list the same lawn /
    pond / pathway shape twice.
    """
    def _canonical(poly):
        rounded = tuple(sorted((round(x / tol) * tol, round(y / tol) * tol)
                                for x, y in poly))
        return rounded

    seen = {}
    unique = []
    n_dupes = 0
    for poly in polygons:
        key = _canonical(poly)
        if key in seen:
            n_dupes += 1
            continue
        seen[key] = True
        unique.append(poly)

    if n_dupes:
        print(f"Note: removed {n_dupes} duplicate exclusion polygon(s) "
              f"from site JSON (identical coordinates found more than once)",
              file=sys.stderr)

    return unique


def read_obj_vertices_2d(path):
    """Return a deduplicated list of (x, y) vertex positions from an OBJ file."""
    if not path or not os.path.exists(path):
        return []
    seen = set()
    pts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                xy = (float(p[1]), float(p[2]))
                if xy not in seen:
                    seen.add(xy)
                    pts.append(xy)
    return pts


def extract_boundary_polygon_2d(path):
    """
    Extract the ordered boundary polygon(s) from a triangulated OBJ file by
    finding edges shared by exactly one triangle (boundary edges) and walking
    them into an ordered ring.  Returns a list of polygons, each a list of
    (x, y) tuples.
    """
    from collections import defaultdict

    if not path or not os.path.exists(path):
        return []

    raw_xy = []
    tris = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                raw_xy.append((float(p[1]), float(p[2])))
            elif line.startswith("f "):
                idxs = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                if len(idxs) >= 3:
                    tris.append(tuple(idxs[:3]))

    if not tris:
        return []

    # Count how many triangles share each edge
    edge_count = defaultdict(int)
    for tri in tris:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            edge_count[(min(a, b), max(a, b))] += 1

    boundary_set = {e for e, c in edge_count.items() if c == 1}
    if not boundary_set:
        return []

    # Build undirected adjacency restricted to boundary edges
    adj = defaultdict(list)
    for a, b in boundary_set:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges = set()
    polygons = []

    for start in sorted(adj):
        # Find an unvisited boundary edge from this vertex
        unvisited_nb = None
        for nb in adj[start]:
            if (min(start, nb), max(start, nb)) not in visited_edges:
                unvisited_nb = nb
                break
        if unvisited_nb is None:
            continue

        ring = [start]
        prev, current = start, unvisited_nb
        while True:
            e = (min(prev, current), max(prev, current))
            if e in visited_edges:
                break
            visited_edges.add(e)
            ring.append(current)
            nbs = [n for n in adj[current] if n != prev]
            if not nbs or nbs[0] == start:
                break
            prev, current = current, nbs[0]

        if len(ring) >= 3:
            polygons.append([raw_xy[i] for i in ring])

    return polygons


def _polygon_interior_point(polygon):
    """
    Return a point (x, y) that is guaranteed to be strictly inside the polygon.

    The simple vertex-average centroid fails for concave polygons (C-shape,
    L-shape, etc.) because it can land outside.  This function tries the
    centroid first, then falls back to a grid scan over the bounding box.
    """
    def _pip(px, py, poly):
        n = len(poly); inside = False; j = n - 1
        for i in range(n):
            xi, yi = poly[i]; xj, yj = poly[j]
            denom = (yj - yi) or 1e-12
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / denom + xi):
                inside = not inside
            j = i
        return inside

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    cx = sum(xs) / len(polygon)
    cy = sum(ys) / len(polygon)
    if _pip(cx, cy, polygon):
        return cx, cy

    # Grid scan: try a 20×20 grid over the bounding box
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    for ni in range(1, 21):
        for nj in range(1, 21):
            px = x0 + ni * (x1 - x0) / 21
            py = y0 + nj * (y1 - y0) / 21
            if _pip(px, py, polygon):
                return px, py

    # Last resort: midpoint of the first edge's inward normal
    return cx, cy


def _mesh_ground_constrained_plain(xMin, xMax, yMin, yMax, zMin, mesh_size,
                                   exclusion_polygons):
    """
    Generate the ground mesh as a Constrained Delaunay Triangulation.

    exclusion_polygons : list of polygon coordinate lists [(x,y), ...]
        Polygon boundaries read directly from the site JSON.  Each polygon
        becomes a hole in the ground mesh so the CDT edges align exactly
        with lawn / pond / pathway patch boundaries.

    Requires the 'triangle' library (pip install triangle).
    Falls back to mesh_ground_delaunay (scipy-based) when unavailable.
    """
    try:
        import triangle as tr
        import numpy as np
    except ImportError:
        print("'triangle' library not found — falling back to scipy Delaunay",
              file=sys.stderr)
        return mesh_ground_delaunay(xMin, xMax, yMin, yMax, zMin, mesh_size,
                                    [], n_workers=1)

    pts = []   # [x, y]
    segs = []  # [i, j]
    holes = [] # [[hx, hy]] — one point inside each hole polygon

    # --- outer rectangle ---
    dx = xMax - xMin
    dy = yMax - yMin
    # Subdivide rectangle edges at mesh_size spacing.
    # Collect ALL boundary points first; segments are generated afterwards so
    # cross-edge connections (last intermediate → next corner) are never missed.
    rect_corners = [
        [xMin, yMin], [xMax, yMin], [xMax, yMax], [xMin, yMax]
    ]
    rect_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    outer_start = 0   # pts is empty here
    for ci, cj in rect_edges:
        p0 = rect_corners[ci]; p1 = rect_corners[cj]
        pts.append(list(p0))
        edge_len = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
        n_edge = max(1, round(edge_len / mesh_size))
        for k in range(1, n_edge):
            t = k / n_edge
            pts.append([p0[0]+t*(p1[0]-p0[0]), p0[1]+t*(p1[1]-p0[1])])
    outer_end = len(pts)   # exclusive
    # Connect consecutive outer-boundary points, closing back to outer_start
    for j in range(outer_start, outer_end):
        segs.append([j, outer_start + (j - outer_start + 1) % (outer_end - outer_start)])

    # --- exclusion holes (from site JSON polygons) ---
    for poly in exclusion_polygons:
        if len(poly) < 3:
            continue
        # Hole point must be strictly INSIDE the polygon.
        # Simple centroid fails for concave shapes (C/L/U-shaped lawns) where
        # the average vertex position can fall outside the polygon boundary.
        hx, hy = _polygon_interior_point(poly)
        holes.append([hx, hy])

        # Add boundary vertices + intermediate edge points.
        ring_start = len(pts)
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            pts.append([p0[0], p0[1]])
            edge_len = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
            n_edge = max(1, round(edge_len / mesh_size))
            for k in range(1, n_edge):
                t = k / n_edge
                pts.append([p0[0]+t*(p1[0]-p0[0]),
                             p0[1]+t*(p1[1]-p0[1])])
        ring_end = len(pts)
        ring_len = ring_end - ring_start
        for j in range(ring_len):
            segs.append([ring_start + j,
                          ring_start + (j + 1) % ring_len])

    max_area = 0.5 * mesh_size * mesh_size

    data = {
        'vertices': np.array(pts,  dtype=np.float64),
        'segments': np.array(segs, dtype=np.int32),
    }
    if holes:
        data['holes'] = np.array(holes, dtype=np.float64)

    # Flags: p = PSLG, a = area constraint.
    # No 'q' flag so no Steiner points are inserted on constrained edges —
    # this keeps boundary vertices exactly as specified, matching the
    # tessellate_polygon_constrained output in json_to_obj.py.
    result = tr.triangulate(data, f'pa{max_area:.10f}')

    out_pts = result['vertices']
    out_tris = result['triangles']

    verts = [(float(p[0]), float(p[1]), zMin) for p in out_pts]
    faces = [tuple(int(v) for v in tri) for tri in out_tris]

    print(f"CDT ground: {len(out_pts)} points → {len(faces)} triangles "
          f"({len(exclusion_polygons)} hole(s))")
    return verts, faces


def _triangulate_guarded(tr, data, max_area, max_mem_gb=8.0):
    """Run Triangle with an address-space cap so a runaway refinement stops this
    process (clear error) instead of eating the whole node's RAM."""
    flags = f'pa{max_area:.10f}'
    old = None
    try:
        import resource
        old = resource.getrlimit(resource.RLIMIT_AS)
        cap = int(max_mem_gb * 1024 ** 3)
        if old[1] == resource.RLIM_INFINITY or cap < old[1]:
            resource.setrlimit(resource.RLIMIT_AS, (cap, old[1]))
        else:
            old = None
    except Exception:
        old = None
    try:
        return tr.triangulate(data, flags)
    finally:
        if old is not None:
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, old)
            except Exception:
                pass


def mesh_ground_constrained(xMin, xMax, yMin, yMax, zMin, mesh_size,
                             exclusion_polygons, grid=1e-6, max_mem_gb=8.0):
    """
    Ground CDT with the lawn / pond / pathway outlines NODED first.

    The optimiser lets a lawn and a pond share a boundary (no gap).  Fed to Triangle as
    separate rings, those shared stretches are nearly-coincident, overlapping segments
    (equal only to ~1e-9 m, subdivided differently).  Triangle then splits them again and
    again ("Ran out of precision", crashes, or unbounded memory growth).  Here all outlines
    are snapped to a `grid` (1 micron) lattice and merged with shapely (overlapping pieces
    become ONE chain of segments, crossings and T-junctions get a shared vertex), so the PSLG
    handed to Triangle is conforming.  Falls back to the plain per-ring PSLG if shapely or
    Triangle is missing.
    """
    try:
        import triangle as tr
        import numpy as np
        import shapely
        from shapely.geometry import LineString
        from shapely.ops import unary_union
    except ImportError:
        print("shapely/triangle not available - using the plain (non-noded) PSLG",
              file=sys.stderr)
        return _mesh_ground_constrained_plain(xMin, xMax, yMin, yMax, zMin,
                                              mesh_size, exclusion_polygons)

    def densify(poly):
        out = []
        n = len(poly)
        for i in range(n):
            p0 = poly[i]; p1 = poly[(i + 1) % n]
            L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            ne = max(1, round(L / mesh_size))
            for k in range(ne):
                t = k / ne
                out.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
        out.append(out[0])
        return out

    rect = [(xMin, yMin), (xMax, yMin), (xMax, yMax), (xMin, yMax)]
    lines = [LineString(densify(rect))]
    holes = []
    skipped = 0
    for poly in exclusion_polygons:
        if len(poly) < 3:
            continue
        area = 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                         - poly[(i + 1) % len(poly)][0] * poly[i][1]
                         for i in range(len(poly)))
        if abs(area) < 1e-9:            # zero-area / collinear polygon
            skipped += 1
            continue
        holes.append(list(_polygon_interior_point(poly)))
        lines.append(LineString(densify(poly)))

    noded = unary_union(shapely.set_precision(unary_union(lines), grid))
    index, verts_l, segs = {}, [], set()

    def vid(c):
        key = (round(c[0] / grid), round(c[1] / grid))
        if key not in index:
            index[key] = len(verts_l)
            verts_l.append((c[0], c[1]))
        return index[key]

    for g in getattr(noded, 'geoms', [noded]):
        cs = list(g.coords)
        for a, b in zip(cs[:-1], cs[1:]):
            ia, ib = vid(a), vid(b)
            if ia != ib:
                segs.add((min(ia, ib), max(ia, ib)))

    max_area = 0.5 * mesh_size * mesh_size
    expected = (xMax - xMin) * (yMax - yMin) / max_area
    if expected > 5e6:
        raise SystemExit(f"gen_ground_obj: mesh spacing {mesh_size:g} m would need about "
                         f"{expected:.2e} triangles - check blockParameters / --mesh-size")

    data = {'vertices': np.array(verts_l, dtype=np.float64),
            'segments': np.array(sorted(segs), dtype=np.int32)}
    if holes:
        data['holes'] = np.array(holes, dtype=np.float64)
    try:
        result = _triangulate_guarded(tr, data, max_area, max_mem_gb)
    except (RuntimeError, MemoryError) as exc:
        raise SystemExit(f"gen_ground_obj: Triangle failed on the noded outlines ({exc}). "
                         f"{len(verts_l)} vertices, {len(segs)} segments")

    out_pts = result['vertices']
    out_tris = result['triangles']
    verts = [(float(p[0]), float(p[1]), zMin) for p in out_pts]
    faces = [tuple(int(v) for v in tri) for tri in out_tris]
    print(f"CDT ground (noded outlines): {len(out_pts)} points -> {len(faces)} triangles "
          f"({len(holes)} hole(s), {skipped} degenerate polygon(s) skipped)")
    return verts, faces


def _point_in_triangle_2d_xy(px, py, tri, eps=1e-9):
    """2-D point-in-triangle using barycentric coordinates (tri is 3 x (x,y,...))."""
    (ax, ay), (bx, by), (cx, cy) = tri[0][:2], tri[1][:2], tri[2][:2]
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def mesh_ground_delaunay(xMin, xMax, yMin, yMax, zMin, mesh_size,
                          exclusion_objs, n_workers=1):
    """
    Generate a ground mesh using scipy Delaunay triangulation so that ground
    triangle edges align exactly with the boundaries of lawns/ponds/pathways.

    Steps
    -----
    1. Build a Cartesian grid of points over the ground rectangle.
    2. For every exclusion OBJ, collect its 2-D vertex positions and add them
       as forced Steiner points.  Because these vertices sit on the exclusion
       boundary, Delaunay will generate edges that follow that boundary exactly.
    3. Triangulate all combined points with scipy Delaunay.
    4. Keep only triangles whose centroid lies inside the ground rectangle AND
       outside every exclusion surface.

    Falls back to subdivide_quad + remove_excluded_faces_fast when scipy is
    not available.
    """
    try:
        from scipy.spatial import Delaunay
        import numpy as np
    except ImportError:
        print("scipy not available — falling back to grid + centroid-exclusion",
              file=sys.stderr)
        verts, faces = subdivide_quad(
            (xMin, yMin, zMin), (xMax, yMin, zMin),
            (xMax, yMax, zMin), (xMin, yMax, zMin), mesh_size)
        all_excl = []
        for obj_path in exclusion_objs:
            all_excl.extend(read_obj_triangles(obj_path))
        for obj_path in exclusion_objs:
            faces = remove_excluded_faces_fast(
                verts, faces, read_obj_triangles(obj_path),
                os.path.basename(obj_path), n_workers=n_workers)
        return verts, faces

    dx = xMax - xMin
    dy = yMax - yMin

    # 1. Cartesian interior grid
    nx = max(1, round(dx / mesh_size))
    ny = max(1, round(dy / mesh_size))
    gx = xMin + np.arange(nx + 1) * (dx / nx)
    gy = yMin + np.arange(ny + 1) * (dy / ny)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([gxx.ravel(), gyy.ravel()])

    # 2. Forced boundary points: only the BOUNDARY vertices of each exclusion
    #    surface (not interior vertices). This keeps the point count small while
    #    still ensuring Delaunay edges align with the exclusion perimeters.
    extra_pts = []
    for obj_path in exclusion_objs:
        for poly in extract_boundary_polygon_2d(obj_path):
            extra_pts.extend(poly)
    extra_pts = np.array(extra_pts) if extra_pts else np.empty((0, 2))

    if len(extra_pts):
        all_pts = np.vstack([grid_pts, extra_pts])
    else:
        all_pts = grid_pts

    # Deduplicate (avoid degenerate triangles from coincident points)
    all_pts = np.unique(all_pts, axis=0)
    print(f"  Delaunay: {len(all_pts)} points "
          f"({len(grid_pts)} grid + {len(extra_pts)} boundary forced)")

    # 3. Delaunay triangulation
    tri = Delaunay(all_pts)

    # 4. Filter: keep triangles inside ground rect and outside all exclusions.
    #    Use a simple point-in-polygon (ray-casting) test against each exclusion
    #    boundary polygon — much faster than iterating over thousands of triangles.
    excl_polys = []  # list of (polygon_2d, bbox)
    for obj_path in exclusion_objs:
        for poly in extract_boundary_polygon_2d(obj_path):
            if len(poly) < 3:
                continue
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            excl_polys.append((poly, (min(xs), max(xs), min(ys), max(ys))))

    def _pip(px, py, poly):
        """Ray-casting point-in-polygon."""
        n = len(poly); inside = False; j = n - 1
        for i in range(n):
            xi, yi = poly[i]; xj, yj = poly[j]
            denom = (yj - yi) or 1e-12
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / denom + xi):
                inside = not inside
            j = i
        return inside

    verts_3d = [(float(p[0]), float(p[1]), zMin) for p in all_pts]
    faces = []
    removed = 0

    for simplex in tri.simplices:
        a, b, c = all_pts[simplex[0]], all_pts[simplex[1]], all_pts[simplex[2]]
        cx = float(a[0] + b[0] + c[0]) / 3.0
        cy = float(a[1] + b[1] + c[1]) / 3.0

        # Must be inside ground rectangle
        if not (xMin <= cx <= xMax and yMin <= cy <= yMax):
            removed += 1
            continue

        # Must be outside all exclusion boundary polygons (ray-casting PIP)
        inside_excl = False
        for poly, (exmin, exmax, eymin, eymax) in excl_polys:
            if exmin <= cx <= exmax and eymin <= cy <= eymax:
                if _pip(cx, cy, poly):
                    inside_excl = True
                    break
        if inside_excl:
            removed += 1
            continue

        faces.append(tuple(int(v) for v in simplex))

    print(f"Delaunay ground: {len(all_pts)} points → "
          f"{len(tri.simplices)} tris, {removed} excluded, "
          f"{len(faces)} kept")
    return verts_3d, faces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- parse arguments ---------------------------------------------------
    args = sys.argv[1:]
    mesh_size_override = None
    exclude_surfaces = []
    n_workers = 1
    positional = []

    site_json_path = None
    i = 0
    while i < len(args):
        if args[i] in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        elif args[i] == "--mesh-size" and i + 1 < len(args):
            mesh_size_override = float(args[i + 1])
            i += 2
        elif args[i] == "--json" and i + 1 < len(args):
            site_json_path = args[i + 1]
            i += 2
        elif args[i] == "--exclude-surface" and i + 1 < len(args):
            exclude_surfaces.append(args[i + 1])
            i += 2
        elif args[i] == "--workers" and i + 1 < len(args):
            n_workers = int(args[i + 1])
            i += 2
        else:
            positional.append(args[i])
            i += 1

    block_params_path = positional[0] if len(positional) > 0 else "blockParameters"
    output_path       = positional[1] if len(positional) > 1 else os.path.join(
                            "constant", "triSurface", "ground.obj")

    # Auto-detect site JSON if not given explicitly
    if site_json_path is None:
        for candidate in ("site.json", "MassingOBJ_reference.json"):
            if os.path.exists(candidate):
                site_json_path = candidate
                break
        # Also look for any *_full.json in cwd
        if site_json_path is None:
            import glob
            matches = glob.glob("*_full.json") + glob.glob("*.json")
            if matches:
                site_json_path = matches[0]

    if not exclude_surfaces:
        for name in ("lawns.obj", "ponds.obj", "pathways.obj"):
            default_exclude = os.path.join("constant", "triSurface", name)
            if os.path.exists(default_exclude):
                exclude_surfaces.append(default_exclude)

    # --- read blockParameters ----------------------------------------------
    if not os.path.exists(block_params_path):
        print(f"Error: blockParameters file not found: {block_params_path}",
              file=sys.stderr)
        sys.exit(1)

    params = read_block_parameters(block_params_path)

    required = ("xMin", "xMax", "yMin", "yMax", "zMin")
    missing = [k for k in required if k not in params]
    if missing:
        print(f"Error: missing keys in {block_params_path}: {missing}",
              file=sys.stderr)
        sys.exit(1)

    xMin = params["xMin"]
    xMax = params["xMax"]
    yMin = params["yMin"]
    yMax = params["yMax"]
    zMin = params["zMin"]

    dx = xMax - xMin
    dy = yMax - yMin

    print(f"Ground plane extents:")
    print(f"  x : [{xMin:.6g}, {xMax:.6g}]  (Δ = {dx:.6g} m)")
    print(f"  y : [{yMin:.6g}, {yMax:.6g}]  (Δ = {dy:.6g} m)")
    print(f"  z : {zMin:.6g} m")

    # --- mesh size ---------------------------------------------------------
    if mesh_size_override is not None:
        mesh_size = mesh_size_override
    else:
        mesh_size = max(dx, dy) * 0.01   # same 1% rule as building faces
    print(f"Mesh spacing : {mesh_size:.4g} m")

    # --- build ground mesh -------------------------------------------------
    # Preferred path: CDT using polygon boundaries from the site JSON.
    # This guarantees zero gaps and zero overlaps because the same boundary
    # coordinates are used for both the patch OBJs (json_to_obj.py) and the
    # ground hole constraints (here).
    excl_polygons = []
    if site_json_path and os.path.exists(site_json_path):
        excl_polygons = read_site_json_polygons(site_json_path)
        print(f"Site JSON   : {site_json_path}  ({len(excl_polygons)} exclusion polygon(s))")

    if excl_polygons:
        verts, faces = mesh_ground_constrained(
            xMin, xMax, yMin, yMax, zMin, mesh_size, excl_polygons)
    elif exclude_surfaces:
        # Fallback: scipy Delaunay with boundary-forced points from OBJ files
        print(f"Exclusion surfaces: {', '.join(os.path.basename(p) for p in exclude_surfaces)}")
        verts, faces = mesh_ground_delaunay(
            xMin, xMax, yMin, yMax, zMin, mesh_size,
            exclude_surfaces, n_workers=n_workers)
    else:
        # No exclusions — simple uniform quad grid is sufficient.
        v0 = (xMin, yMin, zMin)
        v1 = (xMax, yMin, zMin)
        v2 = (xMax, yMax, zMin)
        v3 = (xMin, yMax, zMin)
        verts, faces = subdivide_quad(v0, v1, v2, v3, mesh_size)
        print(f"Grid : {round(dx/mesh_size)} × {round(dy/mesh_size)} quads"
              f"  →  {len(faces)} triangles")

    write_obj(output_path, verts, faces)

    print("\n \nEnd\n")


if __name__ == "__main__":
    main()