#!/usr/bin/env python3
"""
flip_x_axis.py

Flip 'buildings', 'trees', 'osmBuildings' and 'osm_buildings' entries about
the X-axis (i.e. negate their ground-plane Y coordinate) in a MassingOBJ-style
scene JSON, leaving every other entity (ponds, grassAreas, roads, footprints,
dxfEntities, groups, etc.) untouched.

Regular buildings/trees store ground-plane coordinates in the 'x'/'y' keys
(with 'z' as height), so we negate 'y' wherever it appears inside these
entities: position, dimensions.baseVertices, geometry.footprint,
geometry.centroid, faces[].vertices, faces[].center, faces[].normal.

OSM buildings are different: dimensions.baseVertices stores the ground-plane
coordinates in 'x'/'z' (its 'z' is the horizontal axis, not height - see
convert_osm_building() in json_to_obj.py), while 'position' still uses the
regular 'x'/'y'/'z' convention (position.y is ground-plane, position.z is
vertical). So for OSM buildings we negate position.y AND baseVertices[].z.
"""

import copy
import json
import sys


def _flip_point_y(pt):
    """Negate the 'y' field of a point dict, in place, if present."""
    if isinstance(pt, dict) and 'y' in pt:
        pt['y'] = -pt['y']


def _flip_point_list_y(pts):
    if isinstance(pts, list):
        for p in pts:
            _flip_point_y(p)


def flip_regular_entity(entry):
    """Flip a 'buildings' or 'trees' style entry about the X-axis (negate y)."""
    _flip_point_y(entry.get('position'))

    dims = entry.get('dimensions', {})
    if isinstance(dims, dict):
        _flip_point_list_y(dims.get('baseVertices'))

    geom = entry.get('geometry', {})
    if isinstance(geom, dict):
        _flip_point_y(geom.get('centroid'))
        _flip_point_list_y(geom.get('footprint'))

    for face in entry.get('faces', []) or []:
        _flip_point_list_y(face.get('vertices'))
        _flip_point_y(face.get('center'))
        # Flipping about X inverts the sign of the normal's Y component too.
        _flip_point_y(face.get('normal'))


def flip_osm_entity(entry):
    """
    Flip an 'osmBuildings' / 'osm_buildings' style entry about the X-axis.
    These use x/z for the ground plane inside dimensions.baseVertices, and
    the regular x/y/z convention (y = ground plane, z = vertical) everywhere
    else (position, faces if present).
    """
    pos = entry.get('position')
    if isinstance(pos, dict) and 'y' in pos:
        pos['y'] = -pos['y']

    dims = entry.get('dimensions', {})
    if isinstance(dims, dict):
        for v in dims.get('baseVertices', []) or []:
            if isinstance(v, dict) and 'z' in v:
                v['z'] = -v['z']

    # In case any OSM entries carry populated 'faces' (regular x/y/z format),
    # flip those the normal way too.
    for face in entry.get('faces', []) or []:
        _flip_point_list_y(face.get('vertices'))
        _flip_point_y(face.get('center'))
        _flip_point_y(face.get('normal'))


def flip_scene(data):
    """Return a deep-copied version of data with buildings/trees/osm buildings
    flipped about the X-axis."""
    out = copy.deepcopy(data)

    # Handle both the flat format (keys at top level) and the wrapped
    # 'siteOnly_json' format, matching json_to_obj.py's _extract_site().
    site = out['siteOnly_json'] if 'siteOnly_json' in out else out

    for b in site.get('buildings', []) or []:
        flip_regular_entity(b)

    for t in site.get('trees', []) or []:
        flip_regular_entity(t)

    for key in ('osmBuildings', 'osm_buildings'):
        for b in site.get(key, []) or []:
            flip_osm_entity(b)

    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: python flip_x_axis.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flipped = flip_scene(data)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(flipped, f, indent=2)

    print(f"Flipped buildings, trees, osmBuildings and osm_buildings (Y negated) "
          f"and wrote: {out_path}")


if __name__ == "__main__":
    main()