"""
foam_io.py  --  minimal ASCII OpenFOAM I/O used by the temperature/humidity case.

Only what the case needs:
  * read constant/polyMesh  (points, faces, owner, neighbour, boundary)
  * face centres / area vectors of any list of faces
  * read an internalField (uniform or nonuniform List<scalar|vector>)
  * write a nonuniform patch list

Binary meshes are not supported (the momentum case writes ASCII: writeFormat ascii).
"""
import re
from pathlib import Path

import numpy as np

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"//[^\n]*")

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  8
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    {loc}object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def foam_header(cls, obj, location=None):
    loc = f'location    "{location}";\n    ' if location else ""
    return HEADER.format(cls=cls, obj=obj, loc=loc)


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(errors="ignore")
    m = re.search(r"format\s+(\w+)\s*;", text[:2000])
    if m and m.group(1) != "ascii":
        raise ValueError(f"{path}: format {m.group(1)} is not supported, write the case in ASCII "
                         "(controlDict: writeFormat ascii)")
    text = _COMMENT_BLOCK.sub("", text)
    text = _COMMENT_LINE.sub("", text)
    i = text.find("FoamFile")
    if i >= 0:
        j = text.find("}", i)
        text = text[j + 1:]
    return text


def _list_body(text):
    """Return (count, body) of the first `N ( ... )` list in text."""
    m = re.search(r"(?m)^\s*(\d+)\s*\n?\s*\(", text)
    if not m:
        raise ValueError("no list found")
    n = int(m.group(1))
    start = m.end()
    end = text.rfind(")")
    return n, text[start:end]


def read_points(case):
    t = _read(Path(case) / "constant/polyMesh/points")
    n, body = _list_body(t)
    nums = re.findall(r"\(\s*([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\s*\)", body)
    p = np.array(nums, dtype=float).reshape(-1, 3)
    if len(p) != n:
        raise ValueError(f"points: expected {n}, parsed {len(p)}")
    return p


def read_faces(case):
    t = _read(Path(case) / "constant/polyMesh/faces")
    n, body = _list_body(t)
    out = []
    for m in re.finditer(r"(\d+)\s*\(([^()]*)\)", body):
        out.append(np.array(m.group(2).split(), dtype=np.int64))
    if len(out) != n:
        raise ValueError(f"faces: expected {n}, parsed {len(out)}")
    return out


def read_label_list(path):
    t = _read(path)
    n, body = _list_body(t)
    a = np.array(body.split(), dtype=np.int64)
    if len(a) != n:
        raise ValueError(f"{path}: expected {n}, parsed {len(a)}")
    return a


def read_boundary(case):
    """list of dicts: name, type, nFaces, startFace  (in file order)."""
    t = _read(Path(case) / "constant/polyMesh/boundary")
    n, body = _list_body(t)
    patches = []
    for m in re.finditer(r"(\w+)\s*\{([^{}]*)\}", body):
        d = m.group(2)
        nf = re.search(r"nFaces\s+(\d+)", d)
        sf = re.search(r"startFace\s+(\d+)", d)
        ty = re.search(r"type\s+(\w+)", d)
        if nf and sf and ty:
            patches.append(dict(name=m.group(1), type=ty.group(1),
                                nFaces=int(nf.group(1)), startFace=int(sf.group(1))))
    if len(patches) != n:
        raise ValueError(f"boundary: expected {n} patches, parsed {len(patches)}")
    return patches


def face_geometry(points, faces):
    """Centre, area vector Sf and area for a list of faces (OpenFOAM's definition)."""
    nf = len(faces)
    C = np.zeros((nf, 3))
    S = np.zeros((nf, 3))
    sizes = np.array([len(f) for f in faces])
    for k in np.unique(sizes):
        idx = np.flatnonzero(sizes == k)
        P = points[np.stack([faces[i] for i in idx])]            # (m, k, 3)
        if k == 3:
            c = P.mean(axis=1)
            sf = 0.5 * np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            C[idx], S[idx] = c, sf
            continue
        est = P.mean(axis=1)                                      # (m, 3)
        Pn = np.roll(P, -1, axis=1)
        a = 0.5 * np.cross(Pn - P, est[:, None, :] - P)           # (m, k, 3) triangle area vectors
        am = np.linalg.norm(a, axis=2)
        # sign so that a triangle contributes along the face normal
        nrm = a.sum(axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
        w = np.einsum("mkj,mj->mk", a, nrm)
        cen = (P + Pn + est[:, None, :]) / 3.0
        wsum = np.maximum(w.sum(axis=1), 1e-300)
        C[idx] = (w[:, :, None] * cen).sum(axis=1) / wsum[:, None]
        S[idx] = a.sum(axis=1)
    return C, S, np.linalg.norm(S, axis=1)


def read_mesh(case, with_cells=False):
    pts = read_points(case)
    fcs = read_faces(case)
    bnd = read_boundary(case)
    mesh = dict(points=pts, faces=fcs, patches=bnd)
    if with_cells:
        mesh["owner"] = read_label_list(Path(case) / "constant/polyMesh/owner")
        mesh["neighbour"] = read_label_list(Path(case) / "constant/polyMesh/neighbour")
    return mesh


def patch_faces(mesh, name):
    p = next((x for x in mesh["patches"] if x["name"] == name), None)
    if p is None:
        raise KeyError(f"patch '{name}' not in mesh (have {[x['name'] for x in mesh['patches']]})")
    ids = np.arange(p["startFace"], p["startFace"] + p["nFaces"])
    return ids, [mesh["faces"][i] for i in ids]


ROLE_RULES = [            # (role, substrings of the lower-case patch name); first match wins
    ("trees", ("tree",)),
    ("lawn", ("lawn", "grass")),
    ("pond", ("pond", "water")),
    ("pathway", ("pathway", "road", "asphalt")),
    ("buildings", ("build",)),
    ("ground", ("ground", "terrain", "floor")),
]
FLOW_PATCHES = ("inlet", "outlet", "top", "sky", "side", "front", "back", "left", "right", "bottom")


def classify_patches(mesh, overrides=None):
    """Role of every wall patch of the momentum mesh, from its name (case-insensitive substrings):

        trees      'tree'            tree crowns (solid)
        lawn       'lawn' | 'grass'  flat lawn surfaces at ground level
        pond       'pond' | 'water'  flat water surfaces at ground level
        pathway    'pathway' | 'road' | 'asphalt'  flat asphalt surfaces at ground level
        buildings  'build'           buildings (a legacy merged `buildings` patch may contain the trees as well)
        ground     'ground'          the rest of the ground

    `overrides` = {"ground": [names], "lawn": [...], ...} (the "patches" entry of config/surface_types.json)
    replaces the rule for the names it lists.  Inlet / outlet / top / ... (non-wall patches) get role "flow".
    Returns {"role": {patch name: role}, "by_role": {role: [names in boundary order]}}."""
    over = {}
    for role, names in (overrides or {}).items():
        for n in ([names] if isinstance(names, str) else names):
            over[n] = role
    role_of, unknown = {}, []
    for pt in mesh["patches"]:
        n = pt["name"]
        low = n.lower()
        if n in over:
            role_of[n] = over[n]
        elif pt.get("type") in ("symmetry", "symmetryPlane", "empty", "cyclic", "wedge") or low in FLOW_PATCHES:
            role_of[n] = "flow"
        else:
            for role, subs in ROLE_RULES:
                if any(sub in low for sub in subs):
                    role_of[n] = role
                    break
            else:
                if pt.get("type") == "patch":          # an unnamed open boundary
                    role_of[n] = "flow"
                else:
                    unknown.append(n)
    if unknown:
        raise KeyError(f"cannot tell what the wall patch(es) {unknown} are (patches: {[x['name'] for x in mesh['patches']]}). "
                       'Add  "patches": {"buildings": [...], "trees": [...], "lawn": [...], "pond": [...], "pathway": [...], "ground": [...]}  '
                       "to config/surface_types.json")
    by_role = {}
    for pt in mesh["patches"]:
        by_role.setdefault(role_of[pt["name"]], []).append(pt["name"])
    return dict(role=role_of, by_role=by_role)


def read_internal_field(path, ncells=None, rank="scalar"):
    """internalField of a volField, returned as (ncells,) or (ncells,3)."""
    t = _read(path)
    m = re.search(r"internalField\s+uniform\s+(\([^)]*\)|[^;\s]+)\s*;", t)
    if m:
        val = m.group(1).strip("() ").split()
        v = np.array(val, dtype=float)
        if ncells is None:
            raise ValueError("uniform internalField needs ncells")
        return np.tile(v if rank == "vector" else v[0], (ncells, 1) if rank == "vector" else ncells)
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", t)
    if not m:
        raise ValueError(f"{path}: cannot find internalField")
    n = int(m.group(1))
    start = m.end()
    # find the matching close paren of the list
    depth, i = 1, start
    while depth and i < len(t):
        c = t[i]
        depth += (c == "(") - (c == ")")
        i += 1
    body = t[start:i - 1]
    if rank == "vector":
        a = np.array(re.findall(r"\(\s*([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\s*\)", body), dtype=float)
    else:
        a = np.array(body.split(), dtype=float)
    if len(a) != n:
        raise ValueError(f"{path}: expected {n}, parsed {len(a)}")
    return a


def list_scalar(values, per_line=1):
    v = np.asarray(values, dtype=float)
    lines = "\n".join(f"{x:.8g}" for x in v)
    return f"nonuniform List<scalar>\n{len(v)}\n(\n{lines}\n)"


def latest_time(case):
    """Name of the latest numeric time directory (>0) or None."""
    best, bn = None, -1.0
    for d in Path(case).iterdir():
        if d.is_dir():
            try:
                x = float(d.name)
            except ValueError:
                continue
            if x > bn:
                best, bn = d.name, x
    return best
