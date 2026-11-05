"""
Minimal reader / writer for legacy-format BINARY VTK RECTILINEAR_GRID files.

Arrays are returned with shape (nz, ny, nx) for scalars and (nz, ny, nx, 3)
for vectors (VTK stores x fastest, then y, then z).
"""
import re
import numpy as np

_DTYPES = {
    "float": ">f4", "double": ">f8", "int": ">i4", "unsigned_int": ">u4",
    "long": ">i8", "short": ">i2", "unsigned_char": "u1", "char": "i1",
}


def read_rectilinear_vtk(path):
    """Return dict(x, y, z, dims, fields) where fields maps name -> ndarray.

    Handles POINT_DATA SCALARS / VECTORS and FIELD arrays (any with
    ntuples == number of points).
    """
    with open(path, "rb") as fh:
        b = fh.read()

    pos = 0

    def line():
        nonlocal pos
        e = b.index(b"\n", pos)
        s = b[pos:e].decode("ascii", "replace").strip()
        pos = e + 1
        return s

    def take(count, vtype):
        nonlocal pos
        dt = np.dtype(_DTYPES[vtype])
        arr = np.frombuffer(b, dtype=dt, count=count, offset=pos)
        pos += count * dt.itemsize
        return arr.astype(np.float64 if dt.kind == "f" else dt.newbyteorder("="))

    line()                       # "# vtk DataFile Version x.y"
    line()                       # title
    if line().upper() != "BINARY":
        raise NotImplementedError("Only BINARY legacy VTK files are supported.")

    dims, coords, fields, npts = None, {}, {}, None
    while pos < len(b):
        s = line()
        if not s:
            continue
        tok = s.split()
        key = tok[0].upper()
        if key == "DATASET":
            if tok[1].upper() != "RECTILINEAR_GRID":
                raise NotImplementedError("Only DATASET RECTILINEAR_GRID is supported.")
        elif key == "DIMENSIONS":
            dims = tuple(int(t) for t in tok[1:4])
            npts = dims[0] * dims[1] * dims[2]
        elif key in ("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"):
            coords[key[0].lower()] = take(int(tok[1]), tok[2])
        elif key == "POINT_DATA":
            npts = int(tok[1])
        elif key == "SCALARS":
            name, vtype = tok[1], tok[2]
            nc = int(tok[3]) if len(tok) > 3 else 1
            line()               # LOOKUP_TABLE ...
            a = take(npts * nc, vtype)
            fields[name] = a.reshape(npts, nc) if nc > 1 else a
        elif key == "VECTORS":
            fields[tok[1]] = take(npts * 3, tok[2]).reshape(npts, 3)
        elif key == "FIELD":
            for _ in range(int(tok[2])):
                hdr = line()
                while not hdr:               # blank line after the previous array's data
                    hdr = line()
                name, nc, nt, vtype = hdr.split()
                a = take(int(nc) * int(nt), vtype)
                fields[name] = a.reshape(int(nt), int(nc)) if int(nc) > 1 else a
        elif key in ("CELL_DATA", "CELL_TYPES"):
            break                # cell data not needed here

    nx, ny, nz = dims
    shaped = {}
    for k, a in fields.items():
        if a.shape[0] == nx * ny * nz:
            shaped[k] = a.reshape((nz, ny, nx) + a.shape[1:])
    return dict(x=coords["x"], y=coords["y"], z=coords["z"], dims=dims, fields=shaped)


def write_rectilinear_vtk(path, x, y, z, scalars=None, vectors=None, title="pressure reconstruction"):
    """Write a BINARY legacy VTK rectilinear grid (float32, big-endian).

    scalars: dict name -> (nz, ny, nx);   vectors: dict name -> (nz, ny, nx, 3)
    """
    nx, ny, nz = len(x), len(y), len(z)
    n = nx * ny * nz

    def be(a):
        return np.ascontiguousarray(a, dtype=">f4").tobytes()

    with open(path, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\n" + title.encode() + b"\nBINARY\n")
        fh.write(b"DATASET RECTILINEAR_GRID\n")
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\n".encode())
        for ax, c in zip("XYZ", (x, y, z)):
            fh.write(f"{ax}_COORDINATES {len(c)} float\n".encode() + be(c) + b"\n")
        fh.write(f"POINT_DATA {n}\n".encode())
        for name, a in (vectors or {}).items():
            fh.write(f"VECTORS {name} float\n".encode() + be(np.asarray(a).reshape(n, 3)) + b"\n")
        for name, a in (scalars or {}).items():
            safe = re.sub(r"\s+", "_", name)
            fh.write(f"SCALARS {safe} float 1\nLOOKUP_TABLE default\n".encode()
                     + be(np.asarray(a).reshape(n)) + b"\n")
