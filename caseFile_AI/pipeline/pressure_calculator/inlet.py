"""
Inlet identification from the velocity field using the outward-normal flux

        U . n  <  0   ->  flow enters the domain  (inlet)
        U . n  >  0   ->  flow leaves the domain  (outlet)

The domain is a rectilinear box, so it has six boundary faces with outward
normals  xmin:(-1,0,0)  xmax:(+1,0,0)  ymin:(0,-1,0)  ymax:(0,+1,0)
         zmin:(0,0,-1)  zmax:(0,0,+1).

Two modes
---------
mode="face" (default)
    1. For every non-excluded face compute the area integrals
           Q_in  = int max(-U.n, 0) dA ,  Q_out = int max(U.n, 0) dA ,
           Q_net = int U.n dA
       A face is an INLET face when Q_net < 0 (more flow in than out).
    2. Inlet nodes = nodes of the inlet faces where U.n < -tol.
mode="node"
    Every boundary node of a non-excluded face with U.n < -tol is an inlet node
    (this also catches small back-flow patches on outlet faces).

`exclude_faces` keeps faces that are really walls from being treated as
inflow/outflow.  The lowest sampled layer (zmin) is the ground, so it is
excluded by default.
"""
from dataclasses import dataclass, field
import numpy as np

FACES = {          # name -> (axis, outward sign)
    "xmin": (0, -1), "xmax": (0, +1),
    "ymin": (1, -1), "ymax": (1, +1),
    "zmin": (2, -1), "zmax": (2, +1),
}


@dataclass
class InletResult:
    mask: np.ndarray                      # (nz, ny, nx) bool, True on inlet nodes
    faces: dict = field(default_factory=dict)   # per-face statistics

    def report(self):
        lines = [f"{'face':5s} {'class':9s} {'area[m2]':>9s} {'Q_in':>10s} {'Q_out':>10s} {'Q_net':>10s} {'%in':>5s} {'inlet nodes':>11s}"]
        for n, s in self.faces.items():
            lines.append(f"{n:5s} {s['class']:9s} {s['area']:9.1f} {s['Q_in']:10.2f} {s['Q_out']:10.2f} "
                         f"{s['Q_net']:10.2f} {100 * s['frac_in']:5.0f} {s['n_inlet']:11d}")
        lines.append(f"total inlet nodes: {int(self.mask.sum())}")
        return "\n".join(lines)


def _trap_weights(c):
    """Trapezoidal quadrature weights for node coordinates c."""
    w = np.zeros_like(c, dtype=float)
    d = np.diff(c)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def _face_index(axis, side, shape):
    sl = [slice(None)] * 3                  # array order is (z, y, x) -> axis 0=x -> array axis 2
    arr_axis = 2 - axis
    sl[arr_axis] = 0 if side < 0 else shape[arr_axis] - 1
    return tuple(sl)


def identify_inlet(x, y, z, U, fluid=None, exclude_faces=("zmin",), mode="face", tol=0.0):
    """
    x, y, z : 1-D node coordinates
    U       : (nz, ny, nx, 3) velocity (NaN allowed in solid)
    fluid   : optional (nz, ny, nx) bool mask of valid nodes (default: finite U)
    tol     : nodes count as inflow only if U.n < -tol   [m/s]
    """
    if fluid is None:
        fluid = np.isfinite(U).all(-1)
    Uz = np.where(fluid[..., None], U, 0.0)
    w = {0: _trap_weights(x), 1: _trap_weights(y), 2: _trap_weights(z)}
    mask = np.zeros(fluid.shape, bool)
    stats = {}
    inlet_faces = []

    for name, (axis, sgn) in FACES.items():
        idx = _face_index(axis, sgn, fluid.shape)
        un = sgn * Uz[idx][..., axis]               # U . n on the face
        ok = fluid[idx]
        other = [a for a in (0, 1, 2) if a != axis]
        # face array axes are (z,y,x) minus the normal axis -> remaining coord order
        rem = sorted(other, reverse=True)           # array order (z..x)
        dA = w[rem[0]][:, None] * w[rem[1]][None, :]
        Qin = float((np.maximum(-un, 0) * dA * ok).sum())
        Qout = float((np.maximum(un, 0) * dA * ok).sum())
        s = dict(area=float((dA * ok).sum()), Q_in=Qin, Q_out=Qout, Q_net=Qout - Qin,
                 frac_in=Qin / (Qin + Qout) if Qin + Qout > 0 else 0.0, n_inlet=0)
        if name in exclude_faces:
            s["class"] = "excluded"
        elif mode == "node":
            s["class"] = "inlet" if Qin > Qout else ("mixed" if (un < -tol).any() else "outlet")
            inlet_faces.append(name)
        else:
            s["class"] = "inlet" if Qin > Qout else "outlet"
            if s["class"] == "inlet":
                inlet_faces.append(name)
        stats[name] = (s, idx, un, ok)

    for name in inlet_faces:
        s, idx, un, ok = stats[name]
        m = ok & (un < -tol)
        mask[idx] |= m
    out = {}
    for name, (s, idx, un, ok) in stats.items():
        s["n_inlet"] = int((ok & (un < -tol)).sum()) if name in inlet_faces else 0
        out[name] = s
    return InletResult(mask=mask & fluid, faces=out)
