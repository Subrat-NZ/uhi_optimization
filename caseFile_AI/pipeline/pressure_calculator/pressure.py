"""
Pressure from a velocity field via the steady incompressible momentum equation

        rho (U . grad) U = - grad p + mu laplacian(U)
   =>   grad p = g  :=  - rho (U . grad) U + rho * nu * laplacian(U)

Step 1  (momentum_pressure_gradient)  evaluate g at every fluid node with
        finite differences on the (possibly non-uniform) rectilinear grid.
        Solid nodes (NaN) are masked: stencils never reach into them; next to
        a solid/domain edge the code falls back to one-sided differences
        (2nd order when two fluid neighbours exist, otherwise 1st order).

Step 2  (solve_pressure)  integrate g from the inlet value.
        A sampled g is never exactly curl-free, so integrating along a single
        path gives a path-dependent answer.  Instead we find the p that fits
        g best in the least-squares sense over ALL grid edges,

            minimise   sum_edges ( (p_j - p_i)/h_ij - g_ij )^2 ,   g_ij = (g_i+g_j)/2

        with p = p_inlet fixed on the inlet nodes.  This is a sparse SPD
        (graph-Laplacian) system, solved with AMG-preconditioned CG when
        pyamg is installed, otherwise Jacobi-preconditioned CG.
        Where g is exactly a gradient this reproduces plain path integration.

Neglected: gravity/buoyancy (constant density), Reynolds stresses (if the
velocity is a RANS/LES mean, the turbulent stress divergence is missing),
and unsteadiness (dU/dt).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

RHO_AIR = 1.225        # kg/m3  (ISA sea level, 15 C)
NU_AIR = 1.516e-5      # m2/s   (kinematic viscosity of air at ~20 C)
P_AMBIENT = 101325.0   # Pa


# ----------------------------------------------------------------------------
# finite differences with a fluid mask
# ----------------------------------------------------------------------------
def _shift(a, arr_axis, s, fill):
    """out[i] = a[i+s] along arr_axis (s = +1, +2, -1, -2); `fill` outside."""
    out = np.full_like(a, fill)
    dst = [slice(None)] * a.ndim
    src = [slice(None)] * a.ndim
    if s > 0:
        dst[arr_axis], src[arr_axis] = slice(0, -s), slice(s, None)
    else:
        dst[arr_axis], src[arr_axis] = slice(-s, None), slice(0, s)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _spacing(coord, arr_axis, ndim, s):
    """Distance to the neighbour at offset s (+1/-1) along arr_axis, broadcastable; 1 where undefined."""
    d = np.ones_like(coord)
    if s > 0:
        d[:-1] = np.diff(coord)
    else:
        d[1:] = np.diff(coord)
    shape = [1] * ndim
    shape[arr_axis] = -1
    return d.reshape(shape)


def deriv1(f, fluid, coord, arr_axis):
    """First derivative of f along arr_axis at fluid nodes (0 at nodes with no fluid neighbour)."""
    nd = f.ndim
    f0 = np.where(fluid, f, 0.0)
    fp, fm = _shift(f0, arr_axis, +1, 0.0), _shift(f0, arr_axis, -1, 0.0)
    fp2, fm2 = _shift(f0, arr_axis, +2, 0.0), _shift(f0, arr_axis, -2, 0.0)
    mp, mm = _shift(fluid, arr_axis, +1, False), _shift(fluid, arr_axis, -1, False)
    mp2, mm2 = _shift(fluid, arr_axis, +2, False), _shift(fluid, arr_axis, -2, False)
    hp = _spacing(coord, arr_axis, nd, +1)
    hm = _spacing(coord, arr_axis, nd, -1)
    # second neighbour spacing: h2p = x[i+2]-x[i+1] = hp shifted by +1
    hp2 = _shift(np.broadcast_to(hp, f.shape).copy(), arr_axis, +1, 1.0)
    hm2 = _shift(np.broadcast_to(hm, f.shape).copy(), arr_axis, -1, 1.0)

    out = np.zeros_like(f0)
    # central, non-uniform
    c = fluid & mp & mm
    cen = (hm**2 * fp - hp**2 * fm + (hp**2 - hm**2) * f0) / (hp * hm * (hp + hm))
    out = np.where(c, cen, out)
    # forward one-sided (no minus neighbour)
    fw2 = fluid & mp & ~mm & mp2
    h1, h2 = hp, hp2
    fwd2 = (-(2 * h1 + h2) / (h1 * (h1 + h2)) * f0 + (h1 + h2) / (h1 * h2) * fp
            - h1 / (h2 * (h1 + h2)) * fp2)
    out = np.where(fw2, fwd2, out)
    fw1 = fluid & mp & ~mm & ~mp2
    out = np.where(fw1, (fp - f0) / hp, out)
    # backward one-sided (no plus neighbour)
    bw2 = fluid & mm & ~mp & mm2
    h1, h2 = hm, hm2
    bwd2 = ((2 * h1 + h2) / (h1 * (h1 + h2)) * f0 - (h1 + h2) / (h1 * h2) * fm
            + h1 / (h2 * (h1 + h2)) * fm2)
    out = np.where(bw2, bwd2, out)
    bw1 = fluid & mm & ~mp & ~mm2
    out = np.where(bw1, (f0 - fm) / hm, out)
    return out


def deriv2(f, fluid, coord, arr_axis):
    """Second derivative along arr_axis; central stencil only (0 where a neighbour is missing)."""
    nd = f.ndim
    f0 = np.where(fluid, f, 0.0)
    fp, fm = _shift(f0, arr_axis, +1, 0.0), _shift(f0, arr_axis, -1, 0.0)
    mp, mm = _shift(fluid, arr_axis, +1, False), _shift(fluid, arr_axis, -1, False)
    hp, hm = _spacing(coord, arr_axis, nd, +1), _spacing(coord, arr_axis, nd, -1)
    val = 2.0 / (hp + hm) * ((fp - f0) / hp - (f0 - fm) / hm)
    return np.where(fluid & mp & mm, val, 0.0)


# ----------------------------------------------------------------------------
# step 1: momentum -> pressure gradient
# ----------------------------------------------------------------------------
def momentum_pressure_gradient(x, y, z, U, fluid=None, rho=RHO_AIR, nu=NU_AIR):
    """
    U : (nz, ny, nx, 3).  Returns g (nz, ny, nx, 3) = grad p, ordered (gx, gy, gz); NaN in solid.
    """
    if fluid is None:
        fluid = np.isfinite(U).all(-1)
    coords = (x, y, z)                 # physical axis 0,1,2  <->  array axes 2,1,0
    g = np.full(U.shape, np.nan)
    conv = np.zeros(U.shape)
    visc = np.zeros(U.shape)
    for c in range(3):
        uc = U[..., c]
        for a in range(3):
            arr_axis = 2 - a
            conv[..., c] += np.where(fluid, U[..., a], 0.0) * deriv1(uc, fluid, coords[a], arr_axis)
            visc[..., c] += deriv2(uc, fluid, coords[a], arr_axis)
    g[fluid] = (-rho * conv + rho * nu * visc)[fluid]
    return g


# ----------------------------------------------------------------------------
# step 2: integrate from the inlet (least squares over all edges)
# ----------------------------------------------------------------------------
def solve_pressure(x, y, z, g, fluid, inlet, p_inlet=P_AMBIENT, tol=1e-10, maxiter=2000, verbose=True):
    """
    g      : (nz, ny, nx, 3) pressure gradient (gx, gy, gz)
    fluid  : (nz, ny, nx) bool
    inlet  : (nz, ny, nx) bool, nodes where p = p_inlet is imposed
    p_inlet: scalar (e.g. 101325) or (nz, ny, nx) array of per-node Dirichlet values
    Returns p (nz, ny, nx) with NaN in solid / unreachable nodes, and a dict of diagnostics.
    """
    t0 = time.time()
    shape = fluid.shape
    nf = int(fluid.sum())
    idx = np.full(shape, -1, np.int64)
    idx[fluid] = np.arange(nf)
    coords = (x, y, z)

    I, J, D, H = [], [], [], []
    for a in range(3):
        ax = 2 - a
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        ok = fluid[lo] & fluid[hi]
        h = np.diff(coords[a])
        shp = [1, 1, 1]
        shp[ax] = -1
        h = np.broadcast_to(h.reshape(shp), ok.shape)
        ge = 0.5 * (g[lo][..., a] + g[hi][..., a])
        I.append(idx[lo][ok]); J.append(idx[hi][ok])
        D.append((ge * h)[ok]); H.append(h[ok])
    I, J, D, H = map(np.concatenate, (I, J, D, H))
    w = 1.0 / H**2

    # normal equations of the edge least-squares problem: L p = b
    rows = np.concatenate([I, J, I, J])
    cols = np.concatenate([I, J, J, I])
    vals = np.concatenate([w, w, -w, -w])
    L = sp.csr_matrix((vals, (rows, cols)), shape=(nf, nf))
    b = np.bincount(J, weights=D * w, minlength=nf) - np.bincount(I, weights=D * w, minlength=nf)

    # connectivity: every component must touch an inlet node
    known = inlet[fluid]
    ncomp, comp = connected_components(L, directed=False)
    has_inlet = np.zeros(ncomp, bool)
    has_inlet[comp[known]] = True
    reach = has_inlet[comp]
    if not reach.all() and verbose:
        print(f"[warn] {int((~reach).sum())} fluid nodes in {int((~has_inlet).sum())} component(s) "
              f"not connected to an inlet -> left NaN")
    free = reach & ~known
    kn = reach & known
    if not kn.any():
        raise RuntimeError("No inlet nodes found - nothing to anchor the pressure to.")

    # solve for dp = p - p_base to keep the numbers small; p_base = mean inlet pressure.
    # p_inlet may be a scalar (uniform, e.g. ambient) or a (nz, ny, nx) array of per-node values.
    pin = np.broadcast_to(np.asarray(p_inlet, float), shape)[fluid]
    p_base = float(pin[known].mean())
    dp = np.zeros(nf)
    dp[kn] = pin[kn] - p_base
    Lff = L[free][:, free].tocsr()
    rhs = b[free] - L[free][:, kn] @ dp[kn]
    its = [0]
    cb = lambda xk: its.__setitem__(0, its[0] + 1)
    try:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(Lff)
        sol = ml.solve(rhs, tol=tol, maxiter=maxiter, accel="cg", callback=cb)
        solver = "pyamg-SA + CG"
    except ImportError:
        d = Lff.diagonal()
        M = sp.diags(1.0 / d)
        try:
            sol, info = spla.cg(Lff, rhs, rtol=tol, maxiter=maxiter * 10, M=M, callback=cb)
        except TypeError:                      # scipy < 1.12
            sol, info = spla.cg(Lff, rhs, tol=tol, maxiter=maxiter * 10, M=M, callback=cb)
        solver = f"Jacobi-CG (info={info})"
    dp[free] = sol
    r = Lff @ sol - rhs
    rel_res = float(np.linalg.norm(r) / max(np.linalg.norm(rhs), 1e-300))

    p = np.full(shape, np.nan)
    pf = np.full(nf, np.nan)
    pf[reach] = p_base + dp[reach]
    p[fluid] = pf

    # how well does grad p reproduce g?  (RMS edge misfit, and how rotational g was)
    misfit = (dp[J] - dp[I] - D) / H            # reachable edges only matter; others are 0
    ok = reach[I] & reach[J]
    gmag = np.sqrt(np.mean((D[ok] / H[ok]) ** 2))
    diag = dict(n_fluid=nf, n_edges=len(I), n_inlet=int(kn.sum()), solver=solver, iters=its[0],
                rel_residual=rel_res, edge_misfit_rms=float(np.sqrt(np.mean(misfit[ok] ** 2))),
                grad_rms=float(gmag), seconds=time.time() - t0)
    return p, diag
