#!/usr/bin/env python3
"""
Reconstruct the pressure field from Ux, Uy, Uz in a legacy binary VTK
rectilinear-grid file, anchoring p = 101325 Pa at the inlet.

    python run_pressure.py input.vtk -o output.vtk
    python run_pressure.py input.vtk --rho 1.2 --nu 1.5e-5 --exclude-faces zmin --inlet-mode node

Output VTK point data:  U, p_calc, grad_p (momentum), inlet (0/1), and p_ref / p_error
if the input already holds a pressure array (--compare, default "p").
"""
import argparse
import numpy as np
from vtkio import read_rectilinear_vtk, write_rectilinear_vtk
from inlet import identify_inlet
from pressure import momentum_pressure_gradient, solve_pressure, RHO_AIR, NU_AIR, P_AMBIENT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vtk")
    ap.add_argument("-o", "--output", default="pressure_out.vtk")
    ap.add_argument("--velocity", default="U", help="name of the 3-component velocity array")
    ap.add_argument("--rho", type=float, default=RHO_AIR, help="density [kg/m3] (default: air 1.225)")
    ap.add_argument("--nu", type=float, default=NU_AIR, help="kinematic viscosity [m2/s] (default: air 1.516e-5)")
    ap.add_argument("--p-inlet", type=float, default=P_AMBIENT, help="pressure imposed at the inlet [Pa]")
    ap.add_argument("--exclude-faces", nargs="*", default=["zmin"],
                    help="boundary faces that are walls, not inlet/outlet (default: zmin = ground)")
    ap.add_argument("--inlet-mode", choices=["face", "node"], default="face")
    ap.add_argument("--inlet-tol", type=float, default=0.0, help="U.n must be below -tol [m/s] to count as inflow")
    ap.add_argument("--compare", default="p", help="existing pressure array to validate against ('' to skip)")
    a = ap.parse_args()

    d = read_rectilinear_vtk(a.vtk)
    x, y, z = d["x"], d["y"], d["z"]
    U = d["fields"][a.velocity]
    fluid = np.isfinite(U).all(-1)
    print(f"grid {d['dims']}  fluid nodes {fluid.sum()} / {fluid.size}  "
          f"(dx,dy,dz) = ({np.diff(x).mean():.3f}, {np.diff(y).mean():.3f}, {np.diff(z).mean():.3f}) m")
    print(f"Ux [{np.nanmin(U[...,0]):.2f}, {np.nanmax(U[...,0]):.2f}]  Uy [{np.nanmin(U[...,1]):.2f}, "
          f"{np.nanmax(U[...,1]):.2f}]  Uz [{np.nanmin(U[...,2]):.2f}, {np.nanmax(U[...,2]):.2f}]")

    inl = identify_inlet(x, y, z, U, fluid, exclude_faces=tuple(a.exclude_faces), mode=a.inlet_mode, tol=a.inlet_tol)
    print("\nInlet identification (U.n, n = outward normal):")
    print(inl.report())

    g = momentum_pressure_gradient(x, y, z, U, fluid, rho=a.rho, nu=a.nu)
    p, diag = solve_pressure(x, y, z, g, fluid, inl.mask, p_inlet=a.p_inlet)
    print("\nSolver:", {k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in diag.items()})
    print(f"p_calc: min {np.nanmin(p):.3f}  max {np.nanmax(p):.3f}  mean {np.nanmean(p):.3f} Pa "
          f"(gauge range {np.nanmin(p)-a.p_inlet:.2f} .. {np.nanmax(p)-a.p_inlet:.2f})")

    scalars = {
        name: values.copy()
        for name, values in d["fields"].items()
        if values.ndim == 3
    }
    vectors = {
        name: values.copy()
        for name, values in d["fields"].items()
        if values.ndim == 4 and values.shape[-1] == 3
    }
    scalars["p"] = p
    scalars["p_calc"] = p
    scalars["inlet"] = inl.mask.astype(float)
    if a.compare and a.compare in d["fields"]:
        pr = np.where(fluid, d["fields"][a.compare], np.nan)
        err = p - pr
        m = np.isfinite(err)
        r = np.corrcoef(p[m], pr[m])[0, 1]
        print(f"\nvs stored '{a.compare}':  RMSE {np.sqrt(np.mean(err[m]**2)):.3f} Pa, "
              f"mean error {err[m].mean():.3f} Pa, corr {r:.4f}, "
              f"stored range {np.nanmin(pr):.2f}..{np.nanmax(pr):.2f}")
        scalars["p_ref"] = pr
        scalars["p_error"] = err

    vectors["grad_p"] = g
    write_rectilinear_vtk(a.output, x, y, z, scalars=scalars, vectors=vectors)
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
