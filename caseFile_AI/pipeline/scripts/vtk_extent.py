#!/usr/bin/env python3
"""Print the XY extent of a wind-model VTK as:  x_min x_max y_min y_max
(same x/y coordinates that vtk_to_2d.py resamples from)."""
import sys

import pyvista as pv

grid = pv.read(sys.argv[1])
if not (hasattr(grid, "x") and hasattr(grid, "y")):
    sys.exit(f"{sys.argv[1]}: RectilinearGrid expected, got {type(grid).__name__}")
print(f"{float(grid.x[0])!r} {float(grid.x[-1])!r} {float(grid.y[0])!r} {float(grid.y[-1])!r}")