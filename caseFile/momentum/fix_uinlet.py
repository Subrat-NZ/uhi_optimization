import re
import math
import sys
import shutil
from pathlib import Path
from typing import Tuple  # Required for Python versions < 3.9

def parse_uinlet(content: str) -> Tuple[float, float, float]:
    """Extract (X, Y, Z) from 'Uinlet (X Y Z);' in 0/U."""
    # Added optional semicolon to the search pattern
    pattern = r'Uinlet\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\);?'
    match = re.search(pattern, content)
    if not match:
        raise ValueError("Could not find 'Uinlet (X Y Z)' in 0/U file.")
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def math_to_met(ux: float, uy: float) -> Tuple[float, float]:
    """
    Convert velocity vector from mathematical convention to meteorological.
    """
    speed = math.sqrt(ux**2 + uy**2)
    if speed < 1e-10:
        return 0.0, 0.0

    # Recover the angle used as a math angle (CCW from +X)
    theta_rad = math.atan2(uy, ux)

    # Correct decomposition using meteorological convention:
    # Wind FROM theta_met_rad means the vector points in the opposite direction.
    ux_correct = -speed * math.sin(theta_rad)
    uy_correct = -speed * math.cos(theta_rad)

    return ux_correct, uy_correct


def update_uinlet(content: str, ux: float, uy: float, uz: float) -> str:
    """Replace the Uinlet vector in the file content."""
    new_value = f"Uinlet  ({ux:g} {uy:g} {uz:g});"
    pattern = r'Uinlet\s*\(\s*[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\);?'
    
    updated, n = re.subn(pattern, new_value, content)
    if n == 0:
        raise ValueError("Replacement failed — pattern not found in 0/U.")
    return updated


def main():
    u_file = Path("0/U")
    if not u_file.exists():
        print(f"Error: '{u_file}' not found. Run from your OpenFOAM case root.", file=sys.stderr)
        sys.exit(1)

    content = u_file.read_text()

    try:
        # --- Read current (mathematical) vector ---
        ux_math, uy_math, uz = parse_uinlet(content)
        speed = math.sqrt(ux_math**2 + uy_math**2)
        
        # Calculate the angle that was incorrectly used (for logging)
        theta_deg = math.degrees(math.atan2(uy_math, ux_math))

        print(f"  Read Uinlet        : ({ux_math:.6g}, {uy_math:.6g}, {uz:.6g})")
        print(f"  Speed              : {speed:.4f} m/s")
        print(f"  Interpreted Angle  : {theta_deg:.2f}°")

        # --- Convert to meteorological convention ---
        ux_met, uy_met = math_to_met(ux_math, uy_math)
        print(f"  Corrected Uinlet   : ({ux_met:g}, {uy_met:g}, {uz:g})")

        # --- Backup original file ---
        backup = u_file.with_suffix(".U.bak")
        shutil.copy2(u_file, backup)
        print(f"  Backup saved       : {backup}")

        # --- Write corrected vector back ---
        updated = update_uinlet(content, ux_met, uy_met, uz)
        u_file.write_text(updated)
        print(f"  Updated            : {u_file}  ✓")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
