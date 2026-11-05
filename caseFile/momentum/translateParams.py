import sys
from typing import Tuple

def parse_obj_bounds(path: str) -> Tuple[float, float, float, float, float, float]:
    """
    Parses a Wavefront OBJ file and computes bounding box from vertex positions.
    Considers only lines starting with 'v ' (positions). Ignores 'vn', 'vt'.
    """
    xmin = ymin = zmin = float('inf')
    xmax = ymax = zmax = float('-inf')

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Fast path: most OBJ lines start with a small keyword
            if not line or line[0] not in ('v', ' ', '\t'):
                continue
            line = line.strip()
            # Vertex positions are 'v x y z [w]'
            if line.startswith('v '):
                parts = line.split()
                # Expect at least x,y,z; optional w ignored
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                    except ValueError:
                        continue
                    xmin = min(xmin, x); xmax = max(xmax, x)
                    ymin = min(ymin, y); ymax = max(ymax, y)
                    zmin = min(zmin, z); zmax = max(zmax, z)

    if xmin == float('inf'):
        raise ValueError("No vertex positions ('v') found in OBJ file.")
    return xmin, xmax, ymin, ymax, zmin, zmax


def main():
    if len(sys.argv) < 2:
        print("Usage: python obj_bounds.py <path_to_file.obj>")
        sys.exit(1)

    path = sys.argv[1]
    xmin, xmax, ymin, ymax, zmin, zmax = parse_obj_bounds(path)

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0

    print(f"x_min: {xmin}")
    print(f"x_max: {xmax}")
    print(f"y_min: {ymin}")
    print(f"y_max: {ymax}")
    print(f"z_min: {zmin}")
    print(f"z_max: {zmax}")
    print(f"Δx (width): {dx}")
    print(f"Δy (height): {dy}")
    print(f"Δz (depth): {dz}")
    print(f"Center: ({cx}, {cy}, {cz})")

    with open("translateParameters", "w") as f:
        f.write(f"xMin\t{xmin};\n")
        f.write(f"xMax\t{xmax};\n")
        f.write(f"yMin\t{ymin};\n")
        f.write(f"yMax\t{ymax};\n")
        f.write(f"zMin\t{zmin};\n")
        f.write(f"zMax\t{zmax};\n")





if __name__ == "__main__":
    main()
