import sys
from typing import Iterable, Tuple

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


def combined_obj_bounds(paths: Iterable[str]) -> Tuple[float, float, float, float, float, float]:
    """Return bounds spanning every non-empty OBJ in *paths*."""
    xmin = ymin = zmin = float('inf')
    xmax = ymax = zmax = float('-inf')
    used = []

    for path in paths:
        try:
            bounds = parse_obj_bounds(path)
        except ValueError:
            # json_to_obj.py creates empty placeholders for absent entity types.
            print(f"Skipping empty OBJ: {path}")
            continue
        pxmin, pxmax, pymin, pymax, pzmin, pzmax = bounds
        xmin = min(xmin, pxmin); xmax = max(xmax, pxmax)
        ymin = min(ymin, pymin); ymax = max(ymax, pymax)
        zmin = min(zmin, pzmin); zmax = max(zmax, pzmax)
        used.append(path)

    if not used:
        raise ValueError("No vertex positions ('v') found in any input OBJ file.")
    print("Bounds include: " + ", ".join(used))
    return xmin, xmax, ymin, ymax, zmin, zmax


def main():
    if len(sys.argv) < 2:
        print("Usage: python objBoundingDim.py <surface1.obj> [surface2.obj ...]")
        sys.exit(1)

    try:
        xmin, xmax, ymin, ymax, zmin, zmax = combined_obj_bounds(sys.argv[1:])
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

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


    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    
    # Write meshParameters
    with open("meshParameters", "w") as f:
    
        if dz > dx:
            f.write(f"xMin\t{xmin-1.0*dz};\n")
            f.write(f"xMax\t{xmax+1.0*dz};\n")
            f.write(f"yMin\t{ymin-1.0*dz};\n")
            f.write(f"yMax\t{ymax+1.0*dz};\n")
            f.write(f"zMin\t{0.0};\n")
            f.write(f"zMax\t{zmax*1.5};\n")
    
        else:
            f.write(f"xMin\t{xmin-0.25*(dx+dy)};\n")
            f.write(f"xMax\t{xmax+0.25*(dx+dy)};\n")
            f.write(f"yMin\t{ymin-0.25*(dx+dy)};\n")
            f.write(f"yMax\t{ymax+0.25*(dx+dy)};\n")
            f.write(f"zMin\t{0.0};\n")
            f.write(f"zMax\t{zmax*3};\n")
    

    with open("blockParameters", "w") as f:
    
        if dz > dx:
            f.write(f"xMin\t{xmin-1.0*dz};\n")
            f.write(f"xMax\t{xmax+1.0*dz};\n")
            f.write(f"yMin\t{ymin-1.0*dz};\n")
            f.write(f"yMax\t{ymax+1.0*dz};\n")
            f.write(f"zMin\t{0.0};\n")
            f.write(f"zMax\t{dz*1.5};\n")
            f.write(f"zGrading\t{0.2*dz-0.2*zmin};\n")
    
        else:
            f.write(f"xMin\t{xmin-0.25*(dx+dy)};\n")
            f.write(f"xMax\t{xmax+0.25*(dx+dy)};\n")
            f.write(f"yMin\t{ymin-0.25*(dx+dy)};\n")
            f.write(f"yMax\t{ymax+0.25*(dx+dy)};\n")
            f.write(f"zMin\t{0.0};\n")
            f.write(f"zMax\t{dz*3};\n")
            f.write(f"zGrading\t{0.2*dz-0.2*zmin};\n")

    # Write refinementParam (same for both)
    with open("refinementParam", "w") as f:
        f.write(f"min\t({xmin}\t{ymin}\t{0.0});\n")
        f.write(f"max\t({xmax}\t{ymax}\t{zmax});\n")
    
    
    # Write locationIn
    with open("locationIn", "w") as f:
    
        if dz > dx:
            f.write(f"locationInMesh\t({xmin+1}\t{ymin+1}\t{1.5*zmax-1});\n")
        else:
            f.write(f"locationInMesh\t({xmin+1}\t{ymin+1}\t{2*zmax-1});\n")

#    with open("meshParameters", "w") as f:
#        f.write(f"xMin\t{xmin-1.0*dx};\n")
#        f.write(f"xMax\t{xmax+1.0*dx};\n")
#        f.write(f"yMin\t{ymin-1.0*dy};\n")
#        f.write(f"yMax\t{ymax+1.0*dy};\n")
#        f.write(f"zMin\t{zmin};\n")
#        f.write(f"zMax\t{zmax*2};\n")
#
#    with open("refinementParam", "w") as f:
#        f.write(f"min\t({xmin}\t{ymin}\t{zmin});\n")
#        f.write(f"max\t({xmax}\t{ymax}\t{zmax});\n")
#
#    with open("locationIn", "w") as f:
#        f.write(f"locationInMesh\t({xmin+1}\t{ymin+1}\t{2*zmax-1});\n")





if __name__ == "__main__":
    main()
