#!/usr/bin/env python3
import re

# ── 1. Read Uinlet from 0/U ──────────────────────────────────────────────────
with open("0/U") as f:
    text = f.read()

match = re.search(r"Uinlet\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", text)
Uinlet = [float(match.group(i)) for i in (1, 2, 3)]
print(f"Uinlet = {Uinlet}")

# ── 2. Define all side faces with their outward normals ───────────────────────
faces = {
    "East":  {"vertices": "(1 5 6 2)", "normal": [ 1,  0, 0]},
    "North": {"vertices": "(3 2 6 7)", "normal": [ 0,  1, 0]},
    "South": {"vertices": "(0 4 5 1)", "normal": [ 0, -1, 0]},
    "West":  {"vertices": "(0 3 7 4)", "normal": [-1,  0, 0]},
}

# ── 3. Dot product ────────────────────────────────────────────────────────────
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# ── 4. Classify each face — THREE cases ──────────────────────────────────────
inlet_faces    = []
outlet_faces   = []
symmetry_faces = []  # V·n == 0 → parallel to wind → symmetry

for name, data in faces.items():
    vdn = dot(Uinlet, data["normal"])
    if vdn < 0:
        tag = "INLET"
        inlet_faces.append((name, data["vertices"]))
    elif vdn > 0:
        tag = "OUTLET"
        outlet_faces.append((name, data["vertices"]))
    else:                              # vdn == 0
        tag = "SYMMETRY"
        symmetry_faces.append((name, data["vertices"]))
    print(f"  {name}: V·n = {vdn:+.2f}  → {tag}")

# ── 5. Build new boundary block ───────────────────────────────────────────────
def make_patch(patch_name, patch_type, face_list):
    face_lines = "\n".join(f"            {v}   // {n}" for n, v in face_list)
    return f"""    {patch_name}
    {{
        type {patch_type};
        faces
        (
{face_lines}
        );
    }}"""

# Top face (4 7 6 5) always symmetry — merge with any symmetry side faces
all_symmetry_faces = symmetry_faces + [("Top", "(4 7 6 5)")]

new_boundary = "boundary\n(\n"

if inlet_faces:
    new_boundary += make_patch("inlet",  "patch",    inlet_faces)    + "\n\n"
if outlet_faces:
    new_boundary += make_patch("outlet", "patch",    outlet_faces)   + "\n\n"

new_boundary += make_patch("top",    "symmetry", all_symmetry_faces) + "\n\n"
new_boundary += """\
    ground
    {
        type wall;
        faces ( (0 1 2 3) );   // Bottom
    }

);\n"""

# ── 6. Find and replace boundary block by counting parenthesis depth ─────────
def replace_boundary_block(content, new_block):
    m = re.search(r'\bboundary\s*\(', content)
    if not m:
        raise ValueError("Could not find 'boundary (' in blockMeshDict")

    block_start  = m.start()
    paren_start  = m.end() - 1
    depth        = 0
    block_end    = paren_start

    for i in range(paren_start, len(content)):
        if content[i] == '(':
            depth += 1
        elif content[i] == ')':
            depth -= 1
            if depth == 0:
                block_end = i + 1
                if block_end < len(content) and content[block_end] == ';':
                    block_end += 1
                break
    else:
        raise ValueError("Unmatched parentheses in boundary block")

    return content[:block_start] + new_block + content[block_end:]

# ── 7. Rewrite blockMeshDict ──────────────────────────────────────────────────
bmd_path = "system/blockMeshDict"
with open(bmd_path) as f:
    bmd = f.read()

bmd_new = replace_boundary_block(bmd, new_boundary)

with open(bmd_path, "w") as f:
    f.write(bmd_new)

print("\nblockMeshDict updated successfully. Run blockMesh now.")
