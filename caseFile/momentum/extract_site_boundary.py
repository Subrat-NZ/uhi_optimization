"""
extract_obj_object.py
---------------------
Extracts a named object (e.g. 'site_boundary') from a multi-object .obj file
and saves it to a new .obj file with correctly re-indexed vertex indices.

Usage:
    python extract_obj_object.py
    
Or import and call extract_object() directly for custom input/output paths and names.
"""

def extract_object(input_path: str, output_path: str, object_name: str) -> None:
    """
    Extract a single named object from an OBJ file and write it to a new file.

    OBJ face indices are global across the whole file, so this function:
      1. Parses all vertices and groups faces+vertices by object name.
      2. Finds the target object's faces and the global vertex indices they reference.
      3. Re-maps those global indices to local 1-based indices.
      4. Writes the extracted object to a clean, self-contained OBJ file.

    Args:
        input_path  : Path to the source .obj file.
        output_path : Path for the output .obj file.
        object_name : Exact name of the object to extract (case-sensitive).
    """

    # ── 1. Parse the file ────────────────────────────────────────────────────
    all_vertices = []        # list of raw "v ..." lines (0-indexed → global index 1-based)
    objects = {}             # { name: { "face_lines": [...], "other_lines": [...] } }
    current_object = None

    with open(input_path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("v "):          # vertex (position)
                all_vertices.append(stripped)

            elif stripped.startswith("o "):        # object declaration
                current_object = stripped[2:].strip()
                objects[current_object] = {"face_lines": [], "other_lines": []}

            elif stripped.startswith("f "):        # face
                if current_object is not None:
                    objects[current_object]["face_lines"].append(stripped)

            elif stripped and current_object is not None:
                # vt, vn, usemtl, s, etc. — kept verbatim for the target object
                objects[current_object]["other_lines"].append(stripped)

    # ── 2. Validate ──────────────────────────────────────────────────────────
    if object_name not in objects:
        available = ", ".join(objects.keys())
        raise ValueError(
            f"Object '{object_name}' not found in '{input_path}'.\n"
            f"Available objects: {available}"
        )

    obj_data = objects[object_name]

    # ── 3. Collect the global vertex indices used by this object's faces ─────
    #
    # OBJ face tokens can be:  v   /  v/vt  /  v/vt/vn  /  v//vn
    # We only care about the first component (position index).
    #
    used_global_indices = set()

    for face_line in obj_data["face_lines"]:
        tokens = face_line.split()[1:]   # drop the leading "f"
        for token in tokens:
            v_idx = int(token.split("/")[0])
            # OBJ supports negative indices (relative); normalise to positive
            if v_idx < 0:
                v_idx = len(all_vertices) + v_idx + 1
            used_global_indices.add(v_idx)

    # ── 4. Build a mapping  global_index → new_local_index (1-based) ─────────
    sorted_global = sorted(used_global_indices)
    global_to_local = {g: local for local, g in enumerate(sorted_global, start=1)}

    # ── 5. Re-index face lines ───────────────────────────────────────────────
    def remap_face_token(token: str) -> str:
        parts = token.split("/")
        g = int(parts[0])
        if g < 0:
            g = len(all_vertices) + g + 1
        parts[0] = str(global_to_local[g])
        return "/".join(parts)

    remapped_faces = []
    for face_line in obj_data["face_lines"]:
        tokens = face_line.split()
        new_tokens = ["f"] + [remap_face_token(t) for t in tokens[1:]]
        remapped_faces.append(" ".join(new_tokens))

    # ── 6. Write the output file ─────────────────────────────────────────────
    with open(output_path, "w") as out:
        out.write(f"# Extracted object '{object_name}' from '{input_path}'\n")
        out.write(f"o {object_name}\n")

        # Write only the vertices this object actually uses
        for global_idx in sorted_global:
            out.write(all_vertices[global_idx - 1] + "\n")   # list is 0-based

        # Write any extra directives (vt, vn, usemtl, s …)
        for extra in obj_data["other_lines"]:
            out.write(extra + "\n")

        # Write re-indexed faces
        for face in remapped_faces:
            out.write(face + "\n")

    print(f"✓ Extracted '{object_name}' → '{output_path}'")
    print(f"  Vertices : {len(sorted_global)}")
    print(f"  Faces    : {len(remapped_faces)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_object(
        input_path="buildings.obj",
        output_path="site_boundary.obj",
        object_name="site_boundary",
    )
