input_file = "buildings.obj"
output_file = "buildings_transformed.obj"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        if line.startswith("v "):
            parts = line.split()
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            # swap Y -> Z
            new_x = x
            new_y = z
            new_z = y

            fout.write(f"v {new_x} {new_y} {new_z}\n")
        else:
            fout.write(line)
