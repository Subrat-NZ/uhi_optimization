import re

input_file = "buildings.obj"
output_file = "buildingsCleaned.obj"

remove_patterns = [
    r"site_boundary",
    r"vegetation_zone_\d+",
    r"rainwater_pond_\d+"
]

def should_remove(name):
    return any(re.match(p, name) for p in remove_patterns)

skip = False

with open(input_file) as fin, open(output_file, "w") as fout:

    for line in fin:

        # detect object start
        if line.startswith("o "):
            name = line.split()[1]

            if should_remove(name):
                skip = True
            else:
                skip = False
                fout.write(line)

            continue

        # always keep vertex lines
        if line.startswith("v "):
            fout.write(line)
            continue

        # skip faces only if object removed
        if line.startswith("f ") and skip:
            continue

        if not skip:
            fout.write(line)
