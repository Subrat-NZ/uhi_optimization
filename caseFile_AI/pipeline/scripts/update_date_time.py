#!/usr/bin/env python3
"""
Find the hottest hour in an EPW file and update date_and_time.xyz.

Rules for the data line (MM,DD followed by a list of hours):
  * MM,DD is always replaced with the peak-temperature date.
  * If the line has exactly 3 hours (e.g. 09 13 17), the middle one is
    replaced by the peak hour:      03,20  09  13  17  ->  08,01  09  16  17
  * If the line has a longer list (e.g. 06..18 or a full 24 h list), the
    hours are left untouched:       08,01  06 07 ... 18 -> unchanged hours
    (a warning is printed if the peak hour is not in that list).

Usage:
    python update_date_time.py weather.epw date_and_time.xyz
"""
import re
import sys
import csv


def find_peak(epw_path):
    """Return (temp, month, day, hour) of the first maximum dry-bulb temperature."""
    best = None
    with open(epw_path, newline="") as f:
        for row in csv.reader(f):
            # Data rows start after the 8 header lines; they begin with a 4-digit year
            if len(row) < 7 or not row[0].strip().isdigit():
                continue
            try:
                month, day, hour = int(row[1]), int(row[2]), int(row[3])
                temp = float(row[6])          # Dry Bulb Temperature [C]
            except ValueError:
                continue
            if temp >= 99.0:                  # EPW "missing" flag is 99.9
                continue
            if best is None or temp > best[0]:
                best = (temp, month, day, hour)
    if best is None:
        raise RuntimeError("No valid data rows found in EPW file.")
    return best


DATE_RE = re.compile(r"^(\s*)\d{1,2}\s*,\s*\d{1,2}(.*)$")


def main():
    epw_path = sys.argv[1] if len(sys.argv) > 1 else "weather.epw"
    xyz_path = sys.argv[2] if len(sys.argv) > 2 else "date_and_time.xyz"

    temp, mm, dd, hh = find_peak(epw_path)
    print(f"Peak temperature: {temp:.1f} C on {mm:02d}/{dd:02d} at hour {hh:02d}")

    with open(xyz_path) as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        m = DATE_RE.match(line)
        if line.strip() and not line.lstrip().startswith("#") and m:
            lead, rest = m.group(1), m.group(2)
            parts = re.split(r"(\d+)", rest)      # hours sit at odd indices
            hours = [int(x) for x in parts[1::2]]

            if len(hours) == 3:                    # e.g. 09 13 17 -> replace the middle hour
                parts[3] = f"{hh:02d}"
                print(f"3-hour list: middle hour set to {hh:02d}")
            elif len(hours) > 3:                   # long list: leave hours alone
                if hh not in hours:
                    print(f"WARNING: peak hour {hh:02d} is not in the existing hour list; "
                          "hours left unchanged.")
                else:
                    print("Hour list already covers the peak hour; hours left unchanged.")
            else:
                print(f"WARNING: unexpected hour list {hours}; hours left unchanged.")

            lines[i] = f"{lead}{mm:02d},{dd:02d}" + "".join(parts)
            print(f"Updated {xyz_path}: {lines[i]}")
            break
    else:
        raise RuntimeError(f"No 'MM,DD ...' data line found in {xyz_path}.")

    with open(xyz_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()