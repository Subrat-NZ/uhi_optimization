#!/usr/bin/env python3
"""
set_inlet_wind.py  --  write the momentum inlet velocity of the target hour into momentum/0/U

Reads weatherData/target.json (made by epw_to_weather.py): wind speed at 10 m and the direction the wind blows FROM
(model frame, degrees clockwise from +y = north) and rewrites the `Uinlet (ux uy 0);` line of 0/U.
Flow vector = speed * ( -sin(from), -cos(from) ), e.g. wind from 23 deg -> (-0.39, -0.92) * speed, the same
convention as select_epw_cases.py (wind_flow_unit_vector_xy_model).  The inlet is uniform, as in the momentum case.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True, help="weatherData/target.json")
    ap.add_argument("--u-file", type=Path, required=True, help="momentum/0/U")
    ap.add_argument("--speed-factor", type=float, default=1.0)
    a = ap.parse_args(argv)
    b = json.loads(a.target.read_text())["boundary"]
    speed = float(b["wind_speed_10m_ms"]) * a.speed_factor
    th = math.radians(float(b["wind_from_deg_model"]))
    ux, uy = -speed * math.sin(th), -speed * math.cos(th)
    text = a.u_file.read_text()
    new, n = re.subn(r"Uinlet\s*\(\s*[^)]*\)\s*;", f"Uinlet  ({ux:.5f} {uy:.5f} 0);", text, count=1)
    if n != 1:
        raise SystemExit(f"no `Uinlet (...);` line in {a.u_file}")
    a.u_file.write_text(new)
    print(f"[set_inlet_wind] wind {speed:.2f} m/s from {b['wind_from_deg_model']:.0f} deg -> Uinlet ({ux:.4f} {uy:.4f} 0) in {a.u_file}")


if __name__ == "__main__":
    sys.exit(main())
