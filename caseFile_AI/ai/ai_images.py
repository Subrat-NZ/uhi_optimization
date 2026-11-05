#!/usr/bin/env python3
"""ai_images.py  --  images/{U,T,RH}_z2m.png: the AI fields cropped to the site (+ margin), buildings in grey.  Never needed by the optimisation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, default=Path("."))
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    case = a.case.resolve()
    F = np.load(case / "ai_work" / "fields.npz")
    ext = [float(v) for v in F["extent"]]
    js = a.json or sorted(case.glob("*.json"))[0]
    d = json.loads(Path(js).read_text())
    co = (d.get("siteOnly_json", d).get("site") or {}).get("coordinates") or []
    xs = [p["x"] for p in co]
    ys = [p["y"] for p in co]
    H, W = F["T_K"].shape
    if xs:
        mx, my = 0.1 * (max(xs) - min(xs)), 0.1 * (max(ys) - min(ys))
        x0, x1, y0, y1 = max(min(xs) - mx, ext[0]), min(max(xs) + mx, ext[1]), max(min(ys) - my, ext[2]), min(max(ys) + my, ext[3])
    else:
        x0, x1, y0, y1 = ext
    c0, c1 = int((x0 - ext[0]) / (ext[1] - ext[0]) * (W - 1)), int(np.ceil((x1 - ext[0]) / (ext[1] - ext[0]) * (W - 1)))
    r0, r1 = int((ext[3] - y1) / (ext[3] - ext[2]) * (H - 1)), int(np.ceil((ext[3] - y0) / (ext[3] - ext[2]) * (H - 1)))
    out = case / "images"
    out.mkdir(exist_ok=True)
    for key, name, label, cmap in (("U_ms", "U", "|U| [m/s]", "viridis"), ("T_K", "T", "T [C]", "jet"), ("RH_pct", "RH", "RH [%]", "YlGnBu")):
        arr = F[key][r0:r1 + 1, c0:c1 + 1].astype(float)
        if key == "T_K":
            arr = arr - 273.15
        fig, ax = plt.subplots(figsize=(6, 5.5))
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.75")
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cm, extent=[x0, x1, y0, y1], aspect="equal")
        if xs:
            ax.plot(xs + xs[:1], ys + ys[:1], "k-", lw=0.8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        ax.set_title(f"{label.split(' ')[0]} from the AI models (grey = building)", fontsize=9)
        fig.savefig(out / f"{name}_z2m.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("images written")


if __name__ == "__main__":
    sys.exit(main())
