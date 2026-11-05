# UHI worker: wind + T + RH on one RunPod Serverless endpoint

One public image, `docker.io/jagathmcd/uhi-rht-worker:v2`, holds all three models. Each is loaded once per
container. The API sends two kinds of job to the same endpoint:

```
AWS host (CPU, always on)                               RunPod endpoint (GPU, scales to 0)
POST /rhtAPI (UHI Instant: scene + EPW + wind OBJ)       worker/handler.py
  1. wind OBJ (14 KB, inline)  ──── task "wind" ───────▶ wind model -> result.vtk (stays on worker)
     pressure reconstruction + facade Cp -> updated result.vtk + facade_cp_summary.txt
     wind_2d/Summer/ ◀── U_2d.npy + flow + extent + Cp summary ── vtk_to_2d on the VTK's own extent
  2. flip / json_to_obj / weather / rasterize / inlets
     (prepare_all.sh with WIND_2D_DIR, no VTK, no S3)
  3. per time: rasters ~1 MB  ───── task "rht" ────────▶ T model -> RH model (T field stays on worker)
     rht_results/*.png ◀──────────── PNG + JSON 0.4 MB ──
```

- **No S3 and no WindAI API host.** Everything travels inline in the RunPod JSON, and the worker needs no AWS keys.
- **No GIF rendering.** UHI only uses the wind field.
- **Same results as the old path.** The worker's `U_2d.npy` is bit-identical to the one `prepare_all.sh` used to make
  from the downloaded VTK, because the raster extent is the VTK extent.

## Build and push

The model code is in git, but the weights are not (too large for GitHub). In a fresh checkout, copy them in first:

| File | Size |
|---|---|
| `T_model/latest_net_G.pth` | 268 MB |
| `RH_model/<epoch>_net_G.pth` (today `495_net_G.pth`) | 268 MB |
| `wind_api/model/model/best.ckpt` | 663 MB |

The build stops with "missing or LFS-pointer weights" if one is absent.

Without Docker (no daemon or sudo), which is how `v1`/`v2` were built:

```bash
crane auth login index.docker.io -u jagathmcd --password-stdin   # paste an access token
worker/build_push_crane.sh jagathmcd/uhi-rht-worker:v3
```

With Docker, from the repo root:
`docker build --platform linux/amd64 -f worker/Dockerfile -t jagathmcd/uhi-rht-worker:v3 . && docker push ...`

Use a new tag for every build. Workers cache images by tag.

## Endpoint settings

| Setting | Value |
|---|---|
| Container image | `jagathmcd/uhi-rht-worker:v2` (public, so no registry auth) |
| Start command | empty (the image runs `python -u /app/worker/handler.py`) |
| Container disk | 25 GB |
| GPU | 24 GB and up, several types ticked (A5000, L4, 3090, 4090, A6000, L40S). The wind model samples a 150×150×50 grid |
| Active workers | 0, max workers 2–3, idle timeout 60 s, FlashBoot on |
| Network volume / env vars | none |

Worker environment overrides for the wind post-processing are optional:

| Variable | Default | Purpose |
|---|---:|---|
| `PRESSURE_CALCULATOR_TIMEOUT_SEC` | `1800` | Maximum time for pressure reconstruction from the renormalized velocity field. |
| `FACADE_CP_TIMEOUT_SEC` | `900` | Maximum time for per-building facade Cp mapping and summary. |

The wind task returns `facade_cp_summary.txt` with an overall area-weighted
wall Cp and per-building wall/roof Cp summaries, alongside `U_2d.npy`,
`U_flow_direction.json`, and `wind_extent.txt`. The original VTK's velocity
and solid-mask arrays are preserved when its pressure field is replaced, so
the subsequent VTK-to-2D conversion can read the same velocity-magnitude and
solid arrays as before. Pressure reconstruction runs on worker CPU using the
SciPy sparse solver with PyAMG-preconditioned conjugate gradients; it anchors
the inlet at 101325 Pa. Cp uses the submitted inlet-speed magnitude as
`U_ref` and 101325 Pa as the reference pressure. By default, the facade mapper
includes OBJ object names matching `(?i)building` and excludes other objects
such as trees. The report is saved under the run's
`output/wind_2d/Summer/facade_cp_summary.txt` and its text is included in the
completed job's `rht.facade_cp` status result.

## `.env` on the AWS host

```
RUNPOD_API_KEY=rpa_...
RUNPOD_ENDPOINT_ID=<the endpoint above>
RUNPOD_TIMEOUT_SEC=900
WIND_RUNPOD_TIMEOUT_SEC=1800
```

Apply with `set -a && source .env && set +a && pm2 restart 0 --update-env`. A plain `pm2 restart` keeps the old
environment.

## Test locally (CPU, no Docker or RunPod)

```bash
APP_ROOT=. venv/bin/python worker/handler.py --test-wind runs/<...>/input/wind.obj 2.0 275
APP_ROOT=. venv/bin/python worker/handler.py --test \
  runs/<...>/output/rht_work/0900/raster2d runs/<...>/output/weather/weatherData
```
