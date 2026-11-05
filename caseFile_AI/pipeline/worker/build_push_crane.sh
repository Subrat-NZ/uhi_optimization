#!/usr/bin/env bash
# Build + push the worker image WITHOUT Docker (no daemon, no sudo) -- same result as worker/Dockerfile.
#
#   crane auth login index.docker.io -u <user> -p <access-token>     # once
#   worker/build_push_crane.sh jagathmcd/uhi-rht-worker:v1
#
# Three layers on the official PyTorch CUDA image:
#   1. /app/pydeps   worker/requirements.txt as Python 3.11 manylinux wheels (on PYTHONPATH)
#   2. /app          model weights: T_model/*.pth, RH_model/*_net_G.pth, wind_api/model/model/ (rarely change)
#   3. /app          code: scripts/, pressure_calculator/, predict_T/RH.py + norm_stats, wind model, app/, worker/
# Needs: crane (github.com/google/go-containerregistry), python3 with pip, the .pth weights in this checkout.
set -euo pipefail

IMAGE="${1:?usage: $0 <dockerhub-user>/<repo>:<tag>}"
BASE="${BASE_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"   # python 3.11 in /opt/conda
PY_VERSION="${PY_VERSION:-3.11}"
CRANE="${CRANE:-crane}"
PIP="${PIP:-python3 -m pip}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for w in "$ROOT/T_model/latest_net_G.pth" "$ROOT"/RH_model/*_net_G.pth "$ROOT/wind_api/model/model/best.ckpt"; do
  [ "$(stat -c %s "$w")" -gt 1000000 ] || { echo "missing or LFS-pointer weights: $w" >&2; exit 1; }
done

echo "[build] 1/4 python deps for $PY_VERSION"
mkdir -p "$WORK/deps/app/pydeps"
$PIP install -q --target "$WORK/deps/app/pydeps" --python-version "$PY_VERSION" --implementation cp \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  --only-binary=:all: --no-compile -r "$ROOT/worker/requirements.txt"
# lightning / torchmetrics pull torch + CUDA wheels (~5 GB) again: the base image already has torch 2.7.1
( cd "$WORK/deps/app/pydeps" && rm -rf torch torch-*.dist-info torchgen functorch nvidia nvidia_*.dist-info \
    triton triton-*.dist-info bin/torchrun bin/torchfrtrace )

echo "[build] 2/4 weights + code"
W="$WORK/weights/app" C="$WORK/code/app"
mkdir -p "$W/T_model" "$W/RH_model" "$W/wind_api/model/model" "$C/app" "$C/worker" "$C/wind_api/model"
cp "$ROOT"/T_model/*.pth "$W/T_model/"
cp "$ROOT"/RH_model/*_net_G.pth "$W/RH_model/"
cp "$ROOT"/wind_api/model/model/* "$W/wind_api/model/model/"
cp -r "$ROOT/scripts" "$ROOT/pressure_calculator" "$C/"
mkdir -p "$C/T_model" "$C/RH_model"
cp "$ROOT/T_model/predict_T.py" "$ROOT/T_model/norm_stats.json" "$C/T_model/"
cp "$ROOT/RH_model/predict_RH.py" "$ROOT/RH_model/norm_stats.json" "$C/RH_model/"
cp -r "$ROOT/wind_api/model/__init__.py" "$ROOT/wind_api/model/infrenace.py" "$ROOT/wind_api/model/turbdiff" \
  "$C/wind_api/model/"
cp "$ROOT/app/__init__.py" "$ROOT/app/runpod_client.py" "$ROOT/app/wind_ai.py" "$C/app/"
cp "$ROOT/worker/handler.py" "$C/worker/"
find "$WORK" -name __pycache__ -type d -prune -exec rm -rf {} +

echo "[build] 3/4 layers"
# deterministic tars (fixed order, mtime, owner; gzip -n): an unchanged layer keeps its digest, so a code-only
# rebuild re-uploads a few hundred KB instead of the 1.2 GB of weights
layer() { tar -C "$WORK/$1" --sort=name --mtime=2026-01-01 --owner=0 --group=0 --numeric-owner -cf - app \
            | gzip -n > "$WORK/$1.tar.gz"; }
layer deps; layer weights; layer code
ls -lh "$WORK"/*.tar.gz

echo "[build] 4/4 push $IMAGE (base $BASE)"
$CRANE append --platform linux/amd64 -b "$BASE" -f "$WORK/deps.tar.gz" -f "$WORK/weights.tar.gz" -f "$WORK/code.tar.gz" -t "$IMAGE.tmp"
$CRANE mutate "$IMAGE.tmp" -t "$IMAGE" --workdir /app \
  --env PYTHONPATH=/app/pydeps --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env MPLBACKEND=Agg --env APP_ROOT=/app \
  --entrypoint "" --cmd python,-u,/app/worker/handler.py
$CRANE delete "$IMAGE.tmp" 2>/dev/null || true   # Docker Hub may refuse tag deletes; harmless
echo "[build] pushed: $($CRANE digest "$IMAGE") -> $IMAGE"
