#!/usr/bin/env bash
# prepare_all.sh — ONE script that handles everything for a job.
#
# Usage:
#   scripts/prepare_all.sh <scene.json> <weather.epw> <output_dir>
#
# Steps (in order, stops on the first error):
#   1. flip_json.py          scene.json -> <output_dir>/geometry/scene_flipped.json
#                            (lawns / ponds / pathways flipped about the X-axis)
#   2. json_to_obj.py        flipped json -> <output_dir>/geometry/triSurface/*.obj
#   3. extractWeatherData.py weather.epw -> <output_dir>/weather/weatherData/<MMDD>/*
#                            (sunPosVector, Tambient, wetBulbTemperature, windVelocity, ...)
#   4. rasterize_obj.py      triSurface/*.obj -> <output_dir>/raster2d/geometry_labels.npy (+png, meta)
#   5. field_to_2d_aligned.py weatherData -> <output_dir>/raster2d/<MMDD>_<t>_{T,u,w}_inlet.npy
#   6. U_2d.npy + U_flow_direction.json from WIND_2D_DIR -> raster2d/, then wind_inlets.py
#
# Required environment:
#   WIND_2D_DIR           U_2d.npy + U_flow_direction.json + wind_extent.txt made from the wind model's VTK by the
#                         RunPod worker (app/wind_ai.py). The raster extent is wind_extent.txt, so geometry, U and
#                         inlets share the wind model's full XY domain.
# Optional environment variables:
#   PYTHON_BIN            python to use                       (default: python3)
#   DATE_AND_TIME_FILE    date/hours file for weather extract (default: scripts/date_and_time.xyz)
#   BUILDING_MESH_SIZE    metres, passed as --building-mesh-size
#   TREE_CROWN_SHAPE      sphere|cone|spheroid|ellipsoid
#   RASTER_RES            2D raster resolution                (default: 512)
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <scene.json> <weather.epw> <output_dir>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATE_AND_TIME_FILE="${DATE_AND_TIME_FILE:-$SCRIPT_DIR/date_and_time.xyz}"

SCENE_JSON="$(readlink -f "$1")"
EPW_FILE="$(readlink -f "$2")"
OUT_DIR="$(mkdir -p "$3" && readlink -f "$3")"

GEOM_DIR="$OUT_DIR/geometry"
TRISURFACE_DIR="$GEOM_DIR/triSurface"
FLIPPED_JSON="$GEOM_DIR/scene_flipped.json"
WEATHER_DIR="$OUT_DIR/weather"
RASTER_DIR="$OUT_DIR/raster2d"
RASTER_RES="${RASTER_RES:-512}"

[ -s "$SCENE_JSON" ] || { echo "[prepare] scene json missing or empty: $SCENE_JSON" >&2; exit 1; }
[ -s "$EPW_FILE" ]   || { echo "[prepare] epw file missing or empty: $EPW_FILE" >&2; exit 1; }
[ -s "$DATE_AND_TIME_FILE" ] || { echo "[prepare] date_and_time file missing: $DATE_AND_TIME_FILE" >&2; exit 1; }
[ -n "${WIND_2D_DIR:-}" ] && [ -s "$WIND_2D_DIR/U_2d.npy" ] && [ -s "$WIND_2D_DIR/wind_extent.txt" ] || {
  echo "[prepare] WIND_2D_DIR missing or incomplete: ${WIND_2D_DIR:-<unset>}" >&2; exit 1; }
mkdir -p "$TRISURFACE_DIR" "$WEATHER_DIR"

# ── 1. flip ────────────────────────────────────────────────────────────────────
echo "[prepare] 1/6 flip: $SCENE_JSON -> $FLIPPED_JSON"
"$PYTHON_BIN" "$SCRIPT_DIR/flip_json.py" "$SCENE_JSON" "$FLIPPED_JSON"
[ -s "$FLIPPED_JSON" ] || { echo "[prepare] flip did not create $FLIPPED_JSON" >&2; exit 1; }

# ── 2. json -> obj ─────────────────────────────────────────────────────────────
echo "[prepare] 2/6 json_to_obj: $FLIPPED_JSON -> $TRISURFACE_DIR"
EXTRA=()
[ -n "${BUILDING_MESH_SIZE:-}" ] && EXTRA+=(--building-mesh-size "$BUILDING_MESH_SIZE")
"$PYTHON_BIN" "$SCRIPT_DIR/json_to_obj.py" "${EXTRA[@]}" "$FLIPPED_JSON" "$TRISURFACE_DIR"
[ -s "$TRISURFACE_DIR/buildings_refined.obj" ] || {
  echo "[prepare] buildings_refined.obj was not created or is empty" >&2; exit 1; }

# ── 3. weather extraction ──────────────────────────────────────────────────────
# extractWeatherData.py reads *.epw and date_and_time.xyz from the CURRENT
# directory and writes ./weatherData/<MMDD>/, so run it inside its own folder.
echo "[prepare] 3/6 extractWeatherData: $EPW_FILE -> $WEATHER_DIR/weatherData"
rm -rf "$WEATHER_DIR/weatherData"
find "$WEATHER_DIR" -maxdepth 1 -name '*.epw' -delete
cp "$EPW_FILE" "$WEATHER_DIR/weather.epw"
cp "$DATE_AND_TIME_FILE" "$WEATHER_DIR/date_and_time.xyz"
( cd "$WEATHER_DIR" && "$PYTHON_BIN" "$SCRIPT_DIR/extractWeatherData.py" )

if [ -z "$(find "$WEATHER_DIR/weatherData" -type f -name 'sunPosVector' 2>/dev/null | head -1)" ]; then
  echo "[prepare] weather extraction produced no weatherData/<MMDD>/sunPosVector" >&2
  exit 1
fi

# ── 3b. raster extent = wind model domain ──────────────────────────────────────
# U_2d.npy was resampled on exactly this extent by the worker -> the geometry raster must use it too
# (the wind domain is the OBJ bbox + margin, so nothing of U is cut off).
read -r EX0 EX1 EY0 EY1 < "$WIND_2D_DIR/wind_extent.txt"
EXTENT_ARGS=(--extent "$EX0" "$EX1" "$EY0" "$EY1")
echo "[prepare] raster extent = wind model domain: x[$EX0,$EX1] y[$EY0,$EY1]"

# ── 4. OBJ -> 2D geometry raster ───────────────────────────────────────────────
echo "[prepare] 4/6 rasterize_obj: $TRISURFACE_DIR -> $RASTER_DIR"
mkdir -p "$RASTER_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/rasterize_obj.py" "$TRISURFACE_DIR" "$RASTER_DIR" --res "$RASTER_RES" "${EXTENT_ARGS[@]}"
[ -s "$RASTER_DIR/geometry_labels.npy" ] || {
  echo "[prepare] geometry_labels.npy was not created" >&2; exit 1; }

# ── 5. weather -> T / u / w inlet 2D npy ───────────────────────────────────────
echo "[prepare] 5/6 field_to_2d_aligned: $WEATHER_DIR/weatherData -> $RASTER_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/field_to_2d_aligned.py" \
  "$WEATHER_DIR/weatherData" "$RASTER_DIR/geometry_meta.json" "$RASTER_DIR"
[ -n "$(find "$RASTER_DIR" -maxdepth 1 -name '*_T_inlet.npy' | head -1)" ] || {
  echo "[prepare] no *_T_inlet.npy created" >&2; exit 1; }

# ── 6. wind field from the worker -> raster2d, inlets on the wind model's inlet edge(s) ──
echo "[prepare] 6/6 U_2d from the RunPod worker: $WIND_2D_DIR -> $RASTER_DIR"
cp "$WIND_2D_DIR/U_2d.npy" "$RASTER_DIR/U_2d.npy"
[ -s "$WIND_2D_DIR/U_flow_direction.json" ] && cp "$WIND_2D_DIR/U_flow_direction.json" "$RASTER_DIR/"
# inlets follow the wind model direction: same inlet edge(s) for T/u/w, values from the EPW per time
"$PYTHON_BIN" "$SCRIPT_DIR/wind_inlets.py" "$RASTER_DIR" "$WEATHER_DIR/weatherData"

echo "[prepare] done."
echo "  geometry: $TRISURFACE_DIR"
ls -la "$TRISURFACE_DIR"
echo "  weather : $WEATHER_DIR/weatherData"
ls -R "$WEATHER_DIR/weatherData" | head -40
echo "  raster2d: $RASTER_DIR"
ls -la "$RASTER_DIR"