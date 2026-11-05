# AI case folder for the NSGA-II template

The simulation of every individual / offspring (momentum -> air T and q) can now be replaced by the AI pipeline (wind model -> T model -> RH model).
Nothing else of the workflow changes: same population, crossover, mutation, mesh gate, objectives (f1 = UTCI heat, f2 = -UTCI cold), Pareto,
top 5, plots, analysis.

    caseFile/       the OpenFOAM case (unchanged)
    caseFile_AI/    NEW: the AI case (wind -> T -> RH), same contract as caseFile/

## 1. How to use it

Any of these (the type of a case folder is detected from its content, see section 4):

    # a) search with the AI case
    python python/optimize.py --case-file caseFile_AI --output-dir output_ai --run-dir run_ai

    # b) search with the AI case, then confirm the best 5 with OpenFOAM  (recommended)
    python python/optimize.py --case-file caseFile_AI --verify-case-file caseFile --verify-run-dir run_full

    # c) literally replace the folder: rename  caseFile -> caseFile_openfoam  and  caseFile_AI -> caseFile .  Then every command of the
    #    template (optimize.py, nsga2.py, nsga_batch.py, run_all_codes) runs the AI case with no option at all.  Rename back for OpenFOAM.

    # d) the OpenFOAM case as before: nothing to change.
    #    run_all_codes: UHI_OPT_ARGS="--case-file caseFile_AI --verify-case-file caseFile" ./run_all_codes   (nothing set = OpenFOAM)

`python python/optimize.py --check --case-file caseFile_AI` tests the set-up (back end, weights, modules) without running anything.

### Set-up of the AI case (once)
1. Weights: copy them into `caseFile_AI/pipeline/` (the zip you sent has 63-byte placeholders):
   `T_model/latest_net_G.pth`, `RH_model/495_net_G.pth` (or another `<n>_net_G.pth`), `wind_api/model/model/best.ckpt` (+ `stats.pickle`).
2. A back end (`UHI_AI_BACKEND`, default `auto`; the choice is printed by the pre-flight and in every `log.ai`):
   * `runpod`: `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in the environment (or in `caseFile_AI/ai/.env` / `caseFile_AI/pipeline/.env`).
     The same endpoint / image as your app (worker/handler.py). Scales with `--batch`.
   * `local`: the models run on this computer's GPU in ONE persistent process (`ai/serve.py`, started by the first case, weights loaded once,
     requests served one at a time, stops after 1 h idle). Needs torch + `pipeline/worker/requirements.txt` and the weights.
   * `mock`: numpy-only fake fields to test the plumbing. NOT a prediction. The pre-flight warns.
3. Python: the template's modules + `pandas`, `pillow`, `pythermalcomfort`, `scipy` (no OpenFOAM, no mpirun needed for the AI case).

## 2. What the AI case does (caseFile_AI/Allrun, same contract as the OpenFOAM Allrun)

Input (made by run_candidates.py, as before): `<NAME>.json`, `date_and_time.xyz`, `../epw/*.epw`.  Output: `UTCI_details.txt` with
`Site area-averaged: UTCI=xx.xx C`, which run_candidates.py parses.

| step | what | replaced by AI? |
|---|---|---|
| 1 | weather of the hour: `epw_to_weather.py` (target hour, `target.json`, sun, 24 h spin-up) | no, same script |
| 2 | geometry: `json_to_obj.py`, `objBoundingDim.py`, `cubic_block.py`, `gen_ground_obj.py` (python only, no mesh) | no, same scripts (the OpenFOAM mesh steps are skipped) |
| 3 | surface temperatures: `run_thermal.py --stage raw` (buildings, ground classes) | no, same script, runs while the models run |
| 4 | **wind**: `ai/wind_obj.py` -> wind model (inlet = EPW speed and direction of the hour) -> `U_2d` | **yes (momentum / simpleFoam)** |
| 4 | **T**: T model, inlets of the hour | **yes (airTqFoam, T)** |
| 4 | **RH**: RH model on that T | **yes (airTqFoam, q)** |
| 5 | Tmrt (sun, sky view, shade, surface T) and UTCI at the pixels of the site: `ai/ai_utci.py`, same formulas as `compute_utci.py` | no (only where Ta, RH, \|U\| come from) |

The weather the models see is `target.json` (the boundary of the OpenFOAM case), not the pipeline's own `extractWeatherData.py`, so the AI case and
the OpenFOAM case get exactly the same hour, including `--north-deg`. The layout is **not** flipped again (the NSGA JSONs are already in the
frame `json_to_obj.py` reads; `run_all_codes` flips the base once).

Result files, same places as the OpenFOAM case, so `analyze_pareto.py`, `plot_utci_only.py`, `nsga_family_video.py` work on AI runs unchanged:
`UTCI_details.txt`, `temperature_humidity/postProcessing/{utci.json,utci_points.csv}`, `temperature_humidity/weatherData/target.json`,
`temperature_humidity/surfaces/{buildings,trees}.obj`, `momentum/<NAME>.json`, `images/{U,T,RH}_z2m.png`. Extra: `ai_work/{fields.npz,ai_report.json,ai.log}`
(T [K], RH [%], |U| [m/s] rasters + extent). `utci.json` has `"source": "ai"`.

## 3. Files changed / added (everything else is yours, untouched)

New: `caseFile_AI/` (whole folder), `python/case_type.py`, this file.
Changed (each change is small; your originals are kept in `ai_integration_originals/`, so `diff ai_integration_originals/nsga2.py python/nsga2.py` shows exactly what changed):

| file | change |
|---|---|
| `python/run_candidates.py` | does not copy `pipeline/` (weights) into every candidate (`case_type.cfg` -> `no_copy`); starts `Allrun` with `UHI_CASE_SOURCE` = the original case folder; stamps `.uhi_status` with the case type and never reuses a result made by the other type |
| `python/optimize.py` | pre-flight asks `case_type.py` (OpenFOAM: OpenFOAM/mpirun/simpleFoam; AI: back end, weights, modules); honours `--case-file` / `--verify-case-file`; runs `nsga2.py verify` when `--verify-case-file` is given |
| `python/nsga2.py` | the state (`output/nsga_state.json`) records the case type; resuming a run with the other type stops with a clear message (use another `--output-dir` / `--run-dir`) |
| `run_all_codes` | optional `$UHI_OPT_ARGS` passed to `optimize.py` |

`mesh_gate.py` needs no change: `caseFile_AI/momentum/` keeps the two surface scripts it tests, so the gate works for AI runs too.

## 4. The contract (what "replace caseFile" means)

`run_candidates.py` needs of a case folder: an executable `Allrun`, no `*.json` in the folder; it copies the folder to `run/NAME/<season>/`, adds
`NAME.json` + `date_and_time.xyz` + `../epw`, runs `./Allrun`, reads `UTCI_details.txt`. Both case folders satisfy that. The type is
`"ai"` when `case_type.cfg` says so (it is a `.cfg`, not `.json`, on purpose), or when `ai/run_ai_case.py` exists and there is no
`momentum/Allrun|of_Allrun`; otherwise `"openfoam"`. A third kind of case (another model) only has to provide the same `Allrun`.

## 5. Speed

The AI step takes seconds (RunPod: seconds to a minute incl. queue; local GPU: seconds). The surface-temperature model (step 3) becomes the slowest
step; in the OpenFOAM case it was hidden behind meshing and simpleFoam. Measured here on 2 CPU cores (sandbox, mock models, a 45 + 47 building / 600 tree layout):
thermal model 195 s, AI 5 s, UTCI 36 s, whole case 236 s. Knobs (default = exactly the OpenFOAM case's settings):
`UHI_SPINUP_HOURS` (24), `UHI_THERMAL_SHADOW_SAMPLES` (3), `ai/config.json` `"points"` (1500), `"sky_rays"` (48). A smaller spin-up / 1 shadow sample is
about 4 x faster but is not the same surface temperature.

## 6. Limits and what was NOT tested  (please read)

* **Not tested with the real models.** The weights in the zip are placeholders and there is no GPU here. Everything around them was tested with
  `ai/mock_models.py` (same payloads, same files, fake fields): geometry, wind OBJ, rasters / alignment, inlets, back-end plumbing incl. the local server,
  UTCI, run_candidates, a small NSGA run. The calls to the real handler / RunPod client are the pipeline's own functions, but they have not run here.
* **The wind OBJ is my own writer** (`ai/wind_obj.py`): the zip has no OBJ producer (the app takes an upload). It writes `o building_1` (all of
  `buildings_refined.obj`) and one `o tree_<i>` octagonal 1 m prism per tree from the ground to the tree height (`ai/config.json`
  `"tree_wind_height": "total"` or `"trunk"`). If your wind model was trained on a different tree / building OBJ convention, adapt this one file.
* The wind field is checked against the geometry raster (building overlap, `align_wind` in `run_ai_case.py`; a mirror is corrected like the pipeline
  does, a mismatch stops the case with an error instead of using misplaced fields).
* UTCI is evaluated on the models' pedestrian layer (the wind model's z-index 1 raster, `"z_index"` in `ai/config.json`), not exactly 2 m, and with the
  models' accuracy: the objective is a **surrogate**. Use `--verify-case-file caseFile` for the top 5.
* One wind direction / speed per case: the hour's EPW values (calm hours use `min_calm_speed_ms` = 0.5 m/s with the EPW direction).
* A site that reaches outside the wind model's domain (bounding box of the buildings + 10 %) is only evaluated inside it; `utci.json` has `site_coverage`, `log.utci` warns below 90 %.
* The AI case repeats the surface-temperature model per candidate, like the OpenFOAM case, so surface temperatures of both agree by construction.
* `compute_utci.py` and `ai_utci.py` share their formulas by copy; if you change the radiation / UTCI part of one, change the other.

## 7. Troubleshooting

`log.ai` / `ai_work/ai.log` (models), `log.geometry`, `log.thermal_raw`, `log.utci` in `run/NAME/<season>/`; the reason of a failed case is the last
`Allrun: ERROR` line, which the console shows. Typical: "wind field does not match the geometry" (OBJ and raster frames differ: check `wind_obj.py`),
"weights are missing", "RunPod is not configured", "no inlet edge" (wind model reported no inflow edge).
Local server: `caseFile_AI/.ai_server/{state.json,server.log}`; stop it with `kill $(cat caseFile_AI/.ai_server/state.json.pid)`.
