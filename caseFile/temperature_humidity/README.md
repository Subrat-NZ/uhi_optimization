# temperature_humidity – air temperature and specific humidity on the momentum mesh

Passive-scalar case that sits next to `caseFile/momentum` and `caseFile/surface_temperature`:

```
caseFile/
  momentum/             RANS (simpleFoam, kOmegaSST) - already there, run first
  surface_temperature/  solar_thermal_model.py        - already there, imported (not modified)
  temperature_humidity/ <this case>
epw/                    the weather file (IND_Hyderabad.431280_ISHRAE.epw)
```

What it does, in the order `./Allrun` runs it

| # | step | tool |
|---|------|------|
| 1 | EPW + target hour → `weatherData/MMDD/{Tambient,windSpeed,IDN,Idif,sunPosVector,…}` (spin-up window + target hour) and `weatherData/target.json` | `tools/epw_to_weather.py` |
| 2 | **two stages**: `--stage raw` (needs no mesh; the top-level `../Allrun` starts it as soon as `gen_ground_obj.py` of the momentum script has finished, alongside meshing and simpleFoam) and `--stage bind` (after the mesh exists, seconds). Geometry = the momentum case's `constant/triSurface/buildings_refined.obj` + `trees.obj` (exactly what snappyHexMesh used, **mesh frame, no shift**; only if they are missing the layout json is converted with `json_to_obj.py` and shifted); ground = the momentum surfaces `ground.obj` (bare ground), `lawns.obj`, `ponds.obj`, `pathways.obj` (one material class per file, mesh frame) in the raw stage, or the mesh's `ground`, `lawn`, `pond`, `pathway` patches in the single-pass `--stage all`; thermal model run; the bind stage hands the ground results to the mesh faces (4 samples per triangle, nearest raw triangle of the same class) | `tools/run_thermal.py` |
| 3 | surface temperatures → one value per face of the `buildings` and `ground` patches; writes `0/T`, `0/q`, `0/qFlux`, `constant/transportProperties` | `tools/map_surface_temperature.py` |
| 4 | momentum mesh (`constant/polyMesh`) and the latest-time `U`, `phi`, `nut` are copied in | `Allrun` |
| 5 | `airTqFoam` is compiled (once) and run – serial by default, `--parallel N` available | `solver/airTqFoam` |
| 6 | site-averaged T, q, RH, \|U\| at 2 m → `postProcessing/site_average.json`; slice images of \|U\|, T and RH at 2 m → `images/`; `foamToVTK` | `tools/site_average.py`, `tools/plot_slice.py` |

```
./Allrun --epw ../../epw/1cdb6f01-IND_Hyderabad.431280_ISHRAE.epw          # date_and_time.xyz + the json are found automatically
./Allrun --cases ../../jsons/IND_Hyderabad..._cases.json --season heat      # hour from select_epw_cases.py instead
./Allrun --help
```
Defaults: momentum case `../momentum`; `date_and_time.xyz` in `./`, the momentum case or `..` (its hour is the hour *start*, the EPW row is `hour_end = HH+1`, as in `run_candidates.py`); the only `*.json` in `./` or in the momentum case; EPW `weather.epw`, `../epw/*.epw`, `../../epw/*.epw`.
Python needs numpy and scipy only (the thermal model needs pandas as before).

## Set-up on the server

1. **Software.** OpenFOAM-8 (`source .../OpenFOAM-8/etc/bashrc`), and python3 with numpy, scipy, pandas and shapely (shapely is needed by the momentum case's `json_to_obj.py`). No other libraries.
2. **Folder.** Copy this folder next to the momentum case, so that the tree is `caseFile/{momentum, surface_temperature, temperature_humidity}` and the weather file is in `epw/` (`../epw` or `../../epw` is found automatically; otherwise pass `--epw`). `surface_temperature/solar_thermal_model.py` is imported from there, not modified.
3. **Momentum run.** It must be finished (`simpleFoam` converged, `U`, `phi`, `nut` written in the latest time) with `writeFormat ascii`, and its `constant/polyMesh` plus `constant/triSurface/{buildings.obj,buildingsOrig.obj}` and the layout `*.json` and `date_and_time.xyz` still in the momentum case. Nothing else is rerun – the mesh and the velocity field are copied.
4. **Run** from inside `temperature_humidity`:

       cd caseFile/temperature_humidity
       ./Allrun --foam-bashrc /path/to/OpenFOAM-8/etc/bashrc --epw ../../epw/<file>.epw

   The target hour comes from `date_and_time.xyz` (or `--cases <epw>_cases.json --season heat|cold`). The first run compiles `airTqFoam` into `$FOAM_USER_APPBIN` (needs `wmake`).
5. **Check** `surfaces/thermal_report.json` (the `geometry_check` must say that the regenerated geometry coincides with the momentum `buildings.obj`), `surfaces/mapping_report.json` (no warning about far samples, plus pond / lawn / tree temperatures and fluxes) and the solver log. Results: `postProcessing/site_average.json` (T, q, RH, |U| at 2 m over the site) and `images/{U,T,RH}_z2m.png` (horizontal slice at 2 m above the ground, whole domain, jet colour map, white where there are no cells) and the latest-time `T`, `q` (also as VTK).
6. **Re-run only the mapping / solver** after changing `config/surface_types.json` or `--latent-scale`: `./Allrun --skip-thermal ...` (the thermal step is skipped only if you did not change the config's thermal blocks – a change of ground / wall materials needs the full run).
7. **Many layouts.** Run `./Allrun --momentum <that layout's momentum case> ...` from a copy of this folder per case; `Allclean` resets one.

## Patch names of the momentum mesh

The role of every wall patch is read from its name (case-insensitive, `tools/foam_io.py: classify_patches`):

| name contains | role | temperature | vapour flux `qFlux` |
|---|---|---|---|
| `build` (e.g. `obj_buildings_refined`, `buildings`) | buildings | mapped from the thermal model's building cells | 0 |
| `tree` (e.g. `obj_trees`) | tree crowns | ambient | Penman–Monteith transpiration (daytime) |
| `lawn` / `grass` (e.g. `obj_lawns`) | wet lawn | surface energy balance (every face is lawn) | evaporation |
| `pond` / `water` (e.g. `obj_ponds`) | pond | surface energy balance (every face is water) | evaporation |
| `pathway` / `road` / `asphalt` (e.g. `obj_pathways`) | asphalt pathway | pathway material (every face is pathway) | – |
| `ground` | rest of the ground | bare soil / asphalt / lawn / pond fractions of each face from the json polygons | evaporation of the lawn / pond fraction |
| `inlet`, `outlet`, `top`, symmetry / empty types | flow boundaries | as below | – |

* A legacy single `buildings` patch that also contains the tree crowns still works: crowns are recognised from the geometry and get the tree flux.
* If a patch has another name, list it in `config/surface_types.json`: `"patches": {"buildings": ["myBuildings"], "trees": [...], "lawn": [...], "pond": [...], "pathway": [...], "ground": [...]}`.
* The log of `map_surface_temperature.py` and `surfaces/thermal_report.json` print the roles found.
* Frame shift: with the current momentum pipeline there is none (`--geometry auto`, the default, uses `constant/triSurface/*.obj` and prints a mesh-fit check). Only when those OBJ files are absent: the shift is taken from `buildingsOrig.obj` → `buildings.obj` when both exist (exact), otherwise **fitted to the building / tree patches of the mesh** (accuracy about one cell size, printed in the log). `--geometry json` forces regeneration from the layout json.

## Equations

The flow is frozen: `U`, `phi` and `nut` come from the converged momentum run.

    div(phi, T) - laplacian(alphaEff, T) = 0     alphaEff = nu/Pr + nut/Prt     Pr 0.71, Prt 0.85
    div(phi, q) - laplacian(DEff,     q) = 0     DEff     = nu/Sc + nut/Sct     Sc 0.60, Sct 0.70

`q` is the specific humidity (kg/kg). No buoyancy, no radiation in the air, no heat of evaporation in `T` (evaporative cooling enters only through the lawn / pond surface temperatures, see below). The values are in `constant/transportProperties` (written by the mapping step, so edit `--Pr/--Prt/--Sc/--Sct` of `map_surface_temperature.py`, or the file after it).

## Boundary conditions

| patch | T | q |
|---|---|---|
| inlet | fixedValue = EPW dry-bulb of the target hour | fixedValue = specific humidity from EPW Ta, RH and pressure |
| outlet | inletOutlet (same value) | inletOutlet |
| top | symmetry | symmetry |
| ground | fixedValue, one value per face (mapped surface temperature, blend of bare soil / pathway / lawn / pond; ponds and lawns from their surface energy balance) | `mixed` with zero value fraction + analytic evaporation flux `qFlux` of the pond and lawn part of the face |
| buildings (buildings **and** tree crowns – both are solid in the momentum mesh) | fixedValue, one value per face (building surface temperature; tree crowns = ambient) | `mixed` with zero value fraction + analytic tree-transpiration flux `qFlux` on crown faces (0 on building walls) |

### Humidity sources (analytic, `tools/humidity_sources.py`)

All fluxes are evaluated from the weather of the target hour (Ta, q_a, p, wind, radiation), without feedback from the local air, and are given in kg m⁻² s⁻¹ of vapour entering the air. The convective coefficient is the thermal model's own, h_c = 5.7 + 3.8 u (10 m wind); the Lewis analogy gives the mass-transfer conductance ρ·g_a = h_c / cp.

* **Pond.** A permanently wet open-water surface, treated exactly like the lawn below but with water properties (`ground.pond.thermal` in the config: albedo 0.06, emissivity 0.96, effective conductivity / depth coupling to the deep water). Sunlight absorbed (shadows included) is shared between sensible exchange with the air, long-wave exchange, conduction into the water and evaporation, so the pond is warm in strong sun (about 29 °C against a 22.7 °C wet-bulb in the 40 °C, 1000 W/m² test hour) and falls to the wet-bulb value when there is no sun. Alternatives in the config: `"method": "wet_bulb"` (surface at the EPW thermodynamic wet-bulb temperature, flux `F = h_c (q_sat(T_w) − q_a)/cp`) or `"method": "fixed"` with `temperature_C`.
* **Lawn.** A permanently wet surface (it is assumed to be re-hydrated as fast as it evaporates, so it never dries out). The surface temperature T_s is the root of the surface balance `Rni − h_r (T_s−Ta) − h_g (T_s−T_deep) − h_c (T_s−Ta) − L·F = 0`, `F = max(h_c (q_sat(T_s) − q_a)/cp, 0)` (bisection, no iteration tolerance issue). `Rni` is the isothermal net radiation of every lawn triangle from `solar_thermal_model.py` (albedo, shading by buildings and trees, long-wave exchange all included), `h_r = 4 ε σ Ta³`, `h_g = k/depth`. The report gives the irrigation rate this implies (`irrigation_needed_to_stay_wet_mm_per_h`).
* **Trees.** Big-leaf Penman–Monteith, only while the sun is above the horizon: `LE = (Δ Rn + ρ cp VPD / r_a) / (Δ + γ (1 + r_c/r_a))`, `r_a = ρ cp / h_c`, `r_c = r_s / LAI`, `r_s = r_s,min / (F1 F3 F4)` (Noilhan–Planton radiation, vapour-pressure-deficit and temperature stress), `Rn = (1−albedo) Rg + ε (HIR − σ Ta⁴)`. The flux is per m² of ground under the crown and is divided by 4 to give the flux per m² of crown surface (sphere area ratio), which is what the crown faces of the `buildings` patch receive. At night, or with `enabled: false`, the flux is zero.
* `--latent-scale` multiplies all of the above (0 = no humidity sources).

`qFlux` becomes the `refGradient` of the `mixed` condition inside `airTqFoam`: `refGradient = qFlux / (rho · DEff_wall)`, with the wall value of `nut` of the momentum run included.

## Mapping surface temperature → patch faces

* **buildings patch.** Every mesh face is sampled with points spread over its area (one per ≈ 4·spacing², 5 … 400 per face). Each sample takes the temperature of the nearest source triangle (exact point–triangle distance) whose outward normal points the same way (dot > 0.3; |dot| for crowns). The face value is the mean of its samples. Source triangles are the thermal model's building cells (floors and bottoms are not outdoor cells in that model) and the tree-crown triangles (ambient). The same sampling handles faces much bigger than the source cells and faces much smaller.
* **ground patch.** `ground.obj` is built from the very faces of the mesh patch, so the mapping is 1 : 1; a face value is the area-weighted mean over its triangles. Triangles under buildings are removed by the thermal model; a face with nothing left borrows the nearest valid face.
* **ground classes.** The pond / lawn / pathway polygons of the layout json (`ponds`, `grassAreas`, `roads`+`pathways`; priority pond > pathway > lawn > bare ground) give the fraction of each ground face in each class (9 sample points per face). The thermal model is run once per class that is present, with that class's material, and the face value is the fraction-weighted mean. Pond faces take the pond temperature.
* `surfaces/mapping_report.json` has the distances, the unmapped / far samples and the area per class; `surfaces/thermal_report.json` has the frame shift and a **geometry check** (do the regenerated vertices coincide with the momentum case's `constant/triSurface/buildings.obj`?); `surfaces/mapped_patches.vtk` shows the mapped T on the two patches in ParaView.
* **Frame shift.** The momentum pipeline moves the geometry (`translate.sh`). It is recovered as the constant difference between `constant/triSurface/buildingsOrig.obj` and `buildings.obj` of the momentum case (an exact copy of what was applied). If these two files are missing, the bounding-box rule of `translate.sh` is used and a warning is printed by the geometry check if the result does not match.

## Assumptions to review (`config/surface_types.json`)

All of these are my placeholders – none came from your files except the wall and soil values that are the thermal model's own defaults:

* Pathway = asphalt: k 0.75, rho 2110, cp 920, albedo 0.12, emissivity 0.93 (typical handbook values).  Lawn: k 1.0, rho 1100, cp 1500, albedo 0.25, emissivity 0.95.  Bare ground: the thermal model's soil (k 1.5, rho 1150, cp 650, albedo 0.20).
* Deep-soil temperature "auto" = annual mean of the EPW dry-bulb (26.7 °C for Hyderabad); wall interior 24 °C, h 8 (thermal model default).
* Lawn: always wet (≥ 5 mm hydration), analytic evaporation as above. Pond: water albedo 0.06, ε 0.96, effective conductivity 3 W/mK over 0.5 m to the deep water (placeholders). Trees: LAI 3, r_s,min 150 s/m, albedo 0.2, ε 0.97, stress constants of Noilhan–Planton – generic broadleaf values, replace them for your species (`trees.transpiration` in the config).
* The 10 m wind in h_c overstates transfer at ground level, so the potential evaporation of lawn and pond is on the high side; use `--latent-scale` to rescale.
* Trunk faces (the tree part of the `buildings` patch is the crown solid) receive the crown transpiration flux.
* Thermal model: `surface_temperature/solar_thermal_model.py` is imported unchanged. Two API generations are recognised: the current one (`load_obj_checked`, `build_layer_faces`, `BuildingMaterials`, per-class layers – building materials are read from the layout json) and the older `build_ground_faces_from_obj` one. The log line `thermal model: ... (api: layers|ground_fn)` says which. Inside it every ground class is run with `wetness = 0`: soil and asphalt stay dry, and lawn / pond evaporation is the analytic balance of `map_surface_temperature.py` (it uses the model's isothermal net radiation), so evaporation is not counted twice. If neither API is complete, `run_thermal.py` prints what is missing and uses `tools/bundled/solar_thermal_model.py`; `--bundled-thermal-model` forces that.
* Spin-up: 24 h of EPW weather before the target hour so the 0.23 m slabs forget their initial 24 °C (`--spinup 48` is safer for walls with high mass).
* Wind: the thermal model uses the EPW 10 m wind speed (floored at 0.5 m/s) as you described; the momentum inlet is whatever your pipeline wrote into `0/U`.

## Verification that was possible here (no OpenFOAM in this environment)

* `solver/airTqFoam/airTqFoam.C` was **type-checked** (`g++ -fsyntax-only`) against the OpenFOAM-8 headers (OpenFOAM/OpenFOAM-8 on GitHub). It has **not been linked or run**; `wmake` on your machine is the real test. Dictionaries (`system/*`, `constant/transportProperties`, `0/*`) follow the OpenFOAM-8 syntax of your momentum case but were not read by OpenFOAM.
* The python half was run on (a) the polyMesh of your zip, (b) synthetic polyMeshes made from your `wind_catcher.json` and from `generation_1.json` (buildings, 92 trees, a pond and a lawn) with `tests/test_synthetic_mesh.py`: mapped face temperatures equal the source-triangle temperatures (mean difference ≤ 0.02 K; the few larger ones are faces that overlap another coplanar cell in stacked blocks, a situation a real snappyHexMesh mesh does not have), pond / lawn fractions and borrowed ground faces behave as described. The mesh inside your zip is a leftover that does **not** match its `buildings.obj` (faces are on average 5 m from the surface), so it could only be used to test the readers, not the mapping quality.
* `Allrun` was dry-run with stub OpenFOAM commands to check the file flow; the real run has to be done by you.

## Known limits

* The thermal model's tree handling is occlusion only; `buildings.obj` of the momentum case merges trees and buildings, which is why the thermal input is regenerated from the json.
* Stacked / overhanging blocks: the thermal model treats a downward-facing horizontal face that is not at the building's lowest level as a roof (normal flipped up), so underside faces of raised blocks get a sunlit roof temperature. Surfaces of that kind are rare in the layouts you generate, but a wind catcher with raised blocks has them.
* Ground class fractions are resolved only down to the mesh face size; pond and lawn areas smaller than a ground face are represented by their fraction of that face.
* Uniform inlet profile for T and q, steady state, one hour (no time history of the air).
* UTCI also needs Tmrt; `site_average.json` gives T, q, RH and |U| only.

## Things I noticed in the momentum case (not changed)

* `of_Allrun` says `fix_uinlet.py` is disabled but still runs it; it turns the (cosθ, sinθ) vector into (−sinθ, −cosθ), i.e. the stored `0/U.U.bak` and `0/U` differ by that conversion.
* `rm *.obj` removes `trees.obj` and `buildings_refined.obj` after the merge, so the thermal inputs have to be regenerated from the json (done here).
* The OpenFOAM path `/home/subratmcd/Softwares/OpenFOAM-8` is hard-coded in `of_Allrun` (`job_script.sh` uses `/opt/openfoam8`).

## Used inside the NSGA-II loop (`../Allrun`, `python/nsga_batch.py`)
* `tools/set_inlet_wind.py` writes the EPW 10 m wind of the target hour into `momentum/0/U` (`Uinlet`).
* `tools/compute_utci.py` turns the T / q / U result into the site area-averaged UTCI at 2 m (Tmrt from sun, sky view, shade and the mapped
  surface temperatures; same SolarCal expression as `select_epw_cases.py`) and writes `UTCI_details.txt`. Exit code 3 = airTqFoam diverged
  or wrote nothing (the caller then retries with first-order upwind). See `../../NSGA_BATCH_README.md`.
* `Allrun` builds `airTqFoam` only when a source file is newer than the installed binary, and runs on 4 cores by default (`--parallel 1` = serial).

## Thermal model before the mesh exists (`run_thermal.py --stage raw | bind`)
```
python3 tools/run_thermal.py --stage raw  --case . --momentum ../momentum --json layout.json   # no mesh needed, needs momentum/constant/triSurface/{buildings_refined,trees,ground,lawns,ponds,pathways}.obj
python3 tools/run_thermal.py --stage bind --case . --momentum ../momentum --json layout.json   # needs constant/polyMesh; writes surfaces/thermal_surfaces.npz as before
```
`../Allrun` does this for every case: the momentum script gets a marker after its `gen_ground_obj.py` line, the raw stage starts on a snapshot of the surfaces when the marker appears,
and `./Allrun` of this folder binds `surfaces/thermal_raw.npz` to the mesh (`--no-raw` forces the old single pass; a raw result made for another target hour, or any bind error, also
falls back to it). Compared with the single pass on the same mesh the ground temperatures differ by 0.4 K on average (95 % of the triangles within 2.4 K, largest values at shadow edges),
because the values are taken from the 3 m ground triangles of `ground.obj` instead of being computed on the mesh faces.

