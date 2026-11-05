The OBJ pipeline calculates shadows and exterior surface temperatures independently for each input triangle. Coplanar triangles retain a common `parent_face_id` for grouping, but their results are never copied from a whole-wall centroid.

Run your existing command from this directory:

```bash
python3 solar_thermal_model.py \
  --buildings_obj ../../uhi_API/runs/subrat_c39f3effe7a5/constant/triSurface/buildings_refined.obj \
  --trees_obj ../../uhi_API/runs/subrat_c39f3effe7a5/constant/triSurface/trees.obj \
  --ground_obj ../../uhi_API/runs/subrat_c39f3effe7a5/constant/triSurface/ground.obj \
  --weather_dir ../../uhi_API/runs/subrat_c39f3effe7a5/weatherData/0320 \
  --vtk_out vtk_output/
```

Open `vtk_output/scene.pvd` in ParaView and select **Cell Data** for coloring:

- `T_surface_C` (also `T_C`): transient exterior surface temperature in Celsius.
- `T_inner_C` / `T_wall_mean_C`: inner-face and capacity-weighted wall temperatures.
- `T_back_surface_C` / `T_slab_mean_C`: corresponding generic fields for
  either a building slab or a ground slab.
- `q_conduction_in_Wm2`: conduction from the outside node into the wall.
- `q_indoor_out_Wm2`: heat delivered from the wall to the 24°C indoor air.
- `q_back_out_Wm2`: heat leaving either slab through its back boundary. For
  ground cells this is also `q_deep_ground_out_Wm2`.
- `q_storage_Wm2`: instantaneous rate of change of heat stored through the slab.
- `stored_heat_Jm2` / `stored_heat_J`: stored energy per area and for the
  complete triangle, relative to its initial uniform wall state.
- `sunlit_fraction`: direct-beam visible fraction, from 0 (unlit) to 1 (fully sunlit).
- `shadow_fraction`: blocked fraction on a surface facing the sun; back-facing cells have zero here and zero `sunlit_fraction`.
- `I_direct_Wm2` / `I_total_Wm2`: incident direct / total solar radiation.
- `q_solar_absorbed_Wm2`, `q_conv_out_Wm2`, and
  `q_longwave_out_Wm2`: exterior boundary heat fluxes. At every output,
  absorbed solar minus exterior convection and longwave loss equals heat
  delivered indoors plus the rate stored in the slab.
- `building_index`, `parent_face_index`: identifiers for selecting a building or wall.
- `is_building` / `is_ground`: select building or ground thermal cells.
  Trees have zero for both and are displayed at ambient temperature.

The default `--shadow_samples 3` traces three equal-weight interior samples per triangle toward the sun, using all building and tree triangles as opaque blockers. This includes occlusion by other parts of the same building. `--shadow_samples 1` uses only the centroid. Fractional coverage is a sampling estimate; sharper boundaries require finer input triangles. `--no_shadows` disables obstruction tests but retains surface orientation and the horizon test. A partially shaded triangle can have both `shadowed` and `sunlit` set to 1; use the fraction fields to inspect coverage.

Each exposed triangle is treated as an independent one-dimensional, 0.23 m
thick concrete slab. Five finite-volume temperature nodes span the thickness,
including the exterior and interior faces. Adjacent nodes exchange heat using
`k / dx`; their heat capacities sum exactly to `density * specific_heat *
0.23` per square metre. The outer boundary receives absorbed solar radiation
and exchanges heat by wind-dependent convection and longwave radiation:

```text
C * dT_outer/dt = (1 - albedo) * I_total
  - h_out * (T_outer - T_air) - q_longwave
  - (k / dx) * (T_outer - T_node_2)
h_out = 5.7 + 3.8 * max(wind_speed, 0)

C * dT_inner/dt = (k / dx) * (T_previous - T_inner)
  - 8 * (T_inner - 24)
```

With `--ground_obj`, every exposed ground triangle is solved in the same way.
A vertical building-intersection test first removes ground cells beneath
building roofs. The remaining cells receive direct, diffuse, and longwave
radiation, outdoor convection, and the same three-ray partial-shadow test
against buildings and trees. The active ground layer defaults to 0.23 m and
five nodes. Its project soil defaults are conductivity 1.5 W/m/K, density
1150 kg/m³, specific heat 650 J/kg/K, albedo 0.20, and emissivity 0.95. The
deep-soil boundary defaults to 24°C with conductance `k/depth` = 6.522 W/m²/K:

```text
C * dT_ground_bottom/dt = (k / dx) * (T_previous - T_ground_bottom)
  - h_deep * (T_ground_bottom - T_deep)
```

Override these assumptions with `--ground_depth_m`, `--ground_nodes`,
`--deep_ground_temp_c`, `--ground_deep_h`, `--ground_initial_temp_c`,
`--ground_conductivity`, `--ground_density`, `--ground_specific_heat`,
`--ground_albedo`, and `--ground_emissivity`.

Within each weather interval, solar input, air temperatures, convection
coefficients, and the ambient-linearized longwave coefficient are constant.
The coupled nodal temperatures are advanced with a matrix exponential, so the
time update is analytical for this spatially discretized linear system. This
state-space treatment is the same general analytical approach described for
wall conduction in the [EnergyPlus Engineering Reference](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.2.0/EngineeringReference.pdf).
The full fourth-power longwave term is nonlinear, so it is tangent-linearized
at the current ambient temperature for each interval.

The first output timestamp is the initial condition. By default, all wall
nodes start at the indoor temperature of 24°C. Use `--initial_temp_c` for a
measured initial wall temperature. Every later weather record supplies the
constant boundary condition over the elapsed interval ending at that timestamp.
This makes each output's shadow, irradiance, and temperature fields correspond
to the same weather record.

The original `T_solair_C` and lumped `T_building_C` remain separate
diagnostic estimates; neither is the exported surface temperature.

OBJ files do not supply wall layers or thermal materials. Current defaults use
thermal conductivity 2.3 W/m/K, density 2400 kg/m³, specific heat 880 J/kg/K,
wall/roof albedos 0.3/0.4, and emissivity 0.9. The inside boundary uses fixed
24°C air and an 8 W/m²/K film coefficient. Override these thermal-boundary
defaults with `--wall_thickness_m`, `--wall_nodes`, `--indoor_temp_c`,
`--indoor_h`, and `--initial_temp_c`.

The model uses unobstructed isotropic diffuse-sky and ground-reflection
factors, a clear-sky effective longwave temperature, and ambient-temperature
surroundings. It does not model lateral conduction between mesh triangles,
diffuse occlusion, foliage transmission, multiple solar reflections, moisture,
windows, HVAC response, or a dynamically calculated indoor-air temperature.
Wall mesh winding must point outward; coordinates and weather vectors must
share axes and use metres. The JSON input path still uses one shadow ray per
polygon.

`results_faces_hourly.csv` contains one row per exposed building or ground
triangle and weather time. Select `surface_group == "building"` or
`surface_group == "ground"`; group by `parent_face_id` with area weights for
wall averages. CSV and VTK output are larger than before. The simulation is
reused for CSV and VTK rather than run twice. `--out_prefix` and `--vtk_out`
can point to separate locations for comparison runs.

Regression checks:

```bash
python3 -m unittest -v test_solar_thermal_model.py
```

## Detailed calculation

The complete step-by-step derivation, numerical one-hour worked example,
equation-to-code map, assumptions, and exact bibliography are in
[`solar_thermal_calculation.tex`](solar_thermal_calculation.tex).
