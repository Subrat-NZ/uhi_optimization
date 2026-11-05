# NSGA-II for the UHI layout optimisation

Unzip over `/mnt/nashzero_storage/uhi_NSGA_II/` (it replaces only the files below; the rest of `momentum/`, and `surface_temperature/`, `input/`, `epw/`, `jsons/` are yours).
`python/nsga_batch.py` (the earlier queue version) is **gone**: `python/nsga2.py` is the driver.

| file | what |
|---|---|
| `python/optimize.py` | one command: pre-flight, peak-heat / peak-cold hour (if missing), NSGA-II, Pareto plot, top 5 |
| `python/nsga2.py` | the NSGA-II loop (ranking, crowding, tournament, crossover, mutation, survivor selection), top 5, plots |
| `python/layout_cross.py` | crossover and mutation of whole layouts (every child satisfies the Stage 1 rules) |
| `python/generate_candidates.py`, `plot_candidates.py`, `check_invariants.py` | your files with the new names: `individual_K` (initial individuals, `individual_0` = base), `offspring_K` (children); commands `individual` / `offspring` (the old `generation` / `mutation` still work and write the new names) |
| `python/run_candidates.py` | your file + skips old results when `caseFile` is copied into `run/NAME/season` (+ symlink-free `epw`, failure reason in the console) |
| `caseFile/momentum/Allrun`, `prepare_optional_regions.py`, `mesh_gate.py`, `Allclean`, `system/snappyHexMeshDict` | **robust momentum meshing** (see the section below) |
| `caseFile/Allrun`, `caseFile/temperature_humidity/` | one evaluation: wind -> momentum -> T, q -> UTCI (4 cores per simulation) |

## Run
```bash
cd /mnt/nashzero_storage/uhi_NSGA_II
source /home/subratmcd/Softwares/OpenFOAM-8/etc/bashrc            # and your python venv
nohup python3 python/optimize.py --retry-failed > output/optimize.log 2>&1 &     # first time after the failed runs; later: without --retry-failed
python3 python/nsga2.py status                                      # progress at any time
```
Same command again = resume. Simulations that finished are never repeated. Options: `--pop 10 --offspring 4 --gens 5 --batch 4 --pc 0.9 --pm 0.25 --comfort 9 26 --seed 1`.

**Parallel generation (`--workers N`).** Making layouts (the two wind seeds, the random individuals, the children) is single-core Python and is the slow part before the first simulation starts.
`--workers 4` makes them in 4 processes at the same time; `--workers 0` uses one process per CPU core; the default `1` is the old single-process behaviour. Do not ask for more workers than free cores.
Layouts of a `--workers N` run differ from a `--workers 1` run (other random streams), but they obey the same rules, pass `check_invariants.py`, and the same N with the same `--seed` gives the same layouts.
The cool-wind-catcher seed is one task (~30 s on a slow core) and bounds the time of the initial population. Example: `python3 python/optimize.py --pop 7 --gens 8 --offspring 4 --workers 4`.
It only speeds up generation; simulations still run `--batch` at a time.

**Names:** initial individuals `individual_K.json`, base `individual_0`, children `offspring_K.json`. Files and finished runs called `generation_K` / `mutation_K` by
earlier versions are renamed automatically (input/ and run/) the first time `nsga2.py run` starts, so nothing that already ran is simulated again.
(`constraints.json` keeps its section name `"generation"`: it is the recipe of the individuals; renaming it would break your file.)

**The base layout.** `individual_0.json` is a byte-for-byte copy of `input/base.json`. It is simulated like every other individual, takes part in the ranking and can be selected as a parent;
it appears as the gold star in `nsga_report.png`, and `top5` prints its UTCI values and how many of the five picks beat it. (An earlier version simulated a modified base, with the lawns and ponds the rules
require added to a base that has none; that is why the "base" looked as if it had not run.) If `individual_0.json` is not identical to `base.json`, `nsga2.py run` sets the old file aside as
`input/individual_0_rules_enforced.json.old` and its simulation as `run/individual_0_rules_enforced`, writes the real base and simulates it (2 simulations); in a run that is already under way the base is simulated
first and then competes for a place in the population.

## The algorithm
**Objectives** (both minimised) - distance of the site-averaged UTCI from the comfortable band 9 ... 26 C:
`f1 = max(0, UTCI_heat - 26)` for the peak-heat hour, `f2 = max(0, 9 - UTCI_cold)` for the peak-cold hour (0 = inside the band).

1. **Initial population** = the real base layout (`individual_0` = `base.json` copied exactly; nothing added, nothing moved) + `--pop` (10) individuals made by the generator (`individual_1...`; existing files are reused).
   Each is simulated for the heat and the cold hour; `--batch 4` layouts (= 8 simulations = 32 cores) run at the same time.
2. **Rank**: fast non-dominated sorting (rank 1 = Pareto front), then crowding distance inside each rank.
3. **Offspring** (`--offspring`, default 4 = one batch), for each: binary tournament picks two parents (lower rank wins, then larger crowding distance);
   layout crossover with probability `--pc` (0.9); mutation (1-3 small operations) with probability `--pm` (0.25; a child that was not crossed is always mutated).
   Children are written to `input/offspring_K.json`; every one satisfies all Stage 1 rules.
4. **Simulate** the offspring (heat + cold).
5. **Survivors**: parents + offspring are ranked together and the best N (11) survive (elitist).
6. Repeat 2-5 for `--gens` (5) generations. `output/nsga_history.csv` gives front size and hypervolume per generation (stop when the hypervolume no longer grows).
7. **Result**: Pareto front of the final population, `output/nsga_report.png`; top 5 = best heat, best cold, knee, two spread along the front
   (`output/top5.json`, `input/top5_1..5.json`, `output/overview.png`). **The list is not a quality ranking**: every layout on the front is a different
   heat / cold trade-off. The order is always: #1 best heat (lowest f1), #2 best cold (lowest f2), #3 knee, then spread; each entry has a `role` in `top5.json`.
   A front with fewer than 5 members is listed completely in that order and filled with the best layouts of the next front (role "not on the front").

Cost with the defaults: 11 + 5 x 4 = 31 layouts = 62 simulations; about 3 batches for the initial population + 1 per generation, roughly 2.5-3 h at 15-20 min per batch.
A layout whose simulation failed gets 1e6 and dies out; if no initial layout completes, the run stops with the reasons (fix, then `--retry-failed`).

## Seed individuals: cool wind catcher on every building + wind blocker belt (initial population)

Besides the base layout and the random individuals, the initial population contains two **designed** individuals (code: `python/layout_seeds.py`):

| individual | template | wind used | what it is |
|---|---|---|---|
| cool wind catcher (every building) | `jsons/cool_wind_catcher.json` | **heat** case (`heat_case` in `jsons/<epw>_cases.json`) | a pond and curved tree rows on **every** building, aligned to the wind: pond and trees stand at the building's downwind face, on the side of the wind that has room |
| wind blocker (windward edge belt) | `jsons/tree_wind_breaker.json` | **cold** case (`cold_case`) | staggered tree rows (the rows of the template, repeated along the edge) along the **complete windward edge or edges** of the site |

**Both** individuals also get **dense trees on both sides of every pathway / road** (crown radius 2.5 m, centre-to-centre at least 1.1 x the crown
diameters = 5.6 m, as close as the tree-count limit allows). Trees never stand on a pathway (crown edge >= 0.5 m from the road edge).

* The templates are drawn for wind **from the west** (air flowing towards +x); the pattern is rotated to
  `boundary.wind_flow_unit_vector_xy_model` of the season (model axes). The catcher never uses the cold wind and the blocker never the heat wind.
* **Cool wind catcher per building.** Buildings are processed largest first. Each gets the template pond (scaled about the building corner) and its tree
  rows, on the side of the wind (template as drawn or mirrored) and with the small rotation (+-20 deg) / shift (+-8 m) that fits best; a pond that
  is cut off by buildings, roads or the site edge or gets a notch scores worse (up to 20 %, then 45 % of the template pond may be cut off).
  **The pond shape adapts to the room**: if the template outline does not fit, the free part of the template pond next to the building is grown (or
  shrunk) to the area it needs, filling the available space (`pond_shape: adapted` in the log; `template` where the outline fitted). The total pond area of the layout (1.0-1.1 x A0, aimed at the middle of the band) is shared between the buildings in proportion
  to their footprints; a building that has no room for its share gets a smaller pond and the others take the rest (up to 4 passes). A building with no
  room at all gets none (the log names it). Ponds replace the base ponds; lawns give way and are re-fitted to their band. Number of ponds <= the
  number of buildings (**`pond.max_patches` is now `max(4, number of buildings)`**, see CHANGES_seeds_v3.md).
* **Blocker belt.** The windward edges of the site are the boundary stretches whose outward normal faces the wind (more than ~22 deg off side-on:
  `-(normal . flow) >= 0.38`); one wind direction gives one or two stretches (e.g. north + east for a north-east wind). Rows of trees run along them at
  `crown radius + 0.5 m`, then one row pitch further in (template: 2.33 x R); at every step the largest crown (6 m down to 1 m) that stands there
  without breaking a rule is planted, so a narrow strip between the site edge and a building gets small trees instead of a hole. Roads crossing
  the edge and buildings that stand close to it leave gaps (the log gives the **share of the edge shaded** by the crowns, seen along the wind).
  Rows (1-3) and crown size are chosen for the most complete edge within the tree budget; the belt is as tall as the height band allows.
* **Tree-count limit.** The limit `floor(max_count_factor x n0) + max_new_trees` is shared out (constraints.json `seeds`): pathway trees `path_share`
  (45 %), catchers / belt `struct_share` (45 %), base trees the rest; the belt may take budget from the pathways (down to `path_share_min` 30 %) until
  it shades `belt_cover_target` (80 %) of the edge. Where a pattern does not fit, trees are thinned (same shape), never allowed to break a rule. **The
  limit is what makes the trees less dense than the geometry could be** - on a base with 86 trees (limit 189) the pathways get 85 trees on 937 m of
  road edge (one per 11 m); raise `tree.max_count_factor` / `tree.pathway_planting.max_new_trees` for denser seeds (that changes the rule for every
  individual, so it is your call).
* Heights obey the height band of the location (8-10 m near buildings, 8-12 m beside pathways, 12-15 m in the open); the catcher keeps the template's
  tall / short order, belt trees are as tall as allowed, pathway trees mid-band.
* The rest of the layout is the **base with the rules enforced** (lawn / pond totals fitted around the reserved crowns); base trees that touch the new
  trees or exceed the limit are replaced; buildings do not move. All Stage 1 rules hold (`validate` and `check_invariants.py` report 0 new violations).
* A season that does not exist in the climate (`heat_case` / `cold_case` = null) gives no individual of that kind; a missing template file or a pattern
  that cannot be placed only skips that individual (message in the log). The individuals count **within `--pop`**.
* Origin: `template:cool_wind_catcher_all` / `template:wind_blocker_edge` (`output/nsga_state.json`, `nsga_history`, top5 files; `+` and `x` markers in
  `nsga_report.png`). They breed like every other individual.
* Seeding takes about 40-60 s (catcher, 12 buildings) and 5 s (blocker).

Options: `--no-templates`, `--template-kinds default|all|<comma list>` (default: `cool_wind_catcher_all,wind_blocker_edge`; the earlier variants
`wind_catcher`, `wind_blocker`, `cool_wind_catcher` (one reference building) are still there), `--cool-catcher-template FILE`, `--blocker-template FILE`,
`--jsons-dir DIR`. By hand: `python python/generate_candidates.py template` (both seeds) or `--kind all` / `--kind wind_blocker_edge`
(writes `individual_K.json`, K = next free number).

### Earlier variants (only with `--template-kinds all` / `--kind all`)
* `wind_catcher` (`tree_wind_catcher.json`): two converging double rows slid to the upwind edge.
* `wind_blocker` (`tree_wind_breaker.json`): the 3-row wall slid to the upwind edge.
* `cool_wind_catcher`: one catcher on one **reference building** (`--cool-reference-building ID` / `--ref-building`), see CHANGES for details.

## Road shade: trees along the roads in EVERY individual
Not only the two seeds: every random individual, every crossover child and every mutated child has to pass the **road-shade rule** (`pathway_shade`, a hard rule - it is never "inherited" from the base,
a layout that fails it is rejected). Only `individual_0` (the untouched base copy) is exempt.

* **Rule.** Samples are taken every 2 m along both edges of every pathway / road. A sample counts as *shaded* when a crown edge is within `reach_m` (3 m) of it. Samples where not even a
  crown of 1.5 m radius fits (next to buildings, ponds, the site edge) are not counted. At least `min_coverage` (80 %) of the counted samples must be shaded.
* **Repair** (`layout_reflow.ensure_pathway_shade`, called by the initial-individual recipe, both seeds, crossover and mutation): when the coverage is below 80 % it is raised to
  `target_coverage` (90 %) by (1) planting trees beside the road - **the crown radius is chosen to suit the room: 2.5 - 5 m where there is space, shrinking in 0.5 m steps down to 1.5 m between buildings**,
  and the centre is slid forward along the road so one crown shades as much road as possible; (2) enlarging crowns of nearby trees (up to 6 m) where no new tree fits; (3) when the tree-count limit is
  reached, moving trees that stand far from every road (never the seed's catcher / belt trees). New street trees get the upper half of the 8 - 12 m band (longer shadows).
* **After mutation / crossover** the repair runs again, so a tree move, a building shift or a tree split line cannot leave a road bare; the operation is rejected only if the repair cannot restore the shade.
* Settings: `tree.pathway_planting` in `constraints.json` (`shade_required`, `min_coverage`, `target_coverage`, `reach_m`, `sample_step_m`, `min_radius_m`, `shade_radius_range_m`, `grow_max_radius_m`,
  `max_relocated_frac`). `shade_required: false` switches the rule off. The tree-count limit still applies; on a site with a lot of road per tree the repair uses larger crowns instead of more trees.
* On the test site (937 m of road edge, limit 189 trees) all individuals / offspring of a mock run had 82 - 96 % shaded road edge with crown radii 1.5 - 5 m; the audit (`check_invariants.py`) re-derives the rule independently.
* Seeds: the cool-wind-catcher / blocker seeds now try up to 5 internal random seeds if one fails (the pond / lawn fitting failed for 2 of 7 seeds before).

## Novelty rule: no simulation of a copy

A child is not written (and not simulated) if it is practically identical to ANY existing individual / offspring: no building moved
by 1 m or more, fewer than 3 trees differ (position or size) AND each of the lawn / pond totals changed by less than 2 %.
Reason: a child made of a 5 cm lawn change (offspring_1 of the first run) costs a full CFD run, and the run-to-run noise of the
simulation (a few tenths of a kelvin) is larger than such a change, so noise, not merit, decides whether it is selected.
Rejected children are counted in the generation line ("N child(ren) rejected as practically identical ...") and replaced by new ones.
Change it in input/constraints.json:  "novelty": {"enabled": true, "min_building_shift_m": 1.0, "min_tree_changes": 3, "min_patch_change_frac": 0.02}
(set "enabled": false to switch it off; raise the numbers to demand bigger differences).

## Momentum meshing (why offspring_20 diverged, and what changed)
**Cause.** The background mesh is 15 x 15 x ~1.8 m cells; every coarse/fine refinement interface therefore has 8 000 - 10 000 faces at 70 - 83 deg non-orthogonality.
With the stock quality limit (`maxNonOrtho 65`, `minTetQuality` off) snappyHexMesh starts with ~10 000 "error faces" it cannot tell from the faces it damages while snapping,
and the final mesh can contain faces above 90 deg and inverted face pyramids (your offspring_20 mesh: 95.6 deg, 6 pyramids, 10 630 illegal faces). simpleFoam then
diverges slowly (k and omega reach 1e16) and nothing in the solver output says so. The ~10 000 illegal faces are in every layout, not only in offspring_20; the other runs converged by luck.

**What changed** (`prepare_optional_regions.py --profile 1|2|3`, `mesh_gate.py`, `Allrun`):
* `system/meshQualityDict` is written by the script (self-contained): `maxNonOrtho 85` (above what the interfaces produce, so only real defects are rolled back),
  `minTetQuality 1e-15`, `minArea 1e-12`, boundary skewness 12, twist, concavity, determinant, face weight and volume ratio checks.
* Explicit feature edges only for the buildings (the ground, lawns, ponds and trees are flat or thin surfaces; snapping their open edges produced the twisted / concave faces at z = 0.1 - 0.2 m).
  `system/snappyHexMeshDict` takes `snapControls` from the generated `system/optionalSnapControls`.
* After meshing, `mesh_gate.py` reads `log.checkMesh` / `log.snappyHexMesh` and rejects the mesh BEFORE simpleFoam starts when: a face is above 87 deg or there are non-orthogonality errors,
  a face pyramid is inverted, skewness is above 25, snappy left more than 0.05 % illegal faces, or the mesh is nearly empty. 70 - 85 deg faces at refinement interfaces are accepted.
* A rejected mesh is rebuilt with the next profile (2 = no feature snapping, finer snap iterations; 3 = additionally no extra refinement of the ground / lawn / pond / tree surfaces;
  about 35 % fewer cells, so coarser near the ground). If all three are rejected the layout fails in about 3 min instead of after a 10 min diverged run (`momentum/log.meshGate`, `log.checkMesh.profile*`).
  `MESH_GATE_ARGS="--max-nonortho 86"` (environment variable) changes the gate limits.
* `caseFile/Allrun` prints the accepted profile and the mesh figures in `log.Allrun`.

**What was tested** (OpenFOAM v1912 stand-in). All 31 layouts of your input folder meshed with the new settings: every one passed the gate (max non-orthogonality 83.0 - 85.0 deg, no inverted pyramids, at most 1 illegal face, skewness 7 - 12). Six layouts meshed with the old settings: all six had 6 000 - 10 400 illegal faces, one a face at 91 deg with an inverted pyramid. Serial and 4-way parallel + reconstructParMesh give the same figures;
the Allrun meshing loop end to end with a forced rejection of profiles 1 and 2; the gate against your own `log.checkMesh` of offspring_20 (rejected).
**What was not tested:** Foundation OpenFOAM-8 itself. Your v8 defects (95.6 deg, 6 pyramids) did not appear in v1912 for offspring_20, although a hard defect (91.3 deg, 1 inverted pyramid) did appear in v1912 with the old settings for offspring_12.
So "no layout can fail" is not guaranteed; what is guaranteed is that a bad mesh is detected and replaced, or the layout fails early with a reason.
**Comparability:** the mesh differs from the old one (ground cells, feature snapping); in a 400-iteration test the mean wind at 1 - 2.5 m differed by about 5 - 8 % between old and new mesh for offspring_20.
Layouts already simulated on the old mesh should be re-simulated before they are compared with new ones: delete (or move away) their `run/NAME` folders, `python3 python/nsga2.py exclude NAME ...`, then run `optimize.py --retry-failed`.

## Things you should know
* **Momentum divergence:** `compute_utci.py` also checks the wind. If |U| at 2 m is above 4 x the inlet speed at more than 5 % of the sample points, simpleFoam blew up: the layout fails
  (exit code 5, `log.utci`) and gets the penalty. T and q cannot show this, they stay inside their wall / inlet bounds in any flow.
  A layout you do not trust can be removed by hand: `python python/nsga2.py exclude offspring_20`, then `top5` and `plot`.
* **Not run here:** Foundation OpenFOAM-8 is not available to me. The NSGA-II loop, run_candidates and the geometry / thermal / UTCI chain were run with stand-ins for the solvers; the meshing was run with OpenFOAM v1912 (ESI), see the next section for what that does and does not prove.
* **Ties at 0:** once both UTCI values are inside 9 ... 26 the layout has f1 = f2 = 0 and cannot be ranked further; if that happens, tighten the band (`--comfort 12 24`).
* **q:** the vapour of lawn / pond / trees is an imposed flux; a few cells next to those surfaces can leave 0 ... saturation. q is judged and clipped at the 2 m sample points (`log.utci`).
* **Wind:** uniform inlet at the EPW 10 m wind; the CFD wind at 2 m is used as it is.
* 10 individuals and 5 generations are a screening, not a converged NSGA-II: `--gens 10` resumes and continues from where it stopped.


## Clearance rule: lawns and ponds keep their distance from buildings

Three hard rules for every layout (never inherited from the base, also for `individual_0`):

1. lawns, ponds and buildings (generated footprints and OSM buildings) never overlap each other, and ponds and lawns never overlap each other;
2. every pond is at least **2 m** from every building (`constraints.json`: `pond.building_clearance_m`);
3. every lawn is at least **1 m** from every building (`lawn.building_clearance_m`).

The distance is measured between outlines; `patch.clearance_tol_m` (0.05 m) is the measuring tolerance. How it is kept: lawn / pond outlines are cut back around the
buildings grown by the clearance (`layout_reflow.static_blockers(scene, cat)`), after a building move, in crossover, in the template / seed layouts and when the base
patches are fitted (`fit_patch_area` now also rebuilds a category when a patch is too close to a building). Mutations of a lawn / pond that end too close to a building are rejected like
any other rule violation. `check_invariants.py` re-checks overlaps and clearances independently (`OVERLAP`, `CLEARANCE`). Set the two values to 0 to go back to "no overlap" only.


## Stacked buildings are ONE building; what stands on a building stays glued to it

**Stacks.** A building on top of a building is one building for the optimiser. Parts belong to one stack when
1. they have the same footprint (floors), or
2. they belong to the same family - ids `Building 3`, `building_3_mass_1`, `building_3_internal_floor_2`, `building_3_mass_1_internal_floor_5` ... all
   are family `building 3` - and their footprints overlap, or
3. their footprints overlap by at least half of the smaller one and their heights touch (gap <= 0.3 m): a mass sitting on a slab, whatever it is called.

(your pasted scene: 24 parts -> 2 buildings, "Building 3" with its 11 floors / masses and "Building 7" with its 11). A stack moves as a whole (same translation for
every part, `footprints[]` included), the shift ring uses the longest span of the union footprint, and for every overlap / clearance rule
(building-building, lawn / pond clearance 1 m / 2 m, tree distance, pathways, site edge, wind seeds, crossover) the building is the **union of its parts' footprints**
(`layout_ops.building_groups`, `group_polygon`). `check_invariants.py` re-derives the stacks independently and reports `stacked parts moved differently` /
`stack structure changed`.

**Items on a building.** Trees, lawns and ponds that stand on a building of the BASE layout (tree centre inside the stack's footprint; lawn / pond with more than half
of its outline inside) are part of that building:
* the optimiser never edits, removes, resizes or re-places them, and they are not counted in the tree / lawn / pond budgets (tree cap, patch limits, lawn water budget,
  area bands A0 .. 1.10 A0 refer to the free items on the ground);
* when a layout is written (`generate_candidates.write_candidate` -> `layout_ops.attach_hosted`) they are put back, moved by exactly the translation of their building
  (x, y of every point; z / elevation, size and everything else unchanged). So in every individual, offspring, seed and mutation they are on the same spot of the
  same roof as in the base. Crossover does not mix them: they follow the building. Ground trees whose id collides with a roof tree are renumbered;
* `check_invariants.py` checks it: `GLUE tree Tree 6 is not rigidly on Building 3` / `... is missing`.

Detection uses only the base layout, so a free tree can never turn into a roof tree by accident. Roof items that should NOT be glued: move them off the building in
`base.json`. Rooftop lawns / ponds are written to the files exactly as they are in the base, i.e. the case scripts treat them as they do for the base.


## Mesh gate and hairline slits (offspring_17 "Ran out of precision")

* A pond / lawn outline that wraps round an obstacle through a channel under 0.2 m wide, or touches itself in a point, can make the constrained triangulation of
  `json_to_obj.py` fail (`Triangulation failed ... Ran out of precision`); the fallback then writes a broken `ponds.obj`, `objBoundingDim.py` reports
  `y_min: -inf` and `cubic_block` stops with `xMin missing in meshParameters`. New: the generator treats such an outline as a violation (`pond_slit` / `lawn_slit`,
  hard rule) and repairs it (erosion by 0.11 m) in reflow, lawn / pond mutations and `finalize_patches`.
* `json_to_obj.py` and `gen_ground_obj.py` (caseFile/momentum) clean pond and lawn outlines identically: coordinates on a 1 mm grid, duplicates / spikes removed,
  slits eroded by 0.11 m (message `Note: pond ... had a hairline slit`).
* `python/mesh_gate.py`: runs the case's own `json_to_obj.py` + `gen_ground_obj.py` on a layout in a temporary folder (about 1 s) and fails it when anything
  falls back or breaks. `nsga2.py` applies it to every child (message "rejected by the mesh gate"); `--no-mesh-gate` switches it off.
  Test a layout yourself:  `python python/mesh_gate.py input/offspring_17.json`
