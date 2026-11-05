# Stacked buildings = one building; roof items glued to their building

Copy `python/*` (11 files that belong together; includes everything of uhi_meshgate.zip) over `python/` of your case. Nothing changes in `caseFile/`.

* A building on top of a building is one building (same footprint, same family id + overlap, or mass on a slab at touching heights). Your scene: 24 parts -> 2 buildings.
  They move together; overlap / clearance rules use the union of the footprints (`layout_ops.building_groups`, `group_polygon`).
* Trees / lawns / ponds standing on a building of the base are glued to it: not edited by the optimiser, not in the budgets, written back rigidly moved with their
  building (`layout_ops.attach_hosted`, called by `generate_candidates.write_candidate`, also the mesh gate input in `nsga2.py`).
* `check_invariants.py`: independent stack detection (`stacked parts moved differently`, `stack structure changed`) and `GLUE ...` checks.
* `Scene(..., split_hosted=True)` is used for the base only; operators work on `working_copy()` data that already lacks the roof items.

Tested: pasted scene + test trees / lawns / ponds on slab and mass roofs; generation, mutation, crossover, wind seeds, mock NSGA-II run (2 generations),
mesh gate, 4 deliberate breakages (tree nudged, lawn deleted, mass moved alone, pond deformed) are all reported by check_invariants.py.

## Speed (same results, less recomputation)
* The OSM buildings grown by the lawn / pond clearance (1 m / 2 m) were rebuilt on every operation: now grown once and kept (`layout_ops._osm_zone`).
* `discretize_patches` clips the keep-out area to the 1 m around each patch; `_expand` clips the free area to the patch box and builds the "other patches" unions once.
* Road-shade samples: pathway, site edge and OSM tests are computed once (`_shade_static`); only buildings and ponds are re-tested.
* `working_copy` uses pickle instead of `copy.deepcopy` (4x faster on the building faces).
Checked: shade samples, plantable masks, keep-out zones and pond / lawn totals are identical to the previous code; every audit still passes.
Measured on the scenes you pasted (2-core test box): 227 OSM buildings: a generated individual 38 -> 14-16 s, 76 -> 32 s; 17 OSM buildings: 7 individuals 83 -> 62 s.

## --workers N (parallel generation)
* `nsga2.py --workers N` (also accepted by `optimize.py`): seed individuals, random individuals and children are made in N forked processes; `0` = all cores, default `1` = unchanged behaviour.
* Children: parents are chosen in the main process, crossover / mutation / rule check run in workers with a per-attempt random stream `default_rng([seed, gen, attempt])`;
  novelty check, mesh gate and file writing stay in the main process, in attempt order -> reproducible for a given N.
* Layouts differ from a `--workers 1` run (other random streams). All `--workers 2` / `4` test runs pass `check_invariants.py`; two runs with the same N and seed gave identical geometry.
* `nsga_batch.py` is not parallelised.
