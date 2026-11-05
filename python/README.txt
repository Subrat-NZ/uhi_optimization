UHI layout optimizer - project folders (same layout as on the server)
  caseFile/ template OpenFOAM case containing Allrun (must contain NO *.json; Allrun uses the only *.json)
  epw/      exactly ONE *.epw (Allrun does: cp ../epw/*.epw weather.epw)
  input/    base.json (your layout, never modified), constraints.json (optional rule overrides),
            initial individuals individual_X.json (individual_0 = base), offspring offspring_X.json
  jsons/    <epw-stem>_cases.json from select_epw_cases.py (peak-heat and peak-cold hour)
  python/   select_epw_cases.py, generate_candidates.py, layout_ops.py, layout_reflow.py, run_candidates.py, check_invariants.py
  run/      run/NAME/heat and run/NAME/cold, each a copy of caseFile (created automatically)
  output/   utci_extremes_report.txt, objectives.csv, run_summary_<time>.json

Step 0 (make candidates by hand; the NSGA-II driver python/optimize.py / nsga2.py makes and breeds them itself):
   python python/generate_candidates.py individual --count 10 --seed 1 --include-base
   python python/generate_candidates.py offspring --parent input/individual_4.json --count 9 --seed 2   # mutated children
   python python/generate_candidates.py template                              # cool wind catcher on every building (heat wind) + wind blocker belt along the windward edge(s) (cold wind), both with dense pathway trees
   python python/generate_candidates.py template --kind all                   # ... and the earlier variants (tree wind catcher, tree wind blocker wall, cool wind catcher on one building)
   python python/generate_candidates.py validate input/individual_4.json [--show-inherited]
   python python/generate_candidates.py plot input/individual_4.json        (-> output/individual_4.png)
   python python/check_invariants.py input/individual_*.json                (independent audit)
   Numbering continues automatically, files are never overwritten, log in output/candidates_log.jsonl.
   Lawn water: irrigation depth is fixed at 5 mm/day; set lawn.water_available_l_per_day in input/constraints.json, or leave null to use the base-lawn need (5 L/day per m2).
   Pathways (= roads = paths) are read from the "roads" list of the layout (outline polygons; also "pathways"/"osmRoads"); no building, pond or lawn may overlap them.
   Needs: pip install numpy shapely matplotlib
Generation candidates follow RANGES (all in input/constraints.json):
   - every building moves at least 0.5 x its longest projected span (max of E-W, N-S) and at most 10 % of the site
     diameter; if a building is boxed in (roads, other buildings, site edge) the minimum is relaxed in steps
     (x1, x0.5, x0.25) and the relaxation is written to output/candidates_log.jsonl
   - lawns / ponds make room for the moved buildings (cut back) and grow back elsewhere, so lawn and pond totals stay in band
   - trees: count = base + 10 %, a share relocated / resized, heights set by location zone (near buildings / pathways / open)
   - no ponds in the base: ponds are created with a total area of 10 % of the open space (depth 2 m default)
   Mutations stay small perturbations of a parent file.
Step 1 (once per EPW):   python python/select_epw_cases.py --occupied 1 24
Step 2 (per batch):      python python/run_candidates.py individual_1 offspring_1      (or no names = all unfinished)
Per candidate and season: copy caseFile -> run/NAME/<season>, copy input/NAME.json into it, write
date_and_time.xyz (hour = EPW hour_end - 1), run ./Allrun, read UTCI_details.txt.
Objectives (both minimized):  f1 = site-averaged UTCI of the heat case;  f2 = -(site-averaged UTCI of the cold case).
Needs for step 1: pip install pythermalcomfort numpy

NOTE ON footprints[] (fixed): in the layout files, footprints[].vertices are stored with the OPPOSITE sign of y
compared with buildings[].geometry.footprint (y_footprints = -y_building).  When a building moves by (dx, dy) the
footprints[] entry now moves by (dx, -dy) (the convention is detected per file).  check_invariants.py verifies this
and also audits every pairwise overlap (building/OSM/road/lawn/pond; only tree-lawn is allowed).
Building-lawn, building-pond and lawn-pond overlap are hard rules in the generator (never inherited from the base).
CLEARANCE FROM BUILDINGS (hard rules, every layout incl. individual_0, generated and OSM buildings):  ponds >= 2 m, lawns >= 1 m from every building
(constraints.json: "pond": {"building_clearance_m": 2.0}, "lawn": {"building_clearance_m": 1.0}; "patch": {"clearance_tol_m": 0.05} = measuring tolerance).
Violations are called pond_building_clearance / lawn_building_clearance (generator) and CLEARANCE (check_invariants.py).

PARALLEL GENERATION: nsga2.py / optimize.py --workers N  (N processes make the seed individuals, random individuals and children at the same time;
  0 = one per CPU core; default 1 = as before). Layouts differ from a --workers 1 run but are valid and reproducible for a given N and --seed.

STACKED BUILDINGS AND ITEMS ON BUILDINGS
A building over a building is ONE building: parts with the same footprint, parts of the same family ("Building 3", "building_3_mass_1",
"building_3_internal_floor_2") whose footprints overlap, and a mass whose footprint overlaps >= half of the one below at touching heights.
The stack moves as a whole and the union of its footprints is what every overlap / clearance rule sees.
Trees, lawns and ponds that stand on a building in base.json (tree centre / most of the outline inside the stack footprint) are glued to it: the optimiser does not
touch them, they are not part of the tree / lawn / pond budgets, and every written layout carries them moved by exactly the translation of their building.
check_invariants.py reports GLUE ... when one is not where it should be.  Details: NSGA2_README.md, section "Stacked buildings ...".

PICTURES (plot_candidates.py)
  python python/plot_candidates.py                     every individual_*/offspring_* in input/ -> output/<name>.png  (base | candidate)
  python python/plot_candidates.py --diff FILE...      adds a 3rd panel with only the changes (moves, lawn/pond gained/lost, trees +/-/moved)
  python python/plot_candidates.py --grid [--cols 4]   one output/overview.png: base + all candidates
  python python/plot_candidates.py --objectives output/objectives.csv   prints UTCI heat/cold, f1, f2 in the titles
  python python/generate_candidates.py individual --count 10 --plot [--diff]    makes the candidates and draws them
Dashed outlines = original building positions, arrows = translations with the shift in metres, dark-outlined trees = new trees.

PATHWAY PLANTING AND TREE-PATHWAY RULE
  * Trees may NOT overlap pathways (tree.tree_on_pathway_allowed = false): the crown must stay off the pathway.
    Generations repair every base tree that breaks this; the audit checks it independently.
  * Every generation plants trees on BOTH sides of every pathway (tree.pathway_planting): slots lie along both
    edges every ~spacing_m (10 m), radius 2-4 m, crown edge >= 0.5 m from the path, and each slot must satisfy
    the usual rules (crown gaps >= 1.1 (Ri+Rj), centre >= 1.1 R from buildings, crown inside the site, off ponds).
    A slot without room is slid +-1.5 / 3 m along the path, then the crown is shrunk down to 1.5 m, then skipped.
    A random share (fill_fraction_range 60-100 %) of the slots is used, at most max_new_trees (60) per candidate;
    the tree-count cap grows by that budget.  Set "enabled": false to switch it off.
  * ROAD SHADE (rule for EVERY individual, offspring and seed; tree.pathway_planting.shade_required): >= 80 % of the plantable
    samples along both road edges (every 2 m) must have a crown edge within 3 m.  ensure_pathway_shade plants trees with a crown
    radius that suits the room (2.5-5 m, down to 1.5 m between buildings), enlarges nearby crowns (<= 6 m) and, at the tree-count
    limit, moves trees that stand far from every road.  Crossover and mutation repair it again; individual_0 (the base copy) is exempt.
  * A tree whose centre is within tree.pathway_zone_distance_m (6 m) of a pathway counts as a street/path tree
    and gets the 8-12 m height band.

BASE WITHOUT TREES / LAWNS / PONDS (e.g. a layout that only has buildings)
  * Trees: canopy cover (tree.canopy_cover_frac_if_missing = 10 %) x open space / mean crown area -> N trees are
    CREATED in every generation (the +10 % rule then refers to N); look = tree.template (copy of a sample tree).
  * Ponds and lawns: A0 = 10 % of the open space each (pond.open_space_frac_if_missing, lawn.open_space_frac_if_missing);
    lawn water budget = 5 L per m2 of that A0 unless lawn.water_available_l_per_day is given.
  * NEW ponds/lawns are seeded in the LARGEST OPEN AREAS of what is left after the buildings moved (pond first, then
    lawns): the point with the most free room around it is rank 1 (probability patch.largest_open_rank_probs = 80 %,
    rank 2 = 20 %; use [1.0] to always take the largest).  The patch is the part of a disc that is visible from that
    point, so it never encloses a building, has no holes and no spikes thinner than patch.min_width_m (4 m).
  * Patches that still have to wrap around a building get a 1 m wide channel instead of a hairline slit, so a viewer
    that only draws the outer ring does not show the building inside the lawn / pond.
  * Created ponds/lawns are not shuffled afterwards by random operations in the same generation.
