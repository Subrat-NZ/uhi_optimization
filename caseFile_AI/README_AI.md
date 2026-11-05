# caseFile_AI  --  the AI case (wind -> T -> RH), drop-in for caseFile/

Full description, usage and limits: `../AI_INTEGRATION.md`.

Before the first run:
1. put the weights into `pipeline/`:  `T_model/latest_net_G.pth`,  `RH_model/<n>_net_G.pth`,  `wind_api/model/model/best.ckpt` (+ `stats.pickle`)
   (the files in this zip are 63-byte placeholders);
2. choose a back end:  `UHI_AI_BACKEND=runpod` (+ `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`),  `local` (GPU, torch),  or `mock` (fake fields, plumbing test only);
3. `python python/optimize.py --check --case-file caseFile_AI`.

Layout of this folder
    Allrun                         the case (same contract as caseFile/Allrun)          case_type.cfg   marks the folder as "ai" and keeps pipeline/ out of the candidate copies
    ai/                            run_ai_case.py (wind -> T -> RH)  ai_utci.py (UTCI)  wind_obj.py  backend.py  serve.py  mock_models.py  ai_images.py  config.json
    pipeline/                      YOUR AI pipeline (app/ scripts/ worker/ T_model/ RH_model/ wind_api/ pressure_calculator/), used from here by every candidate
    momentum/                      geometry scripts of the OpenFOAM case (json_to_obj, gen_ground_obj, cubic_block, objBoundingDim), python only, no mesh
    temperature_humidity/          tools/ + config/ of the OpenFOAM case (epw_to_weather, run_thermal, foam_io ... no solver)
    surface_temperature/           solar_thermal_model.py (the thermal model, as in caseFile/)
The folder must contain no *.json (the layout is the only one in a candidate's folder).
