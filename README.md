# Calibrating Score Banks for Earth Observation Anomaly Detection

This repository is the research companion for the REO2 2026 paper
"Calibrating Score Banks for Earth Observation Anomaly Detection. Multiplicity,
Tail Shape, and Power."

It contains the revised manuscript, a tracked revision, paper figures, the
EarthNet score bank analysis, stronger fusion baselines, the ESA real event
stress test, compact result tables, and a static GitHub Pages website.

## Repository contents

```text
.
├── .github/workflows/pages.yml
├── scripts
│   ├── power_method_pilot
│   ├── run_care_earthnet.py
│   ├── run_care_frozen_fusion_benchmark.py
│   └── run_esa_power_decomposition.py
├── REPRODUCIBILITY.md
└── prepare_complete_repo.sh
```

## Data and checkpoints

EarthNet2021 and ESA ADB are not redistributed here. Forecast caches, model
checkpoints, and raw experiment outputs are also excluded because of size and
source specific distribution terms. See `REPRODUCIBILITY.md` for the expected
inputs and commands.
