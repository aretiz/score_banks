# Calibrating Score Banks for Earth Observation Anomaly Detection

Research code for the REO2 2026 paper **“Calibrating Score Banks for Earth Observation Anomaly Detection: Multiplicity, Tail Shape, and Power.”**

The repository contains the EarthNet2021 score bank analysis, calibrated aggregation baselines, and a site-disjoint evaluation on real DynamicEarthNet land-cover changes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original EarthNet analysis also requires the forecast caches produced by the associated forecasting project. Raw datasets, model checkpoints, and generated caches are not included in this repository.

## Data

Download the datasets from their official sources.

- [EarthNet2021](https://www.earthnet.tech/)
- [DynamicEarthNet](https://mediatum.ub.tum.de/1650201)

For the DynamicEarthNet evaluation, extract the Sentinel-2 images and labels under:

```text
data/dynamic_earthnet/extracted/
├── sentinel2/
└── labels/
```

## Experiments

### EarthNet2021 analysis

```bash
python scripts/run_care_earthnet.py --help
```

The calibrated fusion comparison is run with:

```bash
python scripts/run_care_frozen_fusion_benchmark.py \
  --bootstrap 2000 \
  --out outputs/reo2_fusion_benchmark
```

### DynamicEarthNet real-change evaluation

```bash
python scripts/run_dynamic_earthnet_score_bank.py \
  --data data/dynamic_earthnet/extracted \
  --out outputs/dynamic_earthnet_score_bank
```

The DynamicEarthNet experiment uses disjoint areas for fitting, design, calibration, and testing. Labels define unchanged and changed evaluation patches. They are not used to construct the persistence forecast or residual scores.

## Main files

- `scripts/run_care_earthnet.py` runs the main EarthNet2021 score bank analysis.
- `scripts/run_care_frozen_fusion_benchmark.py` compares calibrated aggregation rules.
- `scripts/run_dynamic_earthnet_score_bank.py` runs the real EO change evaluation.
- `requirements.txt` lists the Python dependencies.

## License

This repository is released under the [MIT License](LICENSE).
