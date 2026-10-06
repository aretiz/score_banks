# Reproducibility guide

## Scope

The repository separates the EarthNet score bank study from the ESA real event
stress test. Data, caches, and checkpoints are not redistributed.

## Environment

The experiments were run from the project virtual environment at `.venv`.
After the full project files are copied into this repository, activate an
equivalent environment and install the project dependencies.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## EarthNet analysis

The main driver is `scripts/run_care_earthnet.py`. It expects the frozen
EarthNet forecast caches produced by the two exporters.

```bash
python scripts/export_caps_earthnet_cache.py --help
python scripts/export_caps_earthnet_image_only_cache.py --help
python scripts/run_care_earthnet.py --help
```

The reviewer requested fusion comparison is implemented in
`scripts/run_care_frozen_fusion_benchmark.py`.

```bash
python scripts/run_care_frozen_fusion_benchmark.py \
  --bootstrap 2000 \
  --out outputs/reo2_fusion_benchmark
```

Every frozen scalar statistic receives its own split conformal calibration on
the same 466 nominal cubes. Matched power at the bank's realized FPR is a
descriptive operating point comparison.

## ESA real event stress test

The ESA analysis requires the official ESA ADB Mission 1 data and the frozen
protocol artifacts. The main evaluation drivers are shown below.

```bash
python scripts/run_esa_power_decomposition.py --help
python scripts/run_esa_quality_sealed_evaluation.py --help
```

The forecast score bank and quality feature detectors use different inputs and
operating points. Their comparison is diagnostic evidence about representation
alignment. It is not a controlled comparison of aggregation rules.

## Compact reported results

- `results/fusion_baselines.csv`
- `results/esa_real_event_summary.csv`
- `docs/downloads/family_summary.csv`

These tables contain the values reported in the revised manuscript. They do not
contain cube level or event level data.

## Missing large artifacts

The following items must be supplied locally.

- EarthNet2021 data
- EarthNet forecast checkpoints
- Frozen image only and multimodal forecast caches
- ESA ADB Mission 1 data
- Frozen ESA protocol artifacts

Do not commit those items unless their licenses and file sizes are suitable for
public redistribution.
