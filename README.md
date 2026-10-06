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

## Required step before public upload

Read `BEFORE_UPLOAD.md`. The manuscript still needs the confirmed author and
affiliation block. A public repository URL must also be inserted into the paper.
No author list or license has been guessed.

The supplied archive contains all paper specific scripts that were available in
the review bundle. The shared `spt` package and `power_decomposition.py` live in
the full project on the research server. To copy them into this repository, run
the following command from the extracted repository folder.

```bash
bash prepare_complete_repo.sh /home/zervou/research/spt_forecast_then_test
```

The script copies source code and environment files only. It does not copy data,
caches, checkpoints, virtual environments, or experiment outputs.

## Upload through the GitHub website

1. Create a new empty repository at <https://github.com/new>.
2. Open the new repository and choose **Add file**, then **Upload files**.
3. Upload the contents of this folder, including `.github`.
4. Commit the uploaded files to the `main` branch.
5. Open **Settings**, then **Pages**.
6. Select **GitHub Actions** as the source.
7. Open **Actions** and run the workflow named **Publish research page**.

The page will appear at
`https://YOUR_USERNAME.github.io/YOUR_REPOSITORY/`.

## Optional terminal upload

After creating an empty repository, run the following commands from this folder.

```bash
git init
git add .
git commit -m "Initial research companion"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

These commands are provided for the author to run. This package has not
connected to GitHub and has not published anything.

## Local website preview

```bash
python3 scripts/verify_site.py
python3 -m http.server 8000 --directory docs
```

Open <http://localhost:8000> in a browser. Stop the preview with `Ctrl+C`.

## Data and checkpoints

EarthNet2021 and ESA ADB are not redistributed here. Forecast caches, model
checkpoints, and raw experiment outputs are also excluded because of size and
source specific distribution terms. See `REPRODUCIBILITY.md` for the expected
inputs and commands.

## License

No license is assigned on behalf of the authors. Choose and add a license before
making the repository public. The NeurIPS style file retains its original
notices.
