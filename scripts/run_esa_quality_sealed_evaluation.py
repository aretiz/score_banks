from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "power_method_pilot"))

import esa_probe_pilot as pilot
import quality_signal_audit as quality


ROOT = Path("outputs/esa_real_event_readiness")
DEVELOPMENT_PATH = Path(
    "outputs/esa_probe_features_v1/development_probe_features.csv"
)
OUTPUT = Path("outputs/esa_quality_sealed_evaluation_v1")
FIGURES = Path("figures/esa_quality_sealed_evaluation_v1")
SEED = 20260916


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_interval(successes, trials):
    lower = 0.0 if successes == 0 else beta.ppf(
        0.025,
        successes,
        trials - successes + 1,
    )
    upper = 1.0 if successes == trials else beta.ppf(
        0.975,
        successes + 1,
        trials - successes,
    )
    return float(lower), float(upper)


def evaluation_plan(plan, roles):
    role_map = roles.set_index("unit_id").model_role
    wanted = {
        "nominal_calibration",
        "nominal_test",
        "anomaly_test",
    }
    ids = set(role_map[role_map.isin(wanted)].index)
    rows = []

    for unit_id, group in plan[plan.unit_id.isin(ids)].groupby(
        "unit_id",
        sort=True,
    ):
        role = str(role_map.loc[unit_id])

        for column in (
            "history_start",
            "window_start",
        ):
            require(
                group[column].nunique() == 1,
                f"Conflicting {column}: {unit_id}",
            )

        item = group.iloc[0]

        if role == "anomaly_test":
            families = sorted(group.target_family.unique())
            require(
                len(families) == 1,
                f"Ambiguous anomaly family: {unit_id}",
            )
            family = str(families[0])
            label = 1
        else:
            family = "nominal"
            label = 0

        rows.append(
            {
                "unit_id": str(unit_id),
                "source_role": role,
                "label": label,
                "family": family,
                "group_id": (
                    f"event:{unit_id}"
                    if label
                    else f"unit:{unit_id}"
                ),
                "window_start": item.window_start,
                "window_end": group.window_end.min(),
            }
        )

    result = pd.DataFrame(rows)
    require(set(result.unit_id) == ids, "Missing evaluation units")
    require(not result.unit_id.duplicated().any(), "Duplicate units")

    expected = {
        "nominal_calibration": 118,
        "nominal_test": 119,
        "anomaly_test": 37,
    }
    require(
        result.source_role.value_counts().to_dict() == expected,
        "Frozen evaluation counts changed",
    )

    anomaly_counts = (
        result[result.source_role == "anomaly_test"]
        .family.value_counts()
        .to_dict()
    )
    require(
        anomaly_counts == {"class_3": 19, "class_7": 18},
        "Frozen family counts changed",
    )
    return result


def extract_features(rows, horizon_hours):
    units = pd.read_csv(
        ROOT / "window_tensor/units.csv",
        dtype={"unit_id": str},
    )
    channels = pd.read_csv(ROOT / "window_tensor/channels.csv")
    model_protocol = json.loads(
        (ROOT / "mission1_frozen_model_protocol.json").read_text(
            encoding="utf-8"
        )
    )

    with np.load(
        ROOT / "forecast_representations/forecaster_parameters.npz",
        allow_pickle=False,
    ) as archive:
        parameters = {
            name: archive[name]
            for name in archive.files
        }

    channel_column = (
        "Channel"
        if "Channel" in channels
        else "channel"
    )
    channel_names = channels[channel_column].astype(str).tolist()

    require(
        channel_names == model_protocol["input_channels"],
        "Channel order changed",
    )
    pilot.validate_parameters(parameters, len(channel_names))

    units["history_start"] = pd.to_datetime(
        units.history_start,
        format="mixed",
        utc=True,
    )
    require(not units.unit_id.duplicated().any(), "Duplicate tensor IDs")

    positions = {
        unit_id: index
        for index, unit_id in enumerate(units.unit_id)
    }
    require(
        set(rows.unit_id) <= set(positions),
        "Evaluation unit missing from tensor",
    )

    tensor = np.load(
        ROOT / "window_tensor/telemetry.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        tensor.shape
        == (
            len(units),
            int(units.length.max()),
            len(channel_names),
        ),
        "Tensor shape changed",
    )

    cadence = 30
    exact_steps = horizon_hours * 3600 / cadence
    require(
        exact_steps == round(exact_steps),
        "Nonintegral horizon",
    )
    steps = int(round(exact_steps))
    records = []

    for number, row in enumerate(
        rows.itertuples(index=False),
        1,
    ):
        position = positions[row.unit_id]
        unit = units.iloc[position]
        offset = (
            row.window_start - unit.history_start
        ).total_seconds() / cadence

        require(
            abs(offset - round(offset)) < 1e-6,
            f"Off-grid window: {row.unit_id}",
        )

        start = int(round(offset))
        end = start + steps

        require(
            start >= 2 and end <= int(unit.length),
            f"Insufficient tensor coverage: {row.unit_id}",
        )
        require(
            row.window_start
            + pd.Timedelta(hours=horizon_hours)
            <= row.window_end,
            f"Insufficient frozen window: {row.unit_id}",
        )

        raw = np.asarray(
            tensor[position, :end],
            dtype=float,
        )
        require(
            np.isfinite(raw).all(),
            f"Non-finite telemetry: {row.unit_id}",
        )

        standardized = (
            raw - parameters["input_mean"]
        ) / parameters["input_scale"]
        window = standardized[start - 1 : end]

        record = {
            "unit_id": row.unit_id,
            "source_role": row.source_role,
            "label": int(row.label),
            "family": row.family,
            "group_id": row.group_id,
            "window_start": row.window_start.isoformat(),
            "horizon_hours": horizon_hours,
        }
        record.update(
            pilot.quality_features(
                standardized[:start],
                window,
                parameters,
            )
        )
        records.append(record)

        if number == 1 or number % 25 == 0 or number == len(rows):
            print(
                f"[{number}/{len(rows)}] "
                "sealed quality features extracted",
                flush=True,
            )

    result = pd.DataFrame(records)
    feature_columns = [
        column
        for column in result
        if column.startswith("C__")
    ]
    require(
        np.isfinite(
            result[feature_columns].to_numpy(float)
        ).all(),
        "Invalid quality features",
    )
    return result


def summarize(predictions):
    rows = []

    for detector, group in predictions.groupby(
        "detector",
        sort=False,
    ):
        strata = [
            (
                "nominal_test_fpr",
                "nominal",
                group[group.source_role == "nominal_test"],
            )
        ]

        for family, subset in group[
            group.source_role == "anomaly_test"
        ].groupby("family", sort=True):
            strata.append(("power", family, subset))

        for metric, family, subset in strata:
            successes = int(subset.rejected.sum())
            lower, upper = exact_interval(
                successes,
                len(subset),
            )
            rows.append(
                {
                    "detector": detector,
                    "metric": metric,
                    "family": family,
                    "estimate": successes / len(subset),
                    "lower_95": lower,
                    "upper_95": upper,
                    "successes": successes,
                    "units": len(subset),
                }
            )

    return pd.DataFrame(rows)


def make_figure(metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=False)

    keys = [
        ("nominal_test_fpr", "nominal"),
        ("power", "class_7"),
        ("power", "class_3"),
    ]
    labels = [
        "Nominal FPR",
        "Class 7 power",
        "Class 3 power",
    ]
    detectors = [
        "quality_W_primary",
        "quality_C_secondary",
    ]
    colors = ["#2474b5", "#d76ab0"]
    x = np.arange(3)
    width = 0.34

    fig, axis = plt.subplots(
        figsize=(8.2, 4.8),
        constrained_layout=True,
    )

    for shift, detector, color in zip(
        (-0.5, 0.5),
        detectors,
        colors,
    ):
        subset = metrics[
            metrics.detector == detector
        ].set_index(["metric", "family"])
        estimate = np.asarray([
            subset.loc[key, "estimate"]
            for key in keys
        ])
        lower = np.asarray([
            subset.loc[key, "lower_95"]
            for key in keys
        ])
        upper = np.asarray([
            subset.loc[key, "upper_95"]
            for key in keys
        ])

        axis.bar(
            x + shift * width,
            estimate,
            width,
            color=color,
            label=detector,
            yerr=np.vstack([
                estimate - lower,
                upper - estimate,
            ]),
            capsize=4,
        )

    axis.axhline(
        0.05,
        color="black",
        linestyle="--",
        linewidth=1,
        label="alpha=0.05",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("Observed fraction")
    axis.set_title("Sealed ESA Mission-1 evaluation")
    axis.legend(frameon=False)

    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES
            / f"esa_quality_sealed_evaluation.{extension}",
            dpi=200,
        )
    plt.close(fig)


def selected_quality_transform(
    train,
    calibration,
    heldout,
    groups,
):
    """
    Apply the frozen quality preprocessing only to feature groups
    used by the selected detector.

    The original development helper eagerly transformed S even for
    W/C detectors. Sealed evaluation rows intentionally contain only
    decision-time quality features, so unused S columns are absent.
    """
    from sklearn.preprocessing import StandardScaler

    frames = [
        train,
        calibration,
        heldout,
    ]

    names = quality.columns(train)
    requested = set(groups)

    allowed = {
        "H",
        "W",
        "A",
        "S",
        "T",
    }

    unknown = requested - allowed

    require(
        not unknown,
        f"Unknown feature groups: {sorted(unknown)}",
    )

    blocks = {}

    for name in ("H", "W", "A", "S"):
        if name not in requested:
            continue

        columns = names[name]

        for frame_index, frame in enumerate(frames):
            missing = [
                column
                for column in columns
                if column not in frame.columns
            ]

            require(
                not missing,
                "Missing requested "
                f"{name} features in frame "
                f"{frame_index}: {missing[:5]}",
            )

        arrays = [
            frame[columns].to_numpy(float)
            for frame in frames
        ]

        arrays = [
            np.sign(array)
            * np.log1p(np.abs(array))
            for array in arrays
        ]

        scaler = StandardScaler().fit(
            arrays[0]
        )

        blocks[name] = [
            scaler.transform(array)
            for array in arrays
        ]

    if "T" in requested:
        arrays = [
            quality.calendar_array(frame)
            for frame in frames
        ]

        scaler = StandardScaler().fit(
            arrays[0]
        )

        blocks["T"] = [
            scaler.transform(array)
            for array in arrays
        ]

    return [
        np.column_stack([
            blocks[group][frame_index]
            for group in groups
        ])
        for frame_index in range(3)
    ]


def main():
    require(not OUTPUT.exists(), f"Output exists: {OUTPUT}")
    require(not FIGURES.exists(), f"Figures exist: {FIGURES}")

    frozen_path = (
        ROOT
        / "frozen_quality_detector/protocol.json"
    )
    frozen = json.loads(
        frozen_path.read_text(encoding="utf-8")
    )

    require(
        frozen["protocol_version"]
        == "esa_quality_detector_v1",
        "Wrong frozen protocol",
    )
    require(
        frozen["sealed_evaluation_used"] is False,
        "Protocol was not prospectively sealed",
    )
    require(
        [
            (
                item["name"],
                item["feature_set"],
                item["learner"],
            )
            for item in frozen["detectors"]
        ]
        == [
            ("quality_W_primary", "W", "logistic"),
            ("quality_C_secondary", "C", "logistic"),
        ],
        "Detector definitions changed",
    )

    for name in (
        "scripts/power_method_pilot/esa_probe_pilot.py",
        "scripts/power_method_pilot/quality_signal_audit.py",
        "outputs/esa_probe_features_v1/development_probe_features.csv",
    ):
        require(
            sha256(name) == frozen["source_hashes"][name],
            f"Frozen source changed: {name}",
        )

    plan = pd.read_csv(
        ROOT / "mission1_frozen_window_plan.csv",
        dtype=str,
        keep_default_na=False,
    )
    roles = pd.read_csv(
        ROOT / "mission1_frozen_model_unit_roles.csv",
        dtype=str,
        keep_default_na=False,
    )
    require(
        not roles.unit_id.duplicated().any(),
        "Duplicate frozen roles",
    )

    for column in (
        "history_start",
        "window_start",
        "window_end",
    ):
        plan[column] = pd.to_datetime(
            plan[column],
            format="mixed",
            utc=True,
        )

    evaluation_rows = evaluation_plan(plan, roles)
    development = pd.read_csv(
        DEVELOPMENT_PATH,
        dtype={"unit_id": str, "group_id": str},
        keep_default_na=False,
    )
    pilot.validate_features(development)

    require(
        set(development.unit_id).isdisjoint(
            set(evaluation_rows.unit_id)
        ),
        "Development/evaluation overlap",
    )

    print("=== OPEN SEALED ESA QUALITY EVALUATION ===")
    print("Frozen protocol verified: YES")
    print("Development/evaluation overlap: 0")
    print("Test-dependent tuning: NO")

    evaluation = extract_features(
        evaluation_rows,
        float(development.horizon_hours.iloc[0]),
    )
    calibration = evaluation[
        evaluation.source_role
        == "nominal_calibration"
    ].copy()
    test = evaluation[
        evaluation.source_role.isin(
            ["nominal_test", "anomaly_test"]
        )
    ].copy()

    require(
        (calibration.label == 0).all(),
        "Anomaly entered calibration",
    )

    predictions = []
    fitted = []
    alpha = float(frozen["alpha"])

    detector_groups = {
        "quality_W_primary": quality.FEATURE_SETS["W"],
        "quality_C_secondary": quality.FEATURE_SETS["C"],
    }

    for detector, groups in detector_groups.items():
        arrays = selected_quality_transform(
            development,
            calibration,
            test,
            groups,
        )
        model = pilot.learner_fit(
            "logistic",
            arrays[0],
            development,
            pilot.stable_seed(SEED, detector),
        )
        calibration_scores = pilot.predict_score(
            model,
            arrays[1],
        )
        test_scores = pilot.predict_score(
            model,
            arrays[2],
        )
        threshold = pilot.conformal_threshold(
            calibration_scores,
            alpha,
        )

        fitted.append(
            {
                "detector": detector,
                "feature_groups": "+".join(groups),
                "dimension": arrays[0].shape[1],
                "training_units": len(development),
                "calibration_units": len(calibration),
                "threshold": threshold,
            }
        )

        for row, score in zip(
            test.itertuples(index=False),
            test_scores,
        ):
            predictions.append(
                {
                    "detector": detector,
                    "unit_id": row.unit_id,
                    "group_id": row.group_id,
                    "source_role": row.source_role,
                    "family": row.family,
                    "label": int(row.label),
                    "score": float(score),
                    "threshold": threshold,
                    "rejected": int(score > threshold),
                }
            )

    predictions = pd.DataFrame(predictions)
    require(
        not predictions.duplicated(
            ["detector", "unit_id"]
        ).any(),
        "Duplicate predictions",
    )

    metrics = summarize(predictions)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    evaluation.to_csv(
        OUTPUT / "sealed_quality_features.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT / "predictions.csv",
        index=False,
    )
    metrics.to_csv(
        OUTPUT / "metrics.csv",
        index=False,
    )
    pd.DataFrame(fitted).to_csv(
        OUTPUT / "fitted_detectors.csv",
        index=False,
    )

    result_protocol = {
        "status": "sealed_evaluation_complete",
        "source_protocol_sha256": sha256(frozen_path),
        "alpha": alpha,
        "seed": SEED,
        "test_retuning": False,
        "intervals": "exact Clopper-Pearson 95 percent",
        "development_units": len(development),
        "calibration_units": len(calibration),
        "nominal_test_units": int(
            (test.source_role == "nominal_test").sum()
        ),
        "anomaly_test_units": int(
            (test.source_role == "anomaly_test").sum()
        ),
    }
    pilot.json_write(
        OUTPUT / "protocol.json",
        result_protocol,
    )

    paths = sorted(OUTPUT.glob("*"))
    (OUTPUT / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path}\n"
            for path in paths
        ),
        encoding="utf-8",
    )

    make_figure(metrics)

    display = metrics.copy()
    for column in (
        "estimate",
        "lower_95",
        "upper_95",
    ):
        display[column] *= 100

    print("\n=== SEALED RESULTS (%) ===")
    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )
    print("\n=== FILES WRITTEN ===")
    for path in sorted(OUTPUT.iterdir()):
        print(f"{path}: {path.stat().st_size:,} bytes")
    for path in sorted(FIGURES.iterdir()):
        print(f"{path}: {path.stat().st_size:,} bytes")
    print("\nFINAL SEALED ESA QUALITY EVALUATION: PASS")


if __name__ == "__main__":
    main()
