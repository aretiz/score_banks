"""Development-only source audit for the ESA quality-feature signal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import esa_probe_pilot as pilot


FEATURE_SETS = {
    "H": ("H",),
    "W": ("W",),
    "A": ("A",),
    "T": ("T",),
    "HW": ("H", "W"),
    "C": ("H", "W", "A"),
    "S": ("S",),
    "SH": ("S", "H"),
    "SW": ("S", "W"),
    "SA": ("S", "A"),
    "ST": ("S", "T"),
    "SC": ("S", "H", "W", "A"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def columns(frame):
    result = {
        "H": sorted(c for c in frame if c.startswith("C__history__")),
        "W": sorted(c for c in frame if c.startswith("C__window__")),
        "A": sorted(c for c in frame if c.startswith("C__forecast_disagreement_")),
        "S": sorted(c for c in frame if c.startswith("S__")),
    }
    require(all(result.values()), "One or more required feature groups are empty")
    return result


def calendar_array(frame):
    value = pd.to_datetime(frame.window_start, utc=True, format="mixed")
    day = (value - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy() / 86400
    phase = 2 * np.pi * (value.dt.dayofyear.to_numpy() - 1) / 365.2425
    return np.column_stack([day, np.sin(phase), np.cos(phase)])


def transform(train, calibration, heldout, groups):
    from sklearn.preprocessing import StandardScaler

    frames = [train, calibration, heldout]
    names = columns(train)
    blocks = {}
    for name in ("H", "W", "A", "S"):
        arrays = [frame[names[name]].to_numpy(float) for frame in frames]
        arrays = [np.sign(a) * np.log1p(np.abs(a)) for a in arrays]
        scaler = StandardScaler().fit(arrays[0])
        blocks[name] = [scaler.transform(a) for a in arrays]
    arrays = [calendar_array(frame) for frame in frames]
    scaler = StandardScaler().fit(arrays[0])
    blocks["T"] = [scaler.transform(a) for a in arrays]
    return [np.column_stack([blocks[group][i] for group in groups]) for i in range(3)]


def summarize(predictions):
    key = ["mode", "held_family", "feature_set"]
    unit = predictions.groupby(
        key + ["unit_id", "group_id", "family", "label", "source_role"],
        as_index=False,
    ).rejected.mean()
    rows = []
    for keys, group in unit.groupby(key, sort=True):
        nominal = group[group.label == 0]
        rows.append(dict(zip(key, keys)) | {
            "metric": "nominal_fpr", "stratum": "all_nominal",
            "estimate": nominal.rejected.mean(), "units": len(nominal),
        })
        for role, subset in nominal.groupby("source_role", sort=True):
            rows.append(dict(zip(key, keys)) | {
                "metric": "nominal_fpr", "stratum": role,
                "estimate": subset.rejected.mean(), "units": len(subset),
            })
        for family, subset in group[group.label == 1].groupby("family", sort=True):
            rows.append(dict(zip(key, keys)) | {
                "metric": "power", "stratum": family,
                "estimate": subset.rejected.mean(), "units": len(subset),
            })
    return pd.DataFrame(rows)


def matched_pairs(predictions):
    key = ["repeat", "mode", "held_family", "fold", "feature_set", "group_id"]
    positive = predictions[predictions.label == 1].copy()
    control = predictions[
        (predictions.label == 0) & predictions.group_id.str.startswith("pair:")
    ].copy()
    joined = positive.merge(control, on=key, validate="one_to_one", suffixes=("_event", "_control"))
    require(len(joined) == len(positive), "A held-out event is missing its paired control")
    joined["score_win"] = np.where(
        joined.score_event > joined.score_control, 1.0,
        np.where(joined.score_event == joined.score_control, 0.5, 0.0),
    )
    joined["rejection_gain"] = joined.rejected_event - joined.rejected_control
    by_pair = joined.groupby(
        ["mode", "held_family", "feature_set", "group_id", "family_event"],
        as_index=False,
    ).agg(score_win=("score_win", "mean"), rejection_gain=("rejection_gain", "mean"))
    return by_pair.groupby(
        ["mode", "held_family", "feature_set", "family_event"], as_index=False
    ).agg(
        pairs=("group_id", "nunique"),
        event_score_win_rate=("score_win", "mean"),
        paired_rejection_gain=("rejection_gain", "mean"),
    ).rename(columns={"family_event": "family"})


def figures(metrics, folder, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    order = list(FEATURE_SETS)
    for (mode, held), group in metrics.groupby(["mode", "held_family"], sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        power = group[group.metric == "power"]
        for family, subset in power.groupby("stratum"):
            values = subset.set_index("feature_set").estimate.reindex(order)
            axes[0].plot(order, values, marker="o", label=family)
        axes[0].set_ylim(-.02, 1.02)
        axes[0].set_ylabel("Development out-of-fold power")
        axes[0].tick_params(axis="x", rotation=40)
        axes[0].legend()
        fpr = group[(group.metric == "nominal_fpr") &
                    (group.stratum.isin(["all_nominal", "paired_nominal_design"]))]
        for stratum, subset in fpr.groupby("stratum"):
            values = subset.set_index("feature_set").estimate.reindex(order)
            axes[1].plot(order, values, marker="o", label=stratum)
        axes[1].axhline(alpha, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Nominal rejection fraction")
        axes[1].tick_params(axis="x", rotation=40)
        axes[1].legend()
        fig.suptitle(f"Quality-signal source audit | {mode} | withheld={held}")
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"quality_sources_{mode}_{held}.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("outputs/esa_probe_features_v1/development_probe_features.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/esa_quality_signal_audit_v1"))
    parser.add_argument("--figures", type=Path, default=Path("figures/esa_quality_signal_audit_v1"))
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--calibration-units", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    manifest_path = args.features.parent / "manifest.json"
    require(manifest_path.is_file(), "Missing feature manifest")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "exploratory_development_only", "Wrong feature provenance")
    require(pilot.digest(args.features) == manifest.get("features_sha256"), "Feature checksum mismatch")
    pilot.protect_source(args.output, Path("outputs/esa_real_event_readiness"))
    pilot.protect_source(args.figures, Path("outputs/esa_real_event_readiness"))
    require(not args.figures.exists() or not any(args.figures.iterdir()),
            "Figure directory is not empty; use a new versioned directory")
    output = pilot.new_output(args.output)

    frame = pd.read_csv(args.features, dtype={"unit_id": str, "group_id": str}, keep_default_na=False)
    frame = frame.sort_values("unit_id").reset_index(drop=True)
    pilot.validate_features(frame)
    require("window_start" in frame, "Missing window_start")
    predictions = []
    jobs = 0
    for repeat in range(args.repeats):
        for mode in ("within", "leave_family_out"):
            for held, fold, split in pilot.build_splits(
                    frame, args.folds, args.calibration_units, args.seed + repeat, mode):
                jobs += 1
                train, calibration, heldout = [
                    frame.iloc[split[name]].copy()
                    for name in ("train", "calibration", "heldout")
                ]
                for feature_set, groups in FEATURE_SETS.items():
                    arrays = transform(train, calibration, heldout, groups)
                    model = pilot.learner_fit(
                        "logistic", arrays[0], train,
                        pilot.stable_seed(args.seed, repeat, mode, held, fold, feature_set),
                    )
                    cal_score = pilot.predict_score(model, arrays[1])
                    test_score = pilot.predict_score(model, arrays[2])
                    threshold = pilot.conformal_threshold(cal_score, args.alpha)
                    for row, score in zip(heldout.itertuples(), test_score):
                        predictions.append({
                            "repeat": repeat, "mode": mode, "held_family": held,
                            "fold": fold, "feature_set": feature_set,
                            "unit_id": row.unit_id, "group_id": row.group_id,
                            "family": row.family, "label": int(row.label),
                            "source_role": row.source_role, "score": float(score),
                            "threshold": threshold, "rejected": int(score > threshold),
                        })
                print(f"[{jobs}/36] {mode}, withheld={held}, repeat={repeat}, fold={fold}", flush=True)

    predictions = pd.DataFrame(predictions)
    key = ["repeat", "mode", "held_family", "fold", "feature_set", "unit_id"]
    require(not predictions.duplicated(key).any(), "Duplicate out-of-fold predictions")
    metrics = summarize(predictions)
    pairs = matched_pairs(predictions)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    pairs.to_csv(output / "paired_event_control.csv", index=False)
    protocol = {
        "status": "development_only_quality_source_audit",
        "feature_sha256": pilot.digest(args.features),
        "script_sha256": pilot.digest(__file__),
        "feature_sets": FEATURE_SETS,
        "alpha": args.alpha, "folds": args.folds, "repeats": args.repeats,
        "calibration_units": args.calibration_units, "seed": args.seed,
        "learner": "fixed logistic C=0.1",
        "test_roles_loaded": False,
    }
    pilot.json_write(output / "protocol.json", protocol)
    figures(metrics, args.figures, args.alpha)

    print("\n=== QUALITY-SOURCE POWER AND FPR (%) ===")
    table = metrics.copy()
    table["percent"] = 100 * table.estimate
    print(table.pivot(index=["mode", "held_family", "feature_set"],
                      columns=["metric", "stratum"], values="percent")
          .to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n=== EVENT VERSUS ITS PAIRED CONTROL ===")
    display = pairs.copy()
    display[["event_score_win_rate", "paired_rejection_gain"]] *= 100
    print(display.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nH=pre-event history, W=observed window, A=forecaster disagreement, T=calendar, S=residual scores.")
    print("Strong H or T performance indicates selection/time confounding or a precursor signal requiring separate study.")
    print("Strong W performance indicates that simple observed-window summaries already detect the events.")
    print("A useful router must improve on C and W at comparable realized FPR.")
    print("Figures:", args.figures)
    print("QUALITY SOURCE AUDIT COMPLETE. The sealed ESA evaluation set was not loaded.")


if __name__ == "__main__":
    main()
