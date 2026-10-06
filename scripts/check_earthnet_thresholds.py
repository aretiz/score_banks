#!/usr/bin/env python3
"""Audit EarthNet null-bank thresholds without refitting the forecaster.

This script reconstructs the frozen head bank from the exported audit inputs,
checks the fitted head normalization, and measures nominal-test false-positive
rates at both the Gaussian population threshold and the empirical split-
conformal threshold.

It reads only:
    <input-dir>/samples.csv
    <input-dir>/heads.json
    <input-dir>/decomposition.csv  (optional, used for saved thresholds)

It writes only to --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from sklearn.covariance import LedoitWolf

import power_decomposition as pdx


REPORTED_THRESHOLDS = {
    "care_joint": {
        "conformal": 2.934366,
        "gaussian": 2.075095,
    },
    "image_only": {
        "conformal": 2.904886,
        "gaussian": 2.051011,
    },
    "multimodal": {
        "conformal": 2.787936,
        "gaussian": 2.082992,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the EarthNet frozen bank and compare Gaussian and "
            "split-conformal thresholds on untouched nominal-test cubes."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/earthnet_power_decomposition_standardized"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/earthnet_threshold_diagnostic"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mc-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--threshold-tolerance",
        type=float,
        default=5e-6,
        help="Allowed absolute difference from a saved conformal threshold.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conformal_threshold(values: np.ndarray, alpha: float) -> tuple[float, int]:
    scores = np.asarray(values, dtype=float)
    require(scores.ndim == 1 and len(scores) > 0, "Empty calibration scores")
    require(np.isfinite(scores).all(), "Non-finite calibration scores")
    k = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
    require(k <= len(scores), "Calibration sample is too small for alpha")
    return float(np.partition(scores, k - 1)[k - 1]), k


def clopper_pearson(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return lower, upper


def nearest_psd_correlation(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum_before = float(eigenvalues.min())
    clipped = np.maximum(eigenvalues, 0.0)
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(repaired), 1e-15))
    repaired = repaired / np.outer(scale, scale)
    repaired = (repaired + repaired.T) / 2.0
    repair_size = float(np.max(np.abs(repaired - symmetric)))
    return repaired, minimum_before, repair_size


def gaussian_bank_draws(
    correlation: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    repaired, minimum_eigenvalue, repair_size = nearest_psd_correlation(correlation)
    eigenvalues, eigenvectors = np.linalg.eigh(repaired)
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    draws = rng.standard_normal((samples, repaired.shape[0])) @ factor.T
    return draws.max(axis=1), minimum_eigenvalue, repair_size


def exact_unique_value(frame: pd.DataFrame, column: str, representation: str) -> float | None:
    if column not in frame.columns:
        return None
    values = (
        frame.loc[frame["representation"] == representation, column]
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(values) == 0:
        return None
    unique = np.unique(np.round(values, 12))
    require(
        len(unique) == 1,
        f"{representation}: {column} has {len(unique)} distinct values",
    )
    return float(unique[0])


def saved_thresholds(
    decomposition: pd.DataFrame | None,
    representation: str,
) -> tuple[float | None, float | None, str]:
    empirical = None
    gaussian = None
    source = "none"
    if decomposition is not None:
        empirical = exact_unique_value(decomposition, "conformal_threshold", representation)
        gaussian = exact_unique_value(decomposition, "population_threshold", representation)
        if empirical is not None or gaussian is not None:
            source = "decomposition.csv"
    fallback = REPORTED_THRESHOLDS.get(representation, {})
    if empirical is None and "conformal" in fallback:
        empirical = float(fallback["conformal"])
        source = "reported_fallback" if source == "none" else source + "+reported_fallback"
    if gaussian is None and "gaussian" in fallback:
        gaussian = float(fallback["gaussian"])
        source = "reported_fallback" if source == "none" else source + "+reported_fallback"
    return empirical, gaussian, source


def nominal_rows(
    frame: pd.DataFrame,
    representation: str,
    split: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    subset = frame[
        (frame["representation"] == representation)
        & (frame["split"] == split)
    ].copy()
    require(not subset.empty, f"{representation}: missing {split}")
    require(
        not subset["unit_id"].duplicated().any(),
        f"{representation}/{split}: duplicate independent units",
    )
    require(
        not subset[features].isna().any().any(),
        f"{representation}/{split}: missing feature values",
    )
    return subset[features].to_numpy(dtype=float), subset["unit_id"].astype(str).to_numpy()


def summarize_bank(
    representation: str,
    split: str,
    bank: np.ndarray,
) -> dict[str, float | int | str]:
    return {
        "representation": representation,
        "split": split,
        "units": len(bank),
        "mean": float(np.mean(bank)),
        "sd": float(np.std(bank, ddof=1)),
        "q50": float(np.quantile(bank, 0.50)),
        "q90": float(np.quantile(bank, 0.90)),
        "q95_linear": float(np.quantile(bank, 0.95)),
        "q99": float(np.quantile(bank, 0.99)),
        "maximum": float(np.max(bank)),
    }


def summarize_heads(
    representation: str,
    split: str,
    scores: np.ndarray,
    names: list[str],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    normal_cutoffs = {
        "tail_gt_1.644854": norm.ppf(0.95),
        "tail_gt_1.959964": norm.ppf(0.975),
        "tail_gt_2.326348": norm.ppf(0.99),
    }
    for index, name in enumerate(names):
        values = scores[:, index]
        row: dict[str, float | int | str] = {
            "representation": representation,
            "split": split,
            "head_index": index,
            "head_name": name,
            "units": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
        }
        for column, cutoff in normal_cutoffs.items():
            row[column] = float(np.mean(values > cutoff))
        rows.append(row)
    return rows


def fpr_row(
    representation: str,
    threshold_name: str,
    threshold: float,
    bank: np.ndarray,
    gaussian_bank: np.ndarray,
) -> dict[str, float | int | str]:
    alarms = int(np.sum(bank > threshold))
    total = len(bank)
    lower, upper = clopper_pearson(alarms, total)
    return {
        "representation": representation,
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "nominal_test_alarms": alarms,
        "nominal_test_units": total,
        "nominal_test_fpr": alarms / total,
        "nominal_test_ci_low": lower,
        "nominal_test_ci_high": upper,
        "fitted_gaussian_tail_probability": float(np.mean(gaussian_bank > threshold)),
    }


def equivalent_independent_exponent(threshold: float, alpha: float) -> float:
    probability = float(norm.cdf(threshold))
    if probability <= 0.0 or probability >= 1.0:
        return math.nan
    return math.log(1.0 - alpha) / math.log(probability)


def head_names(specification: dict, count: int) -> list[str]:
    values = specification.get("head_names")
    if isinstance(values, list) and len(values) == count:
        return [str(value) for value in values]
    return [f"head_{index}" for index in range(count)]


def make_figure(
    plot_payload: list[dict[str, object]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(plot_payload),
        1,
        figsize=(8.0, 3.2 * len(plot_payload)),
        squeeze=False,
    )
    for axis, payload in zip(axes[:, 0], plot_payload):
        representation = str(payload["representation"])
        gaussian = np.sort(np.asarray(payload["gaussian"], dtype=float))
        fit = np.sort(np.asarray(payload["fit"], dtype=float))
        calibration = np.sort(np.asarray(payload["calibration"], dtype=float))
        test = np.sort(np.asarray(payload["test"], dtype=float))

        for label, values, color, width in [
            ("fitted Gaussian bank", gaussian, "black", 1.8),
            ("nominal fit", fit, "#1f77b4", 1.4),
            ("nominal calibration", calibration, "#ff7f0e", 1.4),
            ("nominal test", test, "#2ca02c", 1.4),
        ]:
            survival = 1.0 - np.arange(1, len(values) + 1) / (len(values) + 1)
            axis.plot(values, survival, label=label, color=color, linewidth=width)

        axis.axhline(0.05, color="0.65", linestyle=":", linewidth=1.0)
        axis.axvline(float(payload["gaussian_threshold"]), color="#9467bd", linestyle="--", label="Gaussian q95")
        axis.axvline(float(payload["conformal_threshold"]), color="#d62728", linestyle="--", label="conformal threshold")
        axis.set_yscale("log")
        axis.set_ylim(1e-4, 1.0)
        axis.set_xlabel("bank score")
        axis.set_ylabel("empirical survival")
        axis.set_title(representation)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)

    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    require(0.0 < args.alpha < 1.0, "alpha must be between zero and one")
    require(args.mc_samples >= 10_000, "Use at least 10,000 Monte Carlo samples")

    input_dir = args.input_dir
    sample_path = input_dir / "samples.csv"
    head_path = input_dir / "heads.json"
    decomposition_path = input_dir / "decomposition.csv"
    require(sample_path.is_file(), f"Missing {sample_path}")
    require(head_path.is_file(), f"Missing {head_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== EARTHNET THRESHOLD DIAGNOSTIC ===", flush=True)
    print(f"input: {input_dir}", flush=True)
    print(f"output: {args.output_dir}", flush=True)
    print("No audit input will be modified.", flush=True)

    samples = pd.read_csv(sample_path, dtype={"unit_id": str})
    heads = json.loads(head_path.read_text(encoding="utf-8"))
    decomposition = pd.read_csv(decomposition_path) if decomposition_path.is_file() else None

    require(hasattr(pdx, "representation_features"), "power_decomposition.py lacks representation_features()")
    require(hasattr(pdx, "normalized_head_weights"), "power_decomposition.py lacks normalized_head_weights()")
    if hasattr(pdx, "validate_input"):
        pdx.validate_input(samples, heads)

    manifest = {
        "samples_sha256": sha256(sample_path),
        "heads_sha256": sha256(head_path),
        "decomposition_sha256": sha256(decomposition_path) if decomposition_path.is_file() else None,
        "alpha": args.alpha,
        "mc_samples": args.mc_samples,
        "seed": args.seed,
        "strict_test_rule": "bank_score > threshold",
        "read_only_inputs": True,
    }

    threshold_rows: list[dict[str, object]] = []
    fpr_rows: list[dict[str, object]] = []
    bank_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    plot_payload: list[dict[str, object]] = []
    reconstruction_failures: list[str] = []

    rng = np.random.default_rng(args.seed)

    for representation in sorted(heads):
        print(f"\n--- {representation} ---", flush=True)
        specification = heads[representation]
        representation_frame = samples[samples["representation"] == representation]
        features = list(pdx.representation_features(representation_frame, specification))
        require(features, f"{representation}: no feature columns")

        fit, fit_ids = nominal_rows(samples, representation, "nominal_fit", features)
        calibration, calibration_ids = nominal_rows(samples, representation, "nominal_calibration", features)
        test, test_ids = nominal_rows(samples, representation, "nominal_test", features)
        require(set(fit_ids).isdisjoint(calibration_ids), f"{representation}: fit/calibration overlap")
        require(set(fit_ids).isdisjoint(test_ids), f"{representation}: fit/test overlap")
        require(set(calibration_ids).isdisjoint(test_ids), f"{representation}: calibration/test overlap")

        nominal_mean = fit.mean(axis=0)
        covariance = LedoitWolf().fit(fit).covariance_
        weights = np.asarray(pdx.normalized_head_weights(covariance, specification), dtype=float)
        require(weights.ndim == 2, f"{representation}: weights are not two-dimensional")
        require(weights.shape[1] == len(features), f"{representation}: weight/feature mismatch")

        correlation = weights @ covariance @ weights.T
        diagonal_error = float(np.max(np.abs(np.diag(correlation) - 1.0)))
        require(diagonal_error < 1e-7, f"{representation}: fitted head variances are not one; max error={diagonal_error}")

        scores_by_split = {
            "nominal_fit": (fit - nominal_mean) @ weights.T,
            "nominal_calibration": (calibration - nominal_mean) @ weights.T,
            "nominal_test": (test - nominal_mean) @ weights.T,
        }
        banks = {name: values.max(axis=1) for name, values in scores_by_split.items()}

        empirical_threshold, conformal_k = conformal_threshold(
            banks["nominal_calibration"], args.alpha
        )
        gaussian_bank, minimum_eigenvalue, repair_size = gaussian_bank_draws(
            correlation, args.mc_samples, rng
        )
        gaussian_threshold = float(np.quantile(gaussian_bank, 1.0 - args.alpha))
        saved_empirical, saved_gaussian, saved_source = saved_thresholds(
            decomposition, representation
        )

        empirical_difference = (
            math.nan if saved_empirical is None else empirical_threshold - saved_empirical
        )
        reconstruction_ok = (
            saved_empirical is None
            or abs(empirical_difference) <= args.threshold_tolerance
        )
        if not reconstruction_ok:
            reconstruction_failures.append(
                f"{representation}: recomputed conformal={empirical_threshold:.12g}, "
                f"saved={saved_empirical:.12g}, difference={empirical_difference:.12g}"
            )

        off_diagonal = correlation - np.eye(len(correlation))
        maximum_abs_off_diagonal = float(np.max(np.abs(off_diagonal))) if len(correlation) > 1 else 0.0
        bonferroni_bound = float(norm.ppf(1.0 - args.alpha / len(correlation)))

        threshold_rows.append(
            {
                "representation": representation,
                "features": len(features),
                "heads": len(correlation),
                "nominal_fit_units": len(fit),
                "nominal_calibration_units": len(calibration),
                "nominal_test_units": len(test),
                "conformal_order_k": conformal_k,
                "recomputed_conformal_threshold": empirical_threshold,
                "saved_conformal_threshold": saved_empirical,
                "conformal_difference": empirical_difference,
                "conformal_reconstruction_ok": reconstruction_ok,
                "saved_threshold_source": saved_source,
                "recomputed_gaussian_threshold": gaussian_threshold,
                "saved_gaussian_threshold": saved_gaussian,
                "empirical_to_saved_gaussian_ratio": (
                    math.nan if saved_gaussian is None else empirical_threshold / saved_gaussian
                ),
                "gaussian_equivalent_exponent_at_empirical_threshold": equivalent_independent_exponent(
                    empirical_threshold, args.alpha
                ),
                "gaussian_union_bound_q95": bonferroni_bound,
                "fitted_correlation_min_eigenvalue": minimum_eigenvalue,
                "fitted_correlation_psd_repair_max_abs": repair_size,
                "fitted_head_variance_max_abs_error": diagonal_error,
                "fitted_max_abs_off_diagonal_correlation": maximum_abs_off_diagonal,
            }
        )

        for split, bank in banks.items():
            bank_rows.append(summarize_bank(representation, split, bank))
            head_rows.extend(
                summarize_heads(
                    representation,
                    split,
                    scores_by_split[split],
                    head_names(specification, weights.shape[0]),
                )
            )

        fitted_corr = np.corrcoef(scores_by_split["nominal_fit"], rowvar=False)
        for split in ["nominal_calibration", "nominal_test"]:
            empirical_corr = np.corrcoef(scores_by_split[split], rowvar=False)
            correlation_rows.append(
                {
                    "representation": representation,
                    "split": split,
                    "maximum_absolute_correlation_change_from_fit": float(
                        np.max(np.abs(empirical_corr - fitted_corr))
                    ),
                    "frobenius_correlation_change_from_fit": float(
                        np.linalg.norm(empirical_corr - fitted_corr, ord="fro")
                    ),
                }
            )

        fpr_rows.append(
            fpr_row(
                representation,
                "recomputed_conformal",
                empirical_threshold,
                banks["nominal_test"],
                gaussian_bank,
            )
        )
        fpr_rows.append(
            fpr_row(
                representation,
                "recomputed_gaussian",
                gaussian_threshold,
                banks["nominal_test"],
                gaussian_bank,
            )
        )
        if saved_empirical is not None:
            fpr_rows.append(
                fpr_row(
                    representation,
                    "saved_conformal",
                    saved_empirical,
                    banks["nominal_test"],
                    gaussian_bank,
                )
            )
        if saved_gaussian is not None:
            fpr_rows.append(
                fpr_row(
                    representation,
                    "saved_gaussian",
                    saved_gaussian,
                    banks["nominal_test"],
                    gaussian_bank,
                )
            )

        chosen_gaussian = saved_gaussian if saved_gaussian is not None else gaussian_threshold
        plot_payload.append(
            {
                "representation": representation,
                "gaussian": gaussian_bank,
                "fit": banks["nominal_fit"],
                "calibration": banks["nominal_calibration"],
                "test": banks["nominal_test"],
                "gaussian_threshold": chosen_gaussian,
                "conformal_threshold": empirical_threshold,
            }
        )

        current_fpr = [
            row
            for row in fpr_rows
            if row["representation"] == representation
        ]
        print(
            f"features={len(features)}, heads={weights.shape[0]}, "
            f"fit/cal/test={len(fit)}/{len(calibration)}/{len(test)}"
        )
        print(
            f"conformal threshold: recomputed={empirical_threshold:.6f}, "
            f"saved={saved_empirical if saved_empirical is not None else float('nan'):.6f}, "
            f"match={reconstruction_ok}"
        )
        print(
            f"Gaussian threshold: recomputed={gaussian_threshold:.6f}, "
            f"saved={saved_gaussian if saved_gaussian is not None else float('nan'):.6f}"
        )
        for row in current_fpr:
            print(
                f"{row['threshold_name']:22s} threshold={row['threshold']:.6f} "
                f"nominal-test FPR={row['nominal_test_fpr']:.6f} "
                f"({row['nominal_test_alarms']}/{row['nominal_test_units']}), "
                f"95% CI=[{row['nominal_test_ci_low']:.6f}, "
                f"{row['nominal_test_ci_high']:.6f}]"
            )

    threshold_frame = pd.DataFrame(threshold_rows)
    fpr_frame = pd.DataFrame(fpr_rows)
    bank_frame = pd.DataFrame(bank_rows)
    head_frame = pd.DataFrame(head_rows)
    correlation_frame = pd.DataFrame(correlation_rows)

    threshold_frame.to_csv(args.output_dir / "threshold_reconstruction.csv", index=False)
    fpr_frame.to_csv(args.output_dir / "threshold_fpr.csv", index=False)
    bank_frame.to_csv(args.output_dir / "bank_split_summary.csv", index=False)
    head_frame.to_csv(args.output_dir / "head_split_summary.csv", index=False)
    correlation_frame.to_csv(args.output_dir / "correlation_shift_summary.csv", index=False)
    make_figure(plot_payload, args.output_dir / "null_bank_survival.png")

    manifest["reconstruction_failures"] = reconstruction_failures
    manifest["status"] = "FAIL_RECONSTRUCTION" if reconstruction_failures else "PASS"
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== DECISIVE FPR TABLE ===")
    display_columns = [
        "representation",
        "threshold_name",
        "threshold",
        "nominal_test_alarms",
        "nominal_test_units",
        "nominal_test_fpr",
        "nominal_test_ci_low",
        "nominal_test_ci_high",
        "fitted_gaussian_tail_probability",
    ]
    print(fpr_frame[display_columns].to_string(index=False))

    print("\n=== FILES WRITTEN ===")
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(f"{path}: {path.stat().st_size:,} bytes")

    if reconstruction_failures:
        print("\nRECONSTRUCTION STATUS: FAIL")
        for failure in reconstruction_failures:
            print(f"  {failure}")
        print(
            "Do not interpret Gaussian-threshold FPRs until the active score "
            "construction is reconciled with the saved audit."
        )
        raise SystemExit(2)

    print("\nRECONSTRUCTION STATUS: PASS")
    print(
        "Next decision: inspect saved_gaussian rows in threshold_fpr.csv. "
        "Those rows answer the missing EarthNet null-FPR question."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nFATAL: {type(error).__name__}: {error}", file=sys.stderr)
        raise
