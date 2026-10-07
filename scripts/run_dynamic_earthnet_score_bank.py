#!/usr/bin/env python3
"""DynamicEarthNet real-change validation for a calibrated score bank."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import beta, norm
from sklearn.metrics import average_precision_score, roc_auc_score


AOI_PATTERN = re.compile(r"^(\d+_\d+_\d+)")
DATE_PATTERN = re.compile(r"(20\d{2})[-_](\d{2})")
HEADS = tuple(f"band_{number:02d}" for number in range(1, 13))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/dynamic_earthnet_score_bank"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--positive-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reuse-features", action="store_true")
    return parser.parse_args()


def aoi_id(name: str) -> str | None:
    match = AOI_PATTERN.match(name)
    return match.group(1) if match else None


def date_key(path: Path) -> tuple[int, int] | None:
    match = DATE_PATTERN.search(path.stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def discover(root: Path):
    sentinel_root = root / "sentinel2"
    label_root = root / "labels"
    sentinel = {
        aoi_id(path.name): path
        for path in sentinel_root.iterdir()
        if path.is_dir() and aoi_id(path.name)
    }
    labels = {
        aoi_id(path.name): path
        for path in label_root.iterdir()
        if path.is_dir() and aoi_id(path.name)
    }
    matched = sorted(set(sentinel) & set(labels))
    if len(matched) < 40:
        raise RuntimeError(
            f"Only {len(matched)} AOIs have both imagery and labels under {root}"
        )
    return matched, sentinel, labels


def indexed_files(directory: Path, recursive: bool) -> dict[tuple[int, int], Path]:
    paths = directory.rglob("*.tif") if recursive else directory.glob("*.tif")
    output = {}
    for path in paths:
        key = date_key(path)
        if key is not None:
            output[key] = path
    return output


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read(out_dtype="float32")


def robust_residual(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    if reference.shape != current.shape or reference.shape[0] != 12:
        raise RuntimeError(
            f"Expected matching 12-band rasters, found {reference.shape} and {current.shape}"
        )
    output = np.full(reference.shape, np.nan, dtype=np.float32)
    for band in range(reference.shape[0]):
        before = reference[band]
        after = current[band]
        valid = (
            np.isfinite(before)
            & np.isfinite(after)
            & (before > 0)
            & (after > 0)
        )
        if valid.sum() < 100:
            continue
        before_values = before[valid]
        after_values = after[valid]
        b10, b50, b90 = np.quantile(before_values, (0.10, 0.50, 0.90))
        a10, a50, a90 = np.quantile(after_values, (0.10, 0.50, 0.90))
        before_scale = max(float(b90 - b10), 1.0)
        after_scale = max(float(a90 - a10), 1.0)
        aligned = (after - a50) * (before_scale / after_scale) + b50
        difference = np.abs(aligned - before) / before_scale
        center = float(np.median(difference[valid]))
        mad = float(np.median(np.abs(difference[valid] - center)))
        scale = max(1.4826 * mad, 1e-6)
        standardized = np.maximum((difference - center) / scale, 0.0)
        output[band, valid] = standardized[valid]
    return output


def block_sum(values: np.ndarray, patch: int) -> np.ndarray:
    bands, height, width = values.shape
    height = (height // patch) * patch
    width = (width // patch) * patch
    cropped = values[:, :height, :width]
    return cropped.reshape(bands, height // patch, patch, width // patch, patch).sum(
        axis=(2, 4)
    )


def block_sum_2d(values: np.ndarray, patch: int) -> np.ndarray:
    height, width = values.shape
    height = (height // patch) * patch
    width = (width // patch) * patch
    cropped = values[:height, :width]
    return cropped.reshape(height // patch, patch, width // patch, patch).sum(
        axis=(1, 3)
    )


def extract_pair(
    aoi: str,
    month: int,
    before_image_path: Path,
    after_image_path: Path,
    before_label_path: Path,
    after_label_path: Path,
    patch: int,
    positive_fraction: float,
    minimum_valid_fraction: float,
) -> pd.DataFrame:
    before_image = read_raster(before_image_path)
    after_image = read_raster(after_image_path)
    before_label = read_raster(before_label_path)
    after_label = read_raster(after_label_path)
    if before_label.shape[0] != 7 or after_label.shape[0] != 7:
        raise RuntimeError("DynamicEarthNet labels must contain seven one-hot bands")
    if before_image.shape[1:] != before_label.shape[1:]:
        raise RuntimeError("Sentinel-2 and label rasters do not share a grid")

    label_valid = (before_label.sum(axis=0) > 0) & (after_label.sum(axis=0) > 0)
    image_valid = (
        np.isfinite(before_image).all(axis=0)
        & np.isfinite(after_image).all(axis=0)
        & (before_image > 0).all(axis=0)
        & (after_image > 0).all(axis=0)
    )
    valid = label_valid & image_valid
    before_class = np.argmax(before_label, axis=0)
    after_class = np.argmax(after_label, axis=0)
    changed = valid & (before_class != after_class)

    residual = robust_residual(before_image, after_image)
    finite = np.isfinite(residual).all(axis=0) & valid
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    counts = block_sum_2d(finite.astype(np.float32), patch)
    changed_counts = block_sum_2d((changed & finite).astype(np.float32), patch)
    residual_sums = block_sum(residual * finite[None, :, :], patch)
    means = residual_sums / np.maximum(counts[None, :, :], 1.0)

    total = float(patch * patch)
    records = []
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            valid_fraction = float(counts[row, column] / total)
            if valid_fraction < minimum_valid_fraction:
                continue
            change_fraction = float(
                changed_counts[row, column] / max(counts[row, column], 1.0)
            )
            if 0.0 < change_fraction < positive_fraction:
                continue
            features = means[:, row, column]
            if not np.isfinite(features).all():
                continue
            records.append(
                {
                    "aoi": aoi,
                    "month": month,
                    "row": row,
                    "column": column,
                    "valid_fraction": valid_fraction,
                    "change_fraction": change_fraction,
                    "label": int(change_fraction >= positive_fraction),
                    **dict(zip(HEADS, features)),
                }
            )
    return pd.DataFrame.from_records(records)


def build_features(root: Path, out: Path, config) -> pd.DataFrame:
    matched, sentinel, labels = discover(root)
    cache = out / "aoi_features"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    coverage = []
    for number, aoi in enumerate(matched, 1):
        cache_path = cache / f"{aoi}.csv"
        if cache_path.is_file():
            print(f"AOI {number}/{len(matched)} {aoi} REUSED", flush=True)
            frame = pd.read_csv(cache_path)
            frames.append(frame)
            coverage.append(
                {"aoi": aoi, "usable_months": int(frame["month"].nunique())}
            )
            continue
        print(f"AOI {number}/{len(matched)} {aoi}", flush=True)
        image_files = indexed_files(sentinel[aoi], recursive=False)
        label_files = indexed_files(labels[aoi], recursive=True)
        usable_months = []
        aoi_frames = []
        common_dates = sorted(set(image_files) & set(label_files))
        for before_date, after_date in zip(common_dates, common_dates[1:]):
            before_index = before_date[0] * 12 + before_date[1]
            after_index = after_date[0] * 12 + after_date[1]
            if after_index - before_index != 1:
                continue
            target_period = after_date[0] * 100 + after_date[1]
            usable_months.append(target_period)
            frame = extract_pair(
                aoi=aoi,
                month=target_period,
                before_image_path=image_files[before_date],
                after_image_path=image_files[after_date],
                before_label_path=label_files[before_date],
                after_label_path=label_files[after_date],
                patch=config.patch,
                positive_fraction=config.positive_fraction,
                minimum_valid_fraction=config.minimum_valid_fraction,
            )
            if len(frame):
                aoi_frames.append(frame)
        if aoi_frames:
            aoi_frame = pd.concat(aoi_frames, ignore_index=True)
            aoi_frame.to_csv(cache_path, index=False)
            frames.append(aoi_frame)
        coverage.append({"aoi": aoi, "usable_months": len(usable_months)})
    pd.DataFrame(coverage).to_csv(out / "aoi_coverage.csv", index=False)
    if not frames:
        raise RuntimeError("No matched annual image-label pairs were found")
    features = pd.concat(frames, ignore_index=True)
    features.to_csv(out / "patch_features.csv", index=False)
    return features


def assign_splits(features: pd.DataFrame, seed: int):
    aois = sorted(features["aoi"].unique())
    if len(aois) < 40:
        raise RuntimeError(f"Only {len(aois)} AOIs yielded usable patch features")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(aois))
    split = {
        "fit": shuffled[:18],
        "design": shuffled[18:30],
        "calibration": shuffled[30:42],
        "test": shuffled[42:],
    }
    role = {aoi: name for name, members in split.items() for aoi in members}
    output = features.copy()
    output.insert(0, "split", output["aoi"].map(role))
    return output, split


def empirical_pvalues(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    pvalues = np.empty_like(values, dtype=float)
    for column in range(reference.shape[1]):
        ordered = np.sort(reference[:, column])
        at_least = len(ordered) - np.searchsorted(
            ordered, values[:, column], side="left"
        )
        pvalues[:, column] = (at_least + 1.0) / (len(ordered) + 1.0)
    return np.clip(pvalues, 1e-12, 1.0 - 1e-12)


def method_scores(pvalues: np.ndarray, best_head: int):
    zscores = norm.isf(pvalues)
    ordered = np.sort(pvalues, axis=1)
    count = pvalues.shape[1]
    fractions = np.arange(1, count + 1, dtype=float) / count
    eligible = (ordered >= 1.0 / count) & (ordered <= 0.5)
    terms = np.sqrt(count) * (fractions[None, :] - ordered)
    terms /= np.sqrt(np.maximum(ordered * (1.0 - ordered), 1e-12))
    terms[~eligible] = -np.inf
    higher_criticism = np.max(terms, axis=1)
    higher_criticism[~np.isfinite(higher_criticism)] = 0.0
    return {
        "max_bank": np.max(zscores, axis=1),
        "mean_bank": np.mean(zscores, axis=1),
        "cauchy_bank": np.mean(np.tan(np.pi * (0.5 - pvalues)), axis=1),
        "higher_criticism": higher_criticism,
        "best_single": zscores[:, best_head],
    }


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def exact_interval(successes: int, units: int):
    tail = 0.025
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, units - successes + 1))
    high = 1.0 if successes == units else float(beta.ppf(1.0 - tail, successes + 1, units - successes))
    return low, high


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method, frame in predictions.groupby("method", sort=False):
        for metric, label in (("nominal_test_fpr", 0), ("power", 1)):
            subset = frame[frame["label"] == label]
            successes = int(subset["rejected"].sum())
            units = len(subset)
            low, high = exact_interval(successes, units)
            records.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": successes / units,
                    "lower_95": low,
                    "upper_95": high,
                    "successes": successes,
                    "units": units,
                }
            )
        records.extend(
            [
                {
                    "method": method,
                    "metric": "auroc",
                    "estimate": roc_auc_score(frame["label"], frame["score"]),
                    "lower_95": np.nan,
                    "upper_95": np.nan,
                    "successes": np.nan,
                    "units": len(frame),
                },
                {
                    "method": method,
                    "metric": "auprc",
                    "estimate": average_precision_score(frame["label"], frame["score"]),
                    "lower_95": np.nan,
                    "upper_95": np.nan,
                    "successes": np.nan,
                    "units": len(frame),
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def cluster_bootstrap(predictions: pd.DataFrame, repetitions: int, seed: int):
    rng = np.random.default_rng(seed)
    aois = predictions["aoi"].unique()
    methods = predictions["method"].unique()
    draws = {
        method: {"fpr": [], "power": [], "auroc": [], "auprc": []}
        for method in methods
    }
    for _ in range(repetitions):
        sampled = rng.choice(aois, size=len(aois), replace=True)
        for method in methods:
            source = predictions[predictions["method"] == method]
            frame = pd.concat(
                [source[source["aoi"] == aoi] for aoi in sampled],
                ignore_index=True,
            )
            negative = frame[frame["label"] == 0]
            positive = frame[frame["label"] == 1]
            if not len(negative) or not len(positive):
                continue
            draws[method]["fpr"].append(float(negative["rejected"].mean()))
            draws[method]["power"].append(float(positive["rejected"].mean()))
            draws[method]["auroc"].append(
                float(roc_auc_score(frame["label"], frame["score"]))
            )
            draws[method]["auprc"].append(
                float(average_precision_score(frame["label"], frame["score"]))
            )
    records = []
    for method in methods:
        for metric, values in draws[method].items():
            values = np.asarray(values, dtype=float)
            records.append(
                {
                    "method": method,
                    "metric": metric,
                    "lower_95": float(np.quantile(values, 0.025)),
                    "upper_95": float(np.quantile(values, 0.975)),
                    "successful_draws": len(values),
                }
            )
    return pd.DataFrame.from_records(records)


def main():
    config = parse_args()
    if not 0.0 < config.alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    config.out.mkdir(parents=True, exist_ok=True)
    feature_path = config.out / "patch_features.csv"
    if config.reuse_features and feature_path.is_file():
        print(f"Reusing {feature_path}")
        features = pd.read_csv(feature_path)
    else:
        features = build_features(config.data, config.out, config)

    data, split = assign_splits(features, config.seed)
    data.to_csv(config.out / "patch_features_with_split.csv", index=False)
    (config.out / "aoi_split.json").write_text(json.dumps(split, indent=2))

    fit_nominal = data[(data["split"] == "fit") & (data["label"] == 0)]
    design = data[data["split"] == "design"]
    calibration = data[(data["split"] == "calibration") & (data["label"] == 0)]
    test = data[data["split"] == "test"]
    for name, frame in {
        "fit nominal": fit_nominal,
        "design": design,
        "calibration nominal": calibration,
        "test": test,
    }.items():
        if not len(frame):
            raise RuntimeError(f"No usable observations in {name}")
    if design["label"].nunique() != 2 or test["label"].nunique() != 2:
        raise RuntimeError("Design and test splits must contain both unchanged and changed patches")

    reference = fit_nominal.loc[:, HEADS].to_numpy(float)
    design_p = empirical_pvalues(reference, design.loc[:, HEADS].to_numpy(float))
    design_z = norm.isf(design_p)
    design_auc = np.asarray(
        [roc_auc_score(design["label"], design_z[:, j]) for j in range(len(HEADS))]
    )
    best_head = int(np.argmax(design_auc))
    pd.DataFrame({"head": HEADS, "design_auroc": design_auc}).sort_values(
        "design_auroc", ascending=False
    ).to_csv(config.out / "head_design_auroc.csv", index=False)

    calibration_p = empirical_pvalues(
        reference, calibration.loc[:, HEADS].to_numpy(float)
    )
    test_p = empirical_pvalues(reference, test.loc[:, HEADS].to_numpy(float))
    calibration_scores = method_scores(calibration_p, best_head)
    test_scores = method_scores(test_p, best_head)
    thresholds = {
        method: conformal_threshold(scores, config.alpha)
        for method, scores in calibration_scores.items()
    }

    identity = test[
        ["aoi", "month", "row", "column", "change_fraction", "label"]
    ].reset_index(drop=True)
    prediction_frames = []
    for method, scores in test_scores.items():
        frame = identity.copy()
        frame.insert(0, "method", method)
        frame["score"] = scores
        frame["threshold"] = thresholds[method]
        frame["rejected"] = frame["score"] > frame["threshold"]
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(config.out / "predictions.csv", index=False)

    metrics = metric_table(predictions)
    metrics.to_csv(config.out / "metrics.csv", index=False)
    bootstrap = cluster_bootstrap(predictions, config.bootstrap, config.seed + 1)
    bootstrap.to_csv(config.out / "aoi_cluster_bootstrap.csv", index=False)

    calibration_z = norm.isf(calibration_p)
    correlation = np.corrcoef(calibration_z, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    effective_rank = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    counts = (
        data.groupby(["split", "label"]).size().rename("count").reset_index()
    )
    summary = {
        "dataset": "DynamicEarthNet Sentinel-2",
        "forecast": "previous-month persistence",
        "target": "monthly land-cover class change",
        "real_change_labels": True,
        "labels_used_for_forecast": False,
        "labels_used_for_head_design": True,
        "labels_used_to_select_nominal_calibration_patches": True,
        "alpha": config.alpha,
        "patch_size": config.patch,
        "positive_change_fraction": config.positive_fraction,
        "n_aois": int(data["aoi"].nunique()),
        "n_heads": len(HEADS),
        "best_single_head": HEADS[best_head],
        "effective_rank": effective_rank,
        "thresholds": thresholds,
        "counts": counts.to_dict("records"),
    }
    (config.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DYNAMICEARTHNET RESULTS ===")
    print(metrics.to_string(index=False))
    print(f"\nBest single head: {HEADS[best_head]}")
    print(f"Effective rank: {effective_rank:.3f}")
    print(f"Wrote: {config.out}")


if __name__ == "__main__":
    main()
