from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from power_decomposition import AuditConfig, GAP_COLUMNS, bootstrap_audit


COMPONENTS = list(GAP_COLUMNS)
FAMILIES = ["class_7", "class_3"]
REFERENCE = "forecast_joint"
COLORS = {
    "representation_gap": "#4C78A8",
    "dictionary_gap": "#F58518",
    "combination_effect": "#54A24B",
    "calibration_effect": "#E45756",
    "gaussian_residual": "#B279A2",
    "uncertain": "#D9D9D9",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point-root", type=Path,
        default=Path("outputs/esa_power_decomposition_v1"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/esa_power_decomposition_bootstrap_v1"),
    )
    parser.add_argument(
        "--figures", type=Path,
        default=Path("figures/esa_power_decomposition_bootstrap_v1"),
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--mc-samples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def verify_seal(root):
    checksum_path = root / "SHA256SUMS.txt"
    require(checksum_path.is_file(), f"Missing seal: {checksum_path}")
    verified = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        path = Path(name.strip())
        require(path.is_file(), f"Missing sealed file: {path}")
        require(sha256(path) == expected, f"Seal mismatch: {path}")
        verified += 1
    return verified


def frames_in(value):
    if isinstance(value, pd.DataFrame):
        return [value]
    if isinstance(value, dict):
        frames = []
        for child in value.values():
            frames.extend(frames_in(child))
        return frames
    if isinstance(value, (tuple, list)):
        frames = []
        for child in value:
            frames.extend(frames_in(child))
        return frames
    return []


def replicate_column(frame):
    for name in (
        "bootstrap_replicate", "bootstrap", "replicate", "draw", "boot",
    ):
        if name in frame:
            return name
    matches = [
        name for name in frame.columns
        if "bootstrap" in name.lower() or "replicate" in name.lower()
    ]
    return matches[0] if len(matches) == 1 else None


def extract_replicates(value):
    candidates = []
    for frame in frames_in(value):
        column = replicate_column(frame)
        if column is not None:
            candidates.append((len(frame), frame.copy(), column))
    require(candidates, f"No replicate frame in bootstrap return type {type(value)}")
    _, frame, column = max(candidates, key=lambda item: item[0])
    return frame, column


def run_batch(samples, heads, family, family_index, offset, count, args):
    batch_number = offset // args.batch_size
    path = args.out / "batches" / f"{family}_batch_{batch_number:03d}.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        require(frame["replicate"].nunique() == count,
                f"Incomplete existing batch: {path}")
        print(f"SKIP completed {path.name}", flush=True)
        return frame

    config = AuditConfig(
        alpha=args.alpha,
        mc_samples=args.mc_samples,
        seed=args.seed + family_index * 1_000_000 + offset,
    )
    print(
        f"RUN {family}: replicates {offset + 1}-{offset + count} "
        f"of {args.replicates}", flush=True,
    )
    value = bootstrap_audit(
        samples,
        heads,
        config,
        replicates=count,
        reference_representation=REFERENCE,
        confidence=args.confidence,
    )
    frame, replicate_name = extract_replicates(value)
    local_values = sorted(frame[replicate_name].unique().tolist())
    require(len(local_values) == count,
            f"Bootstrap returned {len(local_values)} draws, expected {count}")
    mapping = {value: offset + index for index, value in enumerate(local_values)}
    frame["replicate"] = frame[replicate_name].map(mapping).astype(int)
    if replicate_name != "replicate":
        frame = frame.drop(columns=[replicate_name])
    frame["family"] = family
    expected_rows = count * len(heads)
    require(len(frame) == expected_rows,
            f"Unexpected batch rows: {len(frame)} != {expected_rows}")
    frame.to_csv(path, index=False)
    return frame


def interval_table(point, bootstrap, confidence):
    metrics = [
        "observed_power", "oracle_power", *COMPONENTS, "total_loss",
    ]
    metrics = [name for name in metrics if name in point and name in bootstrap]
    rows = []
    for row in point.itertuples(index=False):
        mask = (
            (bootstrap["representation"] == row.representation)
            & (bootstrap["family"] == row.family)
        )
        draws = bootstrap.loc[mask]
        for metric in metrics:
            values = draws[metric].to_numpy(float)
            rows.append({
                "representation": row.representation,
                "family": row.family,
                "metric": metric,
                "estimate": float(getattr(row, metric)),
                "lower": float(np.quantile(values, (1 - confidence) / 2)),
                "upper": float(np.quantile(values, 1 - (1 - confidence) / 2)),
            })
    return pd.DataFrame(rows)


def certify(point, bootstrap, confidence):
    rows = []
    for point_row in point.itertuples(index=False):
        mask = (
            (bootstrap["representation"] == point_row.representation)
            & (bootstrap["family"] == point_row.family)
        )
        draws = (
            bootstrap.loc[mask]
            .sort_values("replicate")
            .groupby("replicate", observed=True)[COMPONENTS]
            .mean()
        )
        estimate = np.asarray(
            [getattr(point_row, name) for name in COMPONENTS], dtype=float
        )
        values = draws[COMPONENTS].to_numpy(float)
        require(len(values) >= 100, "Too few bootstrap draws for certification")
        point_difference = estimate[:, None] - estimate[None, :]
        draw_difference = values[:, :, None] - values[:, None, :]
        centered = draw_difference - point_difference[None, :, :]
        pair_mask = ~np.eye(len(COMPONENTS), dtype=bool)
        maximum_error = np.maximum(
            np.max(np.abs(values - estimate[None, :]), axis=1),
            np.max(np.abs(centered[:, pair_mask]), axis=1),
        )
        radius = float(np.quantile(maximum_error, confidence, method="higher"))
        order = np.argsort(estimate)[::-1]
        top, runner = int(order[0]), int(order[1])
        lower_zero = float(estimate[top] - radius)
        lower_runner = float(estimate[top] - estimate[runner] - radius)
        certified = bool(lower_zero > 0 and lower_runner > 0)
        if certified:
            reason = "certified"
        elif lower_zero <= 0:
            reason = "top_not_separated_from_zero"
        else:
            reason = "top_not_separated_from_runner_up"
        rows.append({
            "representation": point_row.representation,
            "family": point_row.family,
            "point_top": COMPONENTS[top],
            "reported_bottleneck": COMPONENTS[top] if certified else "uncertain",
            "runner_up": COMPONENTS[runner],
            "top_estimate": float(estimate[top]),
            "runner_up_estimate": float(estimate[runner]),
            "top_margin": float(estimate[top] - estimate[runner]),
            "simultaneous_radius": radius,
            "lower_top_vs_zero": lower_zero,
            "lower_top_vs_runner_up": lower_runner,
            "certified": certified,
            "abstention_reason": reason,
            "bootstrap_replicates": len(values),
        })
    return pd.DataFrame(rows)


def make_figure(certificates, output):
    families = FAMILIES
    representations = sorted(certificates["representation"].unique())
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    for y, representation in enumerate(representations):
        for x, family in enumerate(families):
            row = certificates[
                (certificates["representation"] == representation)
                & (certificates["family"] == family)
            ].iloc[0]
            label = row["reported_bottleneck"]
            axis.add_patch(plt.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=COLORS[label], edgecolor="white",
            ))
            text = "uncertain" if label == "uncertain" else label.replace("_", "\n")
            axis.text(x, y, text, ha="center", va="center", fontsize=8)
    axis.set_xlim(-0.5, len(families) - 0.5)
    axis.set_ylim(len(representations) - 0.5, -0.5)
    axis.set_xticks(range(len(families)), families)
    axis.set_yticks(range(len(representations)), representations)
    axis.set_title("ESA Mission-1: confidence-certified bottlenecks")
    legend = [
        Patch(facecolor=COLORS[name], label=name.replace("_", " "))
        for name in COMPONENTS + ["uncertain"]
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.17, 1, 1))
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            output / f"esa_certified_bottlenecks.{extension}",
            dpi=220, bbox_inches="tight",
        )
    plt.close(figure)


def main():
    args = parse_args()
    require(args.replicates >= 100, "Need at least 100 bootstrap replicates")
    require(args.batch_size >= 1, "Invalid batch size")
    verified = verify_seal(args.point_root)
    samples = pd.read_csv(args.point_root / "samples.csv", dtype={"unit_id": str})
    point = pd.read_csv(args.point_root / "decomposition.csv")
    heads = json.loads((args.point_root / "heads.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.point_root / "protocol.json").read_text(encoding="utf-8"))
    require(protocol["reference_representation"] == REFERENCE,
            "Reference representation changed")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "batches").mkdir(parents=True, exist_ok=True)

    batch_frames = []
    for family_index, family in enumerate(FAMILIES):
        family_samples = samples[samples["family"] == family].copy()
        require(not family_samples.empty, f"No samples for {family}")
        offset = 0
        while offset < args.replicates:
            count = min(args.batch_size, args.replicates - offset)
            batch_frames.append(run_batch(
                family_samples, heads, family, family_index,
                offset, count, args,
            ))
            offset += count

    bootstrap = pd.concat(batch_frames, ignore_index=True, sort=False)
    require(
        bootstrap.groupby("family", observed=True)["replicate"].nunique().eq(
            args.replicates
        ).all(),
        "Missing final bootstrap replicates",
    )
    maximum_error = float(bootstrap["decomposition_error"].abs().max())
    require(maximum_error <= 1e-12,
            f"Bootstrap decomposition error: {maximum_error}")
    intervals = interval_table(point, bootstrap, args.confidence)
    certificates = certify(point, bootstrap, args.confidence)

    bootstrap.to_csv(args.out / "bootstrap_replicates.csv", index=False)
    intervals.to_csv(args.out / "bootstrap_intervals.csv", index=False)
    certificates.to_csv(args.out / "certified_bottlenecks.csv", index=False)
    make_figure(certificates, args.figures)

    manifest = {
        "status": "esa_mission1_bootstrap_and_certification_complete",
        "replicates_per_family": args.replicates,
        "batch_size": args.batch_size,
        "mc_samples": args.mc_samples,
        "alpha": args.alpha,
        "confidence": args.confidence,
        "reference_representation": REFERENCE,
        "verified_point_seal_entries": verified,
        "maximum_decomposition_error": maximum_error,
        "certification_method": (
            "bootstrap simultaneous absolute component and pairwise band"
        ),
        "evaluation_status": "posthoc_decomposition_of_unsealed_evaluation",
        "point_source_sha256": sha256(args.point_root / "decomposition.csv"),
        "samples_source_sha256": sha256(args.point_root / "samples.csv"),
        "script_sha256": sha256(__file__),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in args.out.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    with (args.out / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path}\n")

    print("\n=== ESA BOOTSTRAP INTERVALS ===")
    print(intervals[intervals["metric"].isin(COMPONENTS)].to_string(index=False))
    print("\n=== ESA CERTIFIED OR ABSTAIN ===")
    print(certificates.to_string(index=False))
    print("\nMaximum decomposition error:", f"{maximum_error:.16g}")
    print("FINAL ESA BOOTSTRAP-CERTIFICATION STATUS: PASS")


if __name__ == "__main__":
    main()
