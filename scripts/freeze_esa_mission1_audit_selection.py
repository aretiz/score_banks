from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import power_decomposition as pdx


ROOT = Path("outputs/esa_real_event_readiness")
FORECAST_ROOT = ROOT / "forecast_representations"
FEATURES_PATH = FORECAST_ROOT / "unit_features.csv"
NLL_PATH = FORECAST_ROOT / "heldout_nll.csv"
FORECAST_SEAL_PATH = FORECAST_ROOT / "SHA256SUMS.txt"
FROZEN_HEADS_PATH = ROOT / "mission1_frozen_metadata_heads.json"
MODEL_PROTOCOL_PATH = ROOT / "mission1_frozen_model_protocol.json"
MODEL_PROTOCOL_SEAL_PATH = ROOT / "FROZEN_MODEL_PROTOCOL_SHA256SUMS.txt"

OUTPUT = ROOT / "frozen_audit_selection"
DEVELOPMENT_PATH = OUTPUT / "development_unit_features.csv"
EVALUATION_PATH = OUTPUT / "SEALED_evaluation_unit_features.csv"
AUDIT_HEADS_PATH = OUTPUT / "audit_heads.json"
CANDIDATE_PATH = OUTPUT / "candidate_design_metrics.csv"
SELECTION_PATH = OUTPUT / "frozen_representation_selections.csv"
MANIFEST_PATH = OUTPUT / "manifest.json"
SEAL_PATH = OUTPUT / "SHA256SUMS.txt"

DEVELOPMENT_ROLES = {
    "audit_nominal_fit",
    "anomaly_design",
    "paired_nominal_design",
    "nominal_calibration",
}
EVALUATION_ROLES = {
    "nominal_test",
    "anomaly_test",
}
IGNORED_ROLES = {
    "forecaster_fit",
    "nll_validation",
}
CANDIDATES = [
    "persistence",
    "univariate_ar1",
    "cross_channel_ridge",
]
REFERENCE = "forecast_joint"
FAMILIES = ["class_7", "class_3"]
ALPHA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(path: Path) -> int:
    checked = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        expected, filename = line.split(maxsplit=1)
        candidate = Path(filename.strip())
        if not candidate.exists():
            raise FileNotFoundError(f"Sealed file is missing: {candidate}")
        observed = sha256(candidate)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {candidate}: "
                f"expected {expected}, observed {observed}"
            )
        checked += 1
    if checked == 0:
        raise ValueError(f"Empty checksum file: {path}")
    return checked


def write_csv_rows(source: Path, development: Path, evaluation: Path):
    counts: dict[str, int] = {}
    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None or "split_role" not in reader.fieldnames:
            raise ValueError("unit_features.csv lacks split_role")
        with development.open("w", encoding="utf-8", newline="") as dev_handle:
            with evaluation.open("w", encoding="utf-8", newline="") as eval_handle:
                dev_writer = csv.DictWriter(dev_handle, fieldnames=reader.fieldnames)
                eval_writer = csv.DictWriter(eval_handle, fieldnames=reader.fieldnames)
                dev_writer.writeheader()
                eval_writer.writeheader()
                for row in reader:
                    role = row["split_role"]
                    counts[role] = counts.get(role, 0) + 1
                    if role in DEVELOPMENT_ROLES:
                        dev_writer.writerow(row)
                    elif role in EVALUATION_ROLES:
                        eval_writer.writerow(row)
                    elif role not in IGNORED_ROLES:
                        raise ValueError(f"Unexpected frozen model role: {role}")
    observed_roles = set(counts)
    expected_roles = DEVELOPMENT_ROLES | EVALUATION_ROLES | IGNORED_ROLES
    if observed_roles != expected_roles:
        raise ValueError(
            f"Feature roles changed: expected {sorted(expected_roles)}, "
            f"observed {sorted(observed_roles)}"
        )
    return counts


def head_specification(
    frozen_representation,
    feature_frame,
):
    feature_names = list(
        frozen_representation["feature_names"]
    )

    directions = [
        head["direction"]
        for head in frozen_representation["heads"]
    ]

    names = [
        head["name"]
        for head in frozen_representation["heads"]
    ]

    kinds = [
        head["kind"]
        for head in frozen_representation["heads"]
    ]

    matrix = np.asarray(
        directions,
        dtype=float,
    )

    expected_shape = (
        len(names),
        len(feature_names),
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            "Frozen head matrix shape does not "
            "match frozen feature names: "
            f"expected {expected_shape}, "
            f"observed {matrix.shape}"
        )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "Frozen representation contains "
            "duplicate feature names"
        )

    missing = [
        name
        for name in feature_names
        if name not in feature_frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Frozen features are absent from "
            "unit_features.csv: "
            f"{missing[:10]}"
        )

    specification = {
        "feature_columns": feature_names,
        "directions": directions,
        "head_names": names,
        "head_kinds": kinds,
    }

    return (
        "feature_columns",
        specification,
    )

def unique_nominal(
    frame,
    role,
    family,
    feature_names,
):
    subset = frame[
        (frame["split_role"] == role)
        & (frame["target_family"] == family)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No rows for {role}, {family}"
        )

    if subset["unit_id"].duplicated().any():
        examples = (
            subset.loc[
                subset["unit_id"].duplicated(
                    keep=False
                ),
                "unit_id",
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Duplicated nominal units within "
            f"{role}, {family}: {examples}"
        )

    if subset[feature_names].isna().any().any():
        raise ValueError(
            f"Missing nominal features for "
            f"{role}, {family}"
        )

    values = subset[
        feature_names
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite nominal features for "
            f"{role}, {family}"
        )

    return (
        subset.sort_values(
            "unit_id",
            kind="stable",
        )
        .reset_index(drop=True)
    )

def paired_design(frame, family, feature_names):
    design = frame[
        (frame["split_role"] == "anomaly_design")
        & (frame["target_family"] == family)
    ].copy()
    controls = frame[
        (frame["split_role"] == "paired_nominal_design")
        & (frame["target_family"] == family)
    ].copy()
    if design.empty or controls.empty:
        raise ValueError(f"Missing design or paired controls for {family}")
    if design["pair_id"].isna().any() or controls["pair_id"].isna().any():
        raise ValueError(f"Missing pair IDs for {family}")
    if design["pair_id"].duplicated().any():
        raise ValueError(f"Duplicate anomaly-design pair IDs for {family}")
    if controls["pair_id"].duplicated().any():
        raise ValueError(f"Duplicate paired-control pair IDs for {family}")

    control_columns = ["pair_id", "unit_id", *feature_names]
    merged = design.merge(
        controls[control_columns],
        on="pair_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "__paired"),
    )
    if len(merged) != len(design):
        raise AssertionError("Paired design merge changed row count")
    paired_columns = [f"{name}__paired" for name in feature_names]
    if merged[paired_columns].isna().any().any():
        raise ValueError(f"Unmatched paired controls for {family}")
    return design, merged


def conformal_threshold(scores, alpha):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Invalid calibration scores")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


def normalized_weights(covariance, directions):
    directions = np.asarray(directions, dtype=float)
    raw = np.linalg.solve(covariance, directions.T).T
    variance = np.einsum("hi,ij,hj->h", raw, covariance, raw)
    if (variance <= 0).any() or not np.isfinite(variance).all():
        raise ValueError("Invalid frozen-head variances")
    return raw / np.sqrt(variance)[:, None]


def representation_metrics(
    development,
    family,
    representation,
    specification,
    heldout_nll,
):
    feature_names = list(
        specification["feature_columns"]
    )
    fit = unique_nominal(
        development,
        "audit_nominal_fit",
        family,
        feature_names,
    )
    calibration = unique_nominal(
        development,
        "nominal_calibration",
        family,
        feature_names,
    )
    design, paired = paired_design(development, family, feature_names)

    x_fit = fit[feature_names].to_numpy(dtype=float)
    x_cal = calibration[feature_names].to_numpy(dtype=float)
    x_design = design[feature_names].to_numpy(dtype=float)
    paired_names = [f"{name}__paired" for name in feature_names]
    x_paired = paired[paired_names].to_numpy(dtype=float)

    nominal_mean = x_fit.mean(axis=0)
    covariance = LedoitWolf().fit(x_fit).covariance_
    shift = np.mean(x_design - x_paired, axis=0)
    precision_shift = np.linalg.solve(covariance, shift)
    gamma = float(shift @ precision_shift)

    directions = np.asarray(specification["directions"], dtype=float)
    weights = normalized_weights(covariance, directions)
    calibration_bank = ((x_cal - nominal_mean) @ weights.T).max(axis=1)
    design_bank = ((x_design - nominal_mean) @ weights.T).max(axis=1)
    threshold = conformal_threshold(calibration_bank, ALPHA)
    design_power = float(np.mean(design_bank > threshold))

    response = weights @ shift
    best_index = int(np.argmax(response))
    head_names = specification["head_names"]

    return {
        "target_family": family,
        "representation": representation,
        "eligible_candidate": representation in CANDIDATES,
        "heldout_nll": heldout_nll,
        "gamma_hat": gamma,
        "sqrt_gamma_hat": float(np.sqrt(max(0.0, gamma))),
        "design_bank_power": design_power,
        "calibration_threshold": threshold,
        "nominal_fit_units": len(fit),
        "calibration_units": len(calibration),
        "design_units": len(design),
        "head_count": directions.shape[0],
        "best_design_head_index": best_index,
        "best_design_head": head_names[best_index],
        "best_design_head_response": float(response[best_index]),
    }


def select_candidates(metrics):
    rows = []
    for family in FAMILIES:
        family_metrics = metrics[
            (metrics["target_family"] == family)
            & metrics["eligible_candidate"]
        ].copy()
        if set(family_metrics["representation"]) != set(CANDIDATES):
            raise ValueError(f"Candidate representations changed for {family}")

        rules = {
            "minimum_nll": family_metrics.sort_values(
                ["heldout_nll", "representation"],
                ascending=[True, True],
                kind="mergesort",
            ).iloc[0],
            "maximum_gamma": family_metrics.sort_values(
                ["gamma_hat", "representation"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0],
            "direct_design_power": family_metrics.sort_values(
                ["design_bank_power", "gamma_hat", "representation"],
                ascending=[False, False, True],
                kind="mergesort",
            ).iloc[0],
        }
        for rule, selected in rules.items():
            rows.append(
                {
                    "target_family": family,
                    "analysis_tier": (
                        "primary_confirmatory"
                        if family == "class_7"
                        else "secondary_exploratory"
                    ),
                    "selection_rule": rule,
                    "selected_representation": selected["representation"],
                    "selected_heldout_nll": selected["heldout_nll"],
                    "selected_gamma_hat": selected["gamma_hat"],
                    "selected_design_bank_power": selected["design_bank_power"],
                    "tie_breaking": (
                        "lexicographic"
                        if rule != "direct_design_power"
                        else "gamma_then_lexicographic"
                    ),
                }
            )
    return pd.DataFrame(rows)


def main():
    required = [
        FEATURES_PATH,
        NLL_PATH,
        FORECAST_SEAL_PATH,
        FROZEN_HEADS_PATH,
        MODEL_PROTOCOL_PATH,
        MODEL_PROTOCOL_SEAL_PATH,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    print("=== FREEZE ESA MISSION-1 AUDIT SELECTION ===")
    print("anomaly-test feature values used in selection: NO")
    print("nominal-test feature values used in selection: NO")
    print("evaluation feature values printed: NO")
    print("test power calculated: NO")
    print("test FPR calculated: NO")
    forecast_entries = verify_seal(FORECAST_SEAL_PATH)
    protocol_entries = verify_seal(MODEL_PROTOCOL_SEAL_PATH)
    print("verified forecast-representation seal entries:", forecast_entries)
    print("verified model-protocol seal entries:", protocol_entries)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    role_counts = write_csv_rows(
        FEATURES_PATH, DEVELOPMENT_PATH, EVALUATION_PATH
    )
    print("\n=== STREAMING ROLE SEPARATION ===")
    for role in sorted(role_counts):
        print(f"{role:26s} {role_counts[role]:4d}")

    development = pd.read_csv(
        DEVELOPMENT_PATH,
        dtype={"unit_id": str, "pair_id": str, "source_cluster_id": str},
    )
    if set(development["split_role"]) != DEVELOPMENT_ROLES:
        raise ValueError("Development file contains an incorrect role set")

    frozen = json.loads(FROZEN_HEADS_PATH.read_text(encoding="utf-8"))
    frozen_representations = frozen["representations"]
    expected_representations = set(CANDIDATES) | {REFERENCE}
    if set(frozen_representations) != expected_representations:
        raise ValueError("Frozen representation set changed")

    heads = {}
    feature_binding = {}
    for representation in sorted(frozen_representations):
        key, specification = head_specification(
            frozen_representations[representation], development
        )
        heads[representation] = specification
        feature_binding[representation] = key

    AUDIT_HEADS_PATH.write_text(
        json.dumps(heads, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    nll = pd.read_csv(NLL_PATH)
    overall = nll[nll["target"] == "__overall__"].copy()
    nll_map = overall.set_index("model")["heldout_nll"].to_dict()
    if set(nll_map) != set(CANDIDATES):
        raise ValueError("Held-out NLL candidate set changed")

    metric_rows = []
    for family in FAMILIES:
        for representation in [*CANDIDATES, REFERENCE]:
            metric_rows.append(
                representation_metrics(
                    development,
                    family,
                    representation,
                    heads[representation],
                    float(nll_map[representation])
                    if representation in nll_map
                    else np.nan,
                )
            )
    metrics = pd.DataFrame(metric_rows)
    selections = select_candidates(metrics)
    metrics.to_csv(CANDIDATE_PATH, index=False)
    selections.to_csv(SELECTION_PATH, index=False)

    print("\n=== DEVELOPMENT-ONLY CANDIDATE METRICS ===")
    print(
        metrics[
            [
                "target_family",
                "representation",
                "eligible_candidate",
                "heldout_nll",
                "gamma_hat",
                "design_bank_power",
                "best_design_head",
            ]
        ].to_string(index=False)
    )
    print("\n=== FROZEN REPRESENTATION SELECTIONS ===")
    print(selections.to_string(index=False))

    manifest = {
        "status": "frozen_before_evaluation_features_are_analyzed",
        "dataset": "ESA-Mission1",
        "primary_family": "class_7",
        "secondary_family": "class_3",
        "candidate_representations": CANDIDATES,
        "reference_representation": REFERENCE,
        "selection_rules": [
            "minimum_nll",
            "maximum_gamma",
            "direct_design_power",
        ],
        "selection_rule_tie_breaking": {
            "minimum_nll": "minimum NLL, then representation name",
            "maximum_gamma": "maximum gamma, then representation name",
            "direct_design_power": (
                "maximum design bank power, then maximum gamma, "
                "then representation name"
            ),
        },
        "alpha": ALPHA,
        "development_roles": sorted(DEVELOPMENT_ROLES),
        "sealed_evaluation_roles": sorted(EVALUATION_ROLES),
        "ignored_upstream_roles": sorted(IGNORED_ROLES),
        "feature_binding_key": feature_binding,
        "anomaly_test_values_used_in_selection": False,
        "nominal_test_values_used_in_selection": False,
        "evaluation_feature_values_printed": False,
        "test_power_calculated": False,
        "test_fpr_calculated": False,
        "sources": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in required
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sealed = [
        DEVELOPMENT_PATH,
        EVALUATION_PATH,
        AUDIT_HEADS_PATH,
        CANDIDATE_PATH,
        SELECTION_PATH,
        MANIFEST_PATH,
    ]
    SEAL_PATH.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in sealed),
        encoding="utf-8",
    )

    print("\n=== FILES WRITTEN ===")
    for path in sealed + [SEAL_PATH]:
        print(f"{path}: {path.stat().st_size:,} bytes")
    print("\nFINAL AUDIT-SELECTION SEAL: PASS")


if __name__ == "__main__":
    main()
