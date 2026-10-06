from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "outputs/esa_real_event_readiness"
)

DEV_PATH = (
    ROOT
    / "frozen_audit_selection"
    / "development_unit_features.csv"
)

EVAL_PATH = (
    ROOT
    / "frozen_audit_selection"
    / "SEALED_evaluation_unit_features.csv"
)

SOURCE_PATH = (
    ROOT
    / "forecast_representations"
    / "unit_features.csv"
)

HEAD_PATH = (
    ROOT
    / "frozen_audit_selection"
    / "audit_heads.json"
)

OUTPUT = Path(
    "outputs/esa_feature_lineage_audit_v1"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

development = pd.read_csv(DEV_PATH)
evaluation = pd.read_csv(EVAL_PATH)
source_features = pd.read_csv(SOURCE_PATH)

heads = json.loads(
    HEAD_PATH.read_text(encoding="utf-8")
)

feature_columns = sorted({
    column
    for specification in heads.values()
    for column in specification[
        "feature_columns"
    ]
})

TOLERANCE = 1e-8


def family_invariance(
    frame,
    source_name,
):
    available = [
        column
        for column in feature_columns
        if column in frame.columns
    ]

    rows = []

    for unit_id, group in frame.groupby(
        "unit_id",
        sort=True,
    ):
        if (
            "target_family" not in group.columns
            or group[
                "target_family"
            ].nunique() < 2
        ):
            continue

        values = group[
            available
        ].to_numpy(float)

        reference = values[0]

        close = np.isclose(
            values,
            reference[None, :],
            rtol=TOLERANCE,
            atol=TOLERANCE,
            equal_nan=True,
        )

        differing = ~close

        absolute_difference = np.abs(
            values - reference[None, :]
        )

        absolute_difference[
            ~np.isfinite(absolute_difference)
        ] = 0.0

        maximum = float(
            absolute_difference.max()
        )

        flat_index = int(
            absolute_difference.argmax()
        )

        _, feature_index = np.unravel_index(
            flat_index,
            absolute_difference.shape,
        )

        rows.append({
            "source_file": source_name,
            "unit_id": unit_id,
            "rows": len(group),
            "families": "|".join(
                sorted(
                    group[
                        "target_family"
                    ].astype(str).unique()
                )
            ),
            "roles": "|".join(
                sorted(
                    group[
                        "split_role"
                    ].astype(str).unique()
                )
            )
            if "split_role" in group.columns
            else "",
            "feature_columns_checked": (
                len(available)
            ),
            "differing_values": int(
                differing.sum()
            ),
            "maximum_absolute_difference": (
                maximum
            ),
            "largest_difference_feature": (
                available[feature_index]
            ),
            "family_invariant": bool(
                not differing.any()
            ),
        })

    return pd.DataFrame(rows)


invariance = pd.concat(
    [
        family_invariance(
            development,
            "development",
        ),
        family_invariance(
            evaluation,
            "evaluation",
        ),
        family_invariance(
            source_features,
            "source_unit_features",
        ),
    ],
    ignore_index=True,
)

combined = pd.concat(
    [
        development.assign(
            source_partition="development"
        ),
        evaluation.assign(
            source_partition="evaluation"
        ),
    ],
    ignore_index=True,
)

overlap_rows = []

for role in sorted(
    combined["split_role"].unique()
):
    role_frame = combined[
        combined["split_role"] == role
    ]

    class_7 = set(
        role_frame[
            role_frame["target_family"]
            == "class_7"
        ]["unit_id"]
    )

    class_3 = set(
        role_frame[
            role_frame["target_family"]
            == "class_3"
        ]["unit_id"]
    )

    union = class_7 | class_3
    intersection = class_7 & class_3

    overlap_rows.append({
        "split_role": role,
        "class_7_units": len(class_7),
        "class_3_units": len(class_3),
        "shared_units": len(intersection),
        "class_7_only": len(
            class_7 - class_3
        ),
        "class_3_only": len(
            class_3 - class_7
        ),
        "jaccard_overlap": (
            len(intersection) / len(union)
            if union
            else np.nan
        ),
    })

role_overlap = pd.DataFrame(
    overlap_rows
)

scale_rows = []

nominal_roles = [
    "audit_nominal_fit",
    "nominal_calibration",
    "nominal_test",
]

for family in [
    "class_7",
    "class_3",
]:
    family_frame = combined[
        combined["target_family"] == family
    ]

    for representation, specification in (
        heads.items()
    ):
        columns = list(
            specification["feature_columns"]
        )

        standard_deviations = {}

        for role in nominal_roles:
            subset = family_frame[
                family_frame["split_role"]
                == role
            ]

            standard_deviations[role] = (
                subset[columns]
                .std(axis=0, ddof=1)
                .to_numpy(float)
            )

        fit_sd = standard_deviations[
            "audit_nominal_fit"
        ]

        for comparison_role in [
            "nominal_calibration",
            "nominal_test",
        ]:
            comparison_sd = (
                standard_deviations[
                    comparison_role
                ]
            )

            ratios = np.divide(
                comparison_sd,
                fit_sd,
                out=np.full_like(
                    comparison_sd,
                    np.nan,
                ),
                where=fit_sd > 0,
            )

            finite = ratios[
                np.isfinite(ratios)
            ]

            scale_rows.append({
                "target_family": family,
                "representation": (
                    representation
                ),
                "comparison_role": (
                    comparison_role
                ),
                "dimension": len(columns),
                "finite_ratios": len(finite),
                "zero_fit_sd_features": int(
                    np.sum(fit_sd <= 0)
                ),
                "median_sd_ratio": float(
                    np.median(finite)
                )
                if len(finite)
                else np.nan,
                "p90_sd_ratio": float(
                    np.quantile(
                        finite,
                        0.90,
                    )
                )
                if len(finite)
                else np.nan,
                "maximum_sd_ratio": float(
                    np.max(finite)
                )
                if len(finite)
                else np.nan,
            })

raw_scale = pd.DataFrame(scale_rows)

invariance.to_csv(
    OUTPUT / "family_invariance.csv",
    index=False,
)

role_overlap.to_csv(
    OUTPUT / "family_role_overlap.csv",
    index=False,
)

raw_scale.to_csv(
    OUTPUT / "raw_feature_scale.csv",
    index=False,
)

summary = (
    invariance.groupby(
        "source_file",
        observed=True,
    )
    .agg(
        shared_units=("unit_id", "size"),
        noninvariant_units=(
            "family_invariant",
            lambda values: int(
                (~values).sum()
            ),
        ),
        maximum_difference=(
            "maximum_absolute_difference",
            "max",
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT / "summary.csv",
    index=False,
)

noninvariant_total = int(
    (~invariance["family_invariant"]).sum()
)

manifest = {
    "status": (
        "FAIL_FAMILY_INVARIANCE"
        if noninvariant_total
        else "PASS_FAMILY_INVARIANCE"
    ),
    "tolerance": TOLERANCE,
    "noninvariant_shared_units": (
        noninvariant_total
    ),
    "scientific_protocol_changed": False,
    "test_retuning": False,
    "source_hashes": {},
}

for path in [
    DEV_PATH,
    EVAL_PATH,
    SOURCE_PATH,
    HEAD_PATH,
]:
    manifest["source_hashes"][str(path)] = (
        hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    )

(OUTPUT / "manifest.json").write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

files = [
    OUTPUT / "family_invariance.csv",
    OUTPUT / "family_role_overlap.csv",
    OUTPUT / "raw_feature_scale.csv",
    OUTPUT / "summary.csv",
    OUTPUT / "manifest.json",
]

checksums = []

for path in files:
    checksums.append(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}"
        f"  {path}"
    )

(OUTPUT / "SHA256SUMS.txt").write_text(
    "\n".join(checksums) + "\n",
    encoding="utf-8",
)

print("=== FAMILY INVARIANCE SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== NOMINAL ROLE OVERLAP ===")
print(role_overlap.to_string(index=False))

print("\n=== RAW FEATURE SCALE ===")
print(raw_scale.to_string(index=False))

print("\n=== LARGEST FAMILY DIFFERENCES ===")
print(
    invariance.sort_values(
        "maximum_absolute_difference",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)

print(
    "\nFINAL FEATURE-LINEAGE STATUS: "
    + manifest["status"]
)
