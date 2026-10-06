from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPOSITORY),
)

from power_decomposition import (
    AuditConfig,
    GAP_COLUMNS,
    run_audit,
)


FAMILIES = [
    "class_7",
    "class_3",
]

REFERENCE = "forecast_joint"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/esa_real_event_readiness/"
            "frozen_audit_selection"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/"
            "esa_power_decomposition_v1"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260918,
    )

    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    result = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1 << 20),
            b"",
        ):
            result.update(block)

    return result.hexdigest()


def make_block(
    source,
    representation,
    split,
    target,
    features,
    paired=None,
):
    source = source.reset_index(
        drop=True
    )

    metadata = pd.DataFrame(
        {
            "representation": (
                representation
            ),
            "split": split,
            "unit_id": (
                source["unit_id"]
                .astype(str)
            ),
            "target": target,
        }
    )

    values = (
        source[features]
        .reset_index(drop=True)
        .copy()
    )

    values.columns = [
        f"psi_{index}"
        for index in range(len(features))
    ]

    paired_names = [
        f"paired_psi_{index}"
        for index in range(len(features))
    ]

    if paired is None:
        paired_values = pd.DataFrame(
            np.nan,
            index=np.arange(
                len(source)
            ),
            columns=paired_names,
        )

    else:
        paired_values = (
            paired[features]
            .reset_index(drop=True)
            .copy()
        )

        paired_values.columns = (
            paired_names
        )

    return pd.concat(
        [
            metadata,
            values,
            paired_values,
        ],
        axis=1,
    )


def build_family_samples(
    development,
    evaluation,
    heads,
    family,
):
    development = development[
        development["target_family"]
        == family
    ].copy()

    evaluation = evaluation[
        evaluation["target_family"]
        == family
    ].copy()

    design = development[
        development["split_role"]
        == "anomaly_design"
    ].copy()

    paired = development[
        development["split_role"]
        == "paired_nominal_design"
    ].copy()

    require(
        design["pair_id"]
        .notna()
        .all(),
        f"Missing design pair IDs: "
        f"{family}",
    )

    require(
        paired["pair_id"]
        .notna()
        .all(),
        f"Missing control pair IDs: "
        f"{family}",
    )

    require(
        design["pair_id"].is_unique,
        f"Duplicate design pair IDs: "
        f"{family}",
    )

    require(
        paired["pair_id"].is_unique,
        f"Duplicate control pair IDs: "
        f"{family}",
    )

    require(
        set(design["pair_id"])
        == set(paired["pair_id"]),
        f"Design/control pair mismatch: "
        f"{family}",
    )

    paired = (
        paired.set_index("pair_id")
        .loc[
            design[
                "pair_id"
            ].tolist()
        ]
        .reset_index()
    )

    role_sources = {
        "nominal_fit": development[
            development["split_role"]
            == "audit_nominal_fit"
        ],
        "anomaly_design": design,
        "nominal_calibration": development[
            development["split_role"]
            == "nominal_calibration"
        ],
        "nominal_test": evaluation[
            evaluation["split_role"]
            == "nominal_test"
        ],
        "anomaly_test": evaluation[
            evaluation["split_role"]
            == "anomaly_test"
        ],
    }

    expected = {
        "class_7": {
            "nominal_fit": 62,
            "anomaly_design": 19,
            "nominal_calibration": 118,
            "nominal_test": 119,
            "anomaly_test": 18,
        },
        "class_3": {
            "nominal_fit": 62,
            "anomaly_design": 8,
            "nominal_calibration": 118,
            "nominal_test": 119,
            "anomaly_test": 19,
        },
    }

    observed = {
        name: len(frame)
        for name, frame
        in role_sources.items()
    }

    require(
        observed == expected[family],
        f"Unexpected role counts "
        f"{family}: {observed}",
    )

    blocks = []

    for (
        representation,
        specification,
    ) in heads.items():
        features = list(
            specification[
                "feature_columns"
            ]
        )

        for feature in features:
            require(
                feature in development,
                "Missing development "
                f"feature: {feature}",
            )

            require(
                feature in evaluation,
                "Missing evaluation "
                f"feature: {feature}",
            )

        for split, source in (
            role_sources.items()
        ):
            block = make_block(
                source,
                representation,
                split,
                family,
                features,
                paired=(
                    paired
                    if split
                    == "anomaly_design"
                    else None
                ),
            )

            blocks.append(block)

    samples = pd.concat(
        blocks,
        ignore_index=True,
        sort=False,
    )

    role_counts = samples.groupby(
        [
            "representation",
            "split",
        ],
        observed=True,
    ).size()

    for representation in heads:
        for (
            split,
            count,
        ) in expected[family].items():
            require(
                int(
                    role_counts.loc[
                        (
                            representation,
                            split,
                        )
                    ]
                )
                == count,
                "Expanded count mismatch: "
                f"{family}, "
                f"{representation}, "
                f"{split}",
            )

    role_per_unit = (
        samples.groupby(
            "unit_id",
            observed=True,
        )["split"]
        .nunique()
    )

    require(
        int(
            (
                role_per_unit > 1
            ).sum()
        )
        == 0,
        f"Units cross roles in "
        f"{family}",
    )

    return samples


def main():
    args = parse_args()

    require(
        not args.out.exists(),
        f"Output exists: {args.out}",
    )

    development_path = (
        args.input
        / "development_unit_features.csv"
    )

    evaluation_path = (
        args.input
        / (
            "SEALED_evaluation_"
            "unit_features.csv"
        )
    )

    heads_path = (
        args.input
        / "audit_heads.json"
    )

    for path in [
        development_path,
        evaluation_path,
        heads_path,
    ]:
        require(
            path.is_file(),
            f"Missing input: {path}",
        )

    development = pd.read_csv(
        development_path,
        dtype={"unit_id": str},
    )

    evaluation = pd.read_csv(
        evaluation_path,
        dtype={"unit_id": str},
    )

    heads = json.loads(
        heads_path.read_text(
            encoding="utf-8"
        )
    )

    expected_representations = {
        "persistence",
        "univariate_ar1",
        "cross_channel_ridge",
        "forecast_joint",
    }

    require(
        set(heads)
        == expected_representations,
        "Unexpected representations: "
        f"{sorted(heads)}",
    )

    config = AuditConfig(
        alpha=args.alpha,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )

    all_samples = []
    all_results = []

    for family in FAMILIES:
        print(
            f"\n=== BUILD {family} ===",
            flush=True,
        )

        samples = (
            build_family_samples(
                development,
                evaluation,
                heads,
                family,
            )
        )

        print(
            samples.groupby(
                [
                    "representation",
                    "split",
                ],
                observed=True,
            )
            .size()
            .to_string()
        )

        print(
            f"\n=== POINT AUDIT "
            f"{family} ===",
            flush=True,
        )

        result = run_audit(
            samples,
            heads,
            config=config,
            reference_representation=(
                REFERENCE
            ),
        )

        result["family"] = family

        all_samples.append(
            samples.assign(
                family=family
            )
        )

        all_results.append(result)

    samples = pd.concat(
        all_samples,
        ignore_index=True,
        sort=False,
    )

    results = pd.concat(
        all_results,
        ignore_index=True,
        sort=False,
    )

    maximum_error = float(
        results[
            "decomposition_error"
        ]
        .abs()
        .max()
    )

    require(
        maximum_error <= 1e-12,
        "Decomposition error: "
        f"{maximum_error}",
    )

    args.out.mkdir(
        parents=True
    )

    samples.to_csv(
        args.out / "samples.csv",
        index=False,
    )

    results.to_csv(
        args.out
        / "decomposition.csv",
        index=False,
    )

    (
        args.out / "heads.json"
    ).write_text(
        json.dumps(
            heads,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    protocol = {
        "status": (
            "esa_mission1_point_"
            "decomposition_complete"
        ),
        "families": FAMILIES,
        "reference_representation": (
            REFERENCE
        ),
        "alpha": args.alpha,
        "mc_samples": args.mc_samples,
        "seed": args.seed,
        "paired_design_controls": True,
        "family_specific_nominal_pools": (
            True
        ),
        "audit_feature_binding": (
            "representation-local frozen feature "
            "order mapped to psi_0, psi_1, ..."
        ),
        "evaluation_unsealed_before_this_audit": (
            True
        ),
        "interpretation": {
            "class_7": (
                "primary_confirmatory_for_"
                "frozen_detector; "
                "decomposition_posthoc"
            ),
            "class_3": "exploratory",
        },
        "source_hashes": {
            str(path): sha256(path)
            for path in [
                development_path,
                evaluation_path,
                heads_path,
            ]
        },
        "script_sha256": sha256(
            __file__
        ),
    }

    (
        args.out / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    files = sorted(
        path
        for path in args.out.iterdir()
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS.txt"
        )
    )

    with (
        args.out
        / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in files:
            handle.write(
                f"{sha256(path)}  "
                f"{path}\n"
            )

    columns = [
        "representation",
        "family",
        "oracle_power",
        "observed_power",
        *GAP_COLUMNS,
        "total_loss",
        "nominal_fpr",
    ]

    columns = [
        column
        for column in columns
        if column in results
    ]

    print(
        "\n=== ESA MISSION-1 "
        "DECOMPOSITION ==="
    )

    print(
        results[
            columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nMaximum decomposition error:",
        f"{maximum_error:.16g}",
    )

    print(
        "FINAL ESA POINT-DECOMPOSITION "
        "STATUS: PASS"
    )


if __name__ == "__main__":
    main()
