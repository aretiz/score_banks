from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(
    "outputs/"
    "esa_real_event_readiness"
)

CLUSTERS_PATH = (
    ROOT
    / "mission1_clusters_24h.csv"
)

DATASETS_PATH = (
    ROOT
    / "esa_adb_datasets.csv"
)

BENCHMARK_COMMIT = (
    "aeebcd9ecd3e7266d6d6a035a8081b3da83dfe33"
)

PRIMARY_MIN_DESIGN = 15
PRIMARY_MIN_TEST = 15

SECONDARY_MIN_DESIGN = 5
SECONDARY_MIN_TEST = 15

EXPECTED_PRIMARY = {
    "class_7",
}

EXPECTED_SECONDARY = {
    "class_3",
}


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def event_ids(frame):
    result = set()

    for value in frame[
        "event_ids"
    ].astype(str):
        result.update(
            item
            for item in value.split("|")
            if item
        )

    return result


def main():
    for path in [
        CLUSTERS_PATH,
        DATASETS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    datasets = pd.read_csv(
        DATASETS_PATH,
        dtype=str,
    )

    official = datasets[
        (
            datasets[
                "collection_name"
            ]
            == "ESA-Mission1"
        )
        & (
            datasets[
                "dataset_name"
            ]
            == "84_months"
        )
    ]

    if len(official) != 1:
        raise ValueError(
            "Official Mission-1 split "
            "definition is not unique"
        )

    cutoff = pd.to_datetime(
        official.iloc[0][
            "split_at"
        ],
        utc=True,
    )

    clusters = pd.read_csv(
        CLUSTERS_PATH,
        dtype={
            "cluster_id": str,
            "event_ids": str,
        },
    )

    for column in [
        "cluster_start",
        "cluster_end",
    ]:
        clusters[column] = (
            pd.to_datetime(
                clusters[column],
                format="mixed",
                errors="raise",
                utc=True,
            )
        )

    if not (
        clusters["gap_hours"]
        == 24
    ).all():
        raise ValueError(
            "Expected 24-hour "
            "incident clusters"
        )

    if clusters[
        "cluster_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate cluster IDs"
        )

    clusters[
        "temporal_role"
    ] = "crosses_cutoff"

    clusters.loc[
        clusters["cluster_end"]
        < cutoff,
        "temporal_role",
    ] = "pre_cutoff"

    clusters.loc[
        clusters["cluster_start"]
        >= cutoff,
        "temporal_role",
    ] = "post_cutoff"

    pure = clusters[
        (
            clusters["categories"]
            == "Anomaly"
        )
        & (
            clusters["n_classes"]
            == 1
        )
        & (
            clusters[
                "n_categories"
            ]
            == 1
        )
    ].copy()

    counts = (
        pure.groupby(
            [
                "classes",
                "temporal_role",
            ],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )

    for column in [
        "pre_cutoff",
        "post_cutoff",
        "crosses_cutoff",
    ]:
        if column not in counts:
            counts[column] = 0

    counts = counts.reset_index()

    primary = set(
        counts.loc[
            (
                counts[
                    "pre_cutoff"
                ]
                >= PRIMARY_MIN_DESIGN
            )
            & (
                counts[
                    "post_cutoff"
                ]
                >= PRIMARY_MIN_TEST
            ),
            "classes",
        ]
    )

    secondary = set(
        counts.loc[
            (
                ~counts[
                    "classes"
                ].isin(primary)
            )
            & (
                counts[
                    "pre_cutoff"
                ]
                >= SECONDARY_MIN_DESIGN
            )
            & (
                counts[
                    "post_cutoff"
                ]
                >= SECONDARY_MIN_TEST
            ),
            "classes",
        ]
    )

    if primary != EXPECTED_PRIMARY:
        raise ValueError(
            "Unexpected primary "
            f"families: {sorted(primary)}"
        )

    if (
        secondary
        != EXPECTED_SECONDARY
    ):
        raise ValueError(
            "Unexpected secondary "
            f"families: "
            f"{sorted(secondary)}"
        )

    selected = pure[
        pure["classes"].isin(
            primary | secondary
        )
        & pure[
            "temporal_role"
        ].isin(
            [
                "pre_cutoff",
                "post_cutoff",
            ]
        )
    ].copy()

    selected[
        "target_family"
    ] = selected["classes"]

    selected[
        "analysis_tier"
    ] = selected[
        "classes"
    ].map({
        **{
            family: (
                "primary_confirmatory"
            )
            for family in primary
        },
        **{
            family: (
                "secondary_exploratory"
            )
            for family in secondary
        },
    })

    selected[
        "split_role"
    ] = selected[
        "temporal_role"
    ].map({
        "pre_cutoff": (
            "anomaly_design"
        ),
        "post_cutoff": (
            "anomaly_test"
        ),
    })

    design = selected[
        selected["split_role"]
        == "anomaly_design"
    ]

    test = selected[
        selected["split_role"]
        == "anomaly_test"
    ]

    overlap = (
        event_ids(design)
        & event_ids(test)
    )

    if overlap:
        raise AssertionError(
            "Event IDs cross roles: "
            f"{sorted(overlap)}"
        )

    if not (
        design["cluster_end"]
        < cutoff
    ).all():
        raise AssertionError(
            "Design incident "
            "crosses cutoff"
        )

    if not (
        test["cluster_start"]
        >= cutoff
    ).all():
        raise AssertionError(
            "Test incident "
            "precedes cutoff"
        )

    columns = [
        "cluster_id",
        "split_role",
        "analysis_tier",
        "target_family",
        "cluster_start",
        "cluster_end",
        "duration_hours",
        "n_event_ids",
        "event_ids",
        "n_label_rows",
        "n_channels",
        "categories",
        "classes",
        "dimensionalities",
        "localities",
    ]

    selected = (
        selected[columns]
        .sort_values(
            [
                "analysis_tier",
                "split_role",
                "target_family",
                "cluster_start",
            ],
            kind="stable",
        )
    )

    split_path = (
        ROOT
        / "mission1_frozen_"
        "event_split.csv"
    )

    manifest_path = (
        ROOT
        / "mission1_frozen_"
        "event_split_manifest.json"
    )

    selected.to_csv(
        split_path,
        index=False,
    )

    metadata = (
        ROOT
        / "metadata"
        / "ESA-Mission1"
    )

    source_paths = [
        CLUSTERS_PATH,
        DATASETS_PATH,
        metadata / "labels.csv",
        metadata
        / "anomaly_types.csv",
        metadata / "channels.csv",
    ]

    sources = {
        str(path): {
            "bytes": (
                path.stat().st_size
            ),
            "sha256": (
                file_sha256(path)
            ),
        }
        for path in source_paths
    }

    manifest = {
        "status": (
            "frozen_before_telemetry_"
            "or_detector_scores"
        ),
        "dataset": (
            "ESA-Mission1"
        ),
        "dataset_version": (
            "Zenodo record 15237121"
        ),
        "benchmark_commit": (
            BENCHMARK_COMMIT
        ),
        "official_cutoff": (
            cutoff.isoformat()
        ),
        "independent_unit": (
            "24-hour-connected "
            "temporal incident cluster"
        ),
        "category": "Anomaly",
        "primary_rule": {
            "minimum_design_clusters": (
                PRIMARY_MIN_DESIGN
            ),
            "minimum_test_clusters": (
                PRIMARY_MIN_TEST
            ),
        },
        "secondary_rule": {
            "minimum_design_clusters": (
                SECONDARY_MIN_DESIGN
            ),
            "minimum_test_clusters": (
                SECONDARY_MIN_TEST
            ),
            "primary_families_excluded": (
                True
            ),
        },
        "primary_families": (
            sorted(primary)
        ),
        "secondary_families": (
            sorted(secondary)
        ),
        "mixed_class_or_category_"
        "clusters": "excluded",
        "cutoff_crossing_clusters": (
            "excluded"
        ),
        "design_test_event_id_overlap": (
            len(overlap)
        ),
        "label_counts_inspected_"
        "before_freeze": True,
        "telemetry_inspected_"
        "before_freeze": False,
        "detector_scores_inspected_"
        "before_freeze": False,
        "sources": sources,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = (
        selected.groupby(
            [
                "analysis_tier",
                "target_family",
                "split_role",
            ],
            observed=True,
        )
        .agg(
            clusters=(
                "cluster_id",
                "size",
            ),
            event_ids=(
                "n_event_ids",
                "sum",
            ),
            median_channels=(
                "n_channels",
                "median",
            ),
            median_duration_hours=(
                "duration_hours",
                "median",
            ),
        )
        .reset_index()
    )

    seal_paths = [
        split_path,
        manifest_path,
        *source_paths,
    ]

    seal_path = (
        ROOT
        / "FROZEN_SHA256SUMS.txt"
    )

    seal_path.write_text(
        "".join(
            f"{file_sha256(path)}  "
            f"{path}\n"
            for path in seal_paths
        ),
        encoding="utf-8",
    )

    print(
        "=== HIERARCHICAL "
        "OFFICIAL SPLIT ==="
    )

    print(
        "official cutoff:",
        cutoff,
    )

    print(
        "benchmark commit:",
        BENCHMARK_COMMIT,
    )

    print(
        "telemetry inspected: NO"
    )

    print(
        "detector scores inspected: NO"
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\ndesign/test event-ID "
        "overlap:",
        len(overlap),
    )

    print(
        "\nPrimary interpretation: "
        "confirmatory"
    )

    print(
        "Secondary interpretation: "
        "exploratory with design-shift "
        "uncertainty"
    )

    print(
        "\n=== FILES WRITTEN ==="
    )

    for path in [
        split_path,
        manifest_path,
        seal_path,
    ]:
        print(
            f"{path}: "
            f"{path.stat().st_size:,} bytes"
        )

    print(
        "\nFINAL SEAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()
