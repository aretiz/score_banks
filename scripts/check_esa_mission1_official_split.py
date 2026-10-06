from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(
    "outputs/"
    "esa_real_event_readiness"
)

CLUSTERS_PATH = (
    ROOT
    / "mission1_clusters_24h.csv"
)

PRIMARY_FAMILIES = (
    "class_7",
    "class_3",
)

MIN_DESIGN = 10
MIN_TEST = 10

REPOSITORY = (
    "kplabs-pl/ESA-ADB"
)

GITHUB_HEADERS = {
    "Accept": (
        "application/"
        "vnd.github+json"
    ),
    "User-Agent": (
        "predictive-power-audit/1.0"
    ),
}


def sha256(path):
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


def github_file():
    commit_url = (
        "https://api.github.com/repos/"
        f"{REPOSITORY}/commits/main"
    )

    response = requests.get(
        commit_url,
        headers=GITHUB_HEADERS,
        timeout=(30, 120),
    )

    response.raise_for_status()

    commit = response.json()["sha"]

    raw_url = (
        "https://raw.githubusercontent.com/"
        f"{REPOSITORY}/{commit}/"
        "data/preprocessed/datasets.csv"
    )

    response = requests.get(
        raw_url,
        headers=GITHUB_HEADERS,
        timeout=(30, 120),
    )

    response.raise_for_status()

    return (
        commit,
        raw_url,
        response.content,
    )


def event_id_set(frame):
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
    if not CLUSTERS_PATH.exists():
        raise FileNotFoundError(
            CLUSTERS_PATH
        )

    (
        commit,
        datasets_url,
        datasets_bytes,
    ) = github_file()

    datasets_path = (
        ROOT
        / "esa_adb_datasets.csv"
    )

    datasets_path.write_bytes(
        datasets_bytes
    )

    datasets = pd.read_csv(
        io.BytesIO(
            datasets_bytes
        ),
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
            "Expected one "
            "ESA-Mission1/84_months "
            "definition; found "
            f"{len(official)}"
        )

    cutoff_text = (
        official.iloc[0][
            "split_at"
        ]
    )

    cutoff = pd.to_datetime(
        cutoff_text,
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
            "Cluster registry is not "
            "uniformly based on a "
            "24-hour gap"
        )

    if clusters[
        "cluster_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate cluster IDs"
        )

    clusters[
        "official_role"
    ] = "crosses_cutoff"

    clusters.loc[
        clusters["cluster_end"]
        < cutoff,
        "official_role",
    ] = "pre_cutoff"

    clusters.loc[
        clusters["cluster_start"]
        >= cutoff,
        "official_role",
    ] = "post_cutoff"

    pure_anomaly = clusters[
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
        pure_anomaly.groupby(
            [
                "classes",
                "official_role",
            ],
            observed=True,
        )
        .size()
        .rename(
            "independent_clusters"
        )
        .reset_index()
    )

    table = (
        counts.pivot(
            index="classes",
            columns="official_role",
            values=(
                "independent_clusters"
            ),
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    for column in [
        "pre_cutoff",
        "post_cutoff",
        "crosses_cutoff",
    ]:
        if column not in table:
            table[column] = 0

    table = table.sort_values(
        [
            "post_cutoff",
            "pre_cutoff",
        ],
        ascending=False,
    )

    primary = table[
        table["classes"].isin(
            PRIMARY_FAMILIES
        )
    ].copy()

    missing_families = sorted(
        set(PRIMARY_FAMILIES)
        - set(primary["classes"])
    )

    if missing_families:
        raise ValueError(
            "Missing primary families: "
            f"{missing_families}"
        )

    primary["design_ok"] = (
        primary["pre_cutoff"]
        >= MIN_DESIGN
    )

    primary["test_ok"] = (
        primary["post_cutoff"]
        >= MIN_TEST
    )

    primary["passes"] = (
        primary["design_ok"]
        & primary["test_ok"]
    )

    print(
        "=== OFFICIAL ESA-ADB "
        "SPLIT SOURCE ==="
    )

    print(
        "repository:",
        REPOSITORY,
    )

    print(
        "commit:",
        commit,
    )

    print(
        "dataset definition:",
        "ESA-Mission1 / 84_months",
    )

    print(
        "official cutoff:",
        cutoff,
    )

    print(
        "source URL:",
        datasets_url,
    )

    print(
        "telemetry loaded: NO"
    )

    print(
        "detector scores loaded: NO"
    )

    print(
        "\n=== ALL PURE ANOMALY "
        "CLASSES BY OFFICIAL ROLE ==="
    )

    print(
        table.to_string(
            index=False
        )
    )

    print(
        "\n=== PRIMARY FAMILY GATE ==="
    )

    print(
        primary.to_string(
            index=False
        )
    )

    feasibility_path = (
        ROOT
        / "mission1_official_"
        "split_feasibility.csv"
    )

    table.to_csv(
        feasibility_path,
        index=False,
    )

    if not primary[
        "passes"
    ].all():
        print(
            "\n=== DECISION ==="
        )

        print(
            "FAIL: the official temporal "
            "cutoff does not provide at "
            f"least {MIN_DESIGN} design "
            f"and {MIN_TEST} test clusters "
            "for both families."
        )

        print(
            "No frozen split was written."
        )

        print(
            "\nFINAL STATUS: PASS"
        )

        return

    selected = pure_anomaly[
        pure_anomaly[
            "classes"
        ].isin(
            PRIMARY_FAMILIES
        )
        & pure_anomaly[
            "official_role"
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
        "split_role"
    ] = selected[
        "official_role"
    ].map({
        "pre_cutoff": (
            "anomaly_design"
        ),
        "post_cutoff": (
            "anomaly_test"
        ),
    })

    design_ids = event_id_set(
        selected[
            selected["split_role"]
            == "anomaly_design"
        ]
    )

    test_ids = event_id_set(
        selected[
            selected["split_role"]
            == "anomaly_test"
        ]
    )

    overlap = (
        design_ids
        & test_ids
    )

    if overlap:
        raise AssertionError(
            "Event IDs cross "
            "design/test roles: "
            f"{sorted(overlap)}"
        )

    design_rows = selected[
        selected["split_role"]
        == "anomaly_design"
    ]

    test_rows = selected[
        selected["split_role"]
        == "anomaly_test"
    ]

    if not (
        design_rows["cluster_end"]
        < cutoff
    ).all():
        raise AssertionError(
            "Design cluster crosses "
            "or follows cutoff"
        )

    if not (
        test_rows["cluster_start"]
        >= cutoff
    ).all():
        raise AssertionError(
            "Test cluster precedes cutoff"
        )

    split_columns = [
        "cluster_id",
        "split_role",
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
        selected[split_columns]
        .sort_values(
            [
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

    selected.to_csv(
        split_path,
        index=False,
    )

    metadata_root = (
        ROOT
        / "metadata"
        / "ESA-Mission1"
    )

    sources = {}

    for path in [
        metadata_root / "labels.csv",
        metadata_root
        / "anomaly_types.csv",
        metadata_root
        / "channels.csv",
        CLUSTERS_PATH,
        datasets_path,
    ]:
        sources[str(path)] = {
            "bytes": (
                path.stat().st_size
            ),
            "sha256": sha256(path),
        }

    manifest = {
        "status": (
            "frozen_before_telemetry_"
            "or_scores"
        ),
        "dataset": "ESA-Mission1",
        "dataset_version": (
            "Zenodo record 15237121"
        ),
        "benchmark_repository": (
            REPOSITORY
        ),
        "benchmark_commit": commit,
        "official_cutoff": (
            cutoff.isoformat()
        ),
        "independent_unit": (
            "24-hour-connected "
            "temporal incident cluster"
        ),
        "primary_category": (
            "Anomaly"
        ),
        "primary_families": list(
            PRIMARY_FAMILIES
        ),
        "design_definition": (
            "pure primary-family "
            "clusters ending before "
            "the official cutoff"
        ),
        "test_definition": (
            "pure primary-family "
            "clusters starting at or "
            "after the official cutoff"
        ),
        "mixed_clusters": "excluded",
        "cutoff_crossing_clusters": (
            "excluded"
        ),
        "minimum_design_clusters_"
        "per_family": MIN_DESIGN,
        "minimum_test_clusters_"
        "per_family": MIN_TEST,
        "design_event_ids": len(
            design_ids
        ),
        "test_event_ids": len(
            test_ids
        ),
        "design_test_event_id_overlap": (
            len(overlap)
        ),
        "sources": sources,
    }

    manifest_path = (
        ROOT
        / "mission1_frozen_"
        "event_split_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_counts = (
        selected.groupby(
            [
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

    print(
        "\n=== FROZEN SPLIT COUNTS ==="
    )

    print(
        final_counts.to_string(
            index=False
        )
    )

    print(
        "design/test event-ID overlap:",
        len(overlap),
    )

    print(
        "\n=== DECISION ==="
    )

    print(
        "PASS: official temporal split "
        "frozen before telemetry or "
        "score access."
    )

    print(
        "\n=== FILES WRITTEN ==="
    )

    for path in [
        feasibility_path,
        split_path,
        manifest_path,
    ]:
        print(
            f"{path}: "
            f"{path.stat().st_size:,} bytes"
        )

    print(
        "\nFINAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()
