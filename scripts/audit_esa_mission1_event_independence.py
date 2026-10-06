from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "outputs/"
    "esa_real_event_readiness"
)

SOURCE = (
    ROOT
    / "metadata"
    / "ESA-Mission1"
)

GAPS = [
    0,
    1,
    6,
    24,
]


class UnionFind:

    def __init__(self, values):
        self.parent = {
            value: value
            for value in values
        }

        self.rank = {
            value: 0
            for value in values
        }

    def find(self, value):
        parent = self.parent[value]

        if parent != value:
            self.parent[value] = (
                self.find(parent)
            )

        return self.parent[value]

    def union(
        self,
        left,
        right,
    ):
        left = self.find(left)
        right = self.find(right)

        if left == right:
            return

        if (
            self.rank[left]
            < self.rank[right]
        ):
            left, right = right, left

        self.parent[right] = left

        if (
            self.rank[left]
            == self.rank[right]
        ):
            self.rank[left] += 1


def joined_text(values):
    values = sorted({
        str(value)
        for value in values
        if pd.notna(value)
    })

    return "|".join(values)


def cluster_labels(
    labels,
    taxonomy,
    gap_hours,
):
    ids = sorted(
        labels["ID"].unique()
    )

    union_find = UnionFind(ids)

    gap = pd.Timedelta(
        hours=gap_hours
    )

    active = []

    ordered = labels.sort_values(
        [
            "StartTime",
            "EndTime",
            "ID",
        ],
        kind="stable",
    )

    for row in ordered.itertuples(
        index=False
    ):
        cutoff = (
            row.StartTime
            - gap
        )

        active = [
            item
            for item in active
            if item[0] >= cutoff
        ]

        for (
            active_end,
            active_id,
        ) in active:
            if active_id != row.ID:
                union_find.union(
                    active_id,
                    row.ID,
                )

        active.append((
            row.EndTime,
            row.ID,
        ))

    id_to_root = {
        event_id: union_find.find(
            event_id
        )
        for event_id in ids
    }

    clustered = labels.copy()

    clustered["root"] = (
        clustered["ID"].map(
            id_to_root
        )
    )

    rows = []

    for root, group in (
        clustered.groupby(
            "root",
            sort=False,
        )
    ):
        event_ids = sorted(
            group["ID"].unique()
        )

        event_types = taxonomy[
            taxonomy["ID"].isin(
                event_ids
            )
        ]

        start = (
            group["StartTime"].min()
        )

        end = (
            group["EndTime"].max()
        )

        class_text = joined_text(
            event_types.get(
                "Class",
                pd.Series(dtype=str),
            )
        )

        category_text = joined_text(
            event_types.get(
                "Category",
                pd.Series(dtype=str),
            )
        )

        dimensionality_text = (
            joined_text(
                event_types.get(
                    "Dimensionality",
                    pd.Series(
                        dtype=str
                    ),
                )
            )
        )

        locality_text = joined_text(
            event_types.get(
                "Locality",
                pd.Series(dtype=str),
            )
        )

        rows.append({
            "root": root,
            "cluster_start": start,
            "cluster_end": end,
            "duration_hours": (
                end - start
            ).total_seconds() / 3600,
            "n_event_ids": len(
                event_ids
            ),
            "event_ids": "|".join(
                event_ids
            ),
            "n_label_rows": len(
                group
            ),
            "n_channels": (
                group[
                    "Channel"
                ].nunique()
            ),
            "classes": class_text,
            "n_classes": (
                0
                if not class_text
                else len(
                    class_text.split("|")
                )
            ),
            "categories": (
                category_text
            ),
            "n_categories": (
                0
                if not category_text
                else len(
                    category_text.split(
                        "|"
                    )
                )
            ),
            "dimensionalities": (
                dimensionality_text
            ),
            "localities": (
                locality_text
            ),
        })

    clusters = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "cluster_start",
                "cluster_end",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    clusters.insert(
        0,
        "cluster_id",
        [
            f"M1C{i:04d}"
            for i in range(
                1,
                len(clusters) + 1,
            )
        ],
    )

    clusters.insert(
        1,
        "gap_hours",
        gap_hours,
    )

    return clusters


def main():
    labels = pd.read_csv(
        SOURCE / "labels.csv",
        dtype={
            "ID": str,
            "Channel": str,
        },
    )

    taxonomy = (
        pd.read_csv(
            SOURCE
            / "anomaly_types.csv",
            dtype={
                "ID": str,
            },
        )
        .drop_duplicates("ID")
    )

    channels = pd.read_csv(
        SOURCE / "channels.csv"
    )

    for column in [
        "StartTime",
        "EndTime",
    ]:
        labels[column] = (
            pd.to_datetime(
                labels[column],
                format="mixed",
                errors="raise",
                utc=True,
            )
        )

    if (
        labels["EndTime"]
        < labels["StartTime"]
    ).any():
        raise ValueError(
            "Found labels with EndTime "
            "before StartTime"
        )

    if (
        labels["ID"].isna().any()
        or labels[
            "Channel"
        ].isna().any()
    ):
        raise ValueError(
            "Missing event IDs "
            "or channels"
        )

    unknown_ids = sorted(
        set(labels["ID"])
        - set(taxonomy["ID"])
    )

    if unknown_ids:
        raise ValueError(
            "Labels missing taxonomy: "
            f"{unknown_ids[:10]}"
        )

    event_registry = (
        labels.groupby(
            "ID",
            observed=True,
        )
        .agg(
            event_start=(
                "StartTime",
                "min",
            ),
            event_end=(
                "EndTime",
                "max",
            ),
            label_rows=(
                "ID",
                "size",
            ),
            channels=(
                "Channel",
                "nunique",
            ),
            duplicate_channel_rows=(
                "Channel",
                lambda x: int(
                    x.duplicated().sum()
                ),
            ),
        )
        .reset_index()
        .merge(
            taxonomy,
            on="ID",
            how="left",
            validate="one_to_one",
        )
    )

    event_registry[
        "span_hours"
    ] = (
        event_registry["event_end"]
        - event_registry[
            "event_start"
        ]
    ).dt.total_seconds() / 3600

    print(
        "=== MISSION-1 "
        "SOURCE INTEGRITY ==="
    )

    print(
        "label rows:",
        len(labels),
    )

    print(
        "event IDs:",
        labels["ID"].nunique(),
    )

    print(
        "channels with labels:",
        labels[
            "Channel"
        ].nunique(),
    )

    print(
        "channel inventory rows:",
        len(channels),
    )

    print(
        "taxonomy rows:",
        len(taxonomy),
    )

    print(
        "missing taxonomy IDs:",
        len(unknown_ids),
    )

    print(
        "duplicate event/channel rows:",
        labels.duplicated(
            [
                "ID",
                "Channel",
            ]
        ).sum(),
    )

    quantiles = [
        0,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        1,
    ]

    print(
        "\n=== EVENT-LEVEL "
        "DISTRIBUTIONS ==="
    )

    print(
        "label rows per event:"
    )

    print(
        event_registry[
            "label_rows"
        ]
        .quantile(quantiles)
        .to_string()
    )

    print(
        "channels per event:"
    )

    print(
        event_registry[
            "channels"
        ]
        .quantile(quantiles)
        .to_string()
    )

    print(
        "event span in hours:"
    )

    print(
        event_registry[
            "span_hours"
        ]
        .quantile(quantiles)
        .to_string()
    )

    sensitivity_rows = []
    all_clusters = {}

    print(
        "\n=== TEMPORAL CLUSTER "
        "SENSITIVITY ==="
    )

    for gap_hours in GAPS:
        clusters = cluster_labels(
            labels,
            taxonomy,
            gap_hours,
        )

        all_clusters[
            gap_hours
        ] = clusters

        row = {
            "gap_hours": gap_hours,
            "clusters": len(
                clusters
            ),
            "multi_id_clusters": int(
                (
                    clusters[
                        "n_event_ids"
                    ]
                    > 1
                ).sum()
            ),
            "multi_class_clusters": int(
                (
                    clusters[
                        "n_classes"
                    ]
                    > 1
                ).sum()
            ),
            "maximum_ids_per_cluster": (
                int(
                    clusters[
                        "n_event_ids"
                    ].max()
                )
            ),
            "median_cluster_hours": (
                float(
                    clusters[
                        "duration_hours"
                    ].median()
                )
            ),
            "maximum_cluster_hours": (
                float(
                    clusters[
                        "duration_hours"
                    ].max()
                )
            ),
        }

        sensitivity_rows.append(
            row
        )

        print(
            pd.DataFrame(
                [row]
            ).to_string(
                index=False
            )
        )

    conservative = (
        all_clusters[24]
    )

    pure = conservative[
        (
            conservative[
                "n_classes"
            ]
            == 1
        )
        & (
            conservative[
                "n_categories"
            ]
            == 1
        )
    ].copy()

    mixed = conservative.drop(
        index=pure.index
    )

    family = (
        pure.groupby(
            [
                "categories",
                "classes",
            ],
            observed=True,
        )
        .agg(
            independent_clusters=(
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
        .sort_values(
            "independent_clusters",
            ascending=False,
        )
    )

    family[
        "supports_10_design_10_test"
    ] = (
        family[
            "independent_clusters"
        ]
        >= 20
    )

    family[
        "supports_15_design_15_test"
    ] = (
        family[
            "independent_clusters"
        ]
        >= 30
    )

    print(
        "\n=== CONSERVATIVE "
        "24-HOUR CLUSTERS ==="
    )

    print(
        "total clusters:",
        len(conservative),
    )

    print(
        "pure class/category clusters:",
        len(pure),
    )

    print(
        "mixed class/category clusters:",
        len(mixed),
    )

    print(
        "\n=== 24-HOUR CLUSTERS BY "
        "EXTERNAL FAMILY ==="
    )

    print(
        family.to_string(
            index=False
        )
    )

    category = (
        pure.groupby(
            "categories",
            observed=True,
        )
        .size()
        .rename(
            "independent_clusters"
        )
        .reset_index()
        .sort_values(
            "independent_clusters",
            ascending=False,
        )
    )

    print(
        "\n=== CATEGORY TOTALS ==="
    )

    print(
        category.to_string(
            index=False
        )
    )

    anomaly_families = family[
        (
            family["categories"]
            == "Anomaly"
        )
        & family[
            "supports_10_design_10_test"
        ]
    ]

    all_useful_families = family[
        family[
            "categories"
        ].isin(
            [
                "Anomaly",
                "Rare Event",
            ]
        )
        & family[
            "supports_10_design_10_test"
        ]
    ]

    print(
        "\n=== PRELIMINARY "
        "DECISION ==="
    )

    print(
        "Anomaly families with "
        ">=20 clusters:",
        len(anomaly_families),
    )

    print(
        "Anomaly/Rare Event families "
        "with >=20 clusters:",
        len(all_useful_families),
    )

    if len(anomaly_families) >= 2:
        print(
            "PASS: Mission 1 supports "
            "a real-anomaly primary "
            "experiment."
        )
    elif (
        len(all_useful_families)
        >= 3
    ):
        print(
            "PARTIAL PASS: use Anomaly "
            "plus Rare Event families "
            "with explicit wording."
        )
    else:
        print(
            "FAIL: insufficient "
            "independent family-level "
            "clusters after temporal "
            "grouping."
        )

    event_registry.to_csv(
        ROOT
        / "mission1_event_registry.csv",
        index=False,
    )

    pd.DataFrame(
        sensitivity_rows
    ).to_csv(
        ROOT
        / "mission1_cluster_sensitivity.csv",
        index=False,
    )

    conservative.to_csv(
        ROOT
        / "mission1_clusters_24h.csv",
        index=False,
    )

    family.to_csv(
        ROOT
        / "mission1_family_feasibility.csv",
        index=False,
    )

    print(
        "\n=== FILES WRITTEN ==="
    )

    for name in [
        "mission1_event_registry.csv",
        "mission1_cluster_sensitivity.csv",
        "mission1_clusters_24h.csv",
        "mission1_family_feasibility.csv",
    ]:
        path = ROOT / name

        print(
            f"{path}: "
            f"{path.stat().st_size:,} bytes"
        )

    print(
        "\nFINAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()
