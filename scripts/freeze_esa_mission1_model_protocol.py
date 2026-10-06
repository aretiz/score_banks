from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "outputs/esa_real_event_readiness"
)

TENSOR_ROOT = (
    ROOT / "window_tensor"
)

UNITS_PATH = (
    TENSOR_ROOT / "units.csv"
)

CHANNELS_PATH = (
    TENSOR_ROOT / "channels.csv"
)

TENSOR_MANIFEST_PATH = (
    TENSOR_ROOT / "manifest.json"
)

DIAGNOSTICS_PATH = (
    TENSOR_ROOT
    / "resampling_diagnostics.csv"
)

TENSOR_SEAL_PATH = (
    TENSOR_ROOT / "SHA256SUMS.txt"
)

WINDOW_PLAN_PATH = (
    ROOT
    / "mission1_frozen_window_plan.csv"
)

ROLE_PATH = (
    ROOT
    / "mission1_frozen_model_unit_roles.csv"
)

HEADS_PATH = (
    ROOT
    / "mission1_frozen_metadata_heads.json"
)

MANIFEST_PATH = (
    ROOT
    / "mission1_frozen_model_protocol.json"
)

SEAL_PATH = (
    ROOT
    / "FROZEN_MODEL_PROTOCOL_SHA256SUMS.txt"
)


SEED = 20260905

FORECASTER_FIT_FRACTION = 0.50
NLL_VALIDATION_FRACTION = 0.20
AUDIT_NOMINAL_FIT_FRACTION = 0.30

MODEL_NAMES = [
    "persistence",
    "univariate_ar1",
    "cross_channel_ridge",
]

RIDGE_LAMBDA = 1.0
VARIANCE_FLOOR = 1e-6
FEATURE_TOP_FRACTION = 0.10

ALPHA = 0.05
POINT_MONTE_CARLO_SAMPLES = 100_000
BOOTSTRAP_REPLICATES = 1_000


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


def stable_hash(value):
    return hashlib.sha256(
        f"{SEED}|{value}".encode(
            "utf-8"
        )
    ).hexdigest()


def normalize(vector):
    norm = float(
        np.linalg.norm(vector)
    )

    if (
        not np.isfinite(norm)
        or norm <= 0
    ):
        raise ValueError(
            "Cannot normalize an empty "
            "head direction"
        )

    return vector / norm


def build_base_heads(
    channels,
    target_names,
):
    target = (
        channels.set_index(
            "Channel"
        )
        .loc[target_names]
        .reset_index()
    )

    dimension = len(
        target_names
    )

    candidates = []

    for index, channel in enumerate(
        target_names
    ):
        vector = np.zeros(
            dimension,
            dtype=float,
        )

        vector[index] = 1.0

        candidates.append({
            "name": (
                f"channel:{channel}"
            ),
            "kind": "single_channel",
            "members": [channel],
            "direction": vector,
        })

    for group, frame in target.groupby(
        "Group",
        observed=True,
        sort=True,
    ):
        members = sorted(
            frame["Channel"].astype(str)
        )

        vector = np.asarray(
            [
                1.0
                if name in members
                else 0.0
                for name in target_names
            ],
            dtype=float,
        )

        candidates.append({
            "name": f"group:{group}",
            "kind": "metadata_group",
            "members": members,
            "direction": normalize(
                vector
            ),
        })

    for subsystem, frame in (
        target.groupby(
            "Subsystem",
            observed=True,
            sort=True,
        )
    ):
        members = sorted(
            frame["Channel"].astype(str)
        )

        vector = np.asarray(
            [
                1.0
                if name in members
                else 0.0
                for name in target_names
            ],
            dtype=float,
        )

        candidates.append({
            "name": (
                f"subsystem:{subsystem}"
            ),
            "kind": (
                "metadata_subsystem"
            ),
            "members": members,
            "direction": normalize(
                vector
            ),
        })

    candidates.append({
        "name": (
            "global:"
            "all_target_channels"
        ),
        "kind": "global",
        "members": target_names,
        "direction": normalize(
            np.ones(
                dimension,
                dtype=float,
            )
        ),
    })

    # Singleton groups can duplicate
    # single-channel directions.
    retained = []
    support_to_index = {}
    aliases = []

    for head in candidates:
        direction = normalize(
            np.asarray(
                head["direction"],
                dtype=float,
            )
        )

        support = tuple(
            np.flatnonzero(
                np.abs(direction) > 0
            ).tolist()
        )

        if support in support_to_index:
            aliases.append({
                "alias": head["name"],
                "retained": retained[
                    support_to_index[
                        support
                    ]
                ]["name"],
            })

            continue

        support_to_index[
            support
        ] = len(retained)

        retained.append({
            "name": head["name"],
            "kind": head["kind"],
            "members": head["members"],
            "direction": (
                direction.tolist()
            ),
        })

    return retained, aliases


def representation_heads(
    base_heads,
    target_names,
):
    dimension = len(
        target_names
    )

    result = {}

    for model in MODEL_NAMES:
        result[model] = {
            "dimension": dimension,
            "feature_names": [
                f"{model}:{name}"
                for name in target_names
            ],
            "heads": [
                {
                    "name": head["name"],
                    "kind": head["kind"],
                    "members": (
                        head["members"]
                    ),
                    "direction": (
                        head["direction"]
                    ),
                }
                for head in base_heads
            ],
        }

    joint_feature_names = [
        f"{model}:{channel}"
        for model in MODEL_NAMES
        for channel in target_names
    ]

    joint_heads = []

    for model_index, model in enumerate(
        MODEL_NAMES
    ):
        for head in base_heads:
            vector = np.zeros(
                len(MODEL_NAMES)
                * dimension,
                dtype=float,
            )

            start = (
                model_index
                * dimension
            )

            vector[
                start:
                start + dimension
            ] = np.asarray(
                head["direction"],
                dtype=float,
            )

            joint_heads.append({
                "name": (
                    f"model:{model}|"
                    f"{head['name']}"
                ),
                "kind": (
                    "model_specific_"
                    + head["kind"]
                ),
                "members": [
                    f"{model}:{member}"
                    for member
                    in head["members"]
                ],
                "direction": (
                    vector.tolist()
                ),
            })

    for head in base_heads:
        base = np.asarray(
            head["direction"],
            dtype=float,
        )

        vector = normalize(
            np.concatenate([
                base
                for _ in MODEL_NAMES
            ])
        )

        joint_heads.append({
            "name": (
                f"consensus:"
                f"{head['name']}"
            ),
            "kind": (
                "cross_model_consensus_"
                + head["kind"]
            ),
            "members": [
                f"{model}:{channel}"
                for model in MODEL_NAMES
                for channel
                in head["members"]
            ],
            "direction": (
                vector.tolist()
            ),
        })

    result["forecast_joint"] = {
        "dimension": len(
            joint_feature_names
        ),
        "feature_names": (
            joint_feature_names
        ),
        "heads": joint_heads,
    }

    return result


def main():
    required = [
        UNITS_PATH,
        CHANNELS_PATH,
        TENSOR_MANIFEST_PATH,
        DIAGNOSTICS_PATH,
        TENSOR_SEAL_PATH,
        WINDOW_PLAN_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    units = pd.read_csv(
        UNITS_PATH,
        dtype={"unit_id": str},
    )

    channels = pd.read_csv(
        CHANNELS_PATH
    )

    tensor_manifest = json.loads(
        TENSOR_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        tensor_manifest.get(
            "status"
        )
        != "complete"
    ):
        raise RuntimeError(
            "Window tensor is not "
            "complete"
        )

    if units[
        "unit_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate independent units"
        )

    if len(units) != 534:
        raise ValueError(
            "Unexpected independent-unit "
            f"count: {len(units)}"
        )

    nominal_fit = units[
        units["split_role"]
        == "nominal_fit"
    ].copy()

    if len(nominal_fit) != 206:
        raise ValueError(
            "Unexpected nominal-fit "
            f"count: {len(nominal_fit)}"
        )

    nominal_fit["hash"] = (
        nominal_fit["unit_id"].map(
            stable_hash
        )
    )

    nominal_fit = (
        nominal_fit.sort_values(
            [
                "hash",
                "unit_id",
            ],
            kind="stable",
        )
    )

    n = len(nominal_fit)

    n_forecaster = int(
        np.floor(
            n
            * FORECASTER_FIT_FRACTION
        )
    )

    n_nll = int(
        np.floor(
            n
            * NLL_VALIDATION_FRACTION
        )
    )

    n_audit = (
        n
        - n_forecaster
        - n_nll
    )

    if min(
        n_forecaster,
        n_nll,
        n_audit,
    ) <= 0:
        raise RuntimeError(
            "Empty nominal subrole"
        )

    model_role = dict(
        zip(
            units["unit_id"],
            units["split_role"],
        )
    )

    ordered_ids = (
        nominal_fit[
            "unit_id"
        ].tolist()
    )

    for unit in ordered_ids[
        :n_forecaster
    ]:
        model_role[unit] = (
            "forecaster_fit"
        )

    for unit in ordered_ids[
        n_forecaster:
        n_forecaster + n_nll
    ]:
        model_role[unit] = (
            "nll_validation"
        )

    for unit in ordered_ids[
        n_forecaster + n_nll:
    ]:
        model_role[unit] = (
            "audit_nominal_fit"
        )

    roles = units.copy()

    roles = roles.rename(
        columns={
            "split_role":
            "window_plan_role"
        }
    )

    roles["model_role"] = (
        roles["unit_id"].map(
            model_role
        )
    )

    if roles[
        "model_role"
    ].isna().any():
        raise ValueError(
            "Missing model role"
        )

    if (
        roles.groupby(
            "unit_id"
        )["model_role"]
        .nunique()
        .max()
        != 1
    ):
        raise ValueError(
            "An independent unit "
            "crosses model roles"
        )

    roles.to_csv(
        ROLE_PATH,
        index=False,
    )

    target_mask = (
        channels["Target_bool"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({
            "true",
            "1",
            "yes",
        })
    )

    target_names = sorted(
        channels.loc[
            target_mask,
            "Channel",
        ].astype(str)
    )

    input_names = sorted(
        channels[
            "Channel"
        ].astype(str)
    )

    if (
        len(target_names) != 58
        or len(input_names) != 76
    ):
        raise ValueError(
            "Unexpected frozen "
            "channel counts"
        )

    (
        base_heads,
        aliases,
    ) = build_base_heads(
        channels,
        target_names,
    )

    representations = (
        representation_heads(
            base_heads,
            target_names,
        )
    )

    heads_payload = {
        "status": (
            "frozen_before_detector_scores"
        ),
        "input_channels": (
            input_names
        ),
        "target_channels": (
            target_names
        ),
        "base_head_count": (
            len(base_heads)
        ),
        "removed_duplicate_aliases": (
            aliases
        ),
        "representations": (
            representations
        ),
    }

    HEADS_PATH.write_text(
        json.dumps(
            heads_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": (
            "frozen_before_event_scores_"
            "or_test_power"
        ),
        "dataset": "ESA-Mission1",
        "seed": SEED,
        "unit_subroles": {
            "forecaster_fit": (
                n_forecaster
            ),
            "nll_validation": (
                n_nll
            ),
            "audit_nominal_fit": (
                n_audit
            ),
        },
        "unit_split_rule": (
            "SHA256(seed|unit_id) "
            "ordering"
        ),
        "forecasters": {
            "persistence": {
                "mean": (
                    "previous 30-second "
                    "held value"
                ),
                "variance": (
                    "per-target residual "
                    "variance on "
                    "forecaster_fit"
                ),
            },
            "univariate_ar1": {
                "mean": (
                    "per-target intercept "
                    "plus slope times "
                    "previous value"
                ),
                "variance": (
                    "per-target residual "
                    "variance on "
                    "forecaster_fit"
                ),
            },
            "cross_channel_ridge": {
                "inputs": (
                    "previous values of all "
                    "76 standardized channels"
                ),
                "outputs": (
                    "58 standardized target "
                    "channels"
                ),
                "lambda": RIDGE_LAMBDA,
                "variance": (
                    "per-target residual "
                    "variance on "
                    "forecaster_fit"
                ),
            },
        },
        "normalization": (
            "mean and SD estimated only "
            "on forecaster_fit"
        ),
        "variance_floor": (
            VARIANCE_FLOOR
        ),
        "unit_feature": {
            "definition": (
                "mean of largest 10 percent "
                "absolute standardized "
                "one-step forecast errors "
                "within the frozen event "
                "horizon"
            ),
            "top_fraction": (
                FEATURE_TOP_FRACTION
            ),
            "minimum_selected_points": 1,
        },
        "candidate_representations": (
            MODEL_NAMES
        ),
        "reference_representation": (
            "forecast_joint"
        ),
        "reference_definition": (
            "concatenation of all three "
            "58-channel forecast-error "
            "feature vectors"
        ),
        "dictionary": {
            "construction": (
                "target-channel, "
                "metadata-group, "
                "metadata-subsystem, and "
                "global directions"
            ),
            "uses_anomaly_values": False,
            "base_head_count": (
                len(base_heads)
            ),
            "joint_bank": (
                "model-specific copies plus "
                "cross-model consensus copies"
            ),
        },
        "audit": {
            "alpha": ALPHA,
            "point_monte_carlo_samples": (
                POINT_MONTE_CARLO_SAMPLES
            ),
            "bootstrap_replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "alternative_unit": (
                "frozen 24-hour-connected "
                "incident cluster"
            ),
            "nominal_unit": (
                "frozen UTC calendar week"
            ),
            "conformal_tail": "upper",
            "test_power": (
                "fraction of untouched "
                "anomaly-test incident "
                "clusters exceeding the "
                "frozen conformal threshold"
            ),
        },
        "input_channels": input_names,
        "target_channels": target_names,
        "telemetry_schema_and_staleness_"
        "diagnostics_inspected": True,
        "event_aligned_detector_scores_"
        "inspected_before_freeze": False,
        "test_power_inspected_before_"
        "freeze": False,
        "annotation_dependent_"
        "restoration": False,
        "sources": {
            str(path): {
                "bytes": (
                    path.stat().st_size
                ),
                "sha256": sha256(path),
            }
            for path in required
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    sealed = [
        ROLE_PATH,
        HEADS_PATH,
        MANIFEST_PATH,
    ]

    SEAL_PATH.write_text(
        "".join(
            f"{sha256(path)}  "
            f"{path}\n"
            for path in sealed
        ),
        encoding="utf-8",
    )

    print(
        "=== FROZEN ESA MISSION-1 "
        "MODEL PROTOCOL ==="
    )

    print(
        "telemetry tensor opened:",
        "NO",
    )

    print(
        "event scores inspected:",
        "NO",
    )

    print(
        "test power inspected:",
        "NO",
    )

    print(
        "input channels:",
        len(input_names),
    )

    print(
        "target channels:",
        len(target_names),
    )

    print(
        "forecasters:",
        ", ".join(MODEL_NAMES),
    )

    print(
        "base metadata heads:",
        len(base_heads),
    )

    print(
        "removed duplicate "
        "head aliases:",
        len(aliases),
    )

    print(
        "joint heads:",
        len(
            representations[
                "forecast_joint"
            ]["heads"]
        ),
    )

    print(
        "alpha:",
        ALPHA,
    )

    print(
        "point Monte Carlo samples:",
        POINT_MONTE_CARLO_SAMPLES,
    )

    print(
        "bootstrap replicates:",
        BOOTSTRAP_REPLICATES,
    )

    print(
        "\n=== MODEL-ROLE COUNTS ==="
    )

    role_counts = (
        roles.groupby(
            "model_role",
            observed=True,
        )
        .size()
        .rename("units")
        .reset_index()
        .sort_values(
            "model_role"
        )
    )

    print(
        role_counts.to_string(
            index=False
        )
    )

    print(
        "\n=== DICTIONARY COUNTS ==="
    )

    base_counts = (
        pd.Series([
            head["kind"]
            for head in base_heads
        ])
        .value_counts()
    )

    print(
        base_counts
        .rename_axis("kind")
        .rename("heads")
        .to_string()
    )

    print(
        "\n=== REPRESENTATION "
        "DIMENSIONS ==="
    )

    for name, payload in (
        representations.items()
    ):
        print(
            f"{name:22s} "
            f"dimension="
            f"{payload['dimension']:3d} "
            f"heads="
            f"{len(payload['heads']):3d}"
        )

    print(
        "\n=== INTEGRITY ==="
    )

    print(
        "independent units:",
        len(roles),
    )

    print(
        "duplicate units:",
        int(
            roles[
                "unit_id"
            ].duplicated().sum()
        ),
    )

    print(
        "units crossing model roles:",
        0,
    )

    print(
        "anomaly-valued dictionary "
        "construction:",
        "NO",
    )

    print(
        "\n=== FILES WRITTEN ==="
    )

    for path in [
        ROLE_PATH,
        HEADS_PATH,
        MANIFEST_PATH,
        SEAL_PATH,
    ]:
        print(
            f"{path}: "
            f"{path.stat().st_size:,} "
            "bytes"
        )

    print(
        "\nFINAL MODEL-PROTOCOL "
        "SEAL: PASS"
    )


if __name__ == "__main__":
    main()
