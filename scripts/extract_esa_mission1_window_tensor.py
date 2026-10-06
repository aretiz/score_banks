from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archive",
        default=(
            "data/esa_ad/v2/"
            "ESA-Mission1.zip"
        ),
    )

    parser.add_argument(
        "--plan",
        default=(
            "outputs/"
            "esa_real_event_readiness/"
            "mission1_frozen_window_plan.csv"
        ),
    )

    parser.add_argument(
        "--channels",
        default=(
            "outputs/"
            "esa_real_event_readiness/"
            "metadata/"
            "ESA-Mission1/"
            "channels.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "esa_real_event_readiness/"
            "window_tensor"
        ),
    )

    parser.add_argument(
        "--max-new-channels",
        type=int,
        default=0,
        help=(
            "0 processes every remaining "
            "channel; use 2 for the pilot."
        ),
    )

    return parser.parse_args()


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


def load_plan(
    path,
    cadence_seconds,
):
    plan = pd.read_csv(
        path,
        dtype={"unit_id": str},
    )

    time_columns = [
        "history_start",
        "window_start",
        "window_end",
    ]

    for column in time_columns:
        plan[column] = (
            pd.to_datetime(
                plan[column],
                utc=True,
                format="mixed",
            )
            .dt.tz_convert(None)
        )

    consistency = (
        plan.groupby(
            "unit_id",
            observed=True,
        )
        .agg(
            history_starts=(
                "history_start",
                "nunique",
            ),
            roles=(
                "split_role",
                "nunique",
            ),
        )
    )

    if (
        consistency[
            "history_starts"
        ] != 1
    ).any():
        raise ValueError(
            "A unit has inconsistent "
            "history starts"
        )

    if (
        consistency["roles"]
        != 1
    ).any():
        raise ValueError(
            "A unit crosses split roles"
        )

    rows = []

    for unit_id, group in plan.groupby(
        "unit_id",
        observed=True,
        sort=True,
    ):
        history_start = (
            group["history_start"].iloc[0]
        )

        maximum_end = (
            group["window_end"].max()
        )

        delta_seconds = (
            maximum_end
            - history_start
        ).total_seconds()

        ratio = (
            delta_seconds
            / cadence_seconds
        )

        if not np.isclose(
            ratio,
            round(ratio),
            atol=1e-10,
        ):
            raise ValueError(
                "Off-grid unit duration: "
                f"{unit_id}"
            )

        length = (
            int(round(ratio))
            + 1
        )

        rows.append({
            "unit_id": unit_id,
            "split_role": (
                group[
                    "split_role"
                ].iloc[0]
            ),
            "target_families": (
                "|".join(
                    sorted(
                        group[
                            "target_family"
                        ].unique()
                    )
                )
            ),
            "analysis_tiers": (
                "|".join(
                    sorted(
                        group[
                            "analysis_tier"
                        ].unique()
                    )
                )
            ),
            "history_start": (
                history_start
            ),
            "maximum_window_end": (
                maximum_end
            ),
            "length": length,
        })

    units = (
        pd.DataFrame(rows)
        .sort_values(
            "unit_id",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if units[
        "unit_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate canonical unit IDs"
        )

    return plan, units


def requested_grid(
    units,
    cadence_seconds,
):
    cadence_ns = (
        cadence_seconds
        * 1_000_000_000
    )

    pieces = []
    offsets = [0]

    for row in units.itertuples(
        index=False
    ):
        start_ns = pd.Timestamp(
            row.history_start
        ).value

        values = (
            start_ns
            + np.arange(
                row.length,
                dtype=np.int64,
            )
            * cadence_ns
        )

        expected_end = pd.Timestamp(
            row.maximum_window_end
        ).value

        if int(values[-1]) != int(
            expected_end
        ):
            raise ValueError(
                "Grid end mismatch for "
                f"{row.unit_id}"
            )

        pieces.append(values)

        offsets.append(
            offsets[-1]
            + len(values)
        )

    return (
        np.concatenate(pieces),
        np.asarray(
            offsets,
            dtype=np.int64,
        ),
    )


def channel_members(outer):
    mapping = {}

    for info in outer.infolist():
        path = PurePosixPath(
            info.filename
        )

        parents = {
            part.lower()
            for part in path.parts[:-1]
        }

        if (
            info.is_dir()
            or path.suffix.lower()
            != ".zip"
        ):
            continue

        if "channels" not in parents:
            continue

        channel = path.stem

        if channel in mapping:
            raise ValueError(
                "Duplicate channel archive: "
                f"{channel}"
            )

        mapping[channel] = info

    return mapping


def load_channel(
    outer,
    outer_info,
    channel,
):
    with tempfile.NamedTemporaryFile(
        prefix=f"esa_{channel}_",
        suffix=".zip",
    ) as temporary:
        with outer.open(
            outer_info
        ) as source:
            shutil.copyfileobj(
                source,
                temporary,
                length=8 * 1024 * 1024,
            )

        temporary.flush()

        with zipfile.ZipFile(
            temporary.name
        ) as nested:
            files = [
                item
                for item
                in nested.infolist()
                if not item.is_dir()
            ]

            if len(files) != 1:
                raise ValueError(
                    f"{channel}: expected "
                    "one nested member, "
                    f"found {len(files)}"
                )

            with nested.open(
                files[0]
            ) as stream:
                frame = pd.read_pickle(
                    stream
                )

    if isinstance(
        frame,
        pd.Series,
    ):
        series = frame

    elif isinstance(
        frame,
        pd.DataFrame,
    ):
        if channel in frame.columns:
            series = frame[channel]

        elif frame.shape[1] == 1:
            series = frame.iloc[:, 0]

        else:
            raise ValueError(
                f"{channel}: ambiguous "
                "DataFrame columns"
            )

    else:
        raise TypeError(
            f"{channel}: unexpected "
            f"pickle type {type(frame)}"
        )

    index = pd.DatetimeIndex(
        pd.to_datetime(
            series.index
        )
    )

    if index.tz is not None:
        index = (
            index.tz_convert("UTC")
            .tz_localize(None)
        )

    original_rows = len(index)

    duplicate_count = int(
        index.duplicated(
            keep="first"
        ).sum()
    )

    if duplicate_count:
        keep = ~index.duplicated(
            keep="first"
        )

        index = index[keep]

        series = series.iloc[
            np.flatnonzero(keep)
        ]

    if not (
        index.is_monotonic_increasing
    ):
        order = np.argsort(
            index.asi8,
            kind="stable",
        )

        index = index[order]
        series = series.iloc[order]

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float32
    )

    if len(values) == 0:
        raise ValueError(
            f"{channel}: empty channel"
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{channel}: non-finite "
            "raw values"
        )

    metadata = {
        "raw_rows": original_rows,
        "deduplicated_rows": (
            len(index)
        ),
        "duplicate_timestamps": (
            duplicate_count
        ),
        "raw_start": str(
            index[0]
        ),
        "raw_end": str(
            index[-1]
        ),
        "raw_dtype": str(
            values.dtype
        ),
    }

    return (
        index,
        values,
        metadata,
    )


def zero_order_hold_sample(
    index,
    values,
    requested_ns,
):
    requested = pd.DatetimeIndex(
        requested_ns
    )

    locations = index.get_indexer(
        requested,
        method="pad",
    )

    leading = locations < 0

    if leading.any():
        replacements = (
            index.get_indexer(
                requested[leading],
                method="backfill",
            )
        )

        if (
            replacements < 0
        ).any():
            raise ValueError(
                "Requested timestamps lie "
                "outside both channel edges"
            )

        locations[leading] = (
            replacements
        )

    sampled = values[locations]

    source_ns = index.asi8[
        locations
    ]

    age_seconds = (
        requested_ns
        - source_ns
    ).astype(
        np.float64
    ) / 1e9

    effective_age = np.maximum(
        age_seconds,
        0.0,
    )

    diagnostics = {
        "backfilled_points": int(
            leading.sum()
        ),
        "age_seconds_median": float(
            np.quantile(
                effective_age,
                0.50,
            )
        ),
        "age_seconds_p90": float(
            np.quantile(
                effective_age,
                0.90,
            )
        ),
        "age_seconds_p99": float(
            np.quantile(
                effective_age,
                0.99,
            )
        ),
        "age_seconds_maximum": float(
            effective_age.max()
        ),
        "fraction_age_gt_5min": (
            float(
                np.mean(
                    effective_age > 300
                )
            )
        ),
        "fraction_age_gt_1hour": (
            float(
                np.mean(
                    effective_age > 3600
                )
            )
        ),
    }

    return sampled, diagnostics


def write_channel_to_tensor(
    tensor,
    channel_index,
    sampled,
    offsets,
    lengths,
):
    matrix = np.full(
        (
            len(lengths),
            tensor.shape[1],
        ),
        np.nan,
        dtype=np.float32,
    )

    for unit_index, length in enumerate(
        lengths
    ):
        start = offsets[unit_index]
        end = offsets[
            unit_index + 1
        ]

        if end - start != length:
            raise ValueError(
                "Offset/length mismatch"
            )

        matrix[
            unit_index,
            :length,
        ] = sampled[start:end]

    valid_mask = (
        np.arange(
            tensor.shape[1]
        )[None, :]
        < lengths[:, None]
    )

    if not np.isfinite(
        matrix[valid_mask]
    ).all():
        raise ValueError(
            "Non-finite sampled telemetry"
        )

    tensor[
        :,
        :,
        channel_index,
    ] = matrix


def save_progress(
    path,
    payload,
):
    temporary = path.with_suffix(
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def main():
    args = parse_args()

    archive_path = Path(
        args.archive
    )

    plan_path = Path(
        args.plan
    )

    channel_path = Path(
        args.channels
    )

    output = Path(
        args.output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        archive_path,
        plan_path,
        channel_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    cadence_seconds = 30

    plan_hash = file_sha256(
        plan_path
    )

    _, units = load_plan(
        plan_path,
        cadence_seconds,
    )

    (
        requested_ns,
        offsets,
    ) = requested_grid(
        units,
        cadence_seconds,
    )

    lengths = units[
        "length"
    ].to_numpy(
        dtype=np.int64
    )

    channels = pd.read_csv(
        channel_path
    )

    channels["Target_bool"] = (
        channels["Target"]
        .astype(str)
        .str.strip()
        .str.upper()
        == "YES"
    )

    channels = (
        channels.sort_values(
            "Channel",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    names = (
        channels["Channel"]
        .astype(str)
        .tolist()
    )

    if (
        len(names) != 76
        or channels[
            "Target_bool"
        ].sum() != 58
    ):
        raise ValueError(
            "Unexpected Mission-1 "
            "channel inventory"
        )

    units_path = (
        output / "units.csv"
    )

    channels_path = (
        output / "channels.csv"
    )

    partial_path = (
        output
        / "telemetry.partial.npy"
    )

    final_path = (
        output / "telemetry.npy"
    )

    progress_path = (
        output / "progress.json"
    )

    diagnostics_path = (
        output
        / "resampling_diagnostics.csv"
    )

    manifest_path = (
        output / "manifest.json"
    )

    seal_path = (
        output / "SHA256SUMS.txt"
    )

    if final_path.exists():
        print(
            "FINAL tensor already exists:",
            final_path,
        )

        print(
            "Delete it manually only if "
            "a deliberate rebuild is required."
        )

        print(
            "FINAL STATUS: "
            "ALREADY COMPLETE"
        )

        return

    maximum_steps = int(
        lengths.max()
    )

    expected_shape = (
        len(units),
        maximum_steps,
        len(names),
    )

    if partial_path.exists():
        if not progress_path.exists():
            raise RuntimeError(
                "Partial tensor exists "
                "without progress metadata"
            )

        progress = json.loads(
            progress_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            progress[
                "plan_sha256"
            ] != plan_hash
        ):
            raise RuntimeError(
                "Frozen plan changed since "
                "partial extraction"
            )

        if tuple(
            progress["shape"]
        ) != expected_shape:
            raise RuntimeError(
                "Partial tensor shape does "
                "not match the plan"
            )

        tensor = (
            np.lib.format.open_memmap(
                partial_path,
                mode="r+",
                dtype=np.float32,
                shape=expected_shape,
            )
        )

    else:
        tensor = (
            np.lib.format.open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float32,
                shape=expected_shape,
            )
        )

        tensor[:] = np.nan
        tensor.flush()

        progress = {
            "status": "partial",
            "plan_sha256": plan_hash,
            "shape": list(
                expected_shape
            ),
            "cadence_seconds": (
                cadence_seconds
            ),
            "requested_grid_points_per_channel": (
                int(len(requested_ns))
            ),
            "completed_channels": [],
        }

        units.to_csv(
            units_path,
            index=False,
        )

        channels.to_csv(
            channels_path,
            index=False,
        )

        save_progress(
            progress_path,
            progress,
        )

    completed = set(
        progress[
            "completed_channels"
        ]
    )

    remaining = [
        name
        for name in names
        if name not in completed
    ]

    if args.max_new_channels > 0:
        remaining = remaining[
            :args.max_new_channels
        ]

    if diagnostics_path.exists():
        diagnostics_rows = (
            pd.read_csv(
                diagnostics_path
            )
            .to_dict("records")
        )

    else:
        diagnostics_rows = []

    print(
        "=== ESA MISSION-1 WINDOW "
        "TENSOR EXTRACTION ==="
    )

    print(
        "archive:",
        archive_path,
    )

    print(
        "frozen plan SHA256:",
        plan_hash,
    )

    print(
        "canonical independent units:",
        len(units),
    )

    print(
        "maximum time steps:",
        maximum_steps,
    )

    print(
        "channels:",
        len(names),
    )

    print(
        "tensor shape:",
        expected_shape,
    )

    print(
        "tensor size GiB:",
        (
            np.prod(expected_shape)
            * 4
            / 1024**3
        ),
    )

    print(
        "requested grid points "
        "per channel:",
        len(requested_ns),
    )

    print(
        "previously completed channels:",
        len(completed),
    )

    print(
        "channels scheduled now:",
        len(remaining),
    )

    print(
        "resampling: official "
        "30-second zero-order hold "
        "plus edge fill"
    )

    print(
        "annotation-dependent sample "
        "restoration: DISABLED"
    )

    with zipfile.ZipFile(
        archive_path
    ) as outer:
        member_map = channel_members(
            outer
        )

        missing = sorted(
            set(names)
            - set(member_map)
        )

        if missing:
            raise ValueError(
                "Missing channel archives: "
                f"{missing}"
            )

        for sequence, channel in enumerate(
            remaining,
            start=1,
        ):
            started = (
                time.perf_counter()
            )

            channel_index = (
                names.index(channel)
            )

            (
                index,
                values,
                raw_metadata,
            ) = load_channel(
                outer,
                member_map[channel],
                channel,
            )

            sampled, age = (
                zero_order_hold_sample(
                    index,
                    values,
                    requested_ns,
                )
            )

            if not np.isfinite(
                sampled
            ).all():
                raise ValueError(
                    f"{channel}: non-finite "
                    "sampled values"
                )

            write_channel_to_tensor(
                tensor,
                channel_index,
                sampled,
                offsets,
                lengths,
            )

            tensor.flush()

            elapsed = (
                time.perf_counter()
                - started
            )

            row = {
                "channel": channel,
                "channel_index": (
                    channel_index
                ),
                "is_target": bool(
                    channels.loc[
                        channel_index,
                        "Target_bool",
                    ]
                ),
                "requested_points": int(
                    len(requested_ns)
                ),
                **raw_metadata,
                **age,
                "elapsed_seconds": (
                    elapsed
                ),
            }

            diagnostics_rows = [
                old
                for old in diagnostics_rows
                if old.get("channel")
                != channel
            ]

            diagnostics_rows.append(
                row
            )

            (
                pd.DataFrame(
                    diagnostics_rows
                )
                .sort_values(
                    "channel_index",
                    kind="stable",
                )
                .to_csv(
                    diagnostics_path,
                    index=False,
                )
            )

            completed.add(channel)

            progress[
                "completed_channels"
            ] = [
                name
                for name in names
                if name in completed
            ]

            progress["status"] = (
                "partial"
            )

            save_progress(
                progress_path,
                progress,
            )

            print(
                f"[{sequence:02d}/"
                f"{len(remaining):02d}] "
                f"{channel}: "
                f"raw={len(index):,}, "
                f"p99_age="
                f"{age['age_seconds_p99']:.1f}s, "
                f">1h="
                f"{age['fraction_age_gt_1hour']:.4f}, "
                f"elapsed={elapsed:.1f}s"
            )

            del (
                index,
                values,
                sampled,
            )

            gc.collect()

    if len(completed) != len(names):
        print(
            "\n=== PILOT/RESUMABLE "
            "EXTRACTION COMPLETE ==="
        )

        print(
            "completed channels:",
            len(completed),
        )

        print(
            "remaining channels:",
            len(names) - len(completed),
        )

        print(
            "partial tensor:",
            partial_path,
        )

        print(
            "progress metadata:",
            progress_path,
        )

        print(
            "PILOT STATUS: PASS"
        )

        return

    tensor.flush()
    del tensor

    os.rename(
        partial_path,
        final_path,
    )

    diagnostics = pd.read_csv(
        diagnostics_path
    )

    if diagnostics[
        "channel"
    ].nunique() != len(names):
        raise RuntimeError(
            "Incomplete resampling "
            "diagnostics"
        )

    manifest = {
        "status": "complete",
        "dataset": "ESA-Mission1",
        "source_archive": str(
            archive_path
        ),
        "frozen_plan": str(
            plan_path
        ),
        "frozen_plan_sha256": (
            plan_hash
        ),
        "tensor": str(
            final_path
        ),
        "tensor_dtype": "float32",
        "tensor_shape": list(
            expected_shape
        ),
        "axis_order": [
            "unit",
            "time",
            "channel",
        ],
        "cadence_seconds": (
            cadence_seconds
        ),
        "resampling": (
            "zero-order hold followed "
            "by leading/trailing edge fill"
        ),
        "annotation_dependent_"
        "restoration": False,
        "annotation_restoration_reason": (
            "Disabled to prevent held-out "
            "event labels from changing "
            "test inputs"
        ),
        "padding": (
            "NaN after each unit-specific "
            "valid length"
        ),
        "unit_metadata": str(
            units_path
        ),
        "channel_metadata": str(
            channels_path
        ),
        "resampling_diagnostics": str(
            diagnostics_path
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    progress["status"] = (
        "complete"
    )

    save_progress(
        progress_path,
        progress,
    )

    sealed = [
        final_path,
        units_path,
        channels_path,
        diagnostics_path,
        manifest_path,
    ]

    seal_path.write_text(
        "".join(
            f"{file_sha256(path)}  "
            f"{path}\n"
            for path in sealed
        ),
        encoding="utf-8",
    )

    print(
        "\n=== RESAMPLING "
        "DIAGNOSTIC SUMMARY ==="
    )

    diagnostic_columns = [
        "age_seconds_median",
        "age_seconds_p90",
        "age_seconds_p99",
        "age_seconds_maximum",
        "fraction_age_gt_5min",
        "fraction_age_gt_1hour",
    ]

    print(
        diagnostics[
            diagnostic_columns
        ]
        .describe()
        .to_string()
    )

    print(
        "\n=== FILES WRITTEN ==="
    )

    for path in (
        sealed
        + [
            progress_path,
            seal_path,
        ]
    ):
        print(
            f"{path}: "
            f"{path.stat().st_size:,} "
            "bytes"
        )

    print(
        "\nFINAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()
