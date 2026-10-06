"""Small adapter to call from the existing frozen EarthNet CAPS-Z runner."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def component_rows(
    *,
    split: str,
    scene_ids: Sequence[object],
    component_scores: Mapping[str, Sequence[float]],
    caps_scores: Sequence[float],
    metadata: Mapping[str, Sequence[object] | object] | None = None,
) -> pd.DataFrame:
    """Build rows without changing or recalibrating any frozen score."""
    scene_ids = np.asarray(scene_ids)
    n = len(scene_ids)
    result: dict[str, object] = {
        "split": np.repeat(split, n),
        "scene_id": scene_ids,
        "caps": np.asarray(caps_scores, dtype=float),
    }
    if len(result["caps"]) != n:
        raise ValueError("caps_scores length does not match scene_ids")
    for name, values in component_scores.items():
        values = np.asarray(values, dtype=float)
        if len(values) != n:
            raise ValueError(f"Component {name!r} length does not match scene_ids")
        result[name] = values
    if metadata:
        for name, values in metadata.items():
            if np.isscalar(values) or isinstance(values, str):
                result[name] = np.repeat(values, n)
            else:
                values = np.asarray(values)
                if len(values) != n:
                    raise ValueError(f"Metadata {name!r} length does not match scene_ids")
                result[name] = values
    return pd.DataFrame(result)


def write_component_table(frames: Sequence[pd.DataFrame], destination: str | Path) -> None:
    """Concatenate scene_cal/null_test/alt_test frames and write one table."""
    table = pd.concat(list(frames), ignore_index=True, sort=False)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        table.to_csv(destination, index=False)
    elif destination.suffix.lower() in (".parquet", ".pq"):
        table.to_parquet(destination, index=False)
    else:
        raise ValueError("Destination must end in .csv, .parquet, or .pq")


# Example inside the existing EarthNet runner:
#
# frames = []
# frames.append(component_rows(
#     split="scene_cal",
#     scene_ids=scene_cal_ids,
#     component_scores={
#         "global": global_cal,
#         "hc": hc_cal,
#         "bj": bj_cal,
#         "scan_1": scan1_cal,
#         "scan_2": scan2_cal,
#     },
#     caps_scores=caps_cal,
#     metadata={"context": scene_cal_context},
# ))
# frames.append(component_rows(
#     split="null_test",
#     scene_ids=null_ids,
#     component_scores={
#         "global": global_null,
#         "hc": hc_null,
#         "bj": bj_null,
#         "scan_1": scan1_null,
#         "scan_2": scan2_null,
#     },
#     caps_scores=caps_null,
#     metadata={"context": null_context},
# ))
# frames.append(component_rows(
#     split="alt_test",
#     scene_ids=alt_ids,
#     component_scores={
#         "global": global_alt,
#         "hc": hc_alt,
#         "bj": bj_alt,
#         "scan_1": scan1_alt,
#         "scan_2": scan2_alt,
#     },
#     caps_scores=caps_alt,
#     metadata={
#         "anomaly_type": anomaly_type,
#         "strength": strength,
#         "support": support,
#         "context": alt_context,
#     },
# ))
# write_component_table(frames, "outputs/earthnet_caps/components.csv")
