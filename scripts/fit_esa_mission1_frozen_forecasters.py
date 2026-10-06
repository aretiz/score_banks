from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/esa_real_event_readiness")
TENSOR_ROOT = ROOT / "window_tensor"
TENSOR_PATH = TENSOR_ROOT / "telemetry.npy"
UNITS_PATH = TENSOR_ROOT / "units.csv"
CHANNELS_PATH = TENSOR_ROOT / "channels.csv"
TENSOR_SEAL_PATH = TENSOR_ROOT / "SHA256SUMS.txt"
PLAN_PATH = ROOT / "mission1_frozen_window_plan.csv"
PLAN_SEAL_PATH = ROOT / "FROZEN_WINDOW_PLAN_SHA256SUMS.txt"
ROLES_PATH = ROOT / "mission1_frozen_model_unit_roles.csv"
PROTOCOL_PATH = ROOT / "mission1_frozen_model_protocol.json"
PROTOCOL_SEAL_PATH = ROOT / "FROZEN_MODEL_PROTOCOL_SHA256SUMS.txt"

OUTPUT = ROOT / "forecast_representations"
PARAMETERS_PATH = OUTPUT / "forecaster_parameters.npz"
NLL_PATH = OUTPUT / "heldout_nll.csv"
FEATURES_PATH = OUTPUT / "unit_features.csv"
MANIFEST_PATH = OUTPUT / "manifest.json"
SEAL_PATH = OUTPUT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(seal_path: Path) -> int:
    checked = 0
    for raw_line in seal_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        expected, filename = line.split(maxsplit=1)
        candidate = Path(filename.strip())
        if not candidate.exists():
            raise FileNotFoundError(
                f"Sealed input is missing: {candidate}"
            )
        observed = sha256(candidate)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {candidate}: "
                f"expected {expected}, observed {observed}"
            )
        checked += 1
    if checked == 0:
        raise ValueError(f"Empty checksum seal: {seal_path}")
    return checked


def load_inputs():
    required = [
        TENSOR_PATH,
        UNITS_PATH,
        CHANNELS_PATH,
        TENSOR_SEAL_PATH,
        PLAN_PATH,
        PLAN_SEAL_PATH,
        ROLES_PATH,
        PROTOCOL_PATH,
        PROTOCOL_SEAL_PATH,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    tensor_seal_entries = verify_seal(TENSOR_SEAL_PATH)
    plan_seal_entries = verify_seal(PLAN_SEAL_PATH)
    protocol_seal_entries = verify_seal(PROTOCOL_SEAL_PATH)

    tensor = np.load(TENSOR_PATH, mmap_mode="r")
    units = pd.read_csv(UNITS_PATH, dtype={"unit_id": str})
    channels = pd.read_csv(CHANNELS_PATH)
    plan = pd.read_csv(PLAN_PATH, dtype={"unit_id": str, "pair_id": str})
    roles = pd.read_csv(ROLES_PATH, dtype={"unit_id": str})
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    if tensor.shape != (len(units), int(units["length"].max()), len(channels)):
        raise ValueError("Tensor axes do not match unit/channel metadata")
    if units["unit_id"].duplicated().any():
        raise ValueError("Duplicate tensor unit IDs")
    if roles["unit_id"].duplicated().any():
        raise ValueError("Duplicate model-role unit IDs")
    if set(units["unit_id"]) != set(roles["unit_id"]):
        raise ValueError("Tensor units and frozen model roles differ")
    if set(plan["unit_id"]) != set(roles["unit_id"]):
        raise ValueError("Window-plan units and frozen model roles differ")

    for column in ["history_start", "window_start", "window_end"]:
        plan[column] = pd.to_datetime(
            plan[column], utc=True, format="mixed"
        ).dt.tz_convert(None)
    units["history_start"] = pd.to_datetime(
        units["history_start"], utc=True, format="mixed"
    ).dt.tz_convert(None)

    seal_counts = {
        "tensor": tensor_seal_entries,
        "window_plan": plan_seal_entries,
        "model_protocol": protocol_seal_entries,
    }
    return required, tensor, units, channels, plan, roles, protocol, seal_counts


def valid_unit_array(tensor, units, unit_index):
    length = int(units.loc[unit_index, "length"])
    values = np.asarray(tensor[unit_index, :length, :], dtype=np.float64)
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid telemetry for unit index {unit_index}")
    return values


def fit_normalization(tensor, units, indices):
    count = 0
    mean = np.zeros(tensor.shape[2], dtype=np.float64)
    sum_squared_deviation = np.zeros(tensor.shape[2], dtype=np.float64)
    for index in indices:
        values = valid_unit_array(tensor, units, index)
        batch_count = len(values)
        batch_mean = values.mean(axis=0, dtype=np.float64)
        batch_sum_squared_deviation = np.square(
            values - batch_mean, dtype=np.float64
        ).sum(axis=0, dtype=np.float64)
        new_count = count + batch_count
        difference = batch_mean - mean
        sum_squared_deviation += (
            batch_sum_squared_deviation
            + np.square(difference)
            * count
            * batch_count
            / new_count
        )
        mean += difference * batch_count / new_count
        count = new_count
    variance = sum_squared_deviation / max(1, count - 1)
    variance = np.maximum(variance, 1e-12)
    scale = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Non-finite forecaster normalization")
    return mean, scale, count


def standardized_pairs(values, mean, scale, target_indices):
    standardized = (values - mean) / scale
    previous = standardized[:-1]
    target = standardized[1:, target_indices]
    return previous, target


def fit_models(tensor, units, fit_indices, mean, scale, target_indices, ridge):
    target_dimension = len(target_indices)
    input_dimension = tensor.shape[2]
    ar_n = 0
    ar_x = np.zeros(target_dimension)
    ar_y = np.zeros(target_dimension)
    ar_xx = np.zeros(target_dimension)
    ar_xy = np.zeros(target_dimension)
    xtx = np.zeros((input_dimension + 1, input_dimension + 1))
    xty = np.zeros((input_dimension + 1, target_dimension))

    for index in fit_indices:
        values = valid_unit_array(tensor, units, index)
        previous, target = standardized_pairs(
            values, mean, scale, target_indices
        )
        target_previous = previous[:, target_indices]
        ar_n += len(previous)
        ar_x += target_previous.sum(axis=0)
        ar_y += target.sum(axis=0)
        ar_xx += np.square(target_previous).sum(axis=0)
        ar_xy += (target_previous * target).sum(axis=0)

        augmented = np.empty((len(previous), input_dimension + 1))
        augmented[:, 0] = 1.0
        augmented[:, 1:] = previous
        xtx += augmented.T @ augmented
        xty += augmented.T @ target

    denominator = ar_xx - np.square(ar_x) / ar_n
    if (denominator <= 1e-10).any():
        raise ValueError("Degenerate target channel in AR(1) fit")
    ar_slope = (ar_xy - ar_x * ar_y / ar_n) / denominator
    ar_intercept = ar_y / ar_n - ar_slope * ar_x / ar_n

    penalty = np.eye(input_dimension + 1) * float(ridge)
    penalty[0, 0] = 0.0
    try:
        ridge_beta = np.linalg.solve(xtx + penalty, xty)
    except np.linalg.LinAlgError:
        ridge_beta = np.linalg.pinv(xtx + penalty) @ xty

    if not (
        np.isfinite(ar_slope).all()
        and np.isfinite(ar_intercept).all()
        and np.isfinite(ridge_beta).all()
    ):
        raise ValueError("Non-finite fitted coefficients")

    return ar_intercept, ar_slope, ridge_beta, ar_n, xtx


def residuals(previous, target, target_indices, ar_intercept, ar_slope, ridge_beta):
    target_previous = previous[:, target_indices]
    persistence = target - target_previous
    ar1 = target - (ar_intercept + target_previous * ar_slope)
    ridge_prediction = ridge_beta[0] + previous @ ridge_beta[1:]
    cross_ridge = target - ridge_prediction
    return {
        "persistence": persistence,
        "univariate_ar1": ar1,
        "cross_channel_ridge": cross_ridge,
    }


def fit_residual_variances(
    tensor,
    units,
    fit_indices,
    mean,
    scale,
    target_indices,
    ar_intercept,
    ar_slope,
    ridge_beta,
    floor,
):
    sums = {
        name: np.zeros(len(target_indices), dtype=np.float64)
        for name in ["persistence", "univariate_ar1", "cross_channel_ridge"]
    }
    count = 0
    for index in fit_indices:
        values = valid_unit_array(tensor, units, index)
        previous, target = standardized_pairs(values, mean, scale, target_indices)
        all_residuals = residuals(
            previous,
            target,
            target_indices,
            ar_intercept,
            ar_slope,
            ridge_beta,
        )
        count += len(previous)
        for name, matrix in all_residuals.items():
            sums[name] += np.square(matrix).sum(axis=0)
    variances = {
        name: np.maximum(values / count, floor)
        for name, values in sums.items()
    }
    return variances, count


def evaluate_nll(
    tensor,
    units,
    indices,
    mean,
    scale,
    target_indices,
    ar_intercept,
    ar_slope,
    ridge_beta,
    variances,
    target_names,
):
    sums = {
        name: np.zeros(len(target_indices), dtype=np.float64)
        for name in variances
    }
    count = 0
    constant = np.log(2 * np.pi)
    for index in indices:
        values = valid_unit_array(tensor, units, index)
        previous, target = standardized_pairs(values, mean, scale, target_indices)
        all_residuals = residuals(
            previous,
            target,
            target_indices,
            ar_intercept,
            ar_slope,
            ridge_beta,
        )
        count += len(previous)
        for name, matrix in all_residuals.items():
            variance = variances[name]
            nll = 0.5 * (
                constant + np.log(variance) + np.square(matrix) / variance
            )
            sums[name] += nll.sum(axis=0)

    rows = []
    for name, values in sums.items():
        per_target = values / count
        rows.append(
            {
                "model": name,
                "target": "__overall__",
                "heldout_nll": float(per_target.mean()),
                "time_pairs": count,
                "target_channels": len(target_names),
            }
        )
        for target_name, estimate in zip(target_names, per_target):
            rows.append(
                {
                    "model": name,
                    "target": target_name,
                    "heldout_nll": float(estimate),
                    "time_pairs": count,
                    "target_channels": 1,
                }
            )
    return pd.DataFrame(rows), count


def top_fraction_mean(values, fraction):
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Expected a nonempty time-by-channel matrix")
    k = max(1, int(np.ceil(len(values) * fraction)))
    partitioned = np.partition(values, len(values) - k, axis=0)
    return partitioned[-k:].mean(axis=0)


def make_features(
    tensor,
    units,
    plan,
    roles,
    mean,
    scale,
    target_indices,
    target_names,
    ar_intercept,
    ar_slope,
    ridge_beta,
    variances,
    fraction,
    cadence_seconds,
):
    unit_position = {unit: index for index, unit in enumerate(units["unit_id"])}
    role_map = roles.set_index("unit_id")["model_role"].to_dict()
    rows = []

    for unit_id, group in plan.groupby("unit_id", observed=True, sort=True):
        tensor_index = unit_position[unit_id]
        values = valid_unit_array(tensor, units, tensor_index)
        previous, target = standardized_pairs(values, mean, scale, target_indices)
        all_residuals = residuals(
            previous,
            target,
            target_indices,
            ar_intercept,
            ar_slope,
            ridge_beta,
        )
        standardized_residuals = {
            name: matrix / np.sqrt(variances[name])
            for name, matrix in all_residuals.items()
        }
        history_start = pd.Timestamp(units.loc[tensor_index, "history_start"])

        for planned in group.itertuples(index=False):
            # Residual row 0 predicts tensor time row 1.
            start_tensor = int(
                round((planned.window_start - history_start).total_seconds() / cadence_seconds)
            )
            end_tensor = int(
                round((planned.window_end - history_start).total_seconds() / cadence_seconds)
            )
            start_residual = max(0, start_tensor - 1)
            end_residual_exclusive = end_tensor
            if start_residual >= end_residual_exclusive:
                raise ValueError(f"Empty evaluation interval for {unit_id}")

            record = {
                "target_family": planned.target_family,
                "analysis_tier": planned.analysis_tier,
                "split_role": role_map[unit_id],
                "window_plan_role": planned.split_role,
                "unit_id": unit_id,
                "pair_id": planned.pair_id if pd.notna(planned.pair_id) else "",
                "source_cluster_id": (
                    planned.source_cluster_id
                    if pd.notna(planned.source_cluster_id)
                    else ""
                ),
                "window_start": planned.window_start,
                "window_end": planned.window_end,
                "evaluation_points": end_residual_exclusive - start_residual,
            }
            for model_name, matrix in standardized_residuals.items():
                absolute = np.abs(
                    matrix[start_residual:end_residual_exclusive]
                )
                feature = top_fraction_mean(absolute, fraction)
                for target_name, value in zip(target_names, feature):
                    record[f"{model_name}:{target_name}"] = float(value)
            rows.append(record)

    features = pd.DataFrame(rows)
    keys = ["target_family", "split_role", "unit_id"]
    if features.duplicated(keys).any():
        duplicate = features.loc[features.duplicated(keys, keep=False), keys]
        raise ValueError(f"Duplicate feature keys:\n{duplicate.head()}")
    feature_columns = [
        f"{model}:{target}"
        for model in ["persistence", "univariate_ar1", "cross_channel_ridge"]
        for target in target_names
    ]
    if not np.isfinite(features[feature_columns].to_numpy()).all():
        raise ValueError("Non-finite representation features")
    return features, feature_columns


def main():
    (
        sources,
        tensor,
        units,
        channels,
        plan,
        roles,
        protocol,
        seal_counts,
    ) = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cadence_seconds = 30
    if "cadence_seconds" in plan.columns:
        observed_cadences = sorted(
            plan["cadence_seconds"].dropna().astype(int).unique().tolist()
        )
        if observed_cadences != [cadence_seconds]:
            raise ValueError(
                "Frozen plan cadence changed: "
                f"expected {[cadence_seconds]}, observed {observed_cadences}"
            )
    ridge = float(protocol["forecasters"]["cross_channel_ridge"]["lambda"])
    variance_floor = float(protocol["variance_floor"])
    top_fraction = float(protocol["unit_feature"]["top_fraction"])

    if "Channel" in channels.columns:
        channel_column = "Channel"
    elif "channel" in channels.columns:
        channel_column = "channel"
    else:
        raise ValueError(
            "channels.csv must contain either 'Channel' or 'channel'"
        )
    channel_position = {
        name: index
        for index, name in enumerate(channels[channel_column].astype(str))
    }
    input_names = list(protocol["input_channels"])
    observed_input_names = channels[channel_column].astype(str).tolist()
    if observed_input_names != input_names:
        raise ValueError(
            "Tensor channel order differs from frozen protocol input order"
        )
    target_names = list(protocol["target_channels"])
    missing_targets = sorted(set(target_names) - set(channel_position))
    if missing_targets:
        raise ValueError(
            f"Frozen target channels absent from tensor metadata: {missing_targets}"
        )
    target_indices = np.asarray(
        [channel_position[name] for name in target_names], dtype=np.int64
    )
    role_by_unit = roles.set_index("unit_id")["model_role"]
    unit_position = {unit: index for index, unit in enumerate(units["unit_id"])}

    def indices_for(role):
        names = role_by_unit[role_by_unit == role].index.tolist()
        return np.asarray([unit_position[name] for name in names], dtype=np.int64)

    fit_indices = indices_for("forecaster_fit")
    nll_indices = indices_for("nll_validation")
    if len(fit_indices) != 103 or len(nll_indices) != 41:
        raise ValueError("Frozen forecaster/NLL roles changed")

    print("=== FIT FROZEN ESA MISSION-1 FORECASTERS ===")
    print("verified tensor seal entries:", seal_counts["tensor"])
    print("verified window-plan seal entries:", seal_counts["window_plan"])
    print("verified model-protocol seal entries:", seal_counts["model_protocol"])
    print("forecaster-fit independent units:", len(fit_indices))
    print("NLL-validation independent units:", len(nll_indices))
    print("anomaly units used for fitting: 0")
    print("nominal calibration units used for fitting: 0")
    print("nominal test units used for fitting: 0")
    print("anomaly test units used for fitting: 0")

    mean, scale, normalization_points = fit_normalization(
        tensor, units, fit_indices
    )
    (
        ar_intercept,
        ar_slope,
        ridge_beta,
        coefficient_pairs,
        xtx,
    ) = fit_models(
        tensor,
        units,
        fit_indices,
        mean,
        scale,
        target_indices,
        ridge,
    )
    variances, variance_pairs = fit_residual_variances(
        tensor,
        units,
        fit_indices,
        mean,
        scale,
        target_indices,
        ar_intercept,
        ar_slope,
        ridge_beta,
        variance_floor,
    )
    nll, nll_pairs = evaluate_nll(
        tensor,
        units,
        nll_indices,
        mean,
        scale,
        target_indices,
        ar_intercept,
        ar_slope,
        ridge_beta,
        variances,
        target_names,
    )

    np.savez_compressed(
        PARAMETERS_PATH,
        input_mean=mean,
        input_scale=scale,
        target_indices=target_indices,
        target_names=np.asarray(target_names),
        ar_intercept=ar_intercept,
        ar_slope=ar_slope,
        ridge_beta=ridge_beta,
        persistence_variance=variances["persistence"],
        univariate_ar1_variance=variances["univariate_ar1"],
        cross_channel_ridge_variance=variances["cross_channel_ridge"],
    )
    nll.to_csv(NLL_PATH, index=False)

    print("normalization time points:", normalization_points)
    print("coefficient-fit time pairs:", coefficient_pairs)
    print("variance-fit time pairs:", variance_pairs)
    print("NLL-validation time pairs:", nll_pairs)
    print("AR(1) slope range:", float(ar_slope.min()), "to", float(ar_slope.max()))
    diagnostic_penalty = np.eye(xtx.shape[0]) * ridge
    diagnostic_penalty[0, 0] = 0.0
    condition = float(np.linalg.cond(xtx + diagnostic_penalty))
    print("ridge normal-matrix condition number:", condition)

    print("\n=== HELD-OUT NOMINAL NLL ===")
    overall = nll[nll["target"] == "__overall__"].sort_values("heldout_nll")
    print(overall.to_string(index=False))
    selected = overall.iloc[0]
    print("minimum-NLL forecaster:", selected["model"])

    print("\n=== RESIDUAL SCALE RANGES ===")
    for name, variance in variances.items():
        standard_deviation = np.sqrt(variance)
        print(
            f"{name:22s} min={standard_deviation.min():.8f} "
            f"median={np.median(standard_deviation):.8f} "
            f"max={standard_deviation.max():.8f}"
        )

    print("\n=== EXPORT FROZEN REPRESENTATIONS ===")
    features, feature_columns = make_features(
        tensor,
        units,
        plan,
        roles,
        mean,
        scale,
        target_indices,
        target_names,
        ar_intercept,
        ar_slope,
        ridge_beta,
        variances,
        top_fraction,
        cadence_seconds,
    )
    features.to_csv(FEATURES_PATH, index=False)

    print("feature rows:", len(features))
    print("feature columns:", len(feature_columns))
    print("test-power calculations performed: 0")
    print("anomaly-test summaries printed: 0")
    print("\nFeature rows by frozen role:")
    print(
        features.groupby(["target_family", "split_role"], observed=True)
        .size().rename("rows").reset_index().to_string(index=False)
    )

    manifest = {
        "status": "complete_before_power_audit",
        "dataset": "ESA-Mission1",
        "models": ["persistence", "univariate_ar1", "cross_channel_ridge"],
        "reference_representation": "forecast_joint",
        "feature_columns": feature_columns,
        "feature_definition": protocol["unit_feature"],
        "target_channels": target_names,
        "input_channels": list(protocol["input_channels"]),
        "normalization_points": int(normalization_points),
        "coefficient_fit_pairs": int(coefficient_pairs),
        "variance_fit_pairs": int(variance_pairs),
        "nll_validation_pairs": int(nll_pairs),
        "minimum_nll_model": str(selected["model"]),
        "minimum_nll": float(selected["heldout_nll"]),
        "test_power_calculated": False,
        "sources": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sources
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sealed = [PARAMETERS_PATH, NLL_PATH, FEATURES_PATH, MANIFEST_PATH]
    SEAL_PATH.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in sealed),
        encoding="utf-8",
    )
    print("\n=== FILES WRITTEN ===")
    for path in sealed + [SEAL_PATH]:
        print(f"{path}: {path.stat().st_size:,} bytes")
    print("\nFINAL FORECAST-REPRESENTATION STATUS: PASS")


if __name__ == "__main__":
    main()
