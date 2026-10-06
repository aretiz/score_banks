#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAMBDA_GRID = np.asarray(
    [
        0.0,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ],
    dtype=float,
)

ALPHA_GRID = np.asarray(
    [
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
    ],
    dtype=float,
)

CONTRASTS = [
    (
        "single_multimodal_minus_image",
        "multimodal_single",
        "image_single",
    ),
    (
        "bank_multimodal_minus_image",
        "multimodal_bank",
        "image_bank",
    ),
    (
        "mixed_minus_image_bank",
        "mixed_bank",
        "image_bank",
    ),
    (
        "mixed_minus_multimodal_bank",
        "mixed_bank",
        "multimodal_bank",
    ),
    (
        "image_aggregation",
        "image_bank",
        "image_single",
    ),
    (
        "multimodal_aggregation",
        "multimodal_bank",
        "multimodal_single",
    ),
]

METHOD_ORDER = [
    "image_single",
    "multimodal_single",
    "image_bank",
    "multimodal_bank",
    "mixed_bank",
]

METHOD_LABELS = {
    "image_single": "Image single",
    "multimodal_single": "Multimodal single",
    "image_bank": "Image bank",
    "multimodal_bank": "Multimodal bank",
    "mixed_bank": "Mixed bank",
}


def conformal_p(reference, values):
    reference = np.sort(
        np.asarray(
            reference,
            dtype=float,
        )
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    greater_equal = (
        len(reference)
        - np.searchsorted(
            reference,
            values,
            side="left",
        )
    )

    return (
        1.0 + greater_equal
    ) / (
        len(reference) + 1.0
    )


def conformal_threshold(reference, alpha):
    reference = np.sort(
        np.asarray(
            reference,
            dtype=float,
        )
    )

    rank = int(
        np.ceil(
            (len(reference) + 1)
            * (1.0 - alpha)
        )
    )

    rank = min(
        max(rank, 1),
        len(reference),
    )

    return float(
        reference[rank - 1]
    )


def fit_bank(
    nominal,
    alternatives,
    alternative_names,
    feature_indices,
    ridge,
):
    nominal_sub = nominal[
        :,
        feature_indices,
    ]

    mu = nominal_sub.mean(axis=0)

    sd = nominal_sub.std(
        axis=0,
        ddof=1,
    )

    sd = np.maximum(
        sd,
        1e-8,
    )

    z0 = (
        nominal_sub - mu
    ) / sd

    covariance = np.cov(
        z0,
        rowvar=False,
        ddof=1,
    )

    covariance = np.atleast_2d(
        covariance
    )

    system = (
        covariance
        + float(ridge)
        * np.eye(
            covariance.shape[0]
        )
    )

    weights = []
    names = []

    for alternative_name, values in zip(
        alternative_names,
        alternatives,
    ):
        za = (
            values[
                :,
                feature_indices,
            ]
            - mu
        ) / sd

        displacement = np.mean(
            za - z0,
            axis=0,
        )

        weight = np.linalg.solve(
            system,
            displacement,
        )

        variance = float(
            weight
            @ covariance
            @ weight
        )

        if variance <= 1e-12:
            continue

        weights.append(weight)
        names.append(
            str(alternative_name)
        )

    if not weights:
        raise RuntimeError(
            "No usable bank heads."
        )

    return {
        "kind": "bank",
        "indices": np.asarray(
            feature_indices,
            dtype=int,
        ),
        "mu": mu,
        "sd": sd,
        "covariance": covariance,
        "weights": np.asarray(
            weights,
            dtype=float,
        ),
        "head_names": np.asarray(
            names,
            dtype=str,
        ),
        "ridge": float(ridge),
    }


def bank_head_scores(values, model):
    subset = values[
        :,
        model["indices"],
    ]

    standardized = (
        subset - model["mu"]
    ) / model["sd"]

    weights = model["weights"]
    covariance = model["covariance"]

    denominator = np.sqrt(
        np.maximum(
            np.einsum(
                "ki,ij,kj->k",
                weights,
                covariance,
                weights,
            ),
            1e-12,
        )
    )

    return (
        standardized
        @ weights.T
    ) / denominator[None, :]


def bank_score(values, model):
    return np.max(
        bank_head_scores(
            values,
            model,
        ),
        axis=1,
    )


def method_score(values, method):
    if method["kind"] == "single":
        index = method["index"]

        return (
            values[:, index]
            - method["mu"]
        ) / method["sd"]

    return bank_score(
        values,
        method,
    )


def bank_geometry(nominal, model):
    heads = bank_head_scores(
        nominal,
        model,
    )

    if heads.shape[1] == 1:
        return {
            "effective_rank": 1.0,
            "median_correlation": 1.0,
            "minimum_correlation": 1.0,
            "maximum_correlation": 1.0,
        }

    correlation = np.corrcoef(
        heads,
        rowvar=False,
    )

    eigenvalues = np.clip(
        np.linalg.eigvalsh(
            correlation
        ),
        0.0,
        None,
    )

    effective_rank = float(
        eigenvalues.sum() ** 2
        / np.sum(
            eigenvalues ** 2
        )
    )

    off_diagonal = correlation[
        np.triu_indices_from(
            correlation,
            k=1,
        )
    ]

    return {
        "effective_rank":
            effective_rank,
        "median_correlation":
            float(
                np.median(
                    off_diagonal
                )
            ),
        "minimum_correlation":
            float(
                np.min(
                    off_diagonal
                )
            ),
        "maximum_correlation":
            float(
                np.max(
                    off_diagonal
                )
            ),
    }


def design_power_for_bank(
    nominal,
    alternatives,
    model,
    alpha,
):
    reference = bank_score(
        nominal,
        model,
    )

    powers = []

    for values in alternatives:
        pvalues = conformal_p(
            reference,
            bank_score(
                values,
                model,
            ),
        )

        powers.append(
            float(
                np.mean(
                    pvalues <= alpha
                )
            )
        )

    powers = np.asarray(
        powers,
        dtype=float,
    )

    return {
        "mean": float(
            powers.mean()
        ),
        "median": float(
            np.median(
                powers
            )
        ),
        "minimum": float(
            powers.min()
        ),
        "maximum": float(
            powers.max()
        ),
    }


def select_bank(
    scope,
    nominal,
    alternatives,
    alternative_names,
    feature_indices,
    alpha,
):
    rows = []
    candidates = []

    for ridge in LAMBDA_GRID:
        model = fit_bank(
            nominal,
            alternatives,
            alternative_names,
            feature_indices,
            ridge,
        )

        power = design_power_for_bank(
            nominal,
            alternatives,
            model,
            alpha,
        )

        geometry = bank_geometry(
            nominal,
            model,
        )

        row = {
            "scope": scope,
            "lambda": float(ridge),
            "mean_design_power":
                power["mean"],
            "median_design_power":
                power["median"],
            "minimum_design_power":
                power["minimum"],
            "maximum_design_power":
                power["maximum"],
            "n_heads":
                len(
                    model["head_names"]
                ),
            **geometry,
        }

        rows.append(row)
        candidates.append(
            (
                model,
                power["mean"],
                float(ridge),
            )
        )

    maximum = max(
        value
        for _, value, _
        in candidates
    )

    tied = [
        item
        for item in candidates
        if np.isclose(
            item[1],
            maximum,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    selected_model, _, selected_lambda = max(
        tied,
        key=lambda item: item[2],
    )

    for row in rows:
        row["selected_on_design"] = bool(
            np.isclose(
                row["lambda"],
                selected_lambda,
            )
        )

    return (
        selected_model,
        rows,
    )


def select_single(
    scope,
    nominal,
    alternatives,
    feature_names,
    feature_indices,
    alpha,
):
    rows = []

    for index in feature_indices:
        powers = []

        for values in alternatives:
            pvalues = conformal_p(
                nominal[:, index],
                values[:, index],
            )

            powers.append(
                float(
                    np.mean(
                        pvalues <= alpha
                    )
                )
            )

        rows.append(
            {
                "scope": scope,
                "feature_index":
                    int(index),
                "feature":
                    str(
                        feature_names[index]
                    ),
                "mean_design_power":
                    float(
                        np.mean(
                            powers
                        )
                    ),
                "median_design_power":
                    float(
                        np.median(
                            powers
                        )
                    ),
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        [
            "mean_design_power",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    )

    selected = frame.iloc[0]

    index = int(
        selected["feature_index"]
    )

    sd = float(
        nominal[:, index].std(
            ddof=1
        )
    )

    model = {
        "kind": "single",
        "scope": scope,
        "index": index,
        "feature_name":
            str(
                feature_names[index]
            ),
        "mu": float(
            nominal[:, index].mean()
        ),
        "sd": max(
            sd,
            1e-8,
        ),
        "design_mean_power":
            float(
                selected[
                    "mean_design_power"
                ]
            ),
    }

    return model, frame


def interpolate_power(
    false_positive_rates,
    powers,
    target=0.05,
):
    frame = pd.DataFrame(
        {
            "fpr":
                np.asarray(
                    false_positive_rates,
                    dtype=float,
                ),
            "power":
                np.asarray(
                    powers,
                    dtype=float,
                ),
        }
    )

    frame = (
        frame.groupby(
            "fpr",
            as_index=False,
        )["power"]
        .mean()
        .sort_values("fpr")
    )

    return float(
        np.interp(
            target,
            frame["fpr"],
            frame["power"],
            left=frame[
                "power"
            ].iloc[0],
            right=frame[
                "power"
            ].iloc[-1],
        )
    )


def simultaneous_intervals(
    scenario_records,
    scenario_metadata,
    contrasts,
    alpha,
    bootstrap,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    first = next(
        iter(
            scenario_records.values()
        )
    )

    n_units = len(first)

    counts = rng.multinomial(
        n_units,
        np.full(
            n_units,
            1.0 / n_units,
        ),
        size=bootstrap,
    ).astype(
        np.float64
    )

    rows = []
    deviations = np.zeros(
        bootstrap,
        dtype=float,
    )

    for scenario_index, metadata in enumerate(
        scenario_metadata
    ):
        for (
            contrast_name,
            method_a,
            method_b,
        ) in contrasts:
            reject_a = (
                scenario_records[
                    (
                        scenario_index,
                        method_a,
                    )
                ]
                <= alpha
            ).astype(float)

            reject_b = (
                scenario_records[
                    (
                        scenario_index,
                        method_b,
                    )
                ]
                <= alpha
            ).astype(float)

            difference = (
                reject_a - reject_b
            )

            estimate = float(
                difference.mean()
            )

            boot = (
                counts @ difference
            ) / float(n_units)

            deviations = np.maximum(
                deviations,
                np.abs(
                    boot - estimate
                ),
            )

            rows.append(
                {
                    **metadata,
                    "contrast":
                        contrast_name,
                    "method_a":
                        method_a,
                    "method_b":
                        method_b,
                    "estimate":
                        estimate,
                }
            )

    radius = float(
        np.quantile(
            deviations,
            0.95,
        )
    )

    for row in rows:
        row["simul_ci_low"] = max(
            -1.0,
            row["estimate"] - radius,
        )

        row["simul_ci_high"] = min(
            1.0,
            row["estimate"] + radius,
        )

        if row["simul_ci_low"] > 0:
            row["result"] = (
                "method_a_positive"
            )
        elif row["simul_ci_high"] < 0:
            row["result"] = (
                "method_b_positive"
            )
        else:
            row["result"] = "unresolved"

        row["bootstrap_unit"] = (
            "cube_id"
        )

        row["bootstrap_reps"] = int(
            bootstrap
        )

        row["simultaneous_radius"] = (
            radius
        )

    return pd.DataFrame(rows)


def make_summary_figure(
    method_summary,
    output,
):
    frame = (
        method_summary
        .set_index("method")
        .loc[METHOD_ORDER]
        .reset_index()
    )

    x = np.arange(
        len(frame)
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 4.8),
        constrained_layout=True,
    )

    bars = ax.bar(
        x,
        100.0
        * frame[
            "mean_matched_5pct_power"
        ],
        color=[
            "#4C78A8",
            "#72A0CF",
            "#F58518",
            "#FFBF79",
            "#54A24B",
        ],
        alpha=0.82,
        width=0.68,
        label="Mean",
    )

    ax.scatter(
        x,
        100.0
        * frame[
            "median_matched_5pct_power"
        ],
        color="black",
        marker="D",
        s=36,
        zorder=4,
        label="Median",
    )

    for index, bar in enumerate(bars):
        fpr = (
            100.0
            * frame.loc[
                index,
                "native_test_fpr",
            ]
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"FPR {fpr:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            METHOD_LABELS[value]
            for value
            in frame["method"]
        ],
        rotation=24,
        ha="right",
    )

    ax.set_ylabel(
        "Power at matched 5% FPR (%)"
    )

    ax.set_title(
        "Representation and aggregation "
        "must be evaluated jointly"
    )

    ax.grid(
        axis="y",
        alpha=0.22,
    )

    ax.legend(
        frameon=False,
    )

    for extension in [
        "pdf",
        "png",
    ]:
        fig.savefig(
            output
            / (
                "representation_"
                "ablation_summary."
                + extension
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "outputs/"
            "care_earthnet_representation_export/"
            "representation_ablation_features.npz"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "outputs/"
            "care_representation_ablation"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260902,
    )

    args = parser.parse_args()

    output = Path(args.out)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        args.input,
        allow_pickle=False,
    ) as data:
        feature_names = data[
            "feature_names"
        ].astype(str)

        design_nominal = data[
            "design_nominal"
        ].astype(float)

        design_names = data[
            "design_alt_names"
        ].astype(str)

        design_alternatives = data[
            "design_alternatives"
        ].astype(float)

        final_conformal = data[
            "final_conformal"
        ].astype(float)

        audit_nominal = data[
            "audit_nominal"
        ].astype(float)

        test_nominal = data[
            "test_nominal"
        ].astype(float)

        scenario_structure = data[
            "scenario_structure"
        ].astype(str)

        scenario_amount = data[
            "scenario_amount"
        ].astype(float)

        scenario_delta = data[
            "scenario_delta"
        ].astype(float)

        scenario_features = data[
            "scenario_features"
        ].astype(float)

    image_indices = np.asarray(
        [
            index
            for index, name
            in enumerate(feature_names)
            if name.startswith(
                "image_only_"
            )
        ],
        dtype=int,
    )

    multimodal_indices = np.asarray(
        [
            index
            for index, name
            in enumerate(feature_names)
            if name.startswith(
                "multimodal_"
            )
        ],
        dtype=int,
    )

    mixed_indices = np.arange(
        len(feature_names),
        dtype=int,
    )

    if (
        len(image_indices) != 9
        or len(multimodal_indices) != 9
    ):
        raise RuntimeError(
            "Expected nine image-only and "
            "nine multimodal features; found "
            f"{len(image_indices)} and "
            f"{len(multimodal_indices)}."
        )

    image_single, image_single_table = (
        select_single(
            "image",
            design_nominal,
            design_alternatives,
            feature_names,
            image_indices,
            args.alpha,
        )
    )

    multimodal_single, multimodal_single_table = (
        select_single(
            "multimodal",
            design_nominal,
            design_alternatives,
            feature_names,
            multimodal_indices,
            args.alpha,
        )
    )

    image_bank, image_sweep = select_bank(
        "image",
        design_nominal,
        design_alternatives,
        design_names,
        image_indices,
        args.alpha,
    )

    multimodal_bank, multimodal_sweep = (
        select_bank(
            "multimodal",
            design_nominal,
            design_alternatives,
            design_names,
            multimodal_indices,
            args.alpha,
        )
    )

    mixed_bank, mixed_sweep = select_bank(
        "mixed",
        design_nominal,
        design_alternatives,
        design_names,
        mixed_indices,
        args.alpha,
    )

    methods = {
        "image_single":
            image_single,
        "multimodal_single":
            multimodal_single,
        "image_bank":
            image_bank,
        "multimodal_bank":
            multimodal_bank,
        "mixed_bank":
            mixed_bank,
    }

    calibration = np.vstack(
        [
            final_conformal,
            audit_nominal,
        ]
    )

    nominal_pvalues = {}
    calibration_scores = {}

    for method_name, method in methods.items():
        reference = method_score(
            calibration,
            method,
        )

        calibration_scores[
            method_name
        ] = reference

        nominal_pvalues[
            method_name
        ] = conformal_p(
            reference,
            method_score(
                test_nominal,
                method,
            ),
        )

    scenario_metadata = [
        {
            "scenario_index": int(index),
            "structure":
                str(
                    scenario_structure[index]
                ),
            "amount":
                float(
                    scenario_amount[index]
                ),
            "delta":
                float(
                    scenario_delta[index]
                ),
        }
        for index in range(
            len(scenario_structure)
        )
    ]

    scenario_records = {}
    native_rows = []
    operating_rows = []
    matched_rows = []

    for scenario_index, metadata in enumerate(
        scenario_metadata
    ):
        values = scenario_features[
            scenario_index
        ]

        for method_name, method in methods.items():
            pvalues = conformal_p(
                calibration_scores[
                    method_name
                ],
                method_score(
                    values,
                    method,
                ),
            )

            scenario_records[
                (
                    scenario_index,
                    method_name,
                )
            ] = pvalues

            native_rows.append(
                {
                    **metadata,
                    "method":
                        method_name,
                    "native_power":
                        float(
                            np.mean(
                                pvalues
                                <= args.alpha
                            )
                        ),
                }
            )

            fprs = []
            powers = []

            for alpha in ALPHA_GRID:
                fpr = float(
                    np.mean(
                        nominal_pvalues[
                            method_name
                        ]
                        <= alpha
                    )
                )

                power = float(
                    np.mean(
                        pvalues <= alpha
                    )
                )

                fprs.append(fpr)
                powers.append(power)

                operating_rows.append(
                    {
                        **metadata,
                        "method":
                            method_name,
                        "alpha":
                            float(alpha),
                        "test_nominal_fpr":
                            fpr,
                        "power":
                            power,
                    }
                )

            matched_rows.append(
                {
                    **metadata,
                    "method":
                        method_name,
                    "matched_5pct_power":
                        interpolate_power(
                            fprs,
                            powers,
                            target=0.05,
                        ),
                }
            )

    native = pd.DataFrame(
        native_rows
    )

    matched = pd.DataFrame(
        matched_rows
    )

    scenarios = native.merge(
        matched,
        on=[
            "scenario_index",
            "structure",
            "amount",
            "delta",
            "method",
        ],
        validate="one_to_one",
    )

    intervals = simultaneous_intervals(
        scenario_records,
        scenario_metadata,
        CONTRASTS,
        args.alpha,
        args.bootstrap,
        args.seed,
    )

    method_rows = []

    for method_name in METHOD_ORDER:
        method = methods[
            method_name
        ]

        matched_method = matched.loc[
            matched["method"].eq(
                method_name
            )
        ]

        native_fpr = float(
            np.mean(
                nominal_pvalues[
                    method_name
                ]
                <= args.alpha
            )
        )

        if method["kind"] == "single":
            method_rows.append(
                {
                    "method":
                        method_name,
                    "scope":
                        method["scope"],
                    "kind":
                        "single",
                    "selected_feature":
                        method[
                            "feature_name"
                        ],
                    "n_features": 1,
                    "n_heads": 1,
                    "selected_lambda":
                        np.nan,
                    "effective_rank": 1.0,
                    "median_head_correlation":
                        1.0,
                    "design_mean_power":
                        method[
                            "design_mean_power"
                        ],
                    "threshold":
                        conformal_threshold(
                            calibration_scores[
                                method_name
                            ],
                            args.alpha,
                        ),
                    "native_test_fpr":
                        native_fpr,
                    "mean_matched_5pct_power":
                        float(
                            matched_method[
                                "matched_5pct_power"
                            ].mean()
                        ),
                    "median_matched_5pct_power":
                        float(
                            matched_method[
                                "matched_5pct_power"
                            ].median()
                        ),
                }
            )
        else:
            geometry = bank_geometry(
                design_nominal,
                method,
            )

            scope = (
                method_name
                .replace(
                    "_bank",
                    "",
                )
            )

            sweep = pd.DataFrame(
                {
                    "image":
                        image_sweep,
                    "multimodal":
                        multimodal_sweep,
                    "mixed":
                        mixed_sweep,
                }[scope]
            )

            selected = sweep.loc[
                sweep[
                    "selected_on_design"
                ]
            ].iloc[0]

            method_rows.append(
                {
                    "method":
                        method_name,
                    "scope":
                        scope,
                    "kind":
                        "bank",
                    "selected_feature":
                        "",
                    "n_features":
                        len(
                            method["indices"]
                        ),
                    "n_heads":
                        len(
                            method[
                                "head_names"
                            ]
                        ),
                    "selected_lambda":
                        method["ridge"],
                    **geometry,
                    "design_mean_power":
                        float(
                            selected[
                                "mean_design_power"
                            ]
                        ),
                    "threshold":
                        conformal_threshold(
                            calibration_scores[
                                method_name
                            ],
                            args.alpha,
                        ),
                    "native_test_fpr":
                        native_fpr,
                    "mean_matched_5pct_power":
                        float(
                            matched_method[
                                "matched_5pct_power"
                            ].mean()
                        ),
                    "median_matched_5pct_power":
                        float(
                            matched_method[
                                "matched_5pct_power"
                            ].median()
                        ),
                }
            )

    method_summary = pd.DataFrame(
        method_rows
    )

    family_summary = (
        scenarios.groupby(
            [
                "method",
                "structure",
            ],
            observed=True,
        )
        .agg(
            scenarios=(
                "scenario_index",
                "nunique",
            ),
            mean_native_power=(
                "native_power",
                "mean",
            ),
            median_native_power=(
                "native_power",
                "median",
            ),
            mean_matched_5pct_power=(
                "matched_5pct_power",
                "mean",
            ),
            median_matched_5pct_power=(
                "matched_5pct_power",
                "median",
            ),
        )
        .reset_index()
    )

    matched_lookup = matched.set_index(
        [
            "scenario_index",
            "method",
        ]
    )["matched_5pct_power"]

    matched_contrast_rows = []

    for metadata in scenario_metadata:
        scenario_index = metadata[
            "scenario_index"
        ]

        for (
            contrast_name,
            method_a,
            method_b,
        ) in CONTRASTS:
            matched_contrast_rows.append(
                {
                    **metadata,
                    "contrast":
                        contrast_name,
                    "method_a":
                        method_a,
                    "method_b":
                        method_b,
                    "matched_5pct_difference":
                        float(
                            matched_lookup.loc[
                                (
                                    scenario_index,
                                    method_a,
                                )
                            ]
                            - matched_lookup.loc[
                                (
                                    scenario_index,
                                    method_b,
                                )
                            ]
                        ),
                }
            )

    matched_contrasts = pd.DataFrame(
        matched_contrast_rows
    )

    ridge_sweep = pd.DataFrame(
        image_sweep
        + multimodal_sweep
        + mixed_sweep
    )

    single_selection = pd.concat(
        [
            image_single_table,
            multimodal_single_table,
        ],
        ignore_index=True,
    )

    operating_curve = pd.DataFrame(
        operating_rows
    )

    scenarios.to_csv(
        output / "scenario_power.csv",
        index=False,
    )

    method_summary.to_csv(
        output / "method_summary.csv",
        index=False,
    )

    family_summary.to_csv(
        output / "family_summary.csv",
        index=False,
    )

    intervals.to_csv(
        output
        / "simultaneous_contrasts.csv",
        index=False,
    )

    matched_contrasts.to_csv(
        output
        / "matched_5pct_contrasts.csv",
        index=False,
    )

    ridge_sweep.to_csv(
        output
        / "ridge_selection.csv",
        index=False,
    )

    single_selection.to_csv(
        output
        / "single_feature_selection.csv",
        index=False,
    )

    operating_curve.to_csv(
        output
        / "operating_curve.csv",
        index=False,
    )

    make_summary_figure(
        method_summary,
        output,
    )

    contrast_counts = (
        intervals.groupby(
            [
                "contrast",
                "result",
            ],
            observed=True,
        )
        .size()
        .rename("scenarios")
        .reset_index()
    )

    decision = {
        "alpha": args.alpha,
        "bootstrap_replicates":
            args.bootstrap,
        "bootstrap_unit": "cube_id",
        "calibration_units":
            int(
                len(calibration)
            ),
        "test_units":
            int(
                len(test_nominal)
            ),
        "methods":
            METHOD_ORDER,
        "contrasts":
            [
                value[0]
                for value in CONTRASTS
            ],
        "interpretation": (
            "Representation claims require "
            "modality contrasts; matched-FPR "
            "results are descriptive and native "
            "simultaneous intervals are primary."
        ),
    }

    (
        output / "audit_metadata.json"
    ).write_text(
        json.dumps(
            decision,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== REPRESENTATION METHODS ==="
    )

    print(
        method_summary[
            [
                "method",
                "selected_feature",
                "n_features",
                "n_heads",
                "selected_lambda",
                "effective_rank",
                "threshold",
                "native_test_fpr",
                "design_mean_power",
                "mean_matched_5pct_power",
                "median_matched_5pct_power",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== SIMULTANEOUS CONTRAST COUNTS ==="
    )

    print(
        contrast_counts.to_string(
            index=False
        )
    )

    resolved = intervals.loc[
        ~intervals[
            "result"
        ].eq("unresolved")
    ]

    print(
        "\n=== RESOLVED CONTRASTS ==="
    )

    if resolved.empty:
        print("None")
    else:
        print(
            resolved[
                [
                    "structure",
                    "amount",
                    "delta",
                    "contrast",
                    "estimate",
                    "simul_ci_low",
                    "simul_ci_high",
                    "result",
                ]
            ]
            .sort_values(
                [
                    "contrast",
                    "estimate",
                ]
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== OUTPUT FILES ==="
    )

    for path in sorted(
        output.iterdir()
    ):
        if path.is_file():
            print(
                f"{path.name}: "
                f"{path.stat().st_size:,} bytes"
            )


if __name__ == "__main__":
    main()
