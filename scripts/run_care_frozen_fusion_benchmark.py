#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from run_caps_z_earthnet_from_cache import (
    load_cache,
    filter_valid_fraction,
    fit_pipeline,
)

from run_care_earthnet import (
    shared_split5,
    feature_matrix,
    choose_one_window_per_cube,
    inject_pair,
    design_scenarios,
    holdout_scenarios,
    care_score,
    conformal_p,
)


SEED = 20260824
ALPHA = 0.05

LEARNED = [
    "shrinkage_lda",
    "logistic",
    "gradient_boosting",
    "mlp",
]

DEPLOYABLE_BASELINES = [
    "shrinkage_lda",
    "logistic",
    "gradient_boosting",
    "mlp",
    "feature_mean",
    "feature_max",
    "best_single",
]


def load_care_fit(path):
    z = np.load(path, allow_pickle=False)

    return {
        "mu": z["mu"].astype(float),
        "sd": z["sd"].astype(float),
        "Sigma": z["Sigma"].astype(float),
        "weights": z["weights"].astype(float),
        "deltas": z["deltas"].astype(float),
        "feature_names":
            z["feature_names"].astype(str).tolist(),
    }


def exact_ci(rej):
    rej = np.asarray(rej, bool)
    k = int(rej.sum())
    n = len(rej)

    ci = binomtest(
        k,
        n,
    ).proportion_ci(0.95)

    return (
        k,
        n,
        float(ci.low),
        float(ci.high),
    )


# ---------------------------------------------------------------------
# Standard learned-fusion baselines
# ---------------------------------------------------------------------

def fit_standard_baselines(
    X0,
    design_alternatives,
    fit,
    seed,
):
    """
    Frozen pooled supervised training.

    Each target scenario contributes one nominal copy and
    one paired alternative copy, so classes are balanced
    and target scenarios have equal weight.
    """
    mu = fit["mu"]
    sd = fit["sd"]

    Z0 = (
        X0 - mu
    ) / sd

    XX = []
    yy = []

    for _, Xa in design_alternatives:
        Za = (
            Xa - mu
        ) / sd

        XX.append(Z0)
        yy.append(
            np.zeros(
                len(Z0),
                dtype=int,
            )
        )

        XX.append(Za)
        yy.append(
            np.ones(
                len(Za),
                dtype=int,
            )
        )

    Xtrain = np.vstack(XX)
    ytrain = np.concatenate(yy)

    rng = np.random.default_rng(seed)

    p = rng.permutation(
        len(ytrain)
    )

    Xtrain = Xtrain[p]
    ytrain = ytrain[p]

    models = {
        # Classical pooled shrinkage discriminant.
        "shrinkage_lda":
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            ),

        # Ordinary L2 logistic regression.
        "logistic":
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                random_state=seed,
            ),

        # Standard sklearn gradient boosting defaults.
        "gradient_boosting":
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=seed,
            ),

        # Prespecified small MLP.
        "mlp":
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=False,
                random_state=seed,
            ),
    }

    for name, model in models.items():
        print("FIT", name)
        model.fit(
            Xtrain,
            ytrain,
        )

    return models


def score_methods(
    X,
    fit,
    models,
    best_j,
):
    Z = (
        X - fit["mu"]
    ) / fit["sd"]

    out = {
        "care":
            care_score(
                X,
                fit,
            ),

        # Simple fixed fusion.
        "feature_mean":
            Z.mean(axis=1),

        "feature_max":
            Z.max(axis=1),

        # Frozen design-selected individual feature.
        "best_single":
            X[:, best_j],
    }

    out["shrinkage_lda"] = (
        models[
            "shrinkage_lda"
        ].decision_function(Z)
    )

    out["logistic"] = (
        models[
            "logistic"
        ].decision_function(Z)
    )

    out["gradient_boosting"] = (
        models[
            "gradient_boosting"
        ].predict_proba(Z)[:, 1]
    )

    out["mlp"] = (
        models[
            "mlp"
        ].predict_proba(Z)[:, 1]
    )

    return out


# ---------------------------------------------------------------------
# Exact empirical matched-FPR rule
# ---------------------------------------------------------------------

def make_matched_rule(
    nominal_score,
    target_alarms,
):
    """
    Descriptive matched-FPR analysis.

    Creates a deterministic threshold/tie rule yielding exactly
    target_alarms on the evaluation nominal cubes.

    This is NOT a deployable calibration procedure.
    """
    x = np.asarray(
        nominal_score,
        float,
    )

    n = len(x)

    k = int(target_alarms)

    tie_key = np.arange(
        n,
        dtype=int,
    )

    if k <= 0:
        return {
            "mode": "none",
            "n": n,
        }

    if k >= n:
        return {
            "mode": "all",
            "n": n,
        }

    # Ascending lexicographic order:
    # score first, deterministic cube position second.
    order = np.lexsort(
        (
            tie_key,
            x,
        )
    )

    boundary = int(
        order[-k]
    )

    threshold = float(
        x[boundary]
    )

    key_cut = int(
        tie_key[boundary]
    )

    return {
        "mode": "threshold",
        "threshold": threshold,
        "key_cut": key_cut,
        "n": n,
    }


def apply_matched_rule(
    score,
    rule,
):
    x = np.asarray(
        score,
        float,
    )

    n = len(x)

    if n != rule["n"]:
        raise RuntimeError(
            "Matched-FPR unit mismatch."
        )

    if rule["mode"] == "none":
        return np.zeros(
            n,
            dtype=bool,
        )

    if rule["mode"] == "all":
        return np.ones(
            n,
            dtype=bool,
        )

    key = np.arange(
        n,
        dtype=int,
    )

    t = rule["threshold"]
    kc = rule["key_cut"]

    return (
        (x > t)
        |
        (
            (x == t)
            & (key >= kc)
        )
    )


# ---------------------------------------------------------------------
# Simultaneous paired bootstrap
# ---------------------------------------------------------------------

def simultaneous_ci(
    cells,
    n_boot,
    seed,
):
    """
    Simultaneous 95% CIs across all scenario × comparator cells.

    Independent resampling unit = cube_id.
    """
    if not cells:
        return pd.DataFrame()

    n = len(
        cells[0]["diff"]
    )

    D = np.column_stack([
        c["diff"].astype(float)
        for c in cells
    ])

    estimates = D.mean(axis=0)

    rng = np.random.default_rng(seed)

    counts = rng.multinomial(
        n,
        np.full(
            n,
            1.0 / n,
        ),
        size=n_boot,
    ).astype(float)

    boot = (
        counts @ D
    ) / float(n)

    maxdev = np.max(
        np.abs(
            boot
            - estimates[None, :]
        ),
        axis=1,
    )

    q = float(
        np.quantile(
            maxdev,
            0.95,
        )
    )

    rows = []

    for j, c in enumerate(cells):
        est = float(
            estimates[j]
        )

        rows.append({
            "metric": c["metric"],
            "structure": c["structure"],
            "amount": c["amount"],
            "delta": c["delta"],
            "baseline": c["baseline"],
            "care_minus_baseline": est,
            "simul_ci_low": max(
                -1.0,
                est - q,
            ),
            "simul_ci_high": min(
                1.0,
                est + q,
            ),
            "bootstrap_unit":
                "cube_id",
            "bootstrap_reps":
                n_boot,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    ap.add_argument(
        "--out",
        default=(
            "outputs/"
            "care_frozen_fusion_benchmark"
        ),
    )

    args = ap.parse_args()

    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Load exact frozen EarthNet caches.
    # ------------------------------------------------------------

    fc = filter_valid_fraction(
        load_cache(
            "outputs/earthnet/"
            "caps_cal.npz",
            "nchw",
        ),
        0.05,
    )

    ft = filter_valid_fraction(
        load_cache(
            "outputs/earthnet/"
            "caps_test.npz",
            "nchw",
        ),
        0.05,
    )

    ic = filter_valid_fraction(
        load_cache(
            "outputs/"
            "earthnet_image_only/"
            "caps_cal.npz",
            "nchw",
        ),
        0.05,
    )

    it = filter_valid_fraction(
        load_cache(
            "outputs/"
            "earthnet_image_only/"
            "caps_test.npz",
            "nchw",
        ),
        0.05,
    )

    fp, ip, groups = (
        shared_split5(
            fc,
            ic,
            SEED,
        )
    )

    # Verify against frozen CARE split.
    frozen = np.load(
        "outputs/care_earthnet/"
        "frozen_cube_splits.npz",
        allow_pickle=False,
    )

    saved_groups = [
        frozen["local"],
        frozen["component"],
        frozen["design"],
        frozen["final_conformal"],
        frozen["audit"],
    ]

    for a, b in zip(
        groups,
        saved_groups,
    ):
        if not np.array_equal(
            a,
            b,
        ):
            raise RuntimeError(
                "Frozen cube split drift."
            )

    (
        f_local,
        f_comp,
        f_design,
        f_final,
        f_audit,
    ) = fp

    (
        i_local,
        i_comp,
        i_design,
        i_final,
        i_audit,
    ) = ip

    windows = [
        1, 2, 4, 8, 16
    ]

    f_local_cal, f_comp_norm, _ = (
        fit_pipeline(
            f_local,
            f_comp,
            f_design,
            windows,
            30,
        )
    )

    i_local_cal, i_comp_norm, _ = (
        fit_pipeline(
            i_local,
            i_comp,
            i_design,
            windows,
            30,
        )
    )

    fit = load_care_fit(
        "outputs/care_earthnet/"
        "care_fit.npz"
    )

    with open(
        "outputs/care_earthnet/"
        "care_kill_decision.json"
    ) as f:
        old_decision = json.load(f)

    best_feature = old_decision[
        "best_single_feature"
    ]

    if best_feature not in (
        fit["feature_names"]
    ):
        raise RuntimeError(
            "Frozen best feature missing."
        )

    best_j = (
        fit["feature_names"]
        .index(best_feature)
    )

    print(
        "\n=== FROZEN DESIGN-SELECTED SINGLE ==="
    )
    print(best_feature)

    # ------------------------------------------------------------
    # Reconstruct nominal design representation.
    # ------------------------------------------------------------

    design_ids, X0, names = (
        feature_matrix(
            f_design,
            i_design,
            f_local_cal,
            f_comp_norm,
            i_local_cal,
            i_comp_norm,
            windows,
        )
    )

    if names != fit[
        "feature_names"
    ]:
        raise RuntimeError(
            "Feature-bank drift."
        )

    # Confirm frozen CARE transformation.
    if not np.allclose(
        X0.mean(axis=0),
        fit["mu"],
        atol=1e-10,
        rtol=1e-8,
    ):
        raise RuntimeError(
            "Frozen CARE mean drift."
        )

    if not np.allclose(
        X0.std(
            axis=0,
            ddof=1,
        ),
        fit["sd"],
        atol=1e-10,
        rtol=1e-8,
    ):
        raise RuntimeError(
            "Frozen CARE scale drift."
        )

    # ------------------------------------------------------------
    # Reconstruct exact frozen DESIGN alternatives.
    # ------------------------------------------------------------

    selected_design = (
        choose_one_window_per_cube(
            f_design,
            SEED + 101,
        )
    )

    design_alt = []

    ds = design_scenarios()

    for si, (
        structure,
        amount,
        delta,
    ) in enumerate(ds):

        print(
            f"DESIGN {si+1}/{len(ds)} "
            f"{structure} "
            f"{amount:g} "
            f"{delta:g}"
        )

        fa, ia, _ = inject_pair(
            f_design,
            i_design,
            selected_design,
            structure,
            amount,
            delta,
            SEED + 1000 + si,
        )

        ids, Xa, names2 = (
            feature_matrix(
                fa,
                ia,
                f_local_cal,
                f_comp_norm,
                i_local_cal,
                i_comp_norm,
                windows,
            )
        )

        assert np.array_equal(
            ids,
            design_ids,
        )

        assert names2 == names

        tag = (
            f"{structure}"
            f"_a{amount:g}"
            f"_d{delta:g}"
        )

        design_alt.append(
            (tag, Xa)
        )

    # ------------------------------------------------------------
    # Fit all STANDARD baselines once.
    # CARE itself remains completely untouched.
    # ------------------------------------------------------------

    models = fit_standard_baselines(
        X0,
        design_alt,
        fit,
        SEED + 30001,
    )

    # ------------------------------------------------------------
    # Frozen final/audit/test scores.
    # ------------------------------------------------------------

    final_ids, Xfinal, _ = (
        feature_matrix(
            f_final,
            i_final,
            f_local_cal,
            f_comp_norm,
            i_local_cal,
            i_comp_norm,
            windows,
        )
    )

    audit_ids, Xaudit, _ = (
        feature_matrix(
            f_audit,
            i_audit,
            f_local_cal,
            f_comp_norm,
            i_local_cal,
            i_comp_norm,
            windows,
        )
    )

    test_ids, Xtest0, _ = (
        feature_matrix(
            ft,
            it,
            f_local_cal,
            f_comp_norm,
            i_local_cal,
            i_comp_norm,
            windows,
        )
    )

    final_scores = score_methods(
        Xfinal,
        fit,
        models,
        best_j,
    )

    audit_scores = score_methods(
        Xaudit,
        fit,
        models,
        best_j,
    )

    test_scores0 = score_methods(
        Xtest0,
        fit,
        models,
        best_j,
    )

    method_names = [
        "care",
        *DEPLOYABLE_BASELINES,
    ]

    # ------------------------------------------------------------
    # Native-alpha validity.
    # ------------------------------------------------------------

    validity_rows = []

    native_nominal_reject = {}

    for split_name, scores in [
        (
            "audit",
            audit_scores,
        ),
        (
            "test_nominal",
            test_scores0,
        ),
    ]:

        for method in method_names:
            p = conformal_p(
                final_scores[method],
                scores[method],
            )

            rej = (
                p <= ALPHA
            )

            k, n, lo, hi = (
                exact_ci(rej)
            )

            validity_rows.append({
                "split": split_name,
                "method": method,
                "n_cubes": n,
                "alarms": k,
                "fpr": float(
                    rej.mean()
                ),
                "exact_ci_low": lo,
                "exact_ci_high": hi,
            })

            if (
                split_name
                == "test_nominal"
            ):
                native_nominal_reject[
                    method
                ] = rej

    validity = pd.DataFrame(
        validity_rows
    )

    validity.to_csv(
        out / "validity.csv",
        index=False,
    )

    print(
        "\n=== FROZEN FUSION VALIDITY ==="
    )
    print(
        validity.to_string(
            index=False
        )
    )

    # CARE's actual native-alpha nominal FPR
    # is the target for descriptive matched-FPR comparison.
    care_nominal_rej = (
        native_nominal_reject[
            "care"
        ]
    )

    matched_alarm_count = int(
        care_nominal_rej.sum()
    )

    matched_target_fpr = (
        matched_alarm_count
        / len(care_nominal_rej)
    )

    print(
        "\nMatched-FPR target =",
        matched_alarm_count,
        "/",
        len(care_nominal_rej),
        "=",
        matched_target_fpr,
    )

    # Fixed evaluation-nominal matched-FPR rules.
    # These are descriptive only.
    matched_rules = {}

    for method in (
        DEPLOYABLE_BASELINES
    ):
        matched_rules[method] = (
            make_matched_rule(
                test_scores0[method],
                matched_alarm_count,
            )
        )

        check = apply_matched_rule(
            test_scores0[method],
            matched_rules[method],
        )

        if (
            int(check.sum())
            != matched_alarm_count
        ):
            raise RuntimeError(
                f"Failed exact matched FPR "
                f"for {method}."
            )

    # Individual-feature matched rules for
    # unattainable oracle best-view reference.
    oracle_matched_rules = [
        make_matched_rule(
            Xtest0[:, j],
            matched_alarm_count,
        )
        for j in range(
            Xtest0.shape[1]
        )
    ]

    # ------------------------------------------------------------
    # Held-out frozen alternatives.
    # ------------------------------------------------------------

    selected_test = (
        choose_one_window_per_cube(
            ft,
            SEED + 202,
        )
    )

    frozen_power = pd.read_csv(
        "outputs/care_earthnet/"
        "care_holdout_power.csv"
    )

    surface_rows = []
    oracle_rows = []

    native_cells = []
    matched_cells = []

    scenarios = holdout_scenarios()

    for si, (
        structure,
        amount,
        delta,
    ) in enumerate(scenarios):

        print(
            f"[{si+1}/{len(scenarios)}] "
            f"TEST {structure} "
            f"{amount:g} "
            f"{delta:g}"
        )

        fa, ia, _ = inject_pair(
            ft,
            it,
            selected_test,
            structure,
            amount,
            delta,
            SEED + 10000 + si,
        )

        ids, Xalt, names2 = (
            feature_matrix(
                fa,
                ia,
                f_local_cal,
                f_comp_norm,
                i_local_cal,
                i_comp_norm,
                windows,
            )
        )

        assert np.array_equal(
            ids,
            test_ids,
        )

        assert names2 == names

        alt_scores = score_methods(
            Xalt,
            fit,
            models,
            best_j,
        )

        # ---------------------------------------------
        # Native alpha.
        # ---------------------------------------------

        native_reject = {}

        for method in method_names:
            native_reject[method] = (
                conformal_p(
                    final_scores[method],
                    alt_scores[method],
                )
                <= ALPHA
            )

        care_native_power = float(
            native_reject[
                "care"
            ].mean()
        )

        # Verify CARE + frozen best single exactly
        # reproduce the previous frozen experiment.
        qold = frozen_power[
            (
                frozen_power[
                    "structure"
                ]
                == structure
            )
            & np.isclose(
                frozen_power[
                    "amount"
                ],
                amount,
            )
            & np.isclose(
                frozen_power[
                    "delta"
                ],
                delta,
            )
        ]

        if len(qold) != 1:
            raise RuntimeError(
                "Frozen CARE scenario "
                "lookup failed."
            )

        old_care = float(
            qold.iloc[0][
                "care_power"
            ]
        )

        old_single = float(
            qold.iloc[0][
                "best_fixed_single_power"
            ]
        )

        if not np.isclose(
            care_native_power,
            old_care,
            atol=1e-12,
        ):
            raise RuntimeError(
                "CARE power drift."
            )

        current_single = float(
            native_reject[
                "best_single"
            ].mean()
        )

        if not np.isclose(
            current_single,
            old_single,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Frozen best-single "
                "power drift."
            )

        # ---------------------------------------------
        # Matched-FPR.
        #
        # CARE stays at its deployed native-alpha
        # threshold. Other methods are evaluated at
        # exactly CARE's realized nominal test FPR.
        # ---------------------------------------------

        matched_reject = {
            "care":
                native_reject[
                    "care"
                ]
        }

        for method in (
            DEPLOYABLE_BASELINES
        ):
            matched_reject[
                method
            ] = apply_matched_rule(
                alt_scores[method],
                matched_rules[method],
            )

        # ---------------------------------------------
        # Individual-view oracle reference.
        # ---------------------------------------------

        individual_native = []
        individual_matched = []

        for j, feature in enumerate(
            names
        ):
            rn = (
                conformal_p(
                    Xfinal[:, j],
                    Xalt[:, j],
                )
                <= ALPHA
            )

            rm = apply_matched_rule(
                Xalt[:, j],
                oracle_matched_rules[j],
            )

            individual_native.append(
                float(rn.mean())
            )

            individual_matched.append(
                float(rm.mean())
            )

        jon = int(
            np.argmax(
                individual_native
            )
        )

        jom = int(
            np.argmax(
                individual_matched
            )
        )

        oracle_rows.append({
            "structure": structure,
            "amount": amount,
            "delta": delta,
            "oracle_native_feature":
                names[jon],
            "oracle_native_power":
                individual_native[jon],
            "oracle_matched_feature":
                names[jom],
            "oracle_matched_power":
                individual_matched[jom],
            "reference":
                "unattainable hindsight oracle",
        })

        # ---------------------------------------------
        # Save surface and bootstrap contrasts.
        # ---------------------------------------------

        for method in method_names:
            pn = float(
                native_reject[
                    method
                ].mean()
            )

            pm = float(
                matched_reject[
                    method
                ].mean()
            )

            surface_rows.append({
                "structure": structure,
                "amount": amount,
                "delta": delta,
                "method": method,
                "native_power": pn,
                "matched_fpr_power": pm,
                "care_minus_method_native":
                    care_native_power - pn,
                "care_minus_method_matched":
                    care_native_power - pm,
                "matched_target_fpr":
                    matched_target_fpr,
            })

            if method == "care":
                continue

            native_cells.append({
                "metric": "native_alpha",
                "structure": structure,
                "amount": amount,
                "delta": delta,
                "baseline": method,
                "diff": (
                    native_reject[
                        "care"
                    ].astype(float)
                    -
                    native_reject[
                        method
                    ].astype(float)
                ),
            })

            matched_cells.append({
                "metric":
                    "matched_to_care_fpr",
                "structure": structure,
                "amount": amount,
                "delta": delta,
                "baseline": method,
                "diff": (
                    native_reject[
                        "care"
                    ].astype(float)
                    -
                    matched_reject[
                        method
                    ].astype(float)
                ),
            })

    surface = pd.DataFrame(
        surface_rows
    )

    oracle = pd.DataFrame(
        oracle_rows
    )

    surface.to_csv(
        out / "holdout_power.csv",
        index=False,
    )

    oracle.to_csv(
        out / "oracle_best_view.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Overall + family summaries.
    # ------------------------------------------------------------

    overall = (
        surface.groupby("method")
        .agg(
            n_scenarios=(
                "delta",
                "size",
            ),
            mean_native_power=(
                "native_power",
                "mean",
            ),
            mean_matched_fpr_power=(
                "matched_fpr_power",
                "mean",
            ),
            mean_care_minus_method_native=(
                "care_minus_method_native",
                "mean",
            ),
            mean_care_minus_method_matched=(
                "care_minus_method_matched",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_native_power",
            ascending=False,
        )
    )

    family = (
        surface.groupby(
            [
                "structure",
                "method",
            ]
        )
        .agg(
            n_scenarios=(
                "delta",
                "size",
            ),
            mean_native_power=(
                "native_power",
                "mean",
            ),
            mean_matched_fpr_power=(
                "matched_fpr_power",
                "mean",
            ),
            mean_care_minus_method_native=(
                "care_minus_method_native",
                "mean",
            ),
            mean_care_minus_method_matched=(
                "care_minus_method_matched",
                "mean",
            ),
        )
        .reset_index()
    )

    overall.to_csv(
        out / "overall_summary.csv",
        index=False,
    )

    family.to_csv(
        out / "family_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Paired simultaneous confidence intervals.
    # ------------------------------------------------------------

    ci_native = simultaneous_ci(
        native_cells,
        args.bootstrap,
        SEED + 60001,
    )

    ci_matched = simultaneous_ci(
        matched_cells,
        args.bootstrap,
        SEED + 60002,
    )

    ci = pd.concat(
        [
            ci_native,
            ci_matched,
        ],
        ignore_index=True,
    )

    ci["positive"] = (
        ci["simul_ci_low"] > 0
    )

    ci["negative"] = (
        ci["simul_ci_high"] < 0
    )

    ci.to_csv(
        out
        / "simultaneous_paired_ci.csv",
        index=False,
    )

    ci_counts = (
        ci.groupby(
            [
                "metric",
                "baseline",
            ]
        )
        .agg(
            scenarios=(
                "delta",
                "size",
            ),
            mean_gain=(
                "care_minus_baseline",
                "mean",
            ),
            median_gain=(
                "care_minus_baseline",
                "median",
            ),
            simultaneous_care_wins=(
                "positive",
                "sum",
            ),
            simultaneous_care_losses=(
                "negative",
                "sum",
            ),
        )
        .reset_index()
    )

    ci_counts.to_csv(
        out
        / "simultaneous_ci_counts.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Frozen algorithmic kill decision.
    #
    # Hard test:
    # CARE must beat the strongest standard learned-fusion
    # baseline on mean held-out power under BOTH:
    #   1. native alpha
    #   2. matched empirical CARE FPR
    #
    # No tuning after this result.
    # ------------------------------------------------------------

    learned_summary = (
        overall[
            overall["method"].isin(
                LEARNED
            )
        ]
        .copy()
    )

    j_native = (
        learned_summary[
            "mean_native_power"
        ].idxmax()
    )

    strongest_native = (
        learned_summary.loc[
            j_native
        ]
    )

    j_matched = (
        learned_summary[
            "mean_matched_fpr_power"
        ].idxmax()
    )

    strongest_matched = (
        learned_summary.loc[
            j_matched
        ]
    )

    care_row = overall[
        overall["method"]
        == "care"
    ].iloc[0]

    care_mean_native = float(
        care_row[
            "mean_native_power"
        ]
    )

    care_mean_matched = float(
        care_row[
            "mean_matched_fpr_power"
        ]
    )

    gain_native = (
        care_mean_native
        -
        float(
            strongest_native[
                "mean_native_power"
            ]
        )
    )

    gain_matched = (
        care_mean_matched
        -
        float(
            strongest_matched[
                "mean_matched_fpr_power"
            ]
        )
    )

    passes = (
        gain_native > 0
        and gain_matched > 0
    )

    decision = {
        "status":
            "frozen EarthNet fusion benchmark",

        "feature_bank":
            "exact frozen CARE features",

        "design_alternatives":
            "exact frozen CARE design alternatives",

        "final_calibration":
            "same untouched cube-level conformal split",

        "matched_fpr_note":
            (
                "descriptive only; non-CARE thresholds "
                "are matched on evaluation nominal cubes "
                "to CARE native test FPR"
            ),

        "care_native_test_fpr":
            matched_target_fpr,

        "strongest_standard_learned_native":
            str(
                strongest_native[
                    "method"
                ]
            ),

        "strongest_standard_learned_native_power":
            float(
                strongest_native[
                    "mean_native_power"
                ]
            ),

        "care_mean_native_power":
            care_mean_native,

        "care_gain_vs_strongest_learned_native":
            gain_native,

        "strongest_standard_learned_matched":
            str(
                strongest_matched[
                    "method"
                ]
            ),

        "strongest_standard_learned_matched_power":
            float(
                strongest_matched[
                    "mean_matched_fpr_power"
                ]
            ),

        "care_mean_matched_power":
            care_mean_matched,

        "care_gain_vs_strongest_learned_matched":
            gain_matched,

        "passes_frozen_algorithmic_kill_criterion":
            bool(passes),

        "next_step": (
            "run sealed repeated-split robustness benchmark"
            if passes
            else
            "do not claim CARE algorithmic superiority"
        ),
    }

    with open(
        out / "kill_decision.json",
        "w",
    ) as f:
        json.dump(
            decision,
            f,
            indent=2,
        )

    metadata = {
        "seed": SEED,
        "alpha": ALPHA,
        "bootstrap_reps":
            args.bootstrap,
        "bootstrap_unit":
            "cube_id",

        "standard_learned_baselines": {
            "shrinkage_lda":
                (
                    "LinearDiscriminantAnalysis("
                    "solver=lsqr, shrinkage=auto)"
                ),
            "logistic":
                (
                    "L2 logistic, C=1, "
                    "lbfgs"
                ),
            "gradient_boosting":
                (
                    "100 trees, learning_rate=.1, "
                    "max_depth=3"
                ),
            "mlp":
                (
                    "one hidden layer of 32, "
                    "alpha=1e-4"
                ),
        },

        "simple_baselines": [
            "mean of standardized CARE features",
            "max of standardized CARE features",
            best_feature,
        ],

        "oracle":
            (
                "per-scenario hindsight best "
                "individual feature; unattainable"
            ),

        "care_modified":
            False,
    }

    with open(
        out / "benchmark_metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n=== FROZEN FUSION HOLDOUT SUMMARY ==="
    )
    print(
        overall.to_string(
            index=False
        )
    )

    print(
        "\n=== FAMILY SUMMARY ==="
    )
    print(
        family.to_string(
            index=False
        )
    )

    print(
        "\n=== SIMULTANEOUS CI COUNTS ==="
    )
    print(
        ci_counts.to_string(
            index=False
        )
    )

    print(
        "\n=== FROZEN FUSION KILL DECISION ==="
    )
    print(
        json.dumps(
            decision,
            indent=2,
        )
    )

    print(
        "\nWROTE:",
        out.resolve(),
    )


if __name__ == "__main__":
    main()
