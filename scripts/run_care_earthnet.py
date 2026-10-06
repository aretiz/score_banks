#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from run_caps_z_earthnet_from_cache import (
    load_cache,
    filter_valid_fraction,
    subset,
    fit_pipeline,
)
from spt.caps_earthnet import all_raw_scores


BASE_FEATURES = [
    "global_nll",
    "standardized_residual",
    "raw_residual",
    "z_global",
    "z_hc",
    "z_bj",
    "z_scan",
    "caps_z",
]

LAMBDA = 10.0
ALPHA = 0.05
SEED = 20260824


def take_cubes(data, cube_ids):
    ix = np.flatnonzero(np.isin(data["cube_id"], cube_ids))
    return subset(data, ix)


def shared_split5(full, image, seed):
    ids = np.unique(full["cube_id"])
    assert np.array_equal(
        np.sort(ids),
        np.sort(np.unique(image["cube_id"]))
    )

    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(ids), 5)

    full_parts = [take_cubes(full, g) for g in groups]
    image_parts = [take_cubes(image, g) for g in groups]

    for a, b in zip(full_parts, image_parts):
        assert np.array_equal(a["cube_id"], b["cube_id"])
        assert np.array_equal(a["valid"], b["valid"])

    return full_parts, image_parts, groups


def clone_data(data):
    out = {}
    for k, v in data.items():
        if k == "contexts":
            out[k] = {a: b.copy() for a, b in v.items()}
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def cube_max(values, cube_id, ids):
    return np.asarray([
        np.max(values[cube_id == cid])
        for cid in ids
    ], dtype=float)


def innovation_window_score(data):
    z = np.asarray(data["z"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)

    out = np.zeros(len(z), dtype=np.float64)

    for i in range(len(z)):
        x = z[i]
        m = valid[i]

        s = np.zeros_like(x, dtype=np.float64)
        n = np.zeros_like(x, dtype=np.float64)

        a = m[:, :, 1:] & m[:, :, :-1]
        s[:, :, 1:] += np.where(a, x[:, :, :-1], 0.0)
        n[:, :, 1:] += a

        a = m[:, :, :-1] & m[:, :, 1:]
        s[:, :, :-1] += np.where(a, x[:, :, 1:], 0.0)
        n[:, :, :-1] += a

        a = m[:, 1:, :] & m[:, :-1, :]
        s[:, 1:, :] += np.where(a, x[:, :-1, :], 0.0)
        n[:, 1:, :] += a

        a = m[:, :-1, :] & m[:, 1:, :]
        s[:, :-1, :] += np.where(a, x[:, 1:, :], 0.0)
        n[:, :-1, :] += a

        good = m & (n > 0)

        if not np.any(good):
            out[i] = 0.0
            continue

        pred = np.zeros_like(s)
        pred[good] = s[good] / n[good]

        e = x.astype(np.float64) - pred
        out[i] = np.sqrt(np.mean(e[good] ** 2))

    return out


def raw_scores(data, local_cal, comp_norm, windows):
    return all_raw_scores(
        data["z"],
        data["log_sigma"],
        data["valid"],
        local_cal,
        comp_norm,
        windows,
    )


def feature_matrix(
    full,
    image,
    full_local,
    full_comp,
    image_local,
    image_comp,
    windows,
):
    assert np.array_equal(full["cube_id"], image["cube_id"])
    assert np.array_equal(full["valid"], image["valid"])

    ids = np.unique(full["cube_id"])

    sf = raw_scores(
        full, full_local, full_comp, windows
    )
    si = raw_scores(
        image, image_local, image_comp, windows
    )

    names = []
    cols = []

    for prefix, scores, data in [
        ("multimodal", sf, full),
        ("image_only", si, image),
    ]:
        for name in BASE_FEATURES:
            if name not in scores:
                raise RuntimeError(
                    f"Missing score {name} for {prefix}"
                )

            cols.append(
                cube_max(
                    scores[name],
                    data["cube_id"],
                    ids,
                )
            )
            names.append(f"{prefix}_{name}")

    for prefix, data in [
        ("multimodal", full),
        ("image_only", image),
    ]:
        iw = innovation_window_score(data)

        cols.append(
            cube_max(
                iw,
                data["cube_id"],
                ids,
            )
        )

        names.append(f"{prefix}_innovation")

    X = np.column_stack(cols).astype(np.float64)

    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        raise RuntimeError(
            f"Non-finite CARE features, first={bad[0]}"
        )

    return ids, X, names


def choose_one_window_per_cube(data, seed):
    rng = np.random.default_rng(seed)

    selected = []

    for cid in np.unique(data["cube_id"]):
        ix = np.flatnonzero(data["cube_id"] == cid)
        selected.append(int(rng.choice(ix)))

    return np.asarray(selected, dtype=int)


def choose_pixels(valid_pix, structure, amount, rng):
    H, W = valid_pix.shape

    chosen = np.zeros((H, W), dtype=bool)
    valid_idx = np.flatnonzero(valid_pix.ravel())

    if len(valid_idx) == 0:
        return chosen

    if structure == "sparse":
        k = min(int(amount), len(valid_idx))

        sel = rng.choice(
            valid_idx,
            size=k,
            replace=False,
        )

        chosen.ravel()[sel] = True

    elif structure == "contiguous":
        side = min(int(amount), H, W)

        r0 = int(rng.integers(0, H - side + 1))
        c0 = int(rng.integers(0, W - side + 1))

        chosen[r0:r0 + side, c0:c0 + side] = True
        chosen &= valid_pix

    elif structure == "dense":
        k = max(
            1,
            min(
                len(valid_idx),
                int(np.ceil(float(amount) * len(valid_idx))),
            ),
        )

        sel = rng.choice(
            valid_idx,
            size=k,
            replace=False,
        )

        chosen.ravel()[sel] = True

    elif structure == "disk":
        radius = int(amount)

        center_flat = int(rng.choice(valid_idx))

        r0, c0 = np.unravel_index(
            center_flat,
            (H, W),
        )

        rr, cc = np.ogrid[:H, :W]

        chosen = (
            (rr - r0) ** 2
            + (cc - c0) ** 2
            <= radius ** 2
        )

        chosen &= valid_pix

    elif structure == "stripe":
        width = min(int(amount), H, W)

        center_flat = int(rng.choice(valid_idx))

        r0, c0 = np.unravel_index(
            center_flat,
            (H, W),
        )

        if rng.random() < 0.5:
            lo = max(0, r0 - width // 2)
            hi = min(H, lo + width)
            chosen[lo:hi, :] = True
        else:
            lo = max(0, c0 - width // 2)
            hi = min(W, lo + width)
            chosen[:, lo:hi] = True

        chosen &= valid_pix

    else:
        raise ValueError(structure)

    return chosen


def inject_pair(
    full,
    image,
    selected,
    structure,
    amount,
    delta,
    seed,
):
    f = clone_data(full)
    g = clone_data(image)

    assert np.array_equal(f["cube_id"], g["cube_id"])
    assert np.array_equal(f["valid"], g["valid"])

    rng = np.random.default_rng(seed)
    affected = []

    for j in selected:
        valid_pix = f["valid"][j].any(axis=0)

        chosen = choose_pixels(
            valid_pix,
            structure,
            amount,
            rng,
        )

        m = f["valid"][j] & chosen[None, :, :]

        affected.append(int(chosen.sum()))

        sf = np.exp(
            np.clip(f["log_sigma"][j], -30, 30)
        )

        sg = np.exp(
            np.clip(g["log_sigma"][j], -30, 30)
        )

        # Same raw target perturbation for both forecasters.
        raw_shift = float(delta) * sf

        f["z"][j][m] += (
            raw_shift[m]
            / np.maximum(sf[m], 1e-8)
        )

        g["z"][j][m] += (
            raw_shift[m]
            / np.maximum(sg[m], 1e-8)
        )

    return f, g, float(np.mean(affected))


def fit_care(X0, design_alternatives, lam):
    mu = X0.mean(axis=0)

    sd = X0.std(axis=0, ddof=1)
    sd = np.maximum(sd, 1e-8)

    Z0 = (X0 - mu) / sd

    Sigma = np.cov(
        Z0,
        rowvar=False,
        ddof=1,
    )

    A = Sigma + lam * np.eye(Sigma.shape[0])

    weights = []
    names = []
    deltas = []

    for name, Xa in design_alternatives:
        Za = (Xa - mu) / sd

        # Paired design displacement.
        delta = np.mean(
            Za - Z0,
            axis=0,
        )

        w = np.linalg.solve(A, delta)

        denom = float(
            np.sqrt(
                max(
                    w @ Sigma @ w,
                    1e-12,
                )
            )
        )

        if denom <= 1e-6:
            continue

        weights.append(w)
        names.append(name)
        deltas.append(delta)

    if not weights:
        raise RuntimeError(
            "No usable CARE directions."
        )

    return {
        "mu": mu,
        "sd": sd,
        "Sigma": Sigma,
        "weights": np.asarray(weights),
        "weight_names": np.asarray(names),
        "deltas": np.asarray(deltas),
        "lambda": float(lam),
    }


def care_head_scores(X, fit):
    Z = (
        X - fit["mu"]
    ) / fit["sd"]

    W = fit["weights"]
    Sigma = fit["Sigma"]

    den = np.sqrt(
        np.maximum(
            np.einsum(
                "ki,ij,kj->k",
                W,
                Sigma,
                W,
            ),
            1e-12,
        )
    )

    return (
        Z @ W.T
    ) / den[None, :]


def care_score(X, fit):
    return np.max(
        reviewer_care_head_scores(
            X,
            fit,
        ),
        axis=1,
    )


def conformal_threshold(
    ref,
    alpha,
):
    ref = np.sort(
        np.asarray(
            ref,
            dtype=float,
        )
    )

    n = len(ref)

    rank = int(
        np.ceil(
            (n + 1)
            * (1.0 - alpha)
        )
    )

    if rank > n:
        return float("inf")

    return float(
        ref[rank - 1]
    )


def conformal_p(ref, x):
    ref = np.sort(
        np.asarray(ref, dtype=float)
    )

    x = np.asarray(
        x,
        dtype=float,
    )

    ge = (
        len(ref)
        - np.searchsorted(
            ref,
            x,
            side="left",
        )
    )

    return (
        1.0 + ge
    ) / (
        len(ref) + 1.0
    )


def exact_ci(reject):
    reject = np.asarray(
        reject,
        dtype=bool,
    )

    r = binomtest(
        int(reject.sum()),
        len(reject),
    )

    ci = r.proportion_ci(0.95)

    return float(ci.low), float(ci.high)


def design_scenarios():
    out = []

    for k in [4, 16]:
        for d in [1.0, 2.0]:
            out.append(
                ("sparse", float(k), d)
            )

    for side in [4, 8]:
        for d in [1.0, 2.0]:
            out.append(
                ("contiguous", float(side), d)
            )

    for frac in [0.25, 0.50]:
        for d in [1.0, 2.0]:
            out.append(
                ("dense", frac, d)
            )

    return out


def holdout_scenarios():
    out = []

    strengths = [1.5, 2.5, 3.0]

    for k in [1, 64]:
        for d in strengths:
            out.append(
                ("sparse", float(k), d)
            )

    for side in [1, 2, 16]:
        for d in strengths:
            out.append(
                ("contiguous", float(side), d)
            )

    for frac in [0.10, 1.0]:
        for d in strengths:
            out.append(
                ("dense", frac, d)
            )

    for d in strengths:
        out.append(
            ("disk", 4.0, d)
        )

    for d in strengths:
        out.append(
            ("stripe", 4.0, d)
        )

    return out


def simultaneous_bootstrap(
    records,
    nboot,
    seed,
):
    rng = np.random.default_rng(seed)

    names = list(records)

    n = len(
        records[names[0]]["care"]
    )

    counts = rng.multinomial(
        n,
        np.full(n, 1.0 / n),
        size=nboot,
    ).astype(np.float64)

    maxdev = np.zeros(
        nboot,
        dtype=float,
    )

    estimates = {}

    for name in names:
        a = records[name]["care"].astype(float)
        b = records[name]["single"].astype(float)

        est = float(
            np.mean(a - b)
        )

        estimates[name] = est

        boot = (
            counts @ (a - b)
        ) / float(n)

        maxdev = np.maximum(
            maxdev,
            np.abs(boot - est),
        )

    q = float(
        np.quantile(
            maxdev,
            0.95,
        )
    )

    rows = []

    for name in names:
        meta = json.loads(name)
        est = estimates[name]

        rows.append({
            **meta,
            "care_minus_best_single": est,
            "simul_ci_low": max(
                -1.0,
                est - q,
            ),
            "simul_ci_high": min(
                1.0,
                est + q,
            ),
            "bootstrap_unit": "cube_id",
            "bootstrap_reps": nboot,
        })

    return pd.DataFrame(rows)



def reviewer_care_head_scores(X, fit):
    """Return every variance-normalized CARE head, before max aggregation."""
    Z = (X - fit["mu"]) / fit["sd"]
    W = fit["weights"]
    Sigma = fit["Sigma"]
    den = np.sqrt(
        np.maximum(
            np.einsum("ki,ij,kj->k", W, Sigma, W),
            1e-12,
        )
    )
    return (Z @ W.T) / den[None, :]


def reviewer_fixed_scenario_paired_interval(values, nboot, seed):
    """Cube bootstrap with the supplied design scenarios treated as fixed."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Paired cube contributions must be one-dimensional")

    n = len(values)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        n,
        np.full(n, 1.0 / n),
        size=nboot,
    ).astype(float)

    estimate = float(values.mean())
    boot = (counts @ values) / float(n)
    radius = float(np.quantile(np.abs(boot - estimate), 0.95))

    return (
        estimate,
        max(-1.0, estimate - radius),
        min(1.0, estimate + radius),
    )


def reviewer_crossfit_head_audit(records, head_names, nboot, seed):
    """
    Select a head on one test-cube fold and evaluate it on the other.

    Bootstrap replicates resample cube IDs within folds, repeat head selection,
    and use one max-deviation radius across every scenario. This removes the
    same-cube maximization bias and makes the reported interval selection-aware.
    """
    keys = list(records)
    if not keys:
        return pd.DataFrame()

    head_names = np.asarray(head_names).astype(str)
    first = records[keys[0]]
    n = len(first["care"])
    k = first["heads"].shape[1]

    if len(head_names) != k:
        raise ValueError("Head-name count does not match rejection matrix")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    n0 = n // 2
    fold0 = np.sort(permutation[:n0])
    fold1 = np.sort(permutation[n0:])
    n1 = len(fold1)

    counts0 = rng.multinomial(
        n0,
        np.full(n0, 1.0 / n0),
        size=nboot,
    ).astype(float)
    counts1 = rng.multinomial(
        n1,
        np.full(n1, 1.0 / n1),
        size=nboot,
    ).astype(float)

    bootstrap_row = np.arange(nboot)
    max_deviation = np.zeros(nboot, dtype=float)
    prepared = []

    for key in keys:
        care = np.asarray(records[key]["care"], dtype=float)
        heads = np.asarray(records[key]["heads"], dtype=float)

        if care.shape != (n,) or heads.shape != (n, k):
            raise ValueError("Scenario rejection arrays have inconsistent shapes")

        h0 = heads[fold0]
        h1 = heads[fold1]
        c0 = care[fold0]
        c1 = care[fold1]

        # A head selected on fold 0 is evaluated only on fold 1, and vice versa.
        selected_on_0 = int(np.argmax(h0.mean(axis=0)))
        selected_on_1 = int(np.argmax(h1.mean(axis=0)))

        head_sum0 = counts0 @ h0
        head_sum1 = counts1 @ h1
        care_sum0 = counts0 @ c0
        care_sum1 = counts1 @ c1

        # Reselect inside every bootstrap replicate.
        boot_selected_on_0 = np.argmax(head_sum0 / float(n0), axis=1)
        boot_selected_on_1 = np.argmax(head_sum1 / float(n1), axis=1)

        boot_difference = (
            head_sum0[bootstrap_row, boot_selected_on_1]
            - care_sum0
            + head_sum1[bootstrap_row, boot_selected_on_0]
            - care_sum1
        ) / float(n)

        crossfit_head_power = float(
            (
                h0[:, selected_on_1].sum()
                + h1[:, selected_on_0].sum()
            )
            / float(n)
        )
        care_power = float(care.mean())
        estimate = crossfit_head_power - care_power

        max_deviation = np.maximum(
            max_deviation,
            np.abs(boot_difference - estimate),
        )

        prepared.append({
            **json.loads(key),
            "n_cubes": n,
            "fold0_units": n0,
            "fold1_units": n1,
            "head_selected_on_fold0": head_names[selected_on_0],
            "head_selected_on_fold1": head_names[selected_on_1],
            "crossfit_head_power": crossfit_head_power,
            "care_power": care_power,
            "crossfit_head_minus_care": estimate,
        })

    radius = float(np.quantile(max_deviation, 0.95))
    rows = []

    for row in prepared:
        estimate = row["crossfit_head_minus_care"]
        low = max(-1.0, estimate - radius)
        high = min(1.0, estimate + radius)

        if low > 0.0:
            result = "crossfit head positive"
        elif high < 0.0:
            result = "CARE positive"
        else:
            result = "unresolved"

        rows.append({
            **row,
            "simul_ci_low": low,
            "simul_ci_high": high,
            "result": result,
            "simultaneous_radius": radius,
            "bootstrap_unit": "cube_id",
            "bootstrap_reps": nboot,
            "selection_correction": (
                "two_fold_crossfit_with_reselection_inside_bootstrap"
            ),
        })

    return pd.DataFrame(rows)


def reviewer_conformal_threshold(
    ref,
    alpha,
):
    """Finite-sample conformal score threshold."""
    ref = np.sort(
        np.asarray(
            ref,
            dtype=float,
        )
    )

    if len(ref) == 0:
        raise ValueError(
            "Empty conformal reference"
        )

    rank = int(
        np.ceil(
            (len(ref) + 1)
            * (1.0 - alpha)
        )
    )

    index = min(
        max(rank - 1, 0),
        len(ref) - 1,
    )

    return float(ref[index])

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--full-cal",
        default="outputs/earthnet/caps_cal.npz",
    )

    ap.add_argument(
        "--full-test",
        default="outputs/earthnet/caps_test.npz",
    )

    ap.add_argument(
        "--image-cal",
        default="outputs/earthnet_image_only/caps_cal.npz",
    )

    ap.add_argument(
        "--image-test",
        default="outputs/earthnet_image_only/caps_test.npz",
    )

    ap.add_argument(
        "--out",
        default="outputs/care_earthnet",
    )

    ap.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )

    ap.add_argument(
        "--min-valid",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--min-position-ref",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--lambda-ridge",
        type=float,
        default=LAMBDA,
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    args = ap.parse_args()

    outdir = Path(args.out)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fc = filter_valid_fraction(
        load_cache(
            args.full_cal,
            "nchw",
        ),
        args.min_valid,
    )

    ft = filter_valid_fraction(
        load_cache(
            args.full_test,
            "nchw",
        ),
        args.min_valid,
    )

    ic = filter_valid_fraction(
        load_cache(
            args.image_cal,
            "nchw",
        ),
        args.min_valid,
    )

    it = filter_valid_fraction(
        load_cache(
            args.image_test,
            "nchw",
        ),
        args.min_valid,
    )

    for a, b, name in [
        (fc, ic, "cal"),
        (ft, it, "test"),
    ]:
        assert np.array_equal(
            a["cube_id"],
            b["cube_id"],
        ), f"{name}: cube mismatch"

        assert np.array_equal(
            a["valid"],
            b["valid"],
        ), f"{name}: mask mismatch"

    assert len(
        np.intersect1d(
            np.unique(fc["cube_id"]),
            np.unique(ft["cube_id"]),
        )
    ) == 0

    fp, ip, groups = shared_split5(
        fc,
        ic,
        args.seed,
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

    print("\n=== CUBE SPLITS ===")

    for name, d in [
        ("local", f_local),
        ("component", f_comp),
        ("design", f_design),
        ("final_conformal", f_final),
        ("audit", f_audit),
        ("test", ft),
    ]:
        print(
            name,
            len(np.unique(d["cube_id"]))
        )

    np.savez(
        outdir / "frozen_cube_splits.npz",
        local=groups[0],
        component=groups[1],
        design=groups[2],
        final_conformal=groups[3],
        audit=groups[4],
        test=np.unique(ft["cube_id"]),
    )

    f_local_cal, f_comp_norm, _ = fit_pipeline(
        f_local,
        f_comp,
        f_design,
        args.windows,
        args.min_position_ref,
    )

    i_local_cal, i_comp_norm, _ = fit_pipeline(
        i_local,
        i_comp,
        i_design,
        args.windows,
        args.min_position_ref,
    )

    design_ids, X0, feature_names = feature_matrix(
        f_design,
        i_design,
        f_local_cal,
        f_comp_norm,
        i_local_cal,
        i_comp_norm,
        args.windows,
    )

    selected_design = choose_one_window_per_cube(
        f_design,
        args.seed + 101,
    )

    design_alt = []
    design_rows = []

    for si, (
        structure,
        amount,
        delta,
    ) in enumerate(design_scenarios()):

        print(
            f"DESIGN {structure} "
            f"amount={amount:g} "
            f"delta={delta:g}"
        )

        fa, ia, affected = inject_pair(
            f_design,
            i_design,
            selected_design,
            structure,
            amount,
            delta,
            args.seed + 1000 + si,
        )

        ids, Xa, names = feature_matrix(
            fa,
            ia,
            f_local_cal,
            f_comp_norm,
            i_local_cal,
            i_comp_norm,
            args.windows,
        )

        assert np.array_equal(
            ids,
            design_ids,
        )

        assert names == feature_names

        tag = (
            f"{structure}"
            f"_a{amount:g}"
            f"_d{delta:g}"
        )

        design_alt.append(
            (tag, Xa)
        )

        design_rows.append({
            "structure": structure,
            "amount": amount,
            "delta": delta,
            "affected_pixels": affected,
        })

    lambda_grid = [
        0.0,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    lambda_models = {}
    lambda_design_rows = []

    for ridge in lambda_grid:
        candidate = fit_care(
            X0,
            design_alt,
            ridge,
        )

        lambda_models[
            float(ridge)
        ] = candidate

        nominal_score = care_score(
            X0,
            candidate,
        )

        scenario_powers = []

        for _, Xa in design_alt:
            alternative_score = (
                care_score(
                    Xa,
                    candidate,
                )
            )

            rejection = (
                conformal_p(
                    nominal_score,
                    alternative_score,
                )
                <= args.alpha
            )

            scenario_powers.append(
                float(
                    rejection.mean()
                )
            )

        W = candidate["weights"]
        Sigma_candidate = (
            candidate["Sigma"]
        )

        head_covariance = (
            W
            @ Sigma_candidate
            @ W.T
        )

        head_sd = np.sqrt(
            np.maximum(
                np.diag(
                    head_covariance
                ),
                1e-12,
            )
        )

        head_correlation = (
            head_covariance
            / np.outer(
                head_sd,
                head_sd,
            )
        )

        head_correlation = (
            head_correlation
            + head_correlation.T
        ) / 2.0

        correlation_eigenvalues = (
            np.linalg.eigvalsh(
                head_correlation
            )
        )

        correlation_eigenvalues = (
            np.clip(
                correlation_eigenvalues,
                0.0,
                None,
            )
        )

        effective_rank = float(
            correlation_eigenvalues.sum()
            ** 2
            / np.sum(
                correlation_eigenvalues
                ** 2
            )
        )

        upper = np.triu_indices(
            len(W),
            1,
        )

        off_diagonal = (
            head_correlation[
                upper
            ]
        )

        lambda_design_rows.append({
            "lambda": float(ridge),
            "mean_design_power": float(
                np.mean(
                    scenario_powers
                )
            ),
            "median_design_power": float(
                np.median(
                    scenario_powers
                )
            ),
            "minimum_design_power": float(
                np.min(
                    scenario_powers
                )
            ),
            "maximum_design_power": float(
                np.max(
                    scenario_powers
                )
            ),
            "design_threshold": (
                reviewer_conformal_threshold(
                    nominal_score,
                    args.alpha,
                )
            ),
            "median_head_correlation": (
                float(
                    np.median(
                        off_diagonal
                    )
                )
            ),
            "minimum_head_correlation": (
                float(
                    np.min(
                        off_diagonal
                    )
                )
            ),
            "maximum_head_correlation": (
                float(
                    np.max(
                        off_diagonal
                    )
                )
            ),
            "effective_rank": (
                effective_rank
            ),
        })

    lambda_design = pd.DataFrame(
        lambda_design_rows
    )

    selected_lambda_row = (
        lambda_design.sort_values(
            [
                "mean_design_power",
                "median_design_power",
                "lambda",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    selected_lambda = float(
        selected_lambda_row[
            "lambda"
        ]
    )

    lambda_design[
        "selected_on_design"
    ] = np.isclose(
        lambda_design[
            "lambda"
        ],
        selected_lambda,
    )

    lambda_design.to_csv(
        outdir
        / "care_lambda_design_sweep.csv",
        index=False,
    )

    care = lambda_models[
        selected_lambda
    ]

    args.lambda_ridge = (
        selected_lambda
    )

    print(
        "\n=== DESIGN-SELECTED "
        "CARE LAMBDA ==="
    )

    print(
        lambda_design.to_string(
            index=False
        )
    )

    print(
        "selected lambda:",
        selected_lambda,
    )

    W = care["weights"]
    Sigma_care = care["Sigma"]

    euclidean_norm = np.linalg.norm(
        W,
        axis=1,
    )

    weight_cosine = (
        W @ W.T
        / np.outer(
            euclidean_norm,
            euclidean_norm,
        )
    )

    head_covariance = (
        W
        @ Sigma_care
        @ W.T
    )

    head_sd = np.sqrt(
        np.maximum(
            np.diag(
                head_covariance
            ),
            1e-12,
        )
    )

    head_correlation = (
        head_covariance
        / np.outer(
            head_sd,
            head_sd,
        )
    )

    head_correlation = (
        head_correlation
        + head_correlation.T
    ) / 2.0

    correlation_eigenvalues = (
        np.linalg.eigvalsh(
            head_correlation
        )
    )

    correlation_eigenvalues = (
        np.clip(
            correlation_eigenvalues,
            0.0,
            None,
        )
    )

    effective_rank = float(
        correlation_eigenvalues.sum()
        ** 2
        / np.sum(
            correlation_eigenvalues
            ** 2
        )
    )

    pair_rows = []

    for first in range(
        len(W)
    ):
        for second in range(
            first + 1,
            len(W),
        ):
            pair_rows.append({
                "head_1": (
                    care[
                        "weight_names"
                    ][first]
                ),
                "head_2": (
                    care[
                        "weight_names"
                    ][second]
                ),
                "weight_cosine": float(
                    weight_cosine[
                        first,
                        second,
                    ]
                ),
                "nominal_score_correlation":
                    float(
                        head_correlation[
                            first,
                            second,
                        ]
                    ),
            })

    pd.DataFrame(
        pair_rows
    ).to_csv(
        outdir
        / "care_head_pair_correlations.csv",
        index=False,
    )

    upper = np.triu_indices(
        len(W),
        1,
    )

    pd.DataFrame(
        [
            {
                "selected_lambda":
                    selected_lambda,
                "named_heads":
                    len(W),
                "matrix_rank": int(
                    np.linalg.matrix_rank(
                        head_correlation,
                        tol=1e-8,
                    )
                ),
                "effective_rank":
                    effective_rank,
                "median_head_correlation":
                    float(
                        np.median(
                            head_correlation[
                                upper
                            ]
                        )
                    ),
                "minimum_head_correlation":
                    float(
                        np.min(
                            head_correlation[
                                upper
                            ]
                        )
                    ),
                "maximum_head_correlation":
                    float(
                        np.max(
                            head_correlation[
                                upper
                            ]
                        )
                    ),
            }
        ]
    ).to_csv(
        outdir
        / "care_head_geometry.csv",
        index=False,
    )

    np.savez(
        outdir / "care_fit.npz",
        mu=care["mu"],
        sd=care["sd"],
        Sigma=care["Sigma"],
        weights=care["weights"],
        deltas=care["deltas"],
        weight_names=care["weight_names"],
        feature_names=np.asarray(feature_names),
        lambda_ridge=np.asarray(
            [args.lambda_ridge]
        ),
    )

    pd.DataFrame(
        design_rows
    ).to_csv(
        outdir / "design_templates.csv",
        index=False,
    )

    feature_design_power = np.zeros(
        len(feature_names),
        dtype=float,
    )

    for _, Xa in design_alt:
        for j in range(
            len(feature_names)
        ):
            pv = conformal_p(
                X0[:, j],
                Xa[:, j],
            )

            feature_design_power[j] += np.mean(
                pv <= args.alpha
            )

    feature_design_power /= len(design_alt)

    best_j = int(
        np.argmax(
            feature_design_power
        )
    )

    best_feature = feature_names[best_j]

    design_selection = pd.DataFrame({
        "feature": feature_names,
        "mean_design_power":
            feature_design_power,
    }).sort_values(
        "mean_design_power",
        ascending=False,
    )

    design_selection.to_csv(
        outdir / "design_feature_selection.csv",
        index=False,
    )


    # Reviewer check 1: paired cube-level interval for the two innovation
    # features. Design alternatives are fixed; cube IDs are resampled jointly
    # across all 12 alternatives and both representations.
    reviewer_image_j = feature_names.index("image_only_innovation")
    reviewer_multimodal_j = feature_names.index("multimodal_innovation")
    reviewer_image_outcomes = []
    reviewer_multimodal_outcomes = []
    reviewer_scenario_rows = []

    for reviewer_name, reviewer_Xa in design_alt:
        reviewer_image_reject = (
            conformal_p(
                X0[:, reviewer_image_j],
                reviewer_Xa[:, reviewer_image_j],
            )
            <= args.alpha
        )
        reviewer_multimodal_reject = (
            conformal_p(
                X0[:, reviewer_multimodal_j],
                reviewer_Xa[:, reviewer_multimodal_j],
            )
            <= args.alpha
        )

        reviewer_image_outcomes.append(reviewer_image_reject.astype(float))
        reviewer_multimodal_outcomes.append(
            reviewer_multimodal_reject.astype(float)
        )
        reviewer_scenario_rows.append({
            "design_alternative": reviewer_name,
            "n_cubes": len(design_ids),
            "image_only_innovation_power": float(
                reviewer_image_reject.mean()
            ),
            "multimodal_innovation_power": float(
                reviewer_multimodal_reject.mean()
            ),
            "image_minus_multimodal": float(
                reviewer_image_reject.mean()
                - reviewer_multimodal_reject.mean()
            ),
        })

    reviewer_image_outcomes = np.vstack(reviewer_image_outcomes)
    reviewer_multimodal_outcomes = np.vstack(reviewer_multimodal_outcomes)
    reviewer_cube_difference = np.mean(
        reviewer_image_outcomes - reviewer_multimodal_outcomes,
        axis=0,
    )

    reviewer_estimate, reviewer_low, reviewer_high = (
        reviewer_fixed_scenario_paired_interval(
            reviewer_cube_difference,
            args.bootstrap,
            args.seed + 61000,
        )
    )

    reviewer_image_power = float(reviewer_image_outcomes.mean())
    reviewer_multimodal_power = float(reviewer_multimodal_outcomes.mean())

    if not np.isclose(
        reviewer_image_power,
        feature_design_power[reviewer_image_j],
        atol=1e-12,
    ):
        raise AssertionError("Image innovation power does not reproduce sweep")
    if not np.isclose(
        reviewer_multimodal_power,
        feature_design_power[reviewer_multimodal_j],
        atol=1e-12,
    ):
        raise AssertionError("Multimodal innovation power does not reproduce sweep")

    if reviewer_low > 0.0:
        reviewer_pair_result = "image_only positive"
    elif reviewer_high < 0.0:
        reviewer_pair_result = "multimodal positive"
    else:
        reviewer_pair_result = "unresolved"

    reviewer_pair_summary = pd.DataFrame([{
        "comparison": "image_only_innovation_minus_multimodal_innovation",
        "n_cubes": len(design_ids),
        "n_fixed_design_alternatives": len(design_alt),
        "image_only_mean_design_power": reviewer_image_power,
        "multimodal_mean_design_power": reviewer_multimodal_power,
        "paired_difference": reviewer_estimate,
        "paired_ci_low": reviewer_low,
        "paired_ci_high": reviewer_high,
        "result": reviewer_pair_result,
        "bootstrap_unit": "cube_id",
        "bootstrap_reps": args.bootstrap,
        "interval": "paired_cube_max_deviation_95",
    }])

    reviewer_pair_summary.to_csv(
        outdir / "reviewer_innovation_pair_paired_ci.csv",
        index=False,
    )
    pd.DataFrame(reviewer_scenario_rows).to_csv(
        outdir / "reviewer_innovation_pair_by_design_scenario.csv",
        index=False,
    )

    print("\n=== PAIRED INNOVATION DESIGN INTERVAL ===")
    print(reviewer_pair_summary.to_string(index=False))

    print(
        "\n=== DESIGN-SELECTED SINGLE FEATURE ==="
    )

    print(best_feature)

    print(
        "design mean power =",
        feature_design_power[best_j],
    )

    final_ids, Xfinal, _ = feature_matrix(
        f_final,
        i_final,
        f_local_cal,
        f_comp_norm,
        i_local_cal,
        i_comp_norm,
        args.windows,
    )

    audit_ids, Xaudit, _ = feature_matrix(
        f_audit,
        i_audit,
        f_local_cal,
        f_comp_norm,
        i_local_cal,
        i_comp_norm,
        args.windows,
    )

    test_ids, Xtest0, _ = feature_matrix(
        ft,
        it,
        f_local_cal,
        f_comp_norm,
        i_local_cal,
        i_comp_norm,
        args.windows,
    )

    care_final = care_score(
        Xfinal,
        care,
    )

    care_audit = care_score(
        Xaudit,
        care,
    )

    care_test0 = care_score(
        Xtest0,
        care,
    )

    single_final = Xfinal[:, best_j]
    single_audit = Xaudit[:, best_j]
    single_test0 = Xtest0[:, best_j]

    single_final_z = (
        single_final
        - care["mu"][best_j]
    ) / care["sd"][best_j]

    single_audit_z = (
        single_audit
        - care["mu"][best_j]
    ) / care["sd"][best_j]

    single_test0_z = (
        single_test0
        - care["mu"][best_j]
    ) / care["sd"][best_j]

    care_final_heads = (
        reviewer_care_head_scores(
            Xfinal,
            care,
        )
    )

    care_audit_heads = (
        reviewer_care_head_scores(
            Xaudit,
            care,
        )
    )

    care_pooled_calibration = (
        np.concatenate(
            [
                care_final,
                care_audit,
            ]
        )
    )

    single_pooled_calibration = (
        np.concatenate(
            [
                single_final,
                single_audit,
            ]
        )
    )

    care_pooled_head_calibration = (
        np.vstack(
            [
                care_final_heads,
                care_audit_heads,
            ]
        )
    )

    care_pooled_nominal_rej = (
        conformal_p(
            care_pooled_calibration,
            care_test0,
        )
        <= args.alpha
    )

    single_pooled_nominal_rej = (
        conformal_p(
            single_pooled_calibration,
            single_test0,
        )
        <= args.alpha
    )

    pooled_validity = pd.DataFrame(
        [
            {
                "method": "CARE",
                "calibration_units": int(
                    len(
                        care_pooled_calibration
                    )
                ),
                "test_nominal_units": int(
                    len(
                        care_test0
                    )
                ),
                "test_nominal_alarms": int(
                    care_pooled_nominal_rej.sum()
                ),
                "test_nominal_fpr": float(
                    care_pooled_nominal_rej.mean()
                ),
                "threshold": (
                    reviewer_conformal_threshold(
                        care_pooled_calibration,
                        args.alpha,
                    )
                ),
            },
            {
                "method": best_feature,
                "calibration_units": int(
                    len(
                        single_pooled_calibration
                    )
                ),
                "test_nominal_units": int(
                    len(
                        single_test0
                    )
                ),
                "test_nominal_alarms": int(
                    single_pooled_nominal_rej.sum()
                ),
                "test_nominal_fpr": float(
                    single_pooled_nominal_rej.mean()
                ),
                "threshold": (
                    reviewer_conformal_threshold(
                        (
                            single_pooled_calibration
                            - care["mu"][best_j]
                        )
                        / care["sd"][best_j],
                        args.alpha,
                    )
                ),
            },
        ]
    )

    pooled_validity[
        "care_minus_baseline_fpr"
    ] = (
        float(
            care_pooled_nominal_rej.mean()
        )
        - float(
            single_pooled_nominal_rej.mean()
        )
    )

    pooled_validity.to_csv(
        outdir
        / "care_pooled_calibration_validity.csv",
        index=False,
    )

    operating_alpha_grid = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        0.15,
    ]

    care_test_nominal_p = (
        conformal_p(
            care_pooled_calibration,
            care_test0,
        )
    )

    fixed_test_nominal_p = (
        conformal_p(
            single_pooled_calibration,
            single_test0,
        )
    )

    care_test_fpr_by_alpha = {}
    fixed_test_fpr_by_alpha = {}

    for operating_alpha in (
        operating_alpha_grid
    ):
        care_test_fpr_by_alpha[
            operating_alpha
        ] = float(
            np.mean(
                care_test_nominal_p
                <= operating_alpha
            )
        )

        fixed_test_fpr_by_alpha[
            operating_alpha
        ] = float(
            np.mean(
                fixed_test_nominal_p
                <= operating_alpha
            )
        )

    print(
        "\n=== POOLED-CALIBRATION "
        "VALIDITY ==="
    )

    print(
        pooled_validity.to_string(
            index=False
        )
    )

    lambda_final_scores = {}
    lambda_test_nominal_scores = {}

    threshold_rows = []

    for ridge, model in (
        lambda_models.items()
    ):
        final_score = care_score(
            Xfinal,
            model,
        )

        test_nominal_score = (
            care_score(
                Xtest0,
                model,
            )
        )

        lambda_final_scores[
            ridge
        ] = final_score

        lambda_test_nominal_scores[
            ridge
        ] = test_nominal_score

        nominal_rejection = (
            conformal_p(
                final_score,
                test_nominal_score,
            )
            <= args.alpha
        )

        threshold_rows.append({
            "method": "CARE",
            "lambda": ridge,
            "selected_on_design":
                bool(
                    np.isclose(
                        ridge,
                        selected_lambda,
                    )
                ),
            "threshold_standardized_units":
                reviewer_conformal_threshold(
                    final_score,
                    args.alpha,
                ),
            "test_nominal_fpr": float(
                nominal_rejection.mean()
            ),
            "test_nominal_alarms": int(
                nominal_rejection.sum()
            ),
            "test_nominal_units": int(
                len(
                    nominal_rejection
                )
            ),
        })

    baseline_nominal_rejection = (
        conformal_p(
            single_final_z,
            single_test0_z,
        )
        <= args.alpha
    )

    threshold_rows.append({
        "method": best_feature,
        "lambda": np.nan,
        "selected_on_design": True,
        "threshold_standardized_units":
            reviewer_conformal_threshold(
                single_final_z,
                args.alpha,
            ),
        "test_nominal_fpr": float(
            baseline_nominal_rejection.mean()
        ),
        "test_nominal_alarms": int(
            baseline_nominal_rejection.sum()
        ),
        "test_nominal_units": int(
            len(
                baseline_nominal_rejection
            )
        ),
    })

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    selected_care_threshold = float(
        threshold_table.loc[
            threshold_table[
                "method"
            ].eq("CARE")
            & threshold_table[
                "selected_on_design"
            ],
            "threshold_standardized_units",
        ].iloc[0]
    )

    baseline_threshold = float(
        threshold_table.loc[
            ~threshold_table[
                "method"
            ].eq("CARE"),
            "threshold_standardized_units",
        ].iloc[0]
    )

    threshold_table[
        "selected_care_minus_baseline_threshold"
    ] = (
        selected_care_threshold
        - baseline_threshold
    )

    threshold_table.to_csv(
        outdir
        / "care_empirical_thresholds.csv",
        index=False,
    )

    print(
        "\n=== EMPIRICAL THRESHOLDS ==="
    )

    print(
        threshold_table.to_string(
            index=False
        )
    )


    # Pooled nominal scores and empirical
    # thresholds for all frozen CARE heads.
    def care_head_scores(X, fitted):
        Z = (
            X - fitted["mu"]
        ) / fitted["sd"]

        W = fitted["weights"]
        Sigma = fitted["Sigma"]

        den = np.sqrt(
            np.maximum(
                np.einsum(
                    "ki,ij,kj->k",
                    W,
                    Sigma,
                    W,
                ),
                1e-12,
            )
        )

        return (
            Z @ W.T
        ) / den[None, :]

    def conformal_threshold(
        values,
        alpha,
    ):
        values = np.sort(
            np.asarray(
                values,
                dtype=float,
            )
        )

        index = (
            int(
                np.ceil(
                    (len(values) + 1)
                    * (1.0 - alpha)
                )
            )
            - 1
        )

        index = int(
            np.clip(
                index,
                0,
                len(values) - 1,
            )
        )

        return float(values[index])

    from statistics import NormalDist
    import os

    final_heads = reviewer_care_head_scores(
        Xfinal,
        care,
    )

    audit_heads = reviewer_care_head_scores(
        Xaudit,
        care,
    )

    pooled_heads = np.vstack(
        [
            final_heads,
            audit_heads,
        ]
    )

    pooled_ids = np.concatenate(
        [
            final_ids,
            audit_ids,
        ]
    )

    pooled_care = np.concatenate(
        [
            care_final,
            care_audit,
        ]
    )

    pooled_fixed = np.concatenate(
        [
            single_final,
            single_audit,
        ]
    )

    head_names = np.asarray(
        care["weight_names"]
    ).astype(str)

    np.savez(
        outdir
        / "care_pooled_nominal_"
        "head_scores.npz",
        scores=pooled_heads,
        cube_id=pooled_ids,
        head_names=head_names,
    )

    normal = NormalDist()

    gaussian_q95 = normal.inv_cdf(
        0.95
    )

    gaussian_q99 = normal.inv_cdf(
        0.99
    )

    care_max_threshold = (
        reviewer_conformal_threshold(
            pooled_care,
            args.alpha,
        )
    )

    fixed_threshold = (
        reviewer_conformal_threshold(
            pooled_fixed,
            args.alpha,
        )
    )

    rows = []

    for j, name in enumerate(
        head_names
    ):
        values = pooled_heads[:, j]

        q95 = float(
            np.quantile(
                values,
                0.95,
                method="higher",
            )
        )

        q99 = float(
            np.quantile(
                values,
                0.99,
                method="higher",
            )
        )

        threshold = (
            reviewer_conformal_threshold(
                values,
                args.alpha,
            )
        )

        rows.append(
            {
                "head": name,
                "n_calibration":
                    len(values),
                "conformal_threshold":
                    threshold,
                "empirical_q95": q95,
                "empirical_q99": q99,
                "q99_over_q95":
                    q99 / q95,
                "gaussian_q99_over_q95":
                    gaussian_q99
                    / gaussian_q95,
                "exceedances_at_"
                "care_max_threshold":
                    int(
                        np.sum(
                            values
                            > care_max_threshold
                        )
                    ),
            }
        )

    head_thresholds = (
        pd.DataFrame(rows)
        .sort_values(
            "conformal_threshold",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    head_thresholds.to_csv(
        outdir / "care_per_head_thresholds.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "statistic": "CARE_max",
                "threshold":
                    care_max_threshold,
            },
            {
                "statistic":
                    best_feature,
                "threshold":
                    fixed_threshold,
            },
            {
                "statistic":
                    "Gaussian_single_0.05",
                "threshold":
                    gaussian_q95,
            },
        ]
    )

    summary.to_csv(
        outdir
        / "care_threshold_summary.csv",
        index=False,
    )

    print(
        "\n=== POOLED THRESHOLD "
        "SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== POOLED PER-HEAD "
        "THRESHOLDS ==="
    )

    print(
        head_thresholds.to_string(
            index=False
        )
    )

    if (
        os.environ.get(
            "CARE_THRESHOLDS_ONLY"
        )
        == "1"
    ):
        print(
            "\nTHRESHOLD-ONLY RUN COMPLETE"
        )
        return

    # Joint upper-tail dependence diagnostic.
    # Each head is calibrated separately on the same pooled nominal
    # calibration units. Evaluation uses untouched nominal test cubes.
    jx_W = np.asarray(care["weights"], dtype=float)
    jx_Sigma = np.asarray(care["Sigma"], dtype=float)

    jx_den = np.sqrt(
        np.maximum(
            np.einsum(
                "ki,ij,kj->k",
                jx_W,
                jx_Sigma,
                jx_W,
            ),
            1e-12,
        )
    )

    jx_final_heads = (
        ((Xfinal - care["mu"]) / care["sd"])
        @ jx_W.T
    ) / jx_den[None, :]

    jx_audit_heads = (
        ((Xaudit - care["mu"]) / care["sd"])
        @ jx_W.T
    ) / jx_den[None, :]

    jx_test_heads = (
        ((Xtest0 - care["mu"]) / care["sd"])
        @ jx_W.T
    ) / jx_den[None, :]

    jx_calibration_heads = np.vstack(
        [
            jx_final_heads,
            jx_audit_heads,
        ]
    )

    jx_head_names = [
        str(value)
        for value in care["weight_names"]
    ]

    jx_head_reject = np.column_stack(
        [
            conformal_p(
                jx_calibration_heads[:, j],
                jx_test_heads[:, j],
            )
            <= args.alpha
            for j in range(len(jx_head_names))
        ]
    )

    jx_exceedance_count = jx_head_reject.sum(
        axis=1
    )

    jx_any_head = jx_exceedance_count > 0

    jx_pooled_care_calibration = np.concatenate(
        [
            care_final,
            care_audit,
        ]
    )

    jx_care_reject = (
        conformal_p(
            jx_pooled_care_calibration,
            care_test0,
        )
        <= args.alpha
    )

    jx_marginal_rates = jx_head_reject.mean(
        axis=0
    )

    jx_mean_marginal = float(
        jx_marginal_rates.mean()
    )

    jx_union_rate = float(
        jx_any_head.mean()
    )

    jx_independent_union = float(
        1.0
        - np.prod(
            1.0 - jx_marginal_rates
        )
    )

    jx_safe_marginal = np.clip(
        jx_mean_marginal,
        1e-12,
        1.0 - 1e-12,
    )

    jx_safe_union = np.clip(
        jx_union_rate,
        1e-12,
        1.0 - 1e-12,
    )

    jx_tail_multiplicity = float(
        np.log1p(-jx_safe_union)
        / np.log1p(-jx_safe_marginal)
    )

    jx_head_covariance = (
        jx_W
        @ jx_Sigma
        @ jx_W.T
    ) / np.outer(jx_den, jx_den)

    jx_eigenvalues = np.clip(
        np.linalg.eigvalsh(
            jx_head_covariance
        ),
        0.0,
        None,
    )

    jx_linear_effective_rank = float(
        jx_eigenvalues.sum() ** 2
        / np.maximum(
            np.square(
                jx_eigenvalues
            ).sum(),
            1e-12,
        )
    )

    # Whole-cube bootstrap interval for tail multiplicity.
    jx_rng = np.random.default_rng(
        args.seed + 88000
    )

    jx_n = len(jx_any_head)
    jx_nboot = max(
        2000,
        int(args.bootstrap),
    )

    jx_counts = jx_rng.multinomial(
        jx_n,
        np.full(
            jx_n,
            1.0 / jx_n,
        ),
        size=jx_nboot,
    ).astype(float)

    jx_boot_marginal = (
        jx_counts
        @ jx_head_reject.astype(float)
    ) / float(jx_n)

    jx_boot_mean_marginal = (
        jx_boot_marginal.mean(axis=1)
    )

    jx_boot_union = (
        jx_counts
        @ jx_any_head.astype(float)
    ) / float(jx_n)

    jx_boot_tail_multiplicity = (
        np.log1p(
            -np.clip(
                jx_boot_union,
                1e-12,
                1.0 - 1e-12,
            )
        )
        / np.log1p(
            -np.clip(
                jx_boot_mean_marginal,
                1e-12,
                1.0 - 1e-12,
            )
        )
    )

    jx_tail_ci_low, jx_tail_ci_high = (
        np.quantile(
            jx_boot_tail_multiplicity,
            [0.025, 0.975],
        )
    )

    jx_summary = pd.DataFrame(
        [
            {
                "alpha": args.alpha,
                "heads": len(jx_head_names),
                "calibration_cubes":
                    len(jx_calibration_heads),
                "test_nominal_cubes": jx_n,
                "linear_effective_rank":
                    jx_linear_effective_rank,
                "mean_individual_head_fpr":
                    jx_mean_marginal,
                "minimum_individual_head_fpr":
                    float(jx_marginal_rates.min()),
                "maximum_individual_head_fpr":
                    float(jx_marginal_rates.max()),
                "any_individual_head_fpr":
                    jx_union_rate,
                "independent_union_reference":
                    jx_independent_union,
                "care_max_fpr":
                    float(jx_care_reject.mean()),
                "tail_effective_multiplicity":
                    jx_tail_multiplicity,
                "tail_multiplicity_ci_low":
                    float(jx_tail_ci_low),
                "tail_multiplicity_ci_high":
                    float(jx_tail_ci_high),
                "mean_exceedance_count":
                    float(jx_exceedance_count.mean()),
                "mean_count_given_any":
                    float(
                        jx_exceedance_count[
                            jx_any_head
                        ].mean()
                    )
                    if np.any(jx_any_head)
                    else 0.0,
                "single_head_fraction_given_any":
                    float(
                        np.mean(
                            jx_exceedance_count[
                                jx_any_head
                            ] == 1
                        )
                    )
                    if np.any(jx_any_head)
                    else 0.0,
                "bootstrap_unit": "cube_id",
                "bootstrap_reps": jx_nboot,
            }
        ]
    )

    jx_summary.to_csv(
        outdir
        / "joint_exceedance_summary.csv",
        index=False,
    )

    jx_head_table = pd.DataFrame(
        {
            "head": jx_head_names,
            "test_nominal_exceedances":
                jx_head_reject.sum(axis=0),
            "test_nominal_fpr":
                jx_marginal_rates,
        }
    ).sort_values(
        "test_nominal_fpr",
        ascending=False,
    )

    jx_head_table.to_csv(
        outdir
        / "joint_exceedance_head_marginals.csv",
        index=False,
    )

    jx_distribution = pd.DataFrame(
        {
            "exceeding_heads": np.arange(
                len(jx_head_names) + 1
            ),
            "test_nominal_cubes": np.bincount(
                jx_exceedance_count,
                minlength=len(jx_head_names) + 1,
            ),
        }
    )

    jx_distribution["fraction"] = (
        jx_distribution[
            "test_nominal_cubes"
        ]
        / float(jx_n)
    )

    jx_distribution.to_csv(
        outdir
        / "joint_exceedance_count_distribution.csv",
        index=False,
    )

    jx_pair_rows = []

    for j in range(len(jx_head_names)):
        for k in range(j + 1, len(jx_head_names)):
            jx_a = jx_head_reject[:, j]
            jx_b = jx_head_reject[:, k]

            jx_both = float(
                np.mean(jx_a & jx_b)
            )

            jx_either = float(
                np.mean(jx_a | jx_b)
            )

            jx_expected = float(
                jx_marginal_rates[j]
                * jx_marginal_rates[k]
            )

            jx_pair_rows.append(
                {
                    "head_1": jx_head_names[j],
                    "head_2": jx_head_names[k],
                    "head_1_fpr":
                        jx_marginal_rates[j],
                    "head_2_fpr":
                        jx_marginal_rates[k],
                    "joint_fpr": jx_both,
                    "independent_joint_reference":
                        jx_expected,
                    "joint_lift":
                        jx_both
                        / max(jx_expected, 1e-12),
                    "jaccard":
                        jx_both
                        / max(jx_either, 1e-12),
                }
            )

    pd.DataFrame(
        jx_pair_rows
    ).to_csv(
        outdir
        / "joint_exceedance_pairs.csv",
        index=False,
    )

    jx_argmax = np.argmax(
        jx_test_heads,
        axis=1,
    )

    jx_driver_counts = np.bincount(
        jx_argmax[jx_care_reject],
        minlength=len(jx_head_names),
    )

    jx_driver_table = pd.DataFrame(
        {
            "head": jx_head_names,
            "care_alarm_driver_count":
                jx_driver_counts,
            "fraction_of_care_alarms":
                jx_driver_counts
                / max(
                    int(jx_care_reject.sum()),
                    1,
                ),
        }
    ).sort_values(
        "care_alarm_driver_count",
        ascending=False,
    )

    jx_driver_table.to_csv(
        outdir
        / "joint_exceedance_alarm_drivers.csv",
        index=False,
    )

    print(
        "\n=== JOINT EXCEEDANCE SUMMARY ==="
    )
    print(
        jx_summary.to_string(index=False)
    )

    print(
        "\n=== EXCEEDANCE COUNT DISTRIBUTION ==="
    )
    print(
        jx_distribution.loc[
            jx_distribution[
                "test_nominal_cubes"
            ] > 0
        ].to_string(index=False)
    )

    print(
        "\n=== INDIVIDUAL HEAD MARGINALS ==="
    )
    print(
        jx_head_table.to_string(index=False)
    )

    validity_rows = []

    for split_name, cs, ss in [
        (
            "audit",
            care_audit,
            single_audit,
        ),
        (
            "test_nominal",
            care_test0,
            single_test0,
        ),
    ]:
        for method, ref, vals in [
            (
                "CARE",
                care_final,
                cs,
            ),
            (
                best_feature,
                single_final,
                ss,
            ),
        ]:
            rej = (
                conformal_p(
                    ref,
                    vals,
                )
                <= args.alpha
            )

            lo, hi = exact_ci(rej)

            validity_rows.append({
                "split": split_name,
                "method": method,
                "n_cubes": len(rej),
                "alarms": int(rej.sum()),
                "fpr": float(rej.mean()),
                "exact_ci_low": lo,
                "exact_ci_high": hi,
            })

    validity = pd.DataFrame(
        validity_rows
    )

    validity.to_csv(
        outdir / "care_validity.csv",
        index=False,
    )

    print("\n=== VALIDITY ===")
    print(validity.to_string(index=False))


    # Reviewer check 2: every head is calibrated separately on the same pooled
    # 466 nominal cubes used by the current audit. The max rule is unchanged.
    reviewer_pooled_calibration_X = np.vstack([Xfinal, Xaudit])
    reviewer_head_calibration = reviewer_care_head_scores(
        reviewer_pooled_calibration_X,
        care,
    )
    reviewer_care_calibration = care_score(
        reviewer_pooled_calibration_X,
        care,
    )

    selected_test = choose_one_window_per_cube(
        ft,
        args.seed + 202,
    )

    rows = []
    records = []
    record_map = {}

    reviewer_head_selection_records = {}
    multiplicity_record_map = {}
    pooled_record_map = {}
    pooled_best_head_record_map = {}
    pooled_rows = []
    operating_curve_rows = []
    per_feature_rows = []
    lambda_holdout_rows = []

    scenarios = holdout_scenarios()

    for si, (
        structure,
        amount,
        delta,
    ) in enumerate(scenarios):

        print(
            f"[{si+1}/{len(scenarios)}] "
            f"TEST {structure} "
            f"amount={amount:g} "
            f"delta={delta:g}"
        )

        fa, ia, affected = inject_pair(
            ft,
            it,
            selected_test,
            structure,
            amount,
            delta,
            args.seed + 10000 + si,
        )

        ids, Xa, names = feature_matrix(
            fa,
            ia,
            f_local_cal,
            f_comp_norm,
            i_local_cal,
            i_comp_norm,
            args.windows,
        )

        assert np.array_equal(
            ids,
            test_ids,
        )

        assert names == feature_names

        care_alt_heads = (
            reviewer_care_head_scores(
                Xa,
                care,
            )
        )

        care_alt_score = np.max(
            care_alt_heads,
            axis=1,
        )

        maximizing_head = np.argmax(
            care_alt_heads,
            axis=1,
        )

        maximizing_counts = np.bincount(
            maximizing_head,
            minlength=care_alt_heads.shape[1],
        )

        dominant_head_j = int(
            np.argmax(
                maximizing_counts
            )
        )

        dominant_head_frequency = float(
            maximizing_counts[
                dominant_head_j
            ]
            / len(
                maximizing_head
            )
        )

        dominant_head_score = (
            care_alt_heads[
                :,
                dominant_head_j,
            ]
        )

        dominant_head_rej = (
            conformal_p(
                care_final_heads[
                    :,
                    dominant_head_j,
                ],
                dominant_head_score,
            )
            <= args.alpha
        )

        care_rej = (
            conformal_p(
                care_final,
                care_alt_score,
            )
            <= args.alpha
        )

        single_rej = (
            conformal_p(
                single_final,
                Xa[:, best_j],
            )
            <= args.alpha
        )

        care_pooled_p = (
            conformal_p(
                care_pooled_calibration,
                care_alt_score,
            )
        )

        single_pooled_p = (
            conformal_p(
                single_pooled_calibration,
                Xa[:, best_j],
            )
        )

        care_pooled_rej = (
            care_pooled_p
            <= args.alpha
        )

        single_pooled_rej = (
            single_pooled_p
            <= args.alpha
        )

        for operating_alpha in (
            operating_alpha_grid
        ):
            operating_curve_rows.append({
                "structure": structure,
                "amount": amount,
                "delta": delta,
                "alpha": operating_alpha,
                "care_test_nominal_fpr":
                    care_test_fpr_by_alpha[
                        operating_alpha
                    ],
                "fixed_test_nominal_fpr":
                    fixed_test_fpr_by_alpha[
                        operating_alpha
                    ],
                "care_power": float(
                    np.mean(
                        care_pooled_p
                        <= operating_alpha
                    )
                ),
                "fixed_power": float(
                    np.mean(
                        single_pooled_p
                        <= operating_alpha
                    )
                ),
            })

        pooled_head_rejections = (
            np.column_stack(
                [
                    (
                        conformal_p(
                            care_pooled_head_calibration[
                                :,
                                head_j,
                            ],
                            care_alt_heads[
                                :,
                                head_j,
                            ],
                        )
                        <= args.alpha
                    )
                    for head_j in range(
                        care_alt_heads.shape[1]
                    )
                ]
            )
        )

        pooled_head_power = (
            pooled_head_rejections.mean(
                axis=0
            )
        )

        best_pooled_head_j = int(
            np.argmax(
                pooled_head_power
            )
        )

        best_pooled_head_rej = (
            pooled_head_rejections[
                :,
                best_pooled_head_j,
            ]
        )

        for ridge, model in (
            lambda_models.items()
        ):
            ridge_alt_score = (
                care_score(
                    Xa,
                    model,
                )
            )

            ridge_rej = (
                conformal_p(
                    lambda_final_scores[
                        ridge
                    ],
                    ridge_alt_score,
                )
                <= args.alpha
            )

            lambda_holdout_rows.append({
                "structure": structure,
                "amount": amount,
                "delta": delta,
                "lambda": ridge,
                "selected_on_design":
                    bool(
                        np.isclose(
                            ridge,
                            selected_lambda,
                        )
                    ),
                "care_power": float(
                    ridge_rej.mean()
                ),
                "fixed_feature_power":
                    float(
                        single_rej.mean()
                    ),
                "care_minus_fixed":
                    float(
                        ridge_rej.mean()
                        - single_rej.mean()
                    ),
            })

        individual_power = []

        for j, fname in enumerate(
            feature_names
        ):
            rej = (
                conformal_p(
                    Xfinal[:, j],
                    Xa[:, j],
                )
                <= args.alpha
            )

            power = float(
                rej.mean()
            )

            individual_power.append(
                power
            )

            per_feature_rows.append({
                "structure": structure,
                "amount": amount,
                "delta": delta,
                "feature": fname,
                "power": power,
            })

        oracle_j = int(
            np.argmax(
                individual_power
            )
        )

        meta = {
            "structure": structure,
            "amount": amount,
            "delta": delta,
        }

        key = json.dumps(
            meta,
            sort_keys=True,
        )

        record_map[key] = {
            "care": care_rej,
            "single": single_rej,
        }

        multiplicity_record_map[
            key
        ] = {
            "care": dominant_head_rej,
            "single": care_rej,
        }

        pooled_record_map[
            key
        ] = {
            "care": care_pooled_rej,
            "single": single_pooled_rej,
        }

        pooled_best_head_record_map[
            key
        ] = {
            "care": best_pooled_head_rej,
            "single": care_pooled_rej,
        }

        pooled_rows.append({
            **meta,
            "affected_pixels": affected,
            "n_cubes": len(test_ids),
            "care_power": float(
                care_pooled_rej.mean()
            ),
            "fixed_feature_power": float(
                single_pooled_rej.mean()
            ),
            "care_minus_fixed": float(
                care_pooled_rej.mean()
                - single_pooled_rej.mean()
            ),
            "best_individual_head": (
                care[
                    "weight_names"
                ][best_pooled_head_j]
            ),
            "best_individual_head_power":
                float(
                    best_pooled_head_rej.mean()
                ),
            "best_head_minus_care": float(
                best_pooled_head_rej.mean()
                - care_pooled_rej.mean()
            ),
        })


        # Store per-cube rejections for a selection-corrected head comparison.
        # No aggregation rule or exceedance diagnostic is changed here.
        reviewer_head_alt = reviewer_care_head_scores(Xa, care)
        reviewer_head_reject = np.column_stack([
            conformal_p(
                reviewer_head_calibration[:, reviewer_j],
                reviewer_head_alt[:, reviewer_j],
            )
            <= args.alpha
            for reviewer_j in range(reviewer_head_alt.shape[1])
        ])
        reviewer_care_reject = (
            conformal_p(
                reviewer_care_calibration,
                care_alt_score,
            )
            <= args.alpha
        )
        reviewer_head_selection_records[key] = {
            "care": reviewer_care_reject,
            "heads": reviewer_head_reject,
        }

        rows.append({
            **meta,
            "affected_pixels": affected,
            "n_cubes": len(test_ids),
            "care_power":
                float(care_rej.mean()),
            "best_fixed_single_feature":
                best_feature,
            "best_fixed_single_power":
                float(single_rej.mean()),
            "care_minus_best_fixed":
                float(
                    care_rej.mean()
                    - single_rej.mean()
                ),
            "dominant_care_head":
                care[
                    "weight_names"
                ][dominant_head_j],
            "dominant_head_frequency":
                dominant_head_frequency,
            "dominant_head_power":
                float(
                    dominant_head_rej.mean()
                ),
            "dominant_head_minus_care":
                float(
                    dominant_head_rej.mean()
                    - care_rej.mean()
                ),
            "oracle_best_feature":
                feature_names[oracle_j],
            "oracle_best_feature_power":
                individual_power[oracle_j],
            "care_minus_oracle_single":
                float(
                    care_rej.mean()
                    - individual_power[oracle_j]
                ),
        })


    reviewer_crossfit = reviewer_crossfit_head_audit(
        reviewer_head_selection_records,
        care["weight_names"],
        args.bootstrap,
        args.seed + 62000,
    )
    reviewer_crossfit.to_csv(
        outdir / "reviewer_crossfit_head_vs_care_simultaneous_ci.csv",
        index=False,
    )

    print("\n=== SELECTION-CORRECTED HEAD VERSUS CARE ===")
    print(
        reviewer_crossfit["result"]
        .value_counts(dropna=False)
        .to_string()
    )
    reviewer_resolved = reviewer_crossfit.loc[
        ~reviewer_crossfit["result"].eq("unresolved")
    ]
    if reviewer_resolved.empty:
        print("No scenario resolves after cross-fitted selection correction.")
    else:
        print(
            reviewer_resolved[
                [
                    "structure",
                    "amount",
                    "delta",
                    "head_selected_on_fold0",
                    "head_selected_on_fold1",
                    "crossfit_head_minus_care",
                    "simul_ci_low",
                    "simul_ci_high",
                    "result",
                ]
            ].to_string(index=False)
        )

    surface = pd.DataFrame(rows)

    pooled_surface = pd.DataFrame(
        pooled_rows
    )

    operating_curve = pd.DataFrame(
        operating_curve_rows
    )

    operating_curve[
        "care_minus_fixed_native"
    ] = (
        operating_curve[
            "care_power"
        ]
        - operating_curve[
            "fixed_power"
        ]
    )

    operating_curve.to_csv(
        outdir
        / "care_operating_curve.csv",
        index=False,
    )

    operating_summary = (
        operating_curve.groupby(
            "alpha",
            observed=True,
        )
        .agg(
            scenarios=(
                "delta",
                "size",
            ),
            care_test_nominal_fpr=(
                "care_test_nominal_fpr",
                "first",
            ),
            fixed_test_nominal_fpr=(
                "fixed_test_nominal_fpr",
                "first",
            ),
            mean_care_power=(
                "care_power",
                "mean",
            ),
            mean_fixed_power=(
                "fixed_power",
                "mean",
            ),
            median_care_power=(
                "care_power",
                "median",
            ),
            median_fixed_power=(
                "fixed_power",
                "median",
            ),
        )
        .reset_index()
    )

    operating_summary[
        "mean_native_gain"
    ] = (
        operating_summary[
            "mean_care_power"
        ]
        - operating_summary[
            "mean_fixed_power"
        ]
    )

    operating_summary.to_csv(
        outdir
        / "care_operating_curve_summary.csv",
        index=False,
    )

    matched_targets = [
        0.025,
        0.04,
        0.05,
        0.075,
        0.10,
    ]

    matched_rows = []

    def collapsed_curve(
        x,
        y,
    ):
        frame = pd.DataFrame({
            "x": np.asarray(
                x,
                dtype=float,
            ),
            "y": np.asarray(
                y,
                dtype=float,
            ),
        })

        frame = (
            frame.groupby(
                "x",
                observed=True,
            )["y"]
            .max()
            .reset_index()
            .sort_values("x")
        )

        return (
            frame[
                "x"
            ].to_numpy(
                dtype=float
            ),
            frame[
                "y"
            ].to_numpy(
                dtype=float
            ),
        )

    for (
        structure,
        amount,
        delta,
    ), frame in operating_curve.groupby(
        [
            "structure",
            "amount",
            "delta",
        ],
        observed=True,
    ):
        care_x, care_y = collapsed_curve(
            frame[
                "care_test_nominal_fpr"
            ],
            frame[
                "care_power"
            ],
        )

        fixed_x, fixed_y = collapsed_curve(
            frame[
                "fixed_test_nominal_fpr"
            ],
            frame[
                "fixed_power"
            ],
        )

        overlap_low = max(
            care_x.min(),
            fixed_x.min(),
        )

        overlap_high = min(
            care_x.max(),
            fixed_x.max(),
        )

        for target_fpr in (
            matched_targets
        ):
            if not (
                overlap_low
                <= target_fpr
                <= overlap_high
            ):
                continue

            care_power = float(
                np.interp(
                    target_fpr,
                    care_x,
                    care_y,
                )
            )

            fixed_power = float(
                np.interp(
                    target_fpr,
                    fixed_x,
                    fixed_y,
                )
            )

            matched_rows.append({
                "structure": structure,
                "amount": amount,
                "delta": delta,
                "target_fpr": target_fpr,
                "care_power_interpolated":
                    care_power,
                "fixed_power_interpolated":
                    fixed_power,
                "care_minus_fixed_interpolated":
                    care_power
                    - fixed_power,
                "care_fpr_min":
                    float(
                        care_x.min()
                    ),
                "care_fpr_max":
                    float(
                        care_x.max()
                    ),
                "fixed_fpr_min":
                    float(
                        fixed_x.min()
                    ),
                "fixed_fpr_max":
                    float(
                        fixed_x.max()
                    ),
                "descriptive_test_fpr_matching":
                    True,
            })

    matched_fpr = pd.DataFrame(
        matched_rows
    )

    matched_fpr.to_csv(
        outdir
        / "care_matched_fpr_interpolation.csv",
        index=False,
    )

    matched_family = (
        matched_fpr.groupby(
            [
                "target_fpr",
                "structure",
            ],
            observed=True,
        )
        .agg(
            scenarios=(
                "delta",
                "size",
            ),
            mean_care_power=(
                "care_power_interpolated",
                "mean",
            ),
            mean_fixed_power=(
                "fixed_power_interpolated",
                "mean",
            ),
            mean_gain=(
                "care_minus_fixed_interpolated",
                "mean",
            ),
            median_gain=(
                "care_minus_fixed_interpolated",
                "median",
            ),
            minimum_gain=(
                "care_minus_fixed_interpolated",
                "min",
            ),
            maximum_gain=(
                "care_minus_fixed_interpolated",
                "max",
            ),
        )
        .reset_index()
    )

    matched_family.to_csv(
        outdir
        / "care_matched_fpr_family_summary.csv",
        index=False,
    )

    print(
        "\n=== OPERATING CURVE "
        "SUMMARY ==="
    )

    print(
        operating_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== DESCRIPTIVE MATCHED-FPR "
        "FAMILY SUMMARY ==="
    )

    print(
        matched_family.to_string(
            index=False
        )
    )

    pooled_ci = (
        simultaneous_bootstrap(
            pooled_record_map,
            args.bootstrap,
            args.seed + 70000,
        )
        .rename(
            columns={
                "care_minus_best_single":
                    "care_minus_fixed",
                "simul_ci_low":
                    "simul_ci_low",
                "simul_ci_high":
                    "simul_ci_high",
            }
        )
    )

    pooled_best_head_ci = (
        simultaneous_bootstrap(
            pooled_best_head_record_map,
            args.bootstrap,
            args.seed + 80000,
        )
        .rename(
            columns={
                "care_minus_best_single":
                    "best_head_minus_care",
                "simul_ci_low":
                    "best_head_simul_ci_low",
                "simul_ci_high":
                    "best_head_simul_ci_high",
            }
        )
    )

    pooled_surface = (
        pooled_surface.merge(
            pooled_ci[
                [
                    "structure",
                    "amount",
                    "delta",
                    "simul_ci_low",
                    "simul_ci_high",
                ]
            ],
            on=[
                "structure",
                "amount",
                "delta",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            pooled_best_head_ci[
                [
                    "structure",
                    "amount",
                    "delta",
                    "best_head_simul_ci_low",
                    "best_head_simul_ci_high",
                ]
            ],
            on=[
                "structure",
                "amount",
                "delta",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    pooled_surface.to_csv(
        outdir
        / "care_pooled_calibration_power.csv",
        index=False,
    )

    pooled_ci.to_csv(
        outdir
        / "care_pooled_calibration_ci.csv",
        index=False,
    )

    pooled_best_head_ci.to_csv(
        outdir
        / "care_pooled_best_head_ci.csv",
        index=False,
    )

    print(
        "\n=== POOLED-CALIBRATION "
        "POWER SUMMARY ==="
    )

    print(
        pooled_surface.groupby(
            "structure",
            observed=True,
        )
        .agg(
            scenarios=(
                "delta",
                "size",
            ),
            care_power=(
                "care_power",
                "mean",
            ),
            fixed_power=(
                "fixed_feature_power",
                "mean",
            ),
            mean_gain=(
                "care_minus_fixed",
                "mean",
            ),
            median_gain=(
                "care_minus_fixed",
                "median",
            ),
            mean_best_head_cost=(
                "best_head_minus_care",
                "mean",
            ),
        )
        .reset_index()
        .to_string(
            index=False
        )
    )

    lambda_holdout = pd.DataFrame(
        lambda_holdout_rows
    )

    lambda_holdout.to_csv(
        outdir
        / "care_lambda_holdout_sensitivity.csv",
        index=False,
    )

    multiplicity_ci = (
        simultaneous_bootstrap(
            multiplicity_record_map,
            args.bootstrap,
            args.seed + 60000,
        )
        .rename(
            columns={
                "care_minus_best_single":
                    "dominant_head_minus_care",
                "simul_ci_low":
                    "multiplicity_simul_ci_low",
                "simul_ci_high":
                    "multiplicity_simul_ci_high",
            }
        )
    )

    multiplicity_ci.to_csv(
        outdir
        / "care_multiplicity_cost_ci.csv",
        index=False,
    )

    surface = surface.merge(
        multiplicity_ci[
            [
                "structure",
                "amount",
                "delta",
                "multiplicity_simul_ci_low",
                "multiplicity_simul_ci_high",
            ]
        ],
        on=[
            "structure",
            "amount",
            "delta",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        "\n=== LAMBDA HOLDOUT "
        "SUMMARY ==="
    )

    print(
        lambda_holdout.groupby(
            [
                "lambda",
                "selected_on_design",
            ],
            observed=True,
        )
        .agg(
            scenarios=(
                "delta",
                "size",
            ),
            mean_care_power=(
                "care_power",
                "mean",
            ),
            median_care_power=(
                "care_power",
                "median",
            ),
            mean_gain=(
                "care_minus_fixed",
                "mean",
            ),
            median_gain=(
                "care_minus_fixed",
                "median",
            ),
        )
        .reset_index()
        .to_string(
            index=False
        )
    )

    print(
        "\n=== MULTIPLICITY COST ==="
    )

    print(
        surface[
            [
                "structure",
                "amount",
                "delta",
                "dominant_care_head",
                "dominant_head_frequency",
                "care_power",
                "dominant_head_power",
                "dominant_head_minus_care",
                "multiplicity_simul_ci_low",
                "multiplicity_simul_ci_high",
            ]
        ].to_string(
            index=False
        )
    )

    surface.to_csv(
        outdir / "care_holdout_power.csv",
        index=False,
    )

    pd.DataFrame(
        per_feature_rows
    ).to_csv(
        outdir / "care_all_individual_power.csv",
        index=False,
    )

    ci = simultaneous_bootstrap(
        record_map,
        args.bootstrap,
        args.seed + 50000,
    )

    ci.to_csv(
        outdir / "care_simultaneous_ci.csv",
        index=False,
    )

    summary = (
        surface.groupby("structure")
        .agg(
            n_scenarios=(
                "delta",
                "size",
            ),
            mean_care_power=(
                "care_power",
                "mean",
            ),
            mean_best_fixed_power=(
                "best_fixed_single_power",
                "mean",
            ),
            mean_gain=(
                "care_minus_best_fixed",
                "mean",
            ),
            median_gain=(
                "care_minus_best_fixed",
                "median",
            ),
            mean_oracle_single_power=(
                "oracle_best_feature_power",
                "mean",
            ),
            mean_gain_vs_oracle=(
                "care_minus_oracle_single",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        outdir / "care_summary.csv",
        index=False,
    )

    overall_gain = float(
        surface[
            "care_minus_best_fixed"
        ].mean()
    )

    positive_groups = int(
        (
            summary["mean_gain"] > 0
        ).sum()
    )

    ci2 = ci.copy()

    ci2["positive"] = (
        ci2["simul_ci_low"] > 0
    )

    ci2["negative"] = (
        ci2["simul_ci_high"] < 0
    )

    ci_counts = (
        ci2.groupby("structure")
        .agg(
            scenarios=(
                "delta",
                "size",
            ),
            mean=(
                "care_minus_best_single",
                "mean",
            ),
            median=(
                "care_minus_best_single",
                "median",
            ),
            simultaneous_positive=(
                "positive",
                "sum",
            ),
            simultaneous_negative=(
                "negative",
                "sum",
            ),
        )
        .reset_index()
    )

    passes = (
        overall_gain > 0
        and positive_groups >= 2
    )

    decision = {
        "lambda": args.lambda_ridge,
        "alpha": args.alpha,
        "best_single_selected_on":
            "design_split_only",
        "best_single_feature":
            best_feature,
        "mean_heldout_gain_vs_best_fixed_single":
            overall_gain,
        "groups_with_positive_mean_gain":
            positive_groups,
        "n_groups":
            int(len(summary)),
        "passes_frozen_kill_criterion":
            bool(passes),
        "next_step": (
            "replicate frozen CARE on astronomy"
            if passes
            else
            "stop CARE and retain diagnostics paper"
        ),
    }

    with open(
        outdir / "care_kill_decision.json",
        "w",
    ) as f:
        json.dump(
            decision,
            f,
            indent=2,
        )

    print("\n=== HOLDOUT SUMMARY ===")
    print(summary.to_string(index=False))

    print(
        "\n=== SIMULTANEOUS CI COUNTS ==="
    )
    print(ci_counts.to_string(index=False))

    print("\n=== CARE KILL DECISION ===")
    print(
        json.dumps(
            decision,
            indent=2,
        )
    )

    print(
        "\nWROTE",
        outdir.resolve(),
    )


if __name__ == "__main__":
    main()
