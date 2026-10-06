#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import t

FULL = "outputs/earthnet/caps_test.npz"
IMAGE = "outputs/earthnet_image_only/caps_test.npz"
OUT = Path("outputs/earthnet_multimodal_ablation")
DF = 5.0
MIN_VALID = 0.05
BOOT = 2000
SEED = 20260824


def load_pair():
    a = np.load(FULL)
    b = np.load(IMAGE)

    assert np.array_equal(a["cube_id"], b["cube_id"])
    assert np.array_equal(a["mask"], b["mask"])
    assert a["z"].shape == b["z"].shape

    mask = a["mask"].astype(bool)
    if mask.shape[1] == 1:
        mask = np.broadcast_to(mask, a["z"].shape)

    keep = mask.reshape(len(mask), -1).mean(1) >= MIN_VALID

    def get(d):
        return {
            "z": d["z"][keep].astype(np.float32),
            "log_sigma": d["log_sigma"][keep].astype(np.float32),
            "mask": mask[keep],
            "cube_id": d["cube_id"][keep],
        }

    return get(a), get(b)


def cube_aggregates(d):
    z = d["z"]
    ls = d["log_sigma"]
    m = d["mask"]
    cid = d["cube_id"]

    ids = np.unique(cid)
    pos = np.searchsorted(ids, cid)

    # n, nll, abs-error, sq-error, z, z^2, cover90, cover95
    A = np.zeros((len(ids), 8), dtype=np.float64)

    q90 = t.ppf(0.95, DF) / np.sqrt(DF / (DF - 2.0))
    q95 = t.ppf(0.975, DF) / np.sqrt(DF / (DF - 2.0))

    c0 = (
        0.5 * np.log((DF - 2.0) * np.pi)
        + gammaln(DF / 2.0)
        - gammaln((DF + 1.0) / 2.0)
    )

    for i in range(len(z)):
        mm = m[i]
        zz = z[i][mm].astype(np.float64)
        ll = ls[i][mm].astype(np.float64)

        sigma = np.exp(ll)
        residual = zz * sigma

        nll = (
            ll
            + c0
            + 0.5 * (DF + 1.0)
            * np.log1p(zz**2 / (DF - 2.0))
        )

        j = pos[i]
        A[j, 0] += len(zz)
        A[j, 1] += nll.sum()
        A[j, 2] += np.abs(residual).sum()
        A[j, 3] += (residual**2).sum()
        A[j, 4] += zz.sum()
        A[j, 5] += (zz**2).sum()
        A[j, 6] += (np.abs(zz) <= q90).sum()
        A[j, 7] += (np.abs(zz) <= q95).sum()

    return ids, A


def summarize(A):
    s = A.sum(0)
    n = s[0]
    zmean = s[4] / n
    zvar = max(0.0, s[5] / n - zmean**2)

    return {
        "mean_nll": s[1] / n,
        "mae": s[2] / n,
        "rmse": np.sqrt(s[3] / n),
        "z_mean": zmean,
        "z_sd": np.sqrt(zvar),
        "coverage90": s[6] / n,
        "coverage95": s[7] / n,
        "coverage90_error": abs(s[6] / n - 0.90),
        "coverage95_error": abs(s[7] / n - 0.95),
    }


def corr_from_sums(n, sx, sy, sxx, syy, sxy):
    if n < 2:
        return np.nan
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    cov = sxy - sx * sy / n
    return cov / np.sqrt(max(vx * vy, 1e-30))


def add_pairs(acc, x, y, valid):
    xx = x[valid].astype(np.float64)
    yy = y[valid].astype(np.float64)

    acc[0] += len(xx)
    acc[1] += xx.sum()
    acc[2] += yy.sum()
    acc[3] += np.dot(xx, xx)
    acc[4] += np.dot(yy, yy)
    acc[5] += np.dot(xx, yy)


def spatial_corr(d, lag):
    z, m = d["z"], d["mask"]
    acc = np.zeros(6, dtype=np.float64)

    for i in range(len(z)):
        # horizontal
        valid = m[i, :, :, :-lag] & m[i, :, :, lag:]
        add_pairs(
            acc,
            z[i, :, :, :-lag],
            z[i, :, :, lag:],
            valid,
        )

        # vertical
        valid = m[i, :, :-lag, :] & m[i, :, lag:, :]
        add_pairs(
            acc,
            z[i, :, :-lag, :],
            z[i, :, lag:, :],
            valid,
        )

    return corr_from_sums(*acc)


def channel_corr(d):
    z, m = d["z"], d["mask"]
    C = z.shape[1]
    R = np.eye(C)

    for a in range(C):
        for b in range(a + 1, C):
            acc = np.zeros(6, dtype=np.float64)

            for i in range(len(z)):
                valid = m[i, a] & m[i, b]
                add_pairs(
                    acc,
                    z[i, a],
                    z[i, b],
                    valid,
                )

            R[a, b] = R[b, a] = corr_from_sums(*acc)

    return R


def bootstrap(full_A, image_A):
    rng = np.random.default_rng(SEED)

    metrics = [
        "mean_nll",
        "mae",
        "rmse",
        "coverage90_error",
        "coverage95_error",
    ]

    f0 = summarize(full_A)
    i0 = summarize(image_A)

    draws = {k: [] for k in metrics}
    n = len(full_A)

    for _ in range(BOOT):
        ix = rng.integers(0, n, n)
        fs = summarize(full_A[ix])
        ims = summarize(image_A[ix])

        for k in metrics:
            draws[k].append(fs[k] - ims[k])

    rows = []
    for k in metrics:
        x = np.asarray(draws[k])
        rows.append({
            "metric": k,
            "multimodal": f0[k],
            "image_only": i0[k],
            "multimodal_minus_image": f0[k] - i0[k],
            "ci_low": np.quantile(x, 0.025),
            "ci_high": np.quantile(x, 0.975),
            "bootstrap_unit": "cube_id",
        })

    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    full, image = load_pair()

    full_ids, full_A = cube_aggregates(full)
    image_ids, image_A = cube_aggregates(image)

    assert np.array_equal(full_ids, image_ids)

    summaries = pd.DataFrame([
        {"model": "multimodal", **summarize(full_A)},
        {"model": "image_only", **summarize(image_A)},
    ])

    paired = bootstrap(full_A, image_A)

    lag_rows = []
    for lag in [1, 2, 4, 8, 16]:
        lag_rows.append({
            "lag": lag,
            "multimodal": spatial_corr(full, lag),
            "image_only": spatial_corr(image, lag),
        })

    lags = pd.DataFrame(lag_rows)

    Rf = channel_corr(full)
    Ri = channel_corr(image)

    summaries.to_csv(OUT / "forecast_calibration.csv", index=False)
    paired.to_csv(OUT / "paired_forecast_ci.csv", index=False)
    lags.to_csv(OUT / "spatial_correlations.csv", index=False)
    pd.DataFrame(Rf).to_csv(OUT / "channel_corr_multimodal.csv", index=False)
    pd.DataFrame(Ri).to_csv(OUT / "channel_corr_image_only.csv", index=False)

    off = ~np.eye(Rf.shape[0], dtype=bool)

    print("\n=== FORECAST + CALIBRATION ===")
    print(summaries.to_string(index=False))

    print("\n=== PAIRED MULTIMODAL - IMAGE ONLY ===")
    print(paired.to_string(index=False))

    print("\n=== SPATIAL RESIDUAL CORRELATION ===")
    print(lags.to_string(index=False))

    print("\n=== CHANNEL DEPENDENCE ===")
    print(
        "multimodal mean |offdiag corr|:",
        float(np.mean(np.abs(Rf[off])))
    )
    print(
        "image_only mean |offdiag corr|:",
        float(np.mean(np.abs(Ri[off])))
    )

    print("\nCUBES:", len(full_ids))


if __name__ == "__main__":
    main()
