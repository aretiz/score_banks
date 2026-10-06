#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from export_components_template import component_rows, write_component_table

from spt.caps_earthnet import (
    MaskedLocalCalibrator,
    all_raw_scores,
    final_pvalues,
    fit_component_normalizer,
    fit_final_calibrators,
    pixel_standardized_energy,
)

PRIMARY_METHODS = [
    "caps_z",
    "global_nll",
    "standardized_residual",
    "raw_residual",
    "z_global",
    "z_hc",
    "z_bj",
    "z_scan",
]


def to_nchw(a: np.ndarray, layout: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        return a[:, None, :, :]
    if a.ndim != 4:
        raise ValueError(f"Expected 3D/4D array, got shape {a.shape}")
    if layout == "nchw":
        return a
    if layout == "nhwc":
        return np.transpose(a, (0, 3, 1, 2))
    raise ValueError(layout)


def load_cache(path: str, layout: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    keys = set(d.files)
    if "z" in keys:
        z = to_nchw(d["z"], layout).astype(np.float32)
        if "log_sigma" in keys:
            log_sigma = to_nchw(d["log_sigma"], layout).astype(np.float32)
        elif "sigma" in keys:
            sigma = to_nchw(d["sigma"], layout).astype(np.float32)
            log_sigma = np.log(np.maximum(sigma, 1e-8)).astype(np.float32)
        else:
            raise ValueError(f"{path}: cache with z must also contain log_sigma or sigma")
    else:
        needed = {"y", "mu", "sigma"}
        if not needed.issubset(keys):
            raise ValueError(
                f"{path}: need either [z, log_sigma/sigma] or [y, mu, sigma]. Found {sorted(keys)}"
            )
        y = to_nchw(d["y"], layout).astype(np.float32)
        mu = to_nchw(d["mu"], layout).astype(np.float32)
        sigma = to_nchw(d["sigma"], layout).astype(np.float32)
        z = ((y - mu) / np.maximum(sigma, 1e-8)).astype(np.float32)
        log_sigma = np.log(np.maximum(sigma, 1e-8)).astype(np.float32)

    if z.shape != log_sigma.shape:
        raise ValueError(f"z {z.shape} and log_sigma {log_sigma.shape} differ")
    N, C, H, W = z.shape

    if "mask" in keys:
        m = np.asarray(d["mask"])
        if m.ndim == 3:
            m = m[:, None, :, :]
        elif m.ndim == 4 and layout == "nhwc":
            m = np.transpose(m, (0, 3, 1, 2))
        if m.shape[0] != N or m.shape[-2:] != (H, W):
            raise ValueError(f"mask shape {m.shape} incompatible with z {z.shape}")
        if m.shape[1] == 1 and C > 1:
            m = np.broadcast_to(m, (N, C, H, W))
        elif m.shape[1] != C:
            raise ValueError(f"mask channels {m.shape[1]} incompatible with z channels {C}")
        valid = m.astype(bool)
    else:
        valid = np.ones_like(z, dtype=bool)

    valid &= np.isfinite(z) & np.isfinite(log_sigma)
    z = np.where(valid, z, 0.0).astype(np.float32)
    log_sigma = np.where(valid, log_sigma, 0.0).astype(np.float32)

    contexts: Dict[str, np.ndarray] = {}
    for k in d.files:
        if k.startswith("context_"):
            a = np.asarray(d[k])
            if a.ndim in (1, 2) and len(a) == N:
                contexts[k[len("context_"):]] = a.astype(float)
    for k in ("cloud", "cloud_fraction", "motion", "motion_score"):
        if k in keys:
            a = np.asarray(d[k])
            if a.ndim == 1 and len(a) == N:
                contexts[k] = a.astype(float)
    contexts["valid_fraction"] = valid.reshape(N, -1).mean(axis=1)

    out = {"z": z, "log_sigma": log_sigma, "valid": valid, "contexts": contexts}
    if "cube_id" in keys:
        out["cube_id"] = np.asarray(d["cube_id"])
    return out


def subset(data: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    out = {
        "z": data["z"][idx],
        "log_sigma": data["log_sigma"][idx],
        "valid": data["valid"][idx],
        "contexts": {k: v[idx] for k, v in data["contexts"].items()},
    }
    if "cube_id" in data:
        out["cube_id"] = data["cube_id"][idx]
    return out


def filter_valid_fraction(data: Dict[str, np.ndarray], minimum: float) -> Dict[str, np.ndarray]:
    vf = data["contexts"]["valid_fraction"]
    idx = np.flatnonzero(vf >= minimum)
    return subset(data, idx)


def split_calibration(data: Dict[str, np.ndarray], seed: int) -> Tuple[Dict, Dict, Dict]:
    n = len(data["z"])
    if n < 300:
        raise ValueError(f"Need at least 300 calibration scenes after filtering, got {n}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = n // 3
    b = 2 * n // 3
    return subset(data, idx[:a]), subset(data, idx[a:b]), subset(data, idx[b:])


def fit_pipeline(local_d: Dict, component_d: Dict, scene_d: Dict, windows: List[int], min_ref: int):
    pix, vpix = pixel_standardized_energy(local_d["z"], local_d["valid"])
    local_cal = MaskedLocalCalibrator(min_position_ref=min_ref).fit(
        pix.reshape(len(pix), -1), vpix.reshape(len(vpix), -1)
    )
    comp_norm = fit_component_normalizer(component_d["z"], component_d["valid"], local_cal, windows)
    scene_scores = all_raw_scores(
        scene_d["z"], scene_d["log_sigma"], scene_d["valid"], local_cal, comp_norm, windows
    )
    final_cal = fit_final_calibrators(scene_scores)
    return local_cal, comp_norm, final_cal


def evaluate_scores(data: Dict, local_cal, comp_norm, final_cal, windows: List[int]):
    raw = all_raw_scores(data["z"], data["log_sigma"], data["valid"], local_cal, comp_norm, windows)
    p = final_pvalues(raw, final_cal)
    return raw, p


def context_table(pvals: Dict[str, np.ndarray], contexts: Dict[str, np.ndarray], alpha: float) -> pd.DataFrame:
    rows = []
    n = len(next(iter(pvals.values())))
    for cname, x in contexts.items():
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            continue
        finite = np.isfinite(x)
        if finite.sum() < 50 or np.unique(x[finite]).size < 2:
            continue
        try:
            bins = pd.qcut(pd.Series(x), 5, labels=False, duplicates="drop").to_numpy()
        except Exception:
            continue
        for method, pv in pvals.items():
            for b in sorted(pd.unique(bins[np.isfinite(bins)])):
                m = (bins == b) & finite
                if m.sum() == 0:
                    continue
                rows.append({
                    "context": cname,
                    "bin": int(b),
                    "method": method,
                    "n": int(m.sum()),
                    "mean_context": float(np.mean(x[m])),
                    "alarm_rate": float(np.mean(pv[m] <= alpha)),
                })
    return pd.DataFrame(rows)


def choose_pixels(valid_pix: np.ndarray, structure: str, amount: float, rng: np.random.Generator) -> np.ndarray:
    H, W = valid_pix.shape
    chosen = np.zeros((H, W), dtype=bool)
    valid_idx = np.flatnonzero(valid_pix.ravel())
    if valid_idx.size == 0:
        return chosen
    if structure == "sparse":
        k = min(int(amount), valid_idx.size)
        if k > 0:
            sel = rng.choice(valid_idx, size=k, replace=False)
            chosen.ravel()[sel] = True
    elif structure == "dense":
        k = max(1, min(valid_idx.size, int(np.ceil(float(amount) * valid_idx.size))))
        sel = rng.choice(valid_idx, size=k, replace=False)
        chosen.ravel()[sel] = True
    elif structure == "contiguous":
        side = min(int(amount), H, W)
        r0 = int(rng.integers(0, H - side + 1))
        c0 = int(rng.integers(0, W - side + 1))
        chosen[r0:r0+side, c0:c0+side] = True
        chosen &= valid_pix
    else:
        raise ValueError(structure)
    return chosen


def inject_z(data: Dict, structure: str, amount: float, delta: float, seed: int) -> Tuple[Dict, float]:
    z = data["z"].copy()
    valid = data["valid"]
    valid_pix = valid.any(axis=1)
    rng = np.random.default_rng(seed)
    affected = []
    for i in range(len(z)):
        chosen = choose_pixels(valid_pix[i], structure, amount, rng)
        affected.append(chosen.sum())
        # Shift all valid channels at selected pixels by +delta in standardized units.
        m = valid[i] & chosen[None, :, :]
        z[i][m] += float(delta)
    out = {
        "z": z,
        "log_sigma": data["log_sigma"],
        "valid": valid,
        "contexts": data["contexts"],
    }
    return out, float(np.mean(affected))


def simultaneous_paired_bootstrap(
    scenario_rejections: Dict[str, Dict[str, np.ndarray]],
    comparisons: List[Tuple[str, str]],
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenario_names = list(scenario_rejections)
    n = len(next(iter(next(iter(scenario_rejections.values())).values())))
    rows = []
    for a, b in comparisons:
        D = np.stack([
            scenario_rejections[s][a].astype(float) - scenario_rejections[s][b].astype(float)
            for s in scenario_names
        ], axis=0)  # [S,N]
        est = D.mean(axis=1)
        maxdev = np.empty(n_boot, dtype=float)
        for ib in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot = D[:, idx].mean(axis=1)
            maxdev[ib] = np.max(np.abs(boot - est))
        q = float(np.quantile(maxdev, level))
        for s, e in zip(scenario_names, est):
            meta = json.loads(s)
            rows.append({
                **meta,
                "method_a": a,
                "method_b": b,
                "paired_power_diff": float(e),
                "ci_low": float(max(-1.0, e - q)),
                "ci_high": float(min(1.0, e + q)),
                "simultaneous_level": level,
                "bootstrap_reps": n_boot,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Frozen CAPS-Z evaluation on EarthNet prediction caches")
    ap.add_argument("--cal-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--out", default="outputs/caps_z_earthnet")
    ap.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-valid-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--min-position-ref", type=int, default=30)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--deltas", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--sparse-k", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--contiguous-sides", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--dense-frac", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.0])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--max-test-scenes", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cal = filter_valid_fraction(load_cache(args.cal_cache, args.layout), args.min_valid_fraction)
    test = filter_valid_fraction(load_cache(args.test_cache, args.layout), args.min_valid_fraction)
    if args.max_test_scenes and len(test["z"]) > args.max_test_scenes:
        rng = np.random.default_rng(args.seed + 17)
        idx = np.sort(rng.choice(len(test["z"]), args.max_test_scenes, replace=False))
        test = subset(test, idx)

    local_d, component_d, scene_d = split_calibration(cal, args.seed)
    print(
        f"retained calibration={len(cal['z'])} -> local/component/scene="
        f"{len(local_d['z'])}/{len(component_d['z'])}/{len(scene_d['z'])}; test={len(test['z'])}"
    )
    print(f"shape NCHW={test['z'].shape}; windows={args.windows}")

    local_cal, comp_norm, final_cal = fit_pipeline(
        local_d, component_d, scene_d, args.windows, args.min_position_ref
    )
    _, nominal_p = evaluate_scores(test, local_cal, comp_norm, final_cal, args.windows)

    nominal_rows = []
    for method in PRIMARY_METHODS:
        pv = nominal_p[method]
        nominal_rows.append({
            "method": method,
            "n": len(pv),
            "alpha": args.alpha,
            "nominal_alarm_rate": float(np.mean(pv <= args.alpha)),
            "abs_error_from_alpha": float(abs(np.mean(pv <= args.alpha) - args.alpha)),
        })
    nominal_df = pd.DataFrame(nominal_rows).sort_values("abs_error_from_alpha")
    nominal_df.to_csv(out / "nominal_validity.csv", index=False)
    context_df = context_table({k: nominal_p[k] for k in PRIMARY_METHODS}, test["contexts"], args.alpha)
    context_df.to_csv(out / "context_validity.csv", index=False)

    scenarios = []
    for k in args.sparse_k:
        for d in args.deltas:
            scenarios.append(("sparse", float(k), float(d)))
    for s in args.contiguous_sides:
        for d in args.deltas:
            scenarios.append(("contiguous", float(s), float(d)))
    for f in args.dense_frac:
        for d in args.deltas:
            scenarios.append(("dense", float(f), float(d)))

    power_rows = []
    scenario_rejections: Dict[str, Dict[str, np.ndarray]] = {}
    for si, (structure, amount, delta) in enumerate(scenarios):
        print(f"[{si+1}/{len(scenarios)}] {structure} amount={amount:g} delta={delta:g}")
        alt, mean_affected = inject_z(test, structure, amount, delta, args.seed + 1000 + si)
        _, alt_p = evaluate_scores(alt, local_cal, comp_norm, final_cal, args.windows)
        meta = {"structure": structure, "amount": amount, "delta": delta}
        skey = json.dumps(meta, sort_keys=True)
        scenario_rejections[skey] = {}
        for method in PRIMARY_METHODS:
            rej = alt_p[method] <= args.alpha
            scenario_rejections[skey][method] = rej
            power_rows.append({
                **meta,
                "mean_affected_pixels": mean_affected,
                "method": method,
                "n": len(rej),
                "alpha": args.alpha,
                "power": float(np.mean(rej)),
                "native_nominal_alarm_rate": float(np.mean(nominal_p[method] <= args.alpha)),
            })

    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(out / "power_surface.csv", index=False)

    ci = simultaneous_paired_bootstrap(
        scenario_rejections,
        comparisons=[
            ("caps_z", "global_nll"),
            ("caps_z", "standardized_residual"),
            ("caps_z", "z_global"),
            ("caps_z", "z_hc"),
            ("caps_z", "z_bj"),
            ("caps_z", "z_scan"),
        ],
        n_boot=args.bootstrap,
        seed=args.seed + 2000,
    )
    ci.to_csv(out / "simultaneous_paired_power_ci.csv", index=False)

    summary_rows = []
    for baseline in ["global_nll", "standardized_residual"]:
        q = ci[(ci.method_a == "caps_z") & (ci.method_b == baseline)].copy()
        q["win"] = q.ci_low > 0
        q["loss"] = q.ci_high < 0
        for structure, s in q.groupby("structure"):
            summary_rows.append({
                "baseline": baseline,
                "structure": structure,
                "n_scenarios": len(s),
                "mean_power_gain": float(s.paired_power_diff.mean()),
                "median_power_gain": float(s.paired_power_diff.median()),
                "significant_wins": int(s.win.sum()),
                "ties": int((~s.win & ~s.loss).sum()),
                "significant_losses": int(s.loss.sum()),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "primary_summary.csv", index=False)

    notes = {
        "method": "Frozen CAPS-Z",
        "guarantee": "Final split-conformal scene calibration gives finite-sample marginal validity under scene exchangeability; no within-scene independence is required.",
        "earthnet_wording": "Because nominal EarthNet scenes are not externally verified event-free, report nominal alarm rate rather than strict FPR unless event-free annotation is available.",
        "transfer": "No arbitrary distribution-shift guarantee is claimed.",
        "local_evidence": "Per-pixel RMS standardized residual across valid channels, externally normalized by a disjoint local-calibration split.",
        "splits": {
            "local_cal": len(local_d["z"]),
            "component_cal": len(component_d["z"]),
            "scene_cal": len(scene_d["z"]),
            "test": len(test["z"]),
        },
        "no_method_tuning": True,
    }
    (out / "notes.json").write_text(json.dumps(notes, indent=2))

    print("\n=== NOMINAL VALIDITY ===")
    print(nominal_df.to_string(index=False))
    print("\n=== PRIMARY POWER SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
