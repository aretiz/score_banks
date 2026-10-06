#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from spt.data.earthnet import EarthNetWindowDataset
from spt.models.image import ImageForecaster
from spt.trainers import fit_forecaster
from spt.evaluation import collect_scores, add_global_conformal
from spt.plotting import plot_conditional_fpr
from spt.utils import get_device, seed_all, ensure_dir, num_workers


# ---------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------

class WithValidFraction(Dataset):
    """
    Adds valid_fraction to an existing EarthNetWindowDataset without
    requiring any modification to spt/data/earthnet.py.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        mask = sample["target_mask"]
        sample["valid_fraction"] = torch.as_tensor(
            float(mask.float().mean()),
            dtype=torch.float32,
        )
        return sample


def cube_level_half_split(
    dataset: EarthNetWindowDataset,
    seed: int,
) -> tuple[Subset, Subset]:
    """
    Split EarthNet validation data by cube, not by individual windows.

    This avoids putting different windows from the same cube into both
    calibration and evaluation sets.
    """
    n_cubes = len(dataset.files)

    if n_cubes < 2:
        raise RuntimeError(
            f"Need at least 2 validation cubes, found {n_cubes}."
        )

    rng = np.random.default_rng(seed)
    cube_ids = np.arange(n_cubes)
    rng.shuffle(cube_ids)

    n_cal_cubes = max(1, n_cubes // 2)
    n_cal_cubes = min(n_cal_cubes, n_cubes - 1)

    cal_cubes = cube_ids[:n_cal_cubes]
    eval_cubes = cube_ids[n_cal_cubes:]

    spc = dataset.samples_per_cube

    def window_indices(cubes):
        indices = []
        for cube_idx in cubes:
            start = int(cube_idx) * spc
            indices.extend(range(start, start + spc))
        return indices

    wrapped = WithValidFraction(dataset)

    return (
        Subset(wrapped, window_indices(cal_cubes)),
        Subset(wrapped, window_indices(eval_cubes)),
    )


# ---------------------------------------------------------------------
# Mondrian conformal
# ---------------------------------------------------------------------

def _difficulty_edges(
    values: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Quantile bins learned ONLY from calibration data.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No finite calibration difficulty values.")

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)

    # Keep unique interior boundaries.
    interior = np.unique(edges[1:-1])

    return np.concatenate(
        ([-np.inf], interior, [np.inf])
    )


def _assign_bins(
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    return np.searchsorted(
        edges[1:-1],
        np.asarray(values, dtype=float),
        side="right",
    )


def add_mondrian_conformal(
    calibration_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    score_col: str,
    difficulty_col: str = "mc_mean_nll",
    n_bins: int = 5,
    min_cal_per_bin: int = 20,
) -> pd.DataFrame:
    """
    Mondrian/group-conditional conformal anomaly score.

    Difficulty bins are defined from calibration data using mc_mean_nll,
    i.e. the model's expected self-NLL / intrinsic predictive difficulty.

    Higher returned score = more anomalous.
    """
    cal = calibration_df.copy()
    out = eval_df.copy()

    edges = _difficulty_edges(
        cal[difficulty_col].to_numpy(float),
        n_bins=n_bins,
    )

    cal_bins = _assign_bins(
        cal[difficulty_col].to_numpy(float),
        edges,
    )

    eval_bins = _assign_bins(
        out[difficulty_col].to_numpy(float),
        edges,
    )

    global_ref = np.sort(
        cal[score_col].to_numpy(float)
    )

    pvals = np.ones(len(out), dtype=float)

    for b in np.unique(eval_bins):
        eval_idx = np.where(eval_bins == b)[0]

        ref = cal.loc[
            cal_bins == b,
            score_col,
        ].to_numpy(float)

        # Fall back to global calibration if a Mondrian cell is too small.
        if len(ref) < min_cal_per_bin:
            ref = global_ref
        else:
            ref = np.sort(ref)

        scores = out.iloc[eval_idx][score_col].to_numpy(float)

        # Number of calibration scores >= each evaluation score.
        left = np.searchsorted(
            ref,
            scores,
            side="left",
        )

        ge = len(ref) - left

        pvals[eval_idx] = (
            1.0 + ge
        ) / (
            len(ref) + 1.0
        )

    name = f"mondrian_{score_col}"

    out[name] = -np.log(
        np.clip(pvals, 1e-12, 1.0)
    )

    out[f"{name}_p"] = pvals
    out[f"{name}_bin"] = eval_bins

    return out


# ---------------------------------------------------------------------
# FPR evaluation
# ---------------------------------------------------------------------

def conditional_fpr_table_fixed(
    eval_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    score_cols: list[str],
    difficulty_col: str,
    alpha: float = 0.05,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Conditional FPR under ONE threshold per score.

    Raw/non-p-value scores:
        threshold estimated from calibration data.
        Strict '>' avoids the zero-score tie pathology.

    SPT/conformal scores:
        score = -log(p), so use the theoretical p <= alpha threshold.
    """

    d = eval_df[difficulty_col].to_numpy(float)

    try:
        bins = pd.qcut(
            d,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        bins = np.zeros(len(eval_df), dtype=int)

    tmp = eval_df.copy()
    tmp["difficulty_bin"] = np.asarray(bins)

    # These are all -log(p) scores.
    p_value_scores = {
        "spt_upper",
        "spt_two_sided",
        "global_conformal_raw_nll",
        "global_conformal_self_z",
        "global_conformal_spt_upper",
        "mondrian_raw_nll",
        "mondrian_self_z",
        "mondrian_spt_upper",
    }

    records = []

    for score in score_cols:
        if score not in tmp.columns:
            print(f"WARNING: skipping missing score column: {score}")
            continue

        if score in p_value_scores:
            threshold = float(-np.log(alpha))
            use_ge = True
        else:
            values = calibration_df[score].to_numpy(float)
            values = values[np.isfinite(values)]

            threshold = float(
                np.quantile(
                    values,
                    1.0 - alpha,
                    method="higher",
                )
            )

            # Important for EarthNet:
            # zero NLL from an all-masked target must NOT trigger when
            # threshold also happens to equal zero.
            use_ge = False

        for b, group in tmp.groupby(
            "difficulty_bin",
            dropna=False,
        ):
            vals = group[score].to_numpy(float)

            if use_ge:
                alarms = vals >= threshold
            else:
                alarms = vals > threshold

            records.append(
                {
                    "score": score,
                    "difficulty_col": difficulty_col,
                    "difficulty_bin": (
                        int(b) if pd.notna(b) else -1
                    ),
                    "n": int(len(group)),
                    "difficulty_mean": float(
                        group[difficulty_col].mean()
                    ),
                    "threshold": threshold,
                    "fpr": float(np.mean(alarms)),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------

def clean_scores(
    df: pd.DataFrame,
    min_valid: float,
    name: str,
) -> pd.DataFrame:
    """
    Remove observations where essentially no target pixels are usable,
    plus any numerical failures.
    """

    before = len(df)

    if "valid_fraction" not in df.columns:
        raise RuntimeError(
            "valid_fraction missing from collected scores."
        )

    keep = (
        df["valid_fraction"].to_numpy(float)
        >= min_valid
    )

    core_cols = [
        "raw_nll",
        "residual_z",
        "excess_nll",
        "self_z",
        "spt_upper",
        "mc_mean_nll",
        "mc_std_nll",
    ]

    for col in core_cols:
        keep &= np.isfinite(
            df[col].to_numpy(float)
        )

    out = df.loc[keep].reset_index(drop=True)

    print(
        f"{name}: retained {len(out)}/{before} "
        f"({100.0 * len(out) / max(before, 1):.1f}%) "
        f"with valid_fraction >= {min_valid:.3f}"
    )

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


class ZeroExoDataset:
    """Same samples/splits, but remove weather/static information."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x = self.base[idx]
        x = dict(x)
        x["exo"] = torch.zeros_like(x["exo"])
        return x


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        required=True,
        help="Folder containing EarthNet2021 training .npz cubes",
    )

    ap.add_argument(
        "--test-data",
        default=None,
        help="Optional separate EarthNet .npz folder for final evaluation",
    )

    ap.add_argument(
        "--out",
        default="outputs/earthnet",
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--history",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--hidden",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--samples-per-cube",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--n-mc",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--difficulty-bins",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--mondrian-bins",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--min-valid",
        type=float,
        default=0.05,
        help="Minimum fraction of valid target pixels",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--device",
        default="auto",
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=num_workers(),
    )

    ap.add_argument(
        "--eval-only",
        action="store_true",
    )
    ap.add_argument(
        "--zero-exo",
        action="store_true",
        help="Image-only ablation: replace weather/static exogenous inputs with zeros.",
    )

    args = ap.parse_args()

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    seed_all(args.seed)
    device = get_device(args.device)
    out = ensure_dir(args.out)

    print(f"Device: {device}")

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------

    train_base = EarthNetWindowDataset(
        args.data,
        split="train",
        history=args.history,
        size=args.size,
        samples_per_cube=args.samples_per_cube,
        seed=args.seed,
    )

    val_base = EarthNetWindowDataset(
        args.data,
        split="val",
        history=args.history,
        size=args.size,
        samples_per_cube=args.samples_per_cube,
        seed=args.seed,
    )

    # Important: calibration/evaluation split at CUBE level.
    cal_ds, eval_ds = cube_level_half_split(
        val_base,
        seed=args.seed,
    )

    if args.test_data:
        test_base = EarthNetWindowDataset(
            args.test_data,
            split="test",
            history=args.history,
            size=args.size,
            samples_per_cube=args.samples_per_cube,
            seed=args.seed,
        )

        eval_ds = WithValidFraction(test_base)

    # Training does not need valid_fraction.
    train_ds = train_base

    # Controlled multimodal ablation: preserve architecture, splits,
    # parameter count, and training budget; remove only exogenous information.
    if args.zero_exo:
        train_ds = ZeroExoDataset(train_ds)
        cal_ds = ZeroExoDataset(cal_ds)
        eval_ds = ZeroExoDataset(eval_ds)
        print("Ablation: IMAGE ONLY (weather/static inputs zeroed)")
    else:
        print("Condition: MULTIMODAL (imagery + weather + static)")

    sample = train_ds[0]
    exo_dim = int(sample["exo"].numel())

    print(f"Train windows: {len(train_ds)}")
    print(f"Calibration windows: {len(cal_ds)}")
    print(f"Evaluation windows: {len(eval_ds)}")
    print(f"Exogenous dimension: {exo_dim}")

    # ------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    cal_loader = DataLoader(
        cal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = ImageForecaster(
        in_channels=4,
        hidden=args.hidden,
        exo_dim=exo_dim,
    )

    ckpt = Path(out) / "model.pt"

    if args.eval_only:
        if not ckpt.exists():
            raise FileNotFoundError(
                f"--eval-only requested but checkpoint missing: {ckpt}"
            )

        payload = torch.load(
            ckpt,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            payload["model"]
        )

        model.to(device)

        print(f"Loaded checkpoint: {ckpt}")

    else:
        fit_forecaster(
            model,
            train_loader,
            cal_loader,
            device,
            epochs=args.epochs,
            checkpoint=ckpt,
            snapshot_epochs=(1, 2, 4, 8, 12, 16, 20, 25),
            snapshot_dir=Path(out) / "checkpoints",
        )

    # ------------------------------------------------------------
    # Score calibration + evaluation data
    # ------------------------------------------------------------

    extra_keys = (
        "label",
        "difficulty",
        "valid_fraction",
    )

    cal_df = collect_scores(
        model,
        cal_loader,
        device,
        n_mc=args.n_mc,
        mc_chunk=4,
        extra_keys=extra_keys,
    )

    eval_df = collect_scores(
        model,
        eval_loader,
        device,
        n_mc=args.n_mc,
        mc_chunk=4,
        extra_keys=extra_keys,
    )

    # ------------------------------------------------------------
    # Remove essentially unobserved targets BEFORE calibration.
    # ------------------------------------------------------------

    cal_df = clean_scores(
        cal_df,
        min_valid=args.min_valid,
        name="Calibration",
    )

    eval_df = clean_scores(
        eval_df,
        min_valid=args.min_valid,
        name="Evaluation",
    )

    if len(cal_df) < 100:
        print(
            "WARNING: fewer than 100 valid calibration examples. "
            "Conformal estimates will be coarse."
        )

    # Model-derived predictive difficulty.
    #
    # mc_mean_nll = expected self-NLL under p_theta(x|c),
    # which is our Monte Carlo conditional-entropy surrogate.
    cal_df["predicted_difficulty"] = cal_df["mc_mean_nll"]
    eval_df["predicted_difficulty"] = eval_df["mc_mean_nll"]

    # ------------------------------------------------------------
    # Global conformal baselines
    # ------------------------------------------------------------

    # Raw NLL conformal.
    eval_df = add_global_conformal(
        cal_df,
        eval_df,
        score_col="raw_nll",
    )

    # Normalized conformal baseline:
    # self_z already removes model-estimated mean/std of self-NLL.
    eval_df = add_global_conformal(
        cal_df,
        eval_df,
        score_col="self_z",
    )

    # Conformalized SPT:
    # repairs systematic misspecification of the model-based rank.
    eval_df = add_global_conformal(
        cal_df,
        eval_df,
        score_col="spt_upper",
    )

    # ------------------------------------------------------------
    # Mondrian conformal baselines
    # ------------------------------------------------------------

    # IMPORTANT:
    # We bin by mc_mean_nll, which is available from the predictive
    # distribution before seeing the realized target.
    for score_col in (
        "raw_nll",
        "self_z",
        "spt_upper",
    ):
        eval_df = add_mondrian_conformal(
            calibration_df=cal_df,
            eval_df=eval_df,
            score_col=score_col,
            difficulty_col="predicted_difficulty",
            n_bins=args.mondrian_bins,
        )

    # ------------------------------------------------------------
    # Save scores
    # ------------------------------------------------------------

    cal_df.to_csv(
        Path(out) / "calibration_scores.csv",
        index=False,
    )

    eval_df.to_csv(
        Path(out) / "scores_fixed.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Main paper comparison
    # ------------------------------------------------------------

    score_cols = [
        # Raw density
        "raw_nll",

        # Simple model normalization
        "residual_z",
        "self_z",

        # Marginal conformal
        "global_conformal_raw_nll",

        # Normalized conformal
        "global_conformal_self_z",

        # Pure model-based SPT
        "spt_upper",

        # Hybrid: SPT + conformal correction
        "global_conformal_spt_upper",

        # Reviewer-2 comparator
        "mondrian_raw_nll",
        "mondrian_self_z",

        # Mondrian-conformalized SPT
        "mondrian_spt_upper",
    ]

    # ------------------------------------------------------------
    # Primary FPR plot:
    # model-predicted intrinsic difficulty.
    # ------------------------------------------------------------

    fpr_pred = conditional_fpr_table_fixed(
        eval_df=eval_df,
        calibration_df=cal_df,
        score_cols=score_cols,
        difficulty_col="predicted_difficulty",
        alpha=args.alpha,
        n_bins=args.difficulty_bins,
    )

    fpr_pred.to_csv(
        Path(out) / "conditional_fpr_predicted_difficulty.csv",
        index=False,
    )

    # Keep original filename too, so downstream scripts still work.
    fpr_pred.to_csv(
        Path(out) / "conditional_fpr.csv",
        index=False,
    )

    plot_conditional_fpr(
        fpr_pred,
        Path(out) / "conditional_fpr.png",
        alpha=args.alpha,
    )

    # ------------------------------------------------------------
    # Secondary diagnostic:
    # original EarthNet cloud+motion difficulty.
    # ------------------------------------------------------------

    fpr_observed = conditional_fpr_table_fixed(
        eval_df=eval_df,
        calibration_df=cal_df,
        score_cols=score_cols,
        difficulty_col="difficulty",
        alpha=args.alpha,
        n_bins=args.difficulty_bins,
    )

    fpr_observed.to_csv(
        Path(out) / "conditional_fpr_cloud_motion.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Summary metrics: deviation from nominal alpha
    # ------------------------------------------------------------

    summary = (
        fpr_pred
        .assign(
            abs_fpr_error=lambda x:
                np.abs(x["fpr"] - args.alpha)
        )
        .groupby("score", as_index=False)
        .agg(
            mean_fpr=("fpr", "mean"),
            mean_abs_fpr_error=("abs_fpr_error", "mean"),
            max_abs_fpr_error=("abs_fpr_error", "max"),
            min_fpr=("fpr", "min"),
            max_fpr=("fpr", "max"),
        )
        .sort_values(
            "mean_abs_fpr_error",
            ascending=True,
        )
    )

    summary.to_csv(
        Path(out) / "calibration_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print("\n=== Conditional FPR: predicted difficulty ===")
    print(
        fpr_pred.to_string(index=False)
    )

    print("\n=== Calibration summary ===")
    print(
        summary.to_string(index=False)
    )

    print("\nFiles written:")
    print(
        Path(out).resolve()
        / "scores_fixed.csv"
    )
    print(
        Path(out).resolve()
        / "conditional_fpr.csv"
    )
    print(
        Path(out).resolve()
        / "conditional_fpr_cloud_motion.csv"
    )
    print(
        Path(out).resolve()
        / "calibration_summary.csv"
    )
    print(
        Path(out).resolve()
        / "conditional_fpr.png"
    )


if __name__ == "__main__":
    main()