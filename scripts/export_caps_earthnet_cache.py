from pathlib import Path
import gc
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from spt.data.earthnet import EarthNetWindowDataset
from spt.models.image import ImageForecaster
from run_earthnet import cube_level_half_split


DATA = "data/earthnet/train"
CKPT = "outputs/earthnet/model.pt"
OUT = Path("outputs/earthnet")

HISTORY = 4
SIZE = 64
SAMPLES_PER_CUBE = 4
HIDDEN = 64
SEED = 42
BATCH_SIZE = 8


def export(ds, path, model, device):
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    zs, logs, masks, difficulties, history_latents, history_motions, cube_ids = [], [], [], [], [], [], []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            bd = {
                k: v.to(device)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }

            # Strictly history-only context: no target/mask/exogenous target-time inputs.
            h_hist = None
            for t in range(bd["history"].shape[1]):
                h_enc = model.encoder(bd["history"][:, t])
                h_hist = model.gru(h_enc, h_hist)

            history_latent = h_hist.mean(dim=(-2, -1))
            if bd["history"].shape[1] > 1:
                history_motion = (
                    bd["history"][:, -1] - bd["history"][:, -2]
                ).abs().mean(dim=(1, 2, 3))
            else:
                history_motion = torch.zeros(
                    len(bd["history"]), device=device
                )

            pred = model.predictive(bd)

            mu = pred.loc
            sigma = (pred.scale * (pred.df / (pred.df - 2.0)).sqrt()).clamp_min(1e-6)
            y = bd["target"]

            z = (y - mu) / sigma
            mask = bd["target_mask"] > 0.5

            zs.append(z.cpu().numpy().astype("float32"))
            logs.append(torch.log(sigma).cpu().numpy().astype("float32"))
            masks.append(mask.cpu().numpy().astype(bool))

            cube_ids.append(np.asarray([
                np.uint64(int(hashlib.sha1(str(x).encode()).hexdigest()[:16], 16))
                for x in batch["filepath"]
            ], dtype=np.uint64))
            difficulties.append(
                bd["difficulty"].cpu().numpy().astype("float32")
            )
            history_latents.append(
                history_latent.cpu().numpy().astype("float32")
            )
            history_motions.append(
                history_motion.cpu().numpy().astype("float32")
            )

            if i % 100 == 0:
                print(f"{path.name}: batch {i}/{len(loader)}")

    np.savez_compressed(
        path,
        z=np.concatenate(zs),
        log_sigma=np.concatenate(logs),
        mask=np.concatenate(masks),
        context_difficulty=np.concatenate(difficulties),
        context_history_latent=np.concatenate(history_latents),
        context_history_motion=np.concatenate(history_motions),
        cube_id=np.concatenate(cube_ids),
    )

    print("WROTE:", path)

    del zs, logs, masks, difficulties, history_latents, history_motions, cube_ids
    gc.collect()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    val = EarthNetWindowDataset(
        DATA,
        split="val",
        history=HISTORY,
        size=SIZE,
        samples_per_cube=SAMPLES_PER_CUBE,
        seed=SEED,
    )

    cal_ds, test_ds = cube_level_half_split(val, seed=SEED)

    sample = val[0]
    exo_dim = int(sample["exo"].numel())

    model = ImageForecaster(
        in_channels=4,
        hidden=HIDDEN,
        exo_dim=exo_dim,
    )

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    OUT.mkdir(parents=True, exist_ok=True)

    print("Calibration:", len(cal_ds))
    print("Test:", len(test_ds))

    export(cal_ds, OUT / "caps_cal.npz", model, device)
    export(test_ds, OUT / "caps_test.npz", model, device)


if __name__ == "__main__":
    main()
