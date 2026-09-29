import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BraTSDataset
from losses import DiceBCELoss, dice_per_region
from model import HARUNet, count_parameters


def split_cases(root, val_fraction, seed):
    cases = sorted(p for p in Path(root).iterdir() if p.is_dir())
    random.Random(seed).shuffle(cases)
    n_val = int(len(cases) * val_fraction)
    return cases[n_val:], cases[:n_val]


def main():
    ap = argparse.ArgumentParser(description="Train HA-RUnet on BraTS-2020")
    ap.add_argument("--data", required=True, help="path to MICCAI_BraTS2020_TrainingData")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--widths", type=int, nargs=5, default=[32, 64, 128, 256, 512])
    ap.add_argument("--no-attention", action="store_true", help="ablation: residual U-Net without attention modules")
    ap.add_argument("--no-se", action="store_true", help="ablation: without squeeze-excitation")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="../Results/run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_cases, val_cases = split_cases(args.data, args.val_fraction, args.seed)
    train_dl = DataLoader(BraTSDataset(args.data, train_cases, args.crop, augment=True),
                          batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(BraTSDataset(args.data, val_cases, args.crop), batch_size=1, num_workers=2)

    model = HARUNet(widths=tuple(args.widths), attention=not args.no_attention, se=not args.no_se).to(device)
    print(f"HA-RUnet parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = DiceBCELoss()
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()

        model.eval()
        scores = []
        with torch.no_grad():
            for x, y in val_dl:
                scores.append(dice_per_region(model(x.to(device)), y.to(device)))
        wt, tc, et = (sum(s[i] for s in scores) / len(scores) for i in range(3))
        mean = (wt + tc + et) / 3
        history.append({"epoch": epoch, "loss": running / len(train_dl), "dice_wt": wt, "dice_tc": tc, "dice_et": et})
        print(f"epoch {epoch:3d} | loss {running / len(train_dl):.4f} | Dice WT {wt:.3f} TC {tc:.3f} ET {et:.3f}")
        if mean > best:
            best = mean
            torch.save(model.state_dict(), out / "best_model.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
