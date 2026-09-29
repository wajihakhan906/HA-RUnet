import argparse
import json

import torch
from torch.utils.data import DataLoader

from dataset import BraTSDataset
from losses import dice_per_region, sensitivity_per_region
from model import HARUNet


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained HA-RUnet checkpoint")
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--crop", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--widths", type=int, nargs=5, default=[32, 64, 128, 256, 512])
    ap.add_argument("--no-attention", action="store_true", help="ablation: residual U-Net without attention modules")
    ap.add_argument("--no-se", action="store_true", help="ablation: without squeeze-excitation")
    ap.add_argument("--out", default="../Results/metrics.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HARUNet(widths=tuple(args.widths), attention=not args.no_attention, se=not args.no_se).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dl = DataLoader(BraTSDataset(args.data, crop=args.crop), batch_size=1)
    dice, sens = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device))
            dice.append(dice_per_region(logits, y.to(device)))
            sens.append(sensitivity_per_region(logits, y.to(device)))

    names = ["WT", "TC", "ET"]
    metrics = {f"dice_{n}": sum(d[i] for d in dice) / len(dice) for i, n in enumerate(names)}
    metrics |= {f"sensitivity_{n}": sum(s[i] for s in sens) / len(sens) for i, n in enumerate(names)}
    print(json.dumps(metrics, indent=2))
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
