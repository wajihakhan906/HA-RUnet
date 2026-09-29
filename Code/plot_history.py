"""Plots training loss and validation Dice from a train.py history.json."""
import argparse
import json

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="../Results/run/history.json")
    ap.add_argument("--out", default="../Figures/training_curves.png")
    args = ap.parse_args()
    h = json.load(open(args.history))
    ep = [r["epoch"] for r in h]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(ep, [r["loss"] for r in h]); a.set_title("Training loss"); a.set_xlabel("epoch")
    for k, lbl in [("dice_wt", "WT"), ("dice_tc", "TC"), ("dice_et", "ET")]:
        b.plot(ep, [r[k] for r in h], label=lbl)
    b.set_title("Validation Dice"); b.set_xlabel("epoch"); b.legend()
    fig.tight_layout(); fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
