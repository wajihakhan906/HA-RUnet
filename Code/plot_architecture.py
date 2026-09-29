"""Draws the HA-RUnet block diagram to ../Figures/architecture.png."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLORS = {"res": "#4C72B0", "up": "#55A868", "gate": "#C44E52", "io": "#8172B2"}
W, H = 1.6, 0.6


def box(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.03", fc=color, ec="black"))
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=8, weight="bold")


def arrow(ax, a, b, **kw):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", lw=1.2, **kw))


def main(base=16, depth=4):
    widths = [base * 2**i for i in range(depth + 1)]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    level_y = [5 - i for i in range(depth + 1)]
    enc_x = [1 + i * 0.7 for i in range(depth + 1)]
    dec_x = [11 - i * 0.7 for i in range(depth)]

    box(ax, 1, 6.2, "Input\n4 × 128³", COLORS["io"])
    arrow(ax, (1, 5.9), (1, 5.3))
    for i, w in enumerate(widths):
        box(ax, enc_x[i], level_y[i], f"Res-SE\n{w} ch" + ("\n(bottleneck)" if i == depth else ""), COLORS["res"])
        if i < depth:
            arrow(ax, (enc_x[i], level_y[i] - H / 2), (enc_x[i + 1], level_y[i + 1] + H / 2))

    for i in range(depth):
        y = level_y[i]
        gx = 6
        box(ax, gx, y, "Attention\nGate", COLORS["gate"])
        box(ax, dec_x[i], y, f"Up + Res-SE\n{widths[i]} ch", COLORS["up"])
        arrow(ax, (enc_x[i] + W / 2, y), (gx - W / 2, y), ls="--")
        arrow(ax, (gx + W / 2, y), (dec_x[i] - W / 2, y), ls="--")
        if i == depth - 1:
            arrow(ax, (enc_x[depth] + W / 2, level_y[depth]), (dec_x[i], y - H / 2))
        else:
            arrow(ax, (dec_x[i + 1], level_y[i + 1] + H / 2), (dec_x[i], y - H / 2))

    box(ax, 11, 6.2, "1×1×1 Conv\nWT / TC / ET", COLORS["io"])
    arrow(ax, (11, 5.3), (11, 5.9))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 6.8); ax.axis("off")
    ax.set_title("HA-RUnet: residual SE encoder · attention-gated skip connections · residual SE decoder", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in COLORS.values()]
    ax.legend(handles, ["Residual block + Squeeze-Excitation", "Transposed conv + residual SE", "Additive attention gate", "Input / output"],
              loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig("../Figures/architecture.png", dpi=200)


if __name__ == "__main__":
    main()
