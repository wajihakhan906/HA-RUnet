"""Draws the HA-RUnet block diagram to ../Figures/architecture.png."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C = {"conv": "#E8A0A8", "res": "#4C9A8A", "att": "#FFFFFF", "se": "#9BD39B", "io": "#8172B2"}
W, H = 1.5, 0.55


def box(ax, x, y, text, color, w=W, ec="black", tc="white", ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - H / 2), w, H, boxstyle="round,pad=0.03", fc=color, ec=ec, ls=ls))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=7.5, weight="bold")


def arrow(ax, a, b, **kw):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", lw=1.1, **kw))


def main():
    sizes, widths = [128, 64, 32, 16, 8, 4], [32, 64, 128, 256, 512, 512]
    fig, ax = plt.subplots(figsize=(13, 7))
    ys = [6 - i for i in range(6)]
    ex = [1.2 + 0.55 * i for i in range(6)]
    dx = [12.2 - 0.55 * i for i in range(5)]

    box(ax, ex[0], 7.1, "Input 4 × 128³\nFLAIR · T1 · T2 · T1ce", C["io"], w=2.0)
    arrow(ax, (ex[0], 6.8), (ex[0], ys[0] + H / 2))
    for i in range(6):
        label = f"{sizes[i]}³ × {widths[i]}"
        if i == 0:
            box(ax, ex[i], ys[i], "3D conv\n" + label, C["conv"], tc="black")
        else:
            box(ax, ex[i], ys[i], ("Res ×2\n" if i == 5 else "Res\n") + label, C["res"])
            arrow(ax, (ex[i - 1], ys[i - 1] - H / 2), (ex[i], ys[i] + H / 2), color="#C8A000")
    box(ax, ex[5] + 1.5, ys[5], "SE", C["se"], w=0.6, tc="black")
    arrow(ax, (ex[5] + W / 2, ys[5]), (ex[5] + 1.2, ys[5]))

    for i in range(5):
        y = ys[i]
        if i == 0:
            ax.plot([ex[0] + W / 2, dx[0] - W / 2 - 0.1], [y, y], "k-", lw=1)
            ax.text(6.7, y + 0.12, "skip connection", ha="center", fontsize=7, style="italic")
        else:
            box(ax, 6.7, y, f"Attention Module {i}", C["att"], w=2.2, tc="black", ls="--")
            ax.plot([ex[i] + W / 2, 6.7 - 1.1], [y, y], "k-", lw=1)
            ax.plot([6.7 + 1.1, dx[i] - W / 2 - 0.1], [y, y], "k-", lw=1)
        label = f"{sizes[i]}³ × {widths[i]}"
        box(ax, dx[i], y, ("concat + conv\n" if i == 0 else "concat + Res\n") + label, C["conv"] if i == 0 else C["res"],
            tc="black" if i == 0 else "white")
        if i > 0:
            box(ax, dx[i] + 1.15, y, "SE", C["se"], w=0.55, tc="black")
            arrow(ax, (dx[i] + W / 2, y), (dx[i] + 0.87, y))
        src = (ex[5] + 1.8, ys[5]) if i == 4 else (dx[i + 1] + 1.15, ys[i + 1] + H / 2)
        arrow(ax, src, (dx[i], y - H / 2), color="#8B5A2B")

    box(ax, dx[0], 7.1, "Output 128³ × classes\nWT / TC / ET", C["io"], w=2.0)
    arrow(ax, (dx[0], ys[0] + H / 2), (dx[0], 6.8))

    handles = [plt.Rectangle((0, 0), 1, 1, fc=C[k], ec="black", ls="--" if k == "att" else "-") for k in ("conv", "res", "att", "se")]
    ax.legend(handles, ["3D (3×3×3) convolution block", "Residual block (3 pre-activation units)",
                        "Attention module (trunk + soft-mask branch)", "Squeeze-Excitation"],
              loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.45, -0.02))
    ax.text(3.2, 0.45, "yellow: 2×2×2 max-pool   brown: 2×2×2 up-sampling", fontsize=7.5, color="#555555")
    ax.set_xlim(0, 13.8); ax.set_ylim(-0.3, 7.7); ax.axis("off")
    ax.set_title("HA-RUnet: residual encoder–decoder with attention-module skips and SE-recalibrated decoder", fontsize=11)
    fig.tight_layout()
    fig.savefig("../Figures/architecture.png", dpi=200)


if __name__ == "__main__":
    main()
