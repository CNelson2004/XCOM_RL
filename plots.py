"""Plot training curves from a history dictionary (or a saved checkpoint) to an image file.

    from plots import plot_history
    plot_history(history, "runs/training_curves.png",
                 baselines={"random": 0.126, "scripted (advance)": 0.166})

    # or from a checkpoint saved with save_checkpoint:
    plot_history("runs/checkpoint_xcom_epoch500.pt", "runs/epoch500_curves.png")
"""
import matplotlib
matplotlib.use("Agg")                      # draw to files only; no window needed in a script
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

# Colours: categorical slots in fixed order, plus quiet chart ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def smooth(values, window):
    """Trailing moving average; the first points average whatever is available so far."""
    v = np.asarray(values, dtype=float)
    if window <= 1 or len(v) == 0:
        return v
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    for i in range(len(v)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)


def _end_label(ax, x, y, text, color):
    """Name a line at its right-hand end, in ink, next to a dot in the line's colour."""
    ax.plot([x], [y], "o", color=color, markersize=5, zorder=4)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)


def plot_history(history, path="runs/training_curves.png", baselines=None, window=10,
                 max_grad_norm=0.5, title=None):
    """history: the dict from ppo_main_xcom, or a path to a .pt checkpoint containing one.
    baselines: optional {"name": win rate} drawn as dashed reference lines on the win-rate panel.
    window: epochs in the moving average (each epoch's win rate is noisy: about +/-9 points
            with 100 battles).
    """
    if isinstance(history, str):
        import torch
        history = torch.load(history)["history"]
    n = len(history.get("win rate", []))
    if n == 0:
        raise ValueError("history has no epochs to plot")
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle(title or f"Training curves ({n} epochs, {window}-epoch moving average)",
                 x=0.06, ha="left", fontsize=13, color=INK)

    # 1. Battle outcomes: the main learning curve
    ax = axes[0, 0]
    _style(ax, "Battle outcomes per epoch")
    ax.plot(x, history["win rate"], color=BLUE, linewidth=1, alpha=0.25)   # raw, faint
    for key, label, color in (("win rate", "win", BLUE), ("loss rate", "loss", ORANGE),
                              ("timeout rate", "timeout", AQUA)):
        if key in history:
            ys = smooth(history[key], window)
            ax.plot(x, ys, color=color, linewidth=2, label=label)
            _end_label(ax, x[-1], ys[-1], f"{label} {ys[-1]:.0%}", color)
    for k, (name, rate) in enumerate(sorted((baselines or {}).items(), key=lambda item: item[1])):
        ax.axhline(rate, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        # label sits just above its own line; labels are staggered left-to-right so close lines don't collide
        ax.annotate(f"{name} {rate:.0%}", (0.03 + 0.3 * (k % 3), rate), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color=INK_2, zorder=1.5,
                    bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none"))
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9,
              labelcolor=INK_2)
    ax.margins(x=0.08)

    # 2. Entropy
    ax = axes[0, 1]
    _style(ax, "Average policy entropy")
    if "avg entropy" in history:
        ax.plot(x, history["avg entropy"], color=BLUE, linewidth=2)
        _end_label(ax, x[-1], history["avg entropy"][-1], f"{history['avg entropy'][-1]:.2f}", BLUE)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08)

    # 3. Gradient norms, with the clipping threshold
    ax = axes[1, 0]
    _style(ax, "Gradient norm before clipping")
    for key, label, color in (("avg grad norm", "average", BLUE), ("max grad norm", "largest", ORANGE)):
        if key in history:
            ax.plot(x, history[key], color=color, linewidth=2, label=label)
            _end_label(ax, x[-1], history[key][-1], label, color)
    ax.axhline(max_grad_norm, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"clip at {max_grad_norm}", (x[-1], max_grad_norm), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK_2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    ax.margins(x=0.08)

    # 4. Illegal actions (should stay at 0)
    ax = axes[1, 1]
    _style(ax, "Illegal actions (should be 0)")
    if "illegals" in history:
        ax.plot(x, history["illegals"], color=BLUE, linewidth=2)
        _end_label(ax, x[-1], history["illegals"][-1], f"{history['illegals'][-1]:.0f}", BLUE)
    ax.set_ylim(bottom=0, top=max(1, max(history.get("illegals", [0])) * 1.2))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.08)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)                          # free the figure's memory
    return path
