"""Plot training curves from a history dictionary (or a saved checkpoint) to an image file.

    from view.plots import plot_history
    plot_history(history, "runs/training_curves.png",
                 baselines={"random": 0.126, "scripted (cautious)": 0.316})

    # or from a checkpoint saved with save_checkpoint:
    plot_history("runs/checkpoint_xcom_epoch500.pt", "runs/epoch500_curves.png")

Panels (all share the epoch axis, so you can read down a column to line events up):
    battle outcomes | average entropy
    value loss      | policy loss
    approximate KL  | gradient norm (both networks together)
    value net norm  | policy net norm (each clipped separately)
A panel whose numbers are not in the history (e.g. an older run) says so instead of failing.
"""
import matplotlib
matplotlib.use("Agg")                      # draw to files only; no window needed in a script
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

# Colours: categorical slots in fixed order, plus quiet chart ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, AXIS, SURFACE, BAND = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                                "#c3c2b7", "#fcfcfb", "#e9f2fc")

# The history keys this file reads (as created in ppo_main_xcom).
KEYS = {
    "win": "win rate", "loss": "loss rate", "timeout": "timeout rate",
    "entropy": "avg entropy", "value_loss": "value_loss", "policy_loss": "policy_loss",
    "kl": "kl", "grad_avg": "avg grad norm", "grad_max": "max grad norm",
    "value_norm": "avg value norm", "policy_norm": "avg policy norm",
}


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
    ax.margins(x=0.08)


def _end_label(ax, x, y, text, color):
    """Name a line at its right-hand end, in ink, next to a dot in the line's colour."""
    ax.plot([x], [y], "o", color=color, markersize=5, zorder=4)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)


def _reference(ax, y, text, x_frac=0.98, ha="right"):
    """A dashed reference line with a small label sitting just above it."""
    ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(text, (x_frac, y), xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", ha=ha, va="bottom", fontsize=8, color=INK_2, zorder=1.5,
                bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none"))


def _missing(ax, what, x):
    ax.text(0.5, 0.5, f"{what} not logged in this run", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_yticks([])
    ax.plot(x, np.zeros(len(x)), alpha=0)    # invisible line: keeps the same epoch axis as the other panels


def _series(history, key, n):
    """The values for a history key, or None if the run did not log them (or logged a different length)."""
    values = history.get(KEYS[key])
    if values is None or len(values) == 0:
        return None
    values = np.asarray(values, dtype=float)
    return values if len(values) == n else None


def _line_panel(ax, history, x, title, key, color=BLUE, label_fmt="{:.3g}", what=None):
    """A single-series panel: one line, labelled at its end with its last value."""
    _style(ax, title)
    ys = _series(history, key, len(x))
    if ys is None:
        _missing(ax, what or title.lower(), x)
        return None
    ax.plot(x, ys, color=color, linewidth=2)
    _end_label(ax, x[-1], ys[-1], label_fmt.format(ys[-1]), color)
    return ys


def plot_history(history, path="runs/training_curves.png", baselines=None, window=10,
                 max_grad_norm=0.5, title=None):
    """history: the dict from ppo_main_xcom, or a path to a .pt checkpoint containing one.
    baselines: optional {"name": win rate} drawn as dashed reference lines on the outcomes panel.
    window: epochs in the win/loss/timeout moving average (each epoch's rates are noisy).
    max_grad_norm: the clipping threshold, drawn on the per-network gradient-norm panels.
    """
    if isinstance(history, str):
        import torch
        history = torch.load(history)["history"]
    n = len(history.get(KEYS["win"], []))
    if n == 0:
        raise ValueError("history has no epochs to plot")
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), facecolor=SURFACE, layout="constrained",
                             gridspec_kw={"height_ratios": [1.5, 1, 1, 1]})
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.15, hspace=0.06, wspace=0.08)
    fig.suptitle(title or f"Training curves ({n} epochs, {window}-epoch moving average on outcomes)",
                 x=0.02, ha="left", fontsize=13, color=INK)

    # Row 1, left: battle outcomes (the main learning curve)
    ax = axes[0, 0]
    _style(ax, "Battle outcomes per epoch")
    raw_win = _series(history, "win", n)
    ax.plot(x, raw_win, color=BLUE, linewidth=1, alpha=0.25)          # raw win rate, faint
    for key, label, color in (("win", "win", BLUE), ("loss", "loss", ORANGE), ("timeout", "timeout", AQUA)):
        ys = _series(history, key, n)
        if ys is not None:
            ys = smooth(ys, window)
            ax.plot(x, ys, color=color, linewidth=2, label=label)
            _end_label(ax, x[-1], ys[-1], f"{label} {ys[-1]:.0%}", color)
    for k, (name, rate) in enumerate(sorted((baselines or {}).items(), key=lambda item: item[1])):
        # labels sit just above their own line, staggered left-to-right so close lines don't collide
        _reference(ax, rate, f"{name} {rate:.0%}", x_frac=0.03 + 0.3 * (k % 3), ha="left")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9, labelcolor=INK_2)

    # Row 1, right: entropy
    ys = _line_panel(axes[0, 1], history, x, "Average policy entropy", "entropy", label_fmt="{:.2f}",
                     what="entropy")
    if ys is not None:
        axes[0, 1].set_ylim(bottom=0)

    # Row 2: value loss and policy loss, on separate panels (very different scales)
    ys = _line_panel(axes[1, 0], history, x, "Value loss", "value_loss", label_fmt="{:.3f}")
    if ys is not None:
        axes[1, 0].set_ylim(bottom=0)
    ys = _line_panel(axes[1, 1], history, x, "Policy loss", "policy_loss", label_fmt="{:.3f}")
    if ys is not None:
        axes[1, 1].axhline(0, color=AXIS, linewidth=1, zorder=1)

    # Row 3, left: approximate KL, with the usual healthy range and a warning line
    ax = axes[2, 0]
    ys = _line_panel(ax, history, x, "Approximate KL per update", "kl", label_fmt="{:.3f}",
                     what="KL")
    if ys is not None:
        ax.axhspan(0.01, 0.02, color=BAND, zorder=0)
        ax.annotate("typical 0.01–0.02", (0.02, 0.02), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points", fontsize=8, color=INK_2, zorder=1.5,
                    bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none"))
        _reference(ax, 0.05, "large steps: 0.05")
        ax.set_ylim(0, max(0.06, float(np.nanmax(ys)) * 1.15))

    # Row 3, right: gradient norms, with the clipping threshold
    ax = axes[2, 1]
    separate = _series(history, "policy_norm", n) is not None      # newer runs clip each network on its own
    _style(ax, "Gradient norm, both networks together" if separate else "Gradient norm before clipping")
    drawn = False
    for key, label, color in (("grad_avg", "average", BLUE), ("grad_max", "largest", ORANGE)):
        ys = _series(history, key, n)
        if ys is not None:
            ax.plot(x, ys, color=color, linewidth=2, label=label)
            _end_label(ax, x[-1], ys[-1], label, color)
            drawn = True
    if drawn:
        if not separate:                                            # one combined clip: draw its threshold
            _reference(ax, max_grad_norm, f"clip at {max_grad_norm}")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    else:
        _missing(ax, "gradient norms", x)

    # Row 4: each network's own gradient norm before its clip (left: value, right: policy,
    # matching the value loss | policy loss row above)
    for col, key, label, color in ((0, "value_norm", "Value network gradient norm", ORANGE),
                                   (1, "policy_norm", "Policy network gradient norm", BLUE)):
        ax = axes[3, col]
        ys = _line_panel(ax, history, x, f"{label} (average)", key, color=color, label_fmt="{:.2f}",
                         what=f"{label.split(' gradient')[0].lower()} gradient norm")
        if ys is not None:
            _reference(ax, max_grad_norm, f"clip at {max_grad_norm}")
            ax.set_ylim(0, max(max_grad_norm * 1.3, float(np.nanmax(ys)) * 1.15))

    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)                          # free the figure's memory
    return path
