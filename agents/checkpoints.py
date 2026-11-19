"""Save and load training checkpoints, and find the checkpoints that belong to one training run.

Usage inside ppo_main_xcom, at the end of an epoch:
    save_checkpoint(f"runs/checkpoint_xcom_epoch{epoch + 1}.pt",
                    policy_network, value_network, optim, epoch + 1, history, config)

Usage when evaluating (only checkpoints from the same run as the final one are returned):
    for path, checkpoint in run_checkpoints():          # or run_checkpoints(name="tu")
        ...

List everything in runs/ (run from the top project folder):
    python -m agents.checkpoints
"""
import glob
import os
import re
import time
import torch


def _plain(value):
    """Turn NumPy / PyTorch numbers into plain Python numbers, and functions into their names."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if callable(value) and hasattr(value, "__name__"):   # e.g. reward_fn: store "tu_reward", not the function
        return value.__name__
    if hasattr(value, "item"):          # NumPy scalars and one-element tensors
        return value.item()
    return value


def save_checkpoint(path, policy_network, value_network, optim, epoch, history, config):
    """Write one file holding the networks, the optimiser, progress and settings.

    config: the settings needed to rebuild the networks (obs_size, action_size, hidden_size)
            plus any others worth recording for the run (lr, gamma, ...).
    History and config are converted to plain Python values: torch.load's safe default
    (weights_only=True) refuses NumPy numbers and references to functions.
    """
    torch.save({
        "policy": policy_network.state_dict(),   # the policy's learned weights
        "value": value_network.state_dict(),     # the value network's learned weights
        "optim": optim.state_dict(),             # Adam's running averages, needed to resume smoothly
        "epoch": epoch,                          # how many epochs were completed
        "history": _plain(history),              # every per-epoch number logged so far
        "config": _plain(config),                # settings, including what rebuilding the networks needs
    }, path)


def load_checkpoint(path, trusted=False):
    """Read a checkpoint back as a dictionary with the keys used in save_checkpoint.

    trusted=True also loads older files that stored a function in their config (e.g. reward_fn).
    Only use it for checkpoints you made yourself: it lets the file run code while loading.
    """
    try:
        return torch.load(path, weights_only=not trusted)
    except Exception as err:
        if trusted or "Weights only load failed" not in str(err):
            raise
        raise RuntimeError(f"{path} was saved by an older version that stored a function in its config. "
                           f"If you made this file, load it with trusted=True.") from None


# ---------- which checkpoints belong to the same run ----------
def _paths(name=None, folder="runs"):
    """(final path or None, {epoch: path}) for checkpoints named with this run name (or none)."""
    tag = f"_{name}" if name else ""
    final = os.path.join(folder, f"checkpoint_xcom_final{tag}.pt")
    pattern = re.compile(rf"checkpoint_xcom_epoch(\d+){re.escape(tag)}\.pt")
    epochs = {}
    for p in glob.glob(os.path.join(folder, "checkpoint_xcom_epoch*.pt")):
        m = pattern.fullmatch(os.path.basename(p))
        if m:
            epochs[int(m.group(1))] = p
    return (final if os.path.exists(final) else None), epochs


def mismatch(checkpoint, reference):
    """Why `checkpoint` is not from the same run as `reference` (an empty string if it is).

    Same run means: identical config, and the checkpoint's history is exactly the first part of the
    reference's history. Two separate runs with the same settings still log different numbers.
    """
    a, b = _plain(checkpoint["config"]), _plain(reference["config"])
    if a != b:
        keys = sorted(k for k in set(a) | set(b) if a.get(k, "<missing>") != b.get(k, "<missing>"))
        return "different settings: " + ", ".join(f"{k} {a.get(k, '<missing>')} vs {b.get(k, '<missing>')}"
                                                  for k in keys)
    h, ref = checkpoint["history"], reference["history"]
    n = checkpoint["epoch"]
    for key, values in h.items():
        if key in ref and list(values[:n]) != list(ref[key][:n]):
            return f"same settings but a different run (its '{key}' history differs)"
    return ""


def run_checkpoints(name=None, folder="runs", trusted=False):
    """[(path, checkpoint), ...] for one run, in training order, ending with its last checkpoint.

    The reference is the run's final checkpoint (or, if training stopped early, its highest epoch).
    Checkpoints left over from other runs are skipped, with a printed reason.
    """
    final, epochs = _paths(name, folder)
    if final is None and not epochs:
        label = f" named '{name}'" if name else ""
        raise FileNotFoundError(f"no checkpoints{label} found in {folder}/")
    ref_path = final or epochs[max(epochs)]
    if final is None:
        print(f"note: no final checkpoint; using {os.path.basename(ref_path)} as the reference")
    reference = load_checkpoint(ref_path, trusted)

    kept, skipped = [], []
    for epoch in sorted(epochs):
        path = epochs[epoch]
        checkpoint = reference if path == ref_path else load_checkpoint(path, trusted)
        reason = mismatch(checkpoint, reference) or (
            "" if checkpoint["epoch"] <= reference["epoch"] else "later than the final checkpoint")
        if reason:
            skipped.append((path, reason))
        else:
            kept.append((path, checkpoint))
    if final and all(c["epoch"] != reference["epoch"] for _, c in kept):
        kept.append((final, reference))      # the final epoch was not also saved as an epoch file
    for path, reason in skipped:
        print(f"skipped {os.path.basename(path)}: {reason}")
    return kept


def describe_checkpoints(folder="runs", trusted=True):
    """Print what each checkpoint file in a folder is, oldest first: when it was saved, its epoch,
    the run length, the reward and the battle size."""
    paths = sorted(glob.glob(os.path.join(folder, "*.pt")), key=os.path.getmtime)
    if not paths:
        print(f"no .pt files in {folder}/")
    for p in paths:
        c = load_checkpoint(p, trusted)
        print(c)
        cfg = _plain(c["config"])
        env = cfg.get("env_kwargs", {})
        saved = time.strftime("%b %d %H:%M", time.localtime(os.path.getmtime(p)))
        print(f"{saved}  {os.path.basename(p):38s} epoch {c['epoch']:>4} of {cfg.get('total_epochs', '?'):>4}  "
              f"reward {cfg.get('reward_fn', 'sparse (not recorded)'):12s} "
              f"history {len(c['history'].get('win rate', [])):>4} epochs  "
              f"{env.get('n_soldiers', '?')}v{env.get('n_aliens', '?')} on {env.get('size', '?')}x{env.get('size', '?')}")


if __name__ == "__main__":
    describe_checkpoints()
