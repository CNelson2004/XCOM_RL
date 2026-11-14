import torch


def _plain(value):
    # making everything plain python numbers (because pytorch gets fussy with numpy numbers)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):          
        return value.item()
    return value


def save_checkpoint(path, policy_network, value_network, optim, epoch, history, config):
    # pytorch saving a dictionary of everything needed to resume training if a crash occurs
    torch.save({
        "policy": policy_network.state_dict(),   # policy's weights
        "value": value_network.state_dict(),     # value's weights
        "optim": optim.state_dict(),             # adam's stuff (needed to work optimizer more smoothly)
        "epoch": epoch,                          # completed epochs
        "history": _plain(history),              # stats for the epochs
        "config": _plain(config),                # settings needed to rebuild
    }, path)


def load_checkpoint(path):
    return torch.load(path)
