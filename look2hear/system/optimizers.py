"""
Optimizer utilities
"""
import torch.optim as optim


def make_optimizer(params, optim_name="adam", **kwargs):
    """
    Create optimizer

    Args:
        params: model parameters
        optim_name: optimizer name
        **kwargs: optimizer arguments

    Returns:
        optimizer
    """
    optim_name = optim_name.lower()

    if optim_name == "adam":
        return optim.Adam(params, **kwargs)
    elif optim_name == "sgd":
        return optim.SGD(params, **kwargs)
    elif optim_name == "rmsprop":
        return optim.RMSprop(params, **kwargs)
    elif optim_name == "adamw":
        return optim.AdamW(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")
