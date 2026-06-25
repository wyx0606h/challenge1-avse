"""
Utility functions
"""
import torch


def tensors_to_device(tensors, device):
    """
    Move tensors to device

    Args:
        tensors: tuple of tensors or single tensor
        device: target device

    Returns:
        Moved tensors
    """
    if isinstance(tensors, (tuple, list)):
        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tensors)
    else:
        return tensors.to(device) if isinstance(tensors, torch.Tensor) else tensors


def print_only(*args, **kwargs):
    """Print function (for compatibility)"""
    print(*args, **kwargs)
