"""
Video models registry
"""
from .resnet_videomodel import ResNetVideoModel
from .avhubert_videomodel import AVHubertVideoModel

__all__ = [
    "ResNetVideoModel",
    "AVHubertVideoModel",
]
