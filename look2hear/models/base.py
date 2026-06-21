"""
Base model class for all models
"""
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class BaseModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Real-World-AVSE/Baseline",
    pipeline_tag="audio-to-audio",
    license="apache-2.0",
):
    """
    Base class for all models.

    Inherits :class:`huggingface_hub.PyTorchModelHubMixin`, which gives every
    subclass the standard ``save_pretrained`` / ``from_pretrained`` /
    ``push_to_hub`` API. Uploaded repos use the community-standard layout
    (``config.json`` + ``model.safetensors`` + ``README.md``): the mixin captures
    the subclass ``__init__`` arguments into ``config.json`` and stores weights as
    safetensors, so a model rebuilds from a HuggingFace repo id alone, e.g.::

        AV_ConvTasNet.from_pretrained("JusperLee/Real-World-AVSE-Baseline-Track1")

    The legacy :meth:`from_pretrain` (no ``d``) is kept for loading LOCAL
    checkpoints in the project's own formats -- a Lightning ``*.ckpt`` or a
    ``serialize()`` ``best_model.pth`` -- which the mixin does not understand.
    """

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, *args, **kwargs):
        """
        Forward pass - must be implemented by subclasses
        """
        raise NotImplementedError

    def serialize(self):
        """
        Serialize model for saving

        Returns:
            dict: Model state dict and configuration
        """
        return {
            'model_name': self.__class__.__name__,
            'state_dict': self.state_dict(),
            'model_args': self.get_model_args(),
        }

    def get_model_args(self):
        """
        Get model arguments for reconstruction

        Returns:
            dict: Model initialization arguments
        """
        return {'sample_rate': self.sample_rate}

    @classmethod
    def from_pretrain(cls, path, *args, **kwargs):
        """
        Load a model from a LOCAL checkpoint in the project's own formats.

        Handles a ``serialize()`` payload (``{model_name, state_dict, model_args}``,
        e.g. ``best_model.pth``) or a bare/legacy ``state_dict``. For loading from
        a HuggingFace repo (``config.json`` + safetensors) use the mixin's
        :meth:`from_pretrained` (with a ``d``) instead.

        Args:
            path: local checkpoint path.
            *args, **kwargs: extra arguments forwarded to the constructor,
                overriding the saved ``model_args``.

        Returns:
            model: the loaded model.
        """
        checkpoint = torch.load(path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'model_args' in checkpoint:
                # Update with saved args
                saved_args = checkpoint['model_args']
                saved_args.update(kwargs)
                model = cls(*args, **saved_args)
            else:
                model = cls(*args, **kwargs)

            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Old style checkpoint
            model = cls(*args, **kwargs)
            model.load_state_dict(checkpoint)

        return model
