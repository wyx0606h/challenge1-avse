"""
Base model class for all models
"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models

    All models should inherit from this class and implement the forward method.
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
        Load pretrained model

        Args:
            path: Path to checkpoint
            *args, **kwargs: Additional arguments for model initialization

        Returns:
            model: Loaded model
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
