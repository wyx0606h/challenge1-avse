"""
Testing script for Audio-Visual Speech Enhancement
"""
import os
import torch
import yaml
import argparse
import warnings
from tqdm import tqdm

import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="Experiments/checkpoint/AVConvTasNet-Baseline/conf.yml",
    help="Path to configuration file"
)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Path to checkpoint (optional, will use best_model.pth if not specified)"
)
parser.add_argument(
    "--save_dir",
    default=None,
    help="Directory to save enhanced audio (optional)"
)
parser.add_argument(
    "--gpus",
    default="0",
    help="GPU IDs to use"
)


def main(args):
    """Main testing function"""

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Load configuration
    with open(args.conf_dir, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("Testing Configuration:")
    print("=" * 50)
    print(f"Experiment: {config['exp']['exp_name']}")
    print(f"Audio Model: {config['audionet']['audionet_name']}")
    print(f"Video Model: {config.get('videonet', {}).get('videonet_name', 'bundled')}")
    print("=" * 50)

    # Setup directories
    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"])

    # Determine checkpoint path
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = os.path.join(exp_dir, "best_model.pth")

    print(f"Loading model from: {model_path}")

    # Load model. Hyper-parameters AND the video encoder weights are stored
    # inside the checkpoint, so from_pretrain rebuilds the whole audio-visual
    # model -- no audionet_config and no separate video net required.
    audio_model = getattr(look2hear.models, config["audionet"]["audionet_name"]).from_pretrain(
        model_path,
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_model = audio_model.to(device)

    print(f"Using device: {device}")

    # Load data
    print("Loading test data...")
    datamodule = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    # Setup save directory
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Enhanced audio will be saved to: {save_dir}")

    # Setup metrics
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, "test_metrics.csv")
    metrics = MetricsTracker(save_file=metrics_file)

    # Set model to eval mode
    audio_model.eval()

    print("=" * 50)
    print("Starting Evaluation...")
    print("=" * 50)

    # Evaluation loop
    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), desc="Testing"):
            # Load data
            mix, clean, mouth, filename = test_set[idx]

            # Move to device
            mix = mix.to(device)
            clean = clean.to(device)

            # Prepare video input
            if isinstance(mouth, torch.Tensor):
                mouth = mouth.to(device)
            else:
                mouth = torch.from_numpy(mouth).float().to(device)

            # Add batch dim: [Tv, H, W] -> [1, Tv, H, W]. The model's forward
            # adds the channel dim and runs the (bundled) video encoder itself.
            if mouth.ndim == 3:
                mouth = mouth.unsqueeze(0)

            # Add batch dimension to audio
            mix_batch = mix.unsqueeze(0)

            # Enhance (video encoding happens inside the model)
            enhanced = audio_model(mix_batch, mouth)

            # Remove batch dimension
            enhanced = enhanced.squeeze(0)

            # Ensure same length
            min_len = min(enhanced.shape[-1], clean.shape[-1])
            enhanced = enhanced[..., :min_len]
            clean = clean[..., :min_len]
            mix = mix[..., :min_len]

            # Calculate metrics
            metrics(mix=mix.cpu(), clean=clean.cpu(), estimate=enhanced.cpu(), key=filename)

            # Save enhanced audio if requested
            if save_dir:
                import torchaudio
                save_path = os.path.join(save_dir, filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torchaudio.save(save_path, enhanced.cpu().unsqueeze(0), config["datamodule"]["data_config"]["sample_rate"])

    # Print final results
    final_metrics = metrics.final()

    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
