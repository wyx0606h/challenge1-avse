"""Objective (reference-based) metrics for AVSE evaluation.

Used when a clean reference IS available (e.g. synthetic VoxCeleb2 mixtures in
test.py). All metrics wrap the official ``torchmetrics.audio`` implementations
rather than hand-rolled formulas, so results match the literature:

    SI-SDR : ScaleInvariantSignalDistortionRatio
    SDR    : SignalDistortionRatio
    PESQ   : PerceptualEvaluationSpeechQuality(16000, 'wb')   (wideband)
    STOI   : ShortTimeObjectiveIntelligibility(16000, False)

Also reports the "improvement" variants (metric on estimate minus metric on the
unprocessed mixture): si_sdr_i / sdr_i.

For no-reference / text / speaker metrics on the real-world test set (no clean
GT), see :mod:`look2hear.metrics.real_metrics`.
"""

import csv

import numpy as np
import torch
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)


class MetricsTracker:
    """Track reference-based metrics during evaluation.

    Args:
        save_file: optional CSV path for the summary (mean per metric).
        sample_rate: waveform sample rate; PESQ 'wb' and STOI require 16 kHz.
    """

    def __init__(self, save_file=None, sample_rate=16000):
        self.save_file = save_file
        self.sample_rate = sample_rate

        # Official torchmetrics implementations (stateless functional use here:
        # we instantiate once and call per-sample, reading the returned scalar).
        self._si_sdr = ScaleInvariantSignalDistortionRatio()
        self._sdr = SignalDistortionRatio()
        # PESQ wideband mode needs 16 kHz; STOI extended=False (standard STOI).
        self._pesq = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
        self._stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)

        self.all_metrics = {
            "si_sdr": [], "sdr": [], "pesq": [], "stoi": [],
            "si_sdr_i": [], "sdr_i": [],
        }

    @staticmethod
    def _to_1d(x):
        """Coerce [C,T]/[1,T]/[T] (tensor or ndarray) to a 1-D float tensor."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.detach().float().cpu()
        if x.ndim > 1:
            x = x.reshape(-1)
        return x.contiguous()

    def _safe(self, fn, *args):
        """Run a torchmetrics call, returning a float or 0.0 on failure.

        PESQ in particular raises on silent / degenerate frames; we keep the
        evaluation going and record 0.0 rather than aborting the whole run.
        """
        try:
            return float(fn(*args))
        except Exception as e:
            print(f"[metric] {fn} failed: {e}")
            return 0.0

    def __call__(self, mix, clean, estimate, key=None):
        """Score one sample. Args are [C,T] / [1,T] / [T] (tensor or ndarray)."""
        mix = self._to_1d(mix)
        clean = self._to_1d(clean)
        estimate = self._to_1d(estimate)

        # Align lengths defensively (model output can differ by a few samples).
        n = min(mix.shape[-1], clean.shape[-1], estimate.shape[-1])
        mix, clean, estimate = mix[:n], clean[:n], estimate[:n]

        # torchmetrics audio convention: metric(preds, target).
        si_sdr = self._safe(self._si_sdr, estimate, clean)
        sdr = self._safe(self._sdr, estimate, clean)
        si_sdr_mix = self._safe(self._si_sdr, mix, clean)
        sdr_mix = self._safe(self._sdr, mix, clean)
        pesq = self._safe(self._pesq, estimate, clean)
        stoi = self._safe(self._stoi, estimate, clean)

        result = {
            "si_sdr": si_sdr,
            "sdr": sdr,
            "pesq": pesq,
            "stoi": stoi,
            "si_sdr_i": si_sdr - si_sdr_mix,
            "sdr_i": sdr - sdr_mix,
        }
        for k, v in result.items():
            self.all_metrics[k].append(v)
        return result

    def get_mean(self):
        return {k: (float(np.mean(v)) if v else 0.0)
                for k, v in self.all_metrics.items()}

    def final(self):
        mean_metrics = self.get_mean()

        print("\n" + "=" * 50)
        print("Final Evaluation Results:")
        print("=" * 50)
        for k, v in mean_metrics.items():
            print(f"{k.upper()}: {v:.4f}")
        print("=" * 50)

        if self.save_file:
            with open(self.save_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in mean_metrics.items():
                    writer.writerow([k, f"{v:.4f}"])
            print(f"Results saved to {self.save_file}")

        return mean_metrics


# Alias for backward compatibility.
ALLMetricsTracker = MetricsTracker
