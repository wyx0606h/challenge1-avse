"""
Loss functions for audio-visual speech enhancement
"""
import torch
import torch.nn as nn


def cal_si_snr(source, estimate_source):
    """
    Calculate SI-SNR
    Args:
        source: [B, T], ground truth
        estimate_source: [B, T], estimated source
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()

    # Zero-mean normalization
    source = source - torch.mean(source, dim=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

    # s_target
    s_target = torch.sum(source * estimate_source, dim=-1, keepdim=True) * source / (
        torch.sum(source ** 2, dim=-1, keepdim=True) + EPS
    )
    # e_noise
    e_noise = estimate_source - s_target

    # SI-SNR
    si_snr = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + EPS) + EPS
    )
    return si_snr


def cal_si_sdr(source, estimate_source):
    """
    Calculate SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    Same as SI-SNR
    """
    return cal_si_snr(source, estimate_source)


def pairwise_neg_sisdr(est_targets, targets):
    """
    Calculate pairwise negative SI-SDR

    Args:
        est_targets: [B, C, T] or [B, T]
        targets: [B, C, T] or [B, T]

    Returns:
        loss: negative SI-SDR
    """
    if est_targets.ndim == 2:
        est_targets = est_targets.unsqueeze(1)
    if targets.ndim == 2:
        targets = targets.unsqueeze(1)

    # [B, C]
    si_sdr = cal_si_sdr(targets, est_targets)
    return -torch.mean(si_sdr)


def pairwise_neg_snr(est_targets, targets):
    """
    Calculate pairwise negative SNR

    Args:
        est_targets: [B, C, T] or [B, T]
        targets: [B, C, T] or [B, T]

    Returns:
        loss: negative SNR
    """
    if est_targets.ndim == 2:
        est_targets = est_targets.unsqueeze(1)
    if targets.ndim == 2:
        targets = targets.unsqueeze(1)

    EPS = 1e-8
    # [B, C, T]
    noise = est_targets - targets
    snr = 10 * torch.log10(
        torch.sum(targets ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + EPS) + EPS
    )
    return -torch.mean(snr)


def singlesrc_neg_sisdr(est_target, target):
    """
    Single source negative SI-SDR
    """
    return pairwise_neg_sisdr(est_target, target)


def singlesrc_neg_snr(est_target, target):
    """
    Single source negative SNR
    """
    return pairwise_neg_snr(est_target, target)


class PITLossWrapper(nn.Module):
    """
    Permutation Invariant Training (PIT) Loss Wrapper

    Args:
        loss_func: Loss function
        pit_from: 'pw_mtx' or 'pw_pt'
    """
    def __init__(self, loss_func, pit_from='pw_mtx', threshold_byloss=False):
        super(PITLossWrapper, self).__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.threshold_byloss = threshold_byloss

    def forward(self, est_targets, targets):
        """
        Args:
            est_targets: [B, C, T] or [B, T]
            targets: [B, C, T] or [B, T]

        Returns:
            loss: PIT loss
        """
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)
        if est_targets.ndim == 2:
            est_targets = est_targets.unsqueeze(1)

        loss = self.loss_func(est_targets, targets)
        return loss


class SingleSrcNegSDR(nn.Module):
    """
    Single source negative SI-SDR loss
    """
    def __init__(self, sdr_type='sisdr', zero_mean=True, reduction='mean'):
        super(SingleSrcNegSDR, self).__init__()
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.reduction = reduction

    def forward(self, est_target, target):
        if self.sdr_type == 'sisdr':
            loss = pairwise_neg_sisdr(est_target, target)
        elif self.sdr_type == 'snr':
            loss = pairwise_neg_snr(est_target, target)
        else:
            raise ValueError(f"Unsupported SDR type: {self.sdr_type}")

        return loss
