"""NT-LDFE model components.

Noise-aware Temporal Label Distribution Fatigue Estimator for EEG-DE-like
feature matrices and noisy continuous PERCLOS observations.

The model accepts sequence tensors with shape:
    x: (batch, seq_len, *feature_shape)

For the current loaders this means:
    My_fatigue: (batch, seq_len, 31, 5)
    SEED-VIG:   (batch, seq_len, 1600, 17)

No command line arguments live in this file. Hyperparameters are supplied by
the training config in nt_ldfe_loso.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NTLDFEConfig:
    """Model hyperparameters."""

    feature_shape: tuple[int, ...]
    encoder_dim: int = 64
    encoder_layers: int = 2
    temporal_hidden_dim: int = 96
    temporal_layers: int = 2
    regression_hidden_dim: int = 64
    dropout: float = 0.2
    temporal_type: str = "gru"  # "gru" or "tcn"
    bidirectional: bool = True
    use_feature_attention: bool = True
    mu_activation: str = "tanh"  # "tanh", "sigmoid", or "linear"
    num_soft_levels: int = 0
    log_var_min: float = -6.0
    log_var_max: float = 2.0


class FeatureMatrixEncoder(nn.Module):
    """Encode one feature matrix per time window.

    The encoder keeps a lightweight channel/band gate when feature_shape is 2D.
    This gives interpretable weights without assuming a fixed EEG montage.
    """

    def __init__(
        self,
        feature_shape: tuple[int, ...],
        output_dim: int,
        dropout: float = 0.2,
        use_attention: bool = True,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if len(feature_shape) == 0:
            raise ValueError("feature_shape must not be empty")
        self.feature_shape = tuple(int(v) for v in feature_shape)
        self.feature_dim = 1
        for value in self.feature_shape:
            self.feature_dim *= value

        self.use_attention = use_attention and len(self.feature_shape) == 2
        if self.use_attention:
            channels, bands = self.feature_shape
            self.channel_logits = nn.Parameter(torch.zeros(channels))
            self.band_logits = nn.Parameter(torch.zeros(bands))
        else:
            self.channel_logits = None
            self.band_logits = None

        hidden_dim = max(output_dim * 2, output_dim)
        layers: list[nn.Module] = [
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        ])
        self.net = nn.Sequential(*layers)

    def attention_weights(self) -> dict[str, Optional[torch.Tensor]]:
        if not self.use_attention:
            return {"channel": None, "band": None}
        assert self.channel_logits is not None
        assert self.band_logits is not None
        return {
            "channel": torch.softmax(self.channel_logits, dim=0),
            "band": torch.softmax(self.band_logits, dim=0),
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Encode x.

        Args:
            x: (batch, seq_len, *feature_shape)

        Returns:
            encoded: (batch, seq_len, output_dim)
            attention: optional channel/band weights
        """
        if x.dim() < 3:
            raise ValueError(f"Expected sequence input, got shape {tuple(x.shape)}")

        batch_size, seq_len = x.shape[:2]
        feature_shape = tuple(x.shape[2:])
        if feature_shape != self.feature_shape:
            raise ValueError(
                f"Expected feature shape {self.feature_shape}, got {feature_shape}"
            )

        attention = self.attention_weights()
        if self.use_attention:
            channel_w = attention["channel"].view(1, 1, self.feature_shape[0], 1)
            band_w = attention["band"].view(1, 1, 1, self.feature_shape[1])
            x = x * channel_w * band_w

        x_flat = x.reshape(batch_size * seq_len, self.feature_dim)
        encoded = self.net(x_flat)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        return encoded, attention


class CausalTrim1d(nn.Module):
    """Trim right padding for causal Conv1d blocks."""

    def __init__(self, trim: int) -> None:
        super().__init__()
        self.trim = trim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.trim <= 0:
            return x
        return x[..., :-self.trim]


class TemporalConvBlock(nn.Module):
    """Small residual TCN block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dilation: int,
        dropout: float,
        causal: bool = False,
    ) -> None:
        super().__init__()
        kernel_size = 3
        if causal:
            padding = dilation * (kernel_size - 1)
            trim = padding
        else:
            padding = dilation
            trim = 0

        self.conv = nn.Sequential(
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            CausalTrim1d(trim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_dim, output_dim, kernel_size=1),
            nn.GELU(),
        )
        self.residual = (
            nn.Identity() if input_dim == output_dim else nn.Conv1d(input_dim, output_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class TemporalEncoder(nn.Module):
    """GRU or TCN temporal encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        temporal_type: str = "gru",
        causal_tcn: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        temporal_type = temporal_type.lower()
        self.temporal_type = temporal_type
        if temporal_type == "gru":
            self.net = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
        elif temporal_type == "tcn":
            blocks: list[nn.Module] = []
            for layer_idx in range(num_layers):
                in_dim = input_dim if layer_idx == 0 else hidden_dim
                blocks.append(
                    TemporalConvBlock(
                        in_dim,
                        hidden_dim,
                        dilation=2**layer_idx,
                        dropout=dropout,
                        causal=causal_tcn,
                    )
                )
            self.net = nn.Sequential(*blocks)
            self.output_dim = hidden_dim
        else:
            raise ValueError("temporal_type must be 'gru' or 'tcn'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_type == "gru":
            out, _ = self.net(x)
            return out
        x_conv = x.transpose(1, 2)
        out = self.net(x_conv)
        return out.transpose(1, 2)


class HeteroscedasticHead(nn.Module):
    """Output fatigue mean and log variance."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        mu_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.mu_activation = mu_activation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, 1)
        self.log_var = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        raw_mu = self.mu(h).squeeze(-1)
        if self.mu_activation == "sigmoid":
            mu = torch.sigmoid(raw_mu)
        elif self.mu_activation == "tanh":
            mu = 0.5 * (torch.tanh(raw_mu) + 1.0)
        elif self.mu_activation == "linear":
            mu = raw_mu
        else:
            raise ValueError("mu_activation must be 'tanh', 'sigmoid', or 'linear'")
        return mu, self.log_var(h).squeeze(-1)


class ModelOutput(dict):
    """Dictionary output that also supports legacy `mu, log_var = model(x)`."""

    def __iter__(self):
        yield self["mu"]
        yield self["log_var"]


class NTLDFE(nn.Module):
    """Noise-aware temporal fatigue estimator."""

    def __init__(
        self,
        config: Optional[NTLDFEConfig] = None,
        *,
        channels: Optional[int] = None,
        bands: Optional[int] = None,
        de_latent_dim: int = 64,
        encoder_dim: Optional[int] = None,
        temporal_hidden_dim: int = 96,
        temporal_num_layers: int = 2,
        regression_hidden_dim: int = 64,
        dropout: float = 0.2,
        use_gru: bool = True,
        temporal_type: Optional[str] = None,
        encoder_layers: int = 2,
        bidirectional: bool = True,
        use_feature_attention: bool = True,
        mu_activation: str = "tanh",
        num_soft_levels: int = 0,
        log_var_min: float = -6.0,
        log_var_max: float = 2.0,
    ) -> None:
        super().__init__()
        if config is None:
            if channels is None or bands is None:
                raise ValueError(
                    "Provide either NTLDFEConfig or legacy channels/bands arguments."
                )
            config = NTLDFEConfig(
                feature_shape=(channels, bands),
                encoder_dim=encoder_dim if encoder_dim is not None else de_latent_dim,
                encoder_layers=encoder_layers,
                temporal_hidden_dim=temporal_hidden_dim,
                temporal_layers=temporal_num_layers,
                regression_hidden_dim=regression_hidden_dim,
                dropout=dropout,
                temporal_type=temporal_type or ("gru" if use_gru else "tcn"),
                bidirectional=bidirectional,
                use_feature_attention=use_feature_attention,
                mu_activation=mu_activation,
                num_soft_levels=num_soft_levels,
                log_var_min=log_var_min,
                log_var_max=log_var_max,
            )
        self.config = config
        self.encoder = FeatureMatrixEncoder(
            feature_shape=config.feature_shape,
            output_dim=config.encoder_dim,
            dropout=config.dropout,
            use_attention=config.use_feature_attention,
            num_layers=config.encoder_layers,
        )
        self.temporal = TemporalEncoder(
            input_dim=config.encoder_dim,
            hidden_dim=config.temporal_hidden_dim,
            num_layers=config.temporal_layers,
            dropout=config.dropout,
            temporal_type=config.temporal_type,
            bidirectional=config.bidirectional,
        )
        self.head = HeteroscedasticHead(
            input_dim=self.temporal.output_dim,
            hidden_dim=config.regression_hidden_dim,
            dropout=config.dropout,
            mu_activation=config.mu_activation,
        )
        self.soft_head = (
            nn.Linear(self.temporal.output_dim, config.num_soft_levels)
            if config.num_soft_levels > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, Optional[torch.Tensor]]]:
        squeeze_sequence = False
        if x.dim() >= 2 and tuple(x.shape[1:]) == self.config.feature_shape:
            x = x.unsqueeze(1)
            squeeze_sequence = True

        encoded, attention = self.encoder(x)
        temporal = self.temporal(encoded)
        mu, log_var = self.head(temporal)
        log_var = torch.clamp(
            log_var,
            min=self.config.log_var_min,
            max=self.config.log_var_max,
        )
        if squeeze_sequence:
            mu = mu.squeeze(1)
            log_var = log_var.squeeze(1)

        output = ModelOutput({
            "mu": mu,
            "log_var": log_var,
            "attention": attention,
        })
        if self.soft_head is not None:
            soft_logits = self.soft_head(temporal)
            if squeeze_sequence:
                soft_logits = soft_logits.squeeze(1)
            output["soft_logits"] = soft_logits
        return output


# Legacy aliases kept so older notes and example imports still resolve.
DEChannelBandEncoder = FeatureMatrixEncoder
HeteroscedasticRegressionHead = HeteroscedasticHead


@dataclass
class NTLDFELossConfig:
    """Loss weights."""

    lambda_nll: float = 0.5
    lambda_reg: float = 0.5
    lambda_corr: float = 0.05
    lambda_ccc: float = 0.0
    lambda_std: float = 0.0
    lambda_mean: float = 0.05
    beta_smooth: float = 1e-3
    beta_acc: float = 1e-3
    lambda_var: float = 1e-5
    eta_soft: float = 0.0
    gamma_rank: float = 0.0
    num_soft_levels: int = 5
    soft_tau: float = 0.1
    rank_margin: float = 0.05
    rank_delta: float = 0.1
    rank_gap: int = 4


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood without the constant term."""
    return (0.5 * torch.exp(-log_var) * (y - mu).pow(2) + 0.5 * log_var).mean()


def robust_regression_loss(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Huber-style point loss to keep the mean estimate numerically useful."""
    return F.smooth_l1_loss(mu, y, beta=0.05)


def correlation_loss(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batch-level Pearson shape loss for trajectory trend alignment."""
    mu_flat = mu.reshape(-1)
    y_flat = y.reshape(-1)
    if mu_flat.numel() < 3:
        return mu.new_tensor(0.0)
    mu_centered = mu_flat - mu_flat.mean()
    y_centered = y_flat - y_flat.mean()
    denominator = torch.sqrt(
        torch.sum(mu_centered.pow(2)) * torch.sum(y_centered.pow(2))
    ).clamp_min(1e-8)
    corr = torch.sum(mu_centered * y_centered) / denominator
    return 1.0 - corr


def ccc_loss(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concordance loss to align correlation, mean, and amplitude."""
    mu_flat = mu.reshape(-1)
    y_flat = y.reshape(-1)
    if mu_flat.numel() < 3:
        return mu.new_tensor(0.0)
    mu_mean = mu_flat.mean()
    y_mean = y_flat.mean()
    covariance = torch.mean((mu_flat - mu_mean) * (y_flat - y_mean))
    mu_var = torch.var(mu_flat, unbiased=False)
    y_var = torch.var(y_flat, unbiased=False)
    denominator = (mu_var + y_var + (mu_mean - y_mean).pow(2)).clamp_min(1e-8)
    ccc = 2.0 * covariance / denominator
    return 1.0 - ccc


def std_alignment_loss(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Discourage collapsed prediction amplitude inside sequence batches."""
    mu_std = torch.std(mu.reshape(-1), unbiased=False)
    y_std = torch.std(y.reshape(-1), unbiased=False)
    return F.smooth_l1_loss(mu_std, y_std, beta=0.02)


def sequence_mean_loss(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Align sequence-level fatigue load without forcing pointwise overfit."""
    if mu.dim() < 2:
        return robust_regression_loss(mu, y)
    return F.smooth_l1_loss(mu.mean(dim=1), y.mean(dim=1), beta=0.05)


def smoothness_loss(mu: torch.Tensor) -> torch.Tensor:
    if mu.shape[1] < 2:
        return mu.new_tensor(0.0)
    return (mu[:, 1:] - mu[:, :-1]).pow(2).mean()


def acceleration_loss(mu: torch.Tensor) -> torch.Tensor:
    if mu.shape[1] < 3:
        return mu.new_tensor(0.0)
    return (mu[:, 2:] - 2.0 * mu[:, 1:-1] + mu[:, :-2]).pow(2).mean()


def variance_regularization(log_var: torch.Tensor) -> torch.Tensor:
    return log_var.pow(2).mean()


def make_soft_labels(
    y: torch.Tensor,
    num_levels: int,
    tau: float,
    centers: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Map continuous labels to Gaussian-kernel soft label distributions."""
    if centers is None:
        centers = torch.linspace(0.0, 1.0, num_levels, device=y.device, dtype=y.dtype)
    else:
        centers = centers.to(device=y.device, dtype=y.dtype)
    distances = (y.unsqueeze(-1) - centers.view(1, 1, -1)).pow(2)
    q = torch.exp(-distances / (2.0 * tau * tau))
    return q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def soft_label_kl(logits: torch.Tensor, y: torch.Tensor, num_levels: int, tau: float) -> torch.Tensor:
    q = make_soft_labels(y, num_levels=num_levels, tau=tau)
    log_p = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def ranking_loss(
    mu: torch.Tensor,
    y: torch.Tensor,
    margin: float,
    delta: float,
    gap: int,
) -> torch.Tensor:
    """Weak trend ranking loss within each sequence window."""
    if mu.shape[1] <= gap:
        return mu.new_tensor(0.0)
    left_mu = mu[:, :-gap]
    right_mu = mu[:, gap:]
    left_y = y[:, :-gap]
    right_y = y[:, gap:]
    valid = (right_y - left_y) > delta
    if not torch.any(valid):
        return mu.new_tensor(0.0)
    losses = torch.relu(margin - (right_mu - left_mu))
    return losses[valid].mean()


class NTLDFELoss(nn.Module):
    """Combined NT-LDFE objective."""

    def __init__(
        self,
        config: Optional[NTLDFELossConfig] = None,
        *,
        beta_smooth: float = 1e-3,
        beta_acc: float = 1e-3,
        lambda_var: float = 1e-5,
        lambda_nll: float = 0.5,
        lambda_reg: float = 0.5,
        lambda_corr: float = 0.05,
        lambda_ccc: float = 0.0,
        lambda_std: float = 0.0,
        lambda_mean: float = 0.05,
        eta_soft: float = 0.0,
        gamma_rank: float = 0.0,
        num_soft_levels: int = 5,
        soft_tau: float = 0.1,
        rank_margin: float = 0.05,
        rank_delta: float = 0.1,
        rank_gap: int = 4,
    ) -> None:
        super().__init__()
        if config is None:
            config = NTLDFELossConfig(
                lambda_nll=lambda_nll,
                lambda_reg=lambda_reg,
                lambda_corr=lambda_corr,
                lambda_ccc=lambda_ccc,
                lambda_std=lambda_std,
                lambda_mean=lambda_mean,
                beta_smooth=beta_smooth,
                beta_acc=beta_acc,
                lambda_var=lambda_var,
                eta_soft=eta_soft,
                gamma_rank=gamma_rank,
                num_soft_levels=num_soft_levels,
                soft_tau=soft_tau,
                rank_margin=rank_margin,
                rank_delta=rank_delta,
                rank_gap=rank_gap,
            )
        self.config = config

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute loss.

        Preferred call:
            criterion(outputs, y)

        Legacy call:
            criterion(mu, log_var, y)
        """
        if len(args) >= 2 and isinstance(args[0], dict):
            outputs = args[0]
            y = args[1]
        elif len(args) >= 3:
            outputs = {"mu": args[0], "log_var": args[1]}
            y = args[2]
        else:
            raise TypeError("Use criterion(outputs, y) or criterion(mu, log_var, y).")

        mu = outputs["mu"]
        log_var = outputs["log_var"]
        if y.shape != mu.shape:
            y = y.reshape_as(mu)

        nll = gaussian_nll(mu, log_var, y)
        reg = robust_regression_loss(mu, y)
        corr = correlation_loss(mu, y)
        ccc = ccc_loss(mu, y)
        std = std_alignment_loss(mu, y)
        mean = sequence_mean_loss(mu, y)
        smooth = smoothness_loss(mu)
        acc = acceleration_loss(mu)
        var = variance_regularization(log_var)

        total = (
            self.config.lambda_nll * nll
            + self.config.lambda_reg * reg
            + self.config.lambda_corr * corr
            + self.config.lambda_ccc * ccc
            + self.config.lambda_std * std
            + self.config.lambda_mean * mean
            + self.config.beta_smooth * smooth
            + self.config.beta_acc * acc
            + self.config.lambda_var * var
        )

        soft = mu.new_tensor(0.0)
        if self.config.eta_soft > 0 and "soft_logits" in outputs:
            soft = soft_label_kl(
                outputs["soft_logits"],
                y,
                num_levels=self.config.num_soft_levels,
                tau=self.config.soft_tau,
            )
            total = total + self.config.eta_soft * soft

        rank = mu.new_tensor(0.0)
        if self.config.gamma_rank > 0:
            rank = ranking_loss(
                mu,
                y,
                margin=self.config.rank_margin,
                delta=self.config.rank_delta,
                gap=self.config.rank_gap,
            )
            total = total + self.config.gamma_rank * rank

        return {
            "total": total,
            "total_loss": total,
            "nll": nll.detach(),
            "nll_loss": nll.detach(),
            "reg": reg.detach(),
            "corr": corr.detach(),
            "ccc_loss": ccc.detach(),
            "std_loss": std.detach(),
            "mean": mean.detach(),
            "smooth": smooth.detach(),
            "smooth_loss": smooth.detach(),
            "acc": acc.detach(),
            "acc_loss": acc.detach(),
            "var": var.detach(),
            "var_loss": var.detach(),
            "soft": soft.detach(),
            "rank": rank.detach(),
        }


class SoftLabelDistributionLoss(nn.Module):
    """Compatibility wrapper for soft label distribution learning."""

    def __init__(self, num_levels: int = 5, tau: float = 0.1) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.tau = tau

    def soft_labels_from_perclos(self, y: torch.Tensor) -> torch.Tensor:
        return make_soft_labels(y, self.num_levels, self.tau)

    def kl_divergence(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (q * (torch.log(q.clamp_min(1e-8)) - torch.log(p.clamp_min(1e-8)))).sum(-1).mean()


class RankingConstraintLoss(nn.Module):
    """Compatibility wrapper for the ranking loss."""

    def __init__(self, margin: float = 0.05, delta_threshold: float = 0.1) -> None:
        super().__init__()
        self.margin = margin
        self.delta_threshold = delta_threshold

    def forward(self, mu: torch.Tensor, y: torch.Tensor, gap: int = 4) -> torch.Tensor:
        return ranking_loss(mu, y, self.margin, self.delta_threshold, gap)


def reshape_data_for_model(data, labels, time_steps: Optional[int] = None):
    """Legacy helper: convert numpy arrays to tensors without changing order."""
    import numpy as np

    data_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    return data_tensor, label_tensor


def create_temporal_batch(data, labels, window_size: int = 10, stride: int = 5):
    """Legacy helper for sliding windows over a single ordered sequence."""
    import numpy as np

    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    windows = []
    y_windows = []
    for start in range(0, data.shape[0] - window_size + 1, stride):
        end = start + window_size
        windows.append(data[start:end])
        y_windows.append(labels[start:end])
    return (
        torch.from_numpy(np.stack(windows).astype(np.float32)),
        torch.from_numpy(np.stack(y_windows).astype(np.float32)),
    )


def smooth_labels(labels, window_size: int = 5, method: str = "gaussian"):
    """Smooth labels within one already-split sequence."""
    import numpy as np

    labels = np.asarray(labels, dtype=np.float32)
    if window_size <= 1:
        return labels.copy()
    if window_size % 2 == 0:
        window_size += 1
    if method == "gaussian":
        radius = window_size // 2
        xs = np.arange(-radius, radius + 1)
        sigma = max(window_size / 3.0, 1e-6)
        kernel = np.exp(-(xs**2) / (2.0 * sigma**2))
    elif method == "hamming":
        kernel = np.hamming(window_size)
    elif method == "hann":
        kernel = np.hanning(window_size)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    kernel = kernel / kernel.sum()
    padded = np.pad(labels, (window_size // 2, window_size // 2), mode="reflect")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)
