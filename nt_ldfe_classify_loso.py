"""Complete LOSO classification pipeline for NT-LDFE on SEED-VIG.

Converts continuous PERCLOS labels to 3-class labels:
    Class 0: [0, 0.35]
    Class 1: (0.35, 0.7]
    Class 2: (0.7, 1.0]

Reuses FeatureMatrixEncoder and TemporalEncoder from nt_ldfe_model.py and
adds a ClassificationHead for 3-way fatigue-level prediction.

Key improvements (v3):
  - Focal Loss for class imbalance
  - Soft voting (probability averaging) for trajectory prediction
  - Mild Gaussian noise augmentation during training
  - Cosine annealing LR with linear warmup
  - Time-based validation for reliable signal
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nt_ldfe_model import (
    FeatureMatrixEncoder,
    TemporalEncoder,
)
from seed_dataloader import find_subject_dirs, load_subject as load_seedvig_subject


# ═══════════════════════════════════════════════════════════════════════════════
# Label conversion
# ═══════════════════════════════════════════════════════════════════════════════

def perclos_to_class(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Convert continuous PERCLOS [0,1] to class labels.

    2-class: Class 0 (awake): [0, 0.35], Class 1 (fatigue): (0.35, 1.0]
    3-class: Class 0 (low):   [0, 0.35]
             Class 1 (medium): (0.35, 0.7]
             Class 2 (high):   (0.7, 1.0]
    """
    labels = np.asarray(labels, dtype=np.float32)
    classes = np.zeros_like(labels, dtype=np.int64)
    if num_classes == 2:
        classes[labels > 0.35] = 1
    elif num_classes == 3:
        classes[labels > 0.35] = 1
        classes[labels > 0.7] = 2
    else:
        raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")
    return classes


def class_labels_to_unit_interval(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Validate class labels and keep as float32."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid_classes = list(range(num_classes))
    if not np.all(np.isin(labels, valid_classes)):
        raise ValueError(f"Class labels must be in {valid_classes}")
    return labels.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Focal Loss for class imbalance
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Down-weights easy examples, focuses on hard ones.
    Reference: Lin et al., ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        targets = targets.long().view(-1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -(smooth * log_probs).sum(dim=-1)
            probs = (smooth * F.softmax(logits, dim=-1)).sum(dim=-1)
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            probs = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - probs).pow(self.gamma)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if self.label_smoothing > 0:
                alpha_t = (smooth * alpha.unsqueeze(0)).sum(dim=-1)
            else:
                alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassifyConfig:
    """All user-editable experiment parameters for classification."""

    dataset_name: str = "seedvig"
    seedvig_root: str = "dataset/seedvig"
    output_dir: str = "outputs/nt_ldfe_classify_loso"

    seed: int = 42
    num_classes: int = 3
    max_folds: int | None = None
    val_fraction: float = 0.2
    val_subject_count: int = 3
    val_selection_strategy: str = "stratified_mean"
    val_strategy: str = "time"

    sequence_length: int = 45
    train_stride: int = 5
    eval_stride: int = 45
    selection_prediction_stride: int = 5

    # SEED-VIG feature compaction
    seedvig_feature_mode: str = "hybrid_de"
    seedvig_de_bins: int = 5
    seedvig_sampling_rate: float = 200.0
    feature_eps: float = 1e-6

    # Training
    batch_size: int = 48
    epochs: int = 250
    min_epochs: int = 15
    patience: int = 40
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0

    # Model
    temporal_type: str = "gru"
    encoder_dim: int = 96
    encoder_layers: int = 2
    temporal_hidden_dim: int = 128
    temporal_layers: int = 2
    classifier_hidden_dim: int = 96
    dropout: float = 0.2
    bidirectional: bool = True
    use_feature_attention: bool = True

    # Focal loss (replaces class_weight_mode)
    focal_gamma: float = 1.0        # 0 = standard CE, 1-2 = focal
    class_weight_mode: str = "balanced"
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Label smoothing for cross-entropy
    label_smoothing: float = 0.1

    # Data augmentation
    aug_noise_std: float = 0.02     # Gaussian noise std (0 = disabled)
    use_weighted_sampler: bool = False  # oversample minority classes
    feature_dropout: float = 0.0    # dropout applied to features before encoder (0=off)

    # Training schedule
    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = True
    use_scheduler: bool = True
    warmup_epochs: int = 5           # linear warmup epochs

    # Validation metric for checkpoint selection
    validation_metric: str = "macro_f1"   # "macro_f1", "accuracy", or "kappa"

    # Persistence
    save_checkpoints: bool = False
    save_predictions: bool = True
    save_history: bool = True
    save_partial_results: bool = True
    resume_completed_folds: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = ClassifyConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# Classification model
# ═══════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """Output 3-class logits per time step."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → logits: (batch, seq_len, num_classes)"""
        return self.net(x)


class ClassifyNTLDFE(nn.Module):
    """NT-LDFE adapted for 3-class fatigue level classification."""

    def __init__(self, config: ClassifyConfig, feature_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.config = config
        self.feature_shape = feature_shape
        self.feature_dropout = nn.Dropout(config.feature_dropout) if config.feature_dropout > 0 else nn.Identity()
        self.encoder = FeatureMatrixEncoder(
            feature_shape=feature_shape,
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
        self.head = ClassificationHead(
            input_dim=self.temporal.output_dim,
            hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (batch, seq_len, *feature_shape) → logits: (batch, seq_len, num_classes)"""
        squeeze_sequence = False
        if x.dim() >= 2 and tuple(x.shape[1:]) == self.feature_shape:
            x = x.unsqueeze(1)
            squeeze_sequence = True

        x = self.feature_dropout(x)
        encoded, attention = self.encoder(x)
        temporal = self.temporal(encoded)
        logits = self.head(temporal)

        if squeeze_sequence:
            logits = logits.squeeze(1)

        return {
            "logits": logits,
            "attention": attention,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SEED-VIG feature preparation
# ═══════════════════════════════════════════════════════════════════════════════

def seedvig_to_spectral_de(
    data: np.ndarray, sampling_rate: float, eps: float = 1e-6,
) -> np.ndarray:
    """Convert SEED-VIG epochs to channel x EEG-band DE-like features."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"SEED-VIG data should be 3D, got {data.shape}")
    n_samples, n_time, n_channels = data.shape
    centered = data - np.mean(data, axis=1, keepdims=True)
    window = np.hanning(n_time).astype(np.float32).reshape(1, n_time, 1)
    fft = np.fft.rfft(centered * window, axis=1)
    power = (np.abs(fft) ** 2).astype(np.float64) / max(n_time, 1)
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sampling_rate)
    bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
    features = np.empty((n_samples, n_channels, len(bands)), dtype=np.float32)
    for band_index, (low, high) in enumerate(bands):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            raise ValueError(f"No FFT bins found for band {low}-{high} Hz")
        band_power = np.mean(power[:, mask, :], axis=1)
        features[:, :, band_index] = np.log(band_power + eps).astype(np.float32)
    return features


def seedvig_to_band_ratios(
    data: np.ndarray, sampling_rate: float, eps: float = 1e-6,
) -> np.ndarray:
    """Compute fatigue-relevant band power ratios per channel.

    Returns per channel: [θ/β, α/β, (θ+α)/β, θ/α, δ/β, δ/α]
    These ratios are robust EEG fatigue markers (less affected by individual differences).
    """
    data = np.asarray(data, dtype=np.float32)
    n_samples, n_time, n_channels = data.shape
    centered = data - np.mean(data, axis=1, keepdims=True)
    window = np.hanning(n_time).astype(np.float32).reshape(1, n_time, 1)
    fft = np.fft.rfft(centered * window, axis=1)
    power = (np.abs(fft) ** 2).astype(np.float64) / max(n_time, 1)
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sampling_rate)

    band_powers = {}
    for name, (lo, hi) in [("delta", (1,4)), ("theta", (4,8)), ("alpha", (8,13)),
                             ("beta", (13,30)), ("gamma", (30,45))]:
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[name] = np.mean(power[:, mask, :], axis=1)  # (n_samples, n_channels)

    ratios = np.empty((n_samples, n_channels, 6), dtype=np.float32)
    ratios[:, :, 0] = np.log(band_powers["theta"] / (band_powers["beta"] + eps) + eps)
    ratios[:, :, 1] = np.log(band_powers["alpha"] / (band_powers["beta"] + eps) + eps)
    ratios[:, :, 2] = np.log((band_powers["theta"] + band_powers["alpha"]) / (band_powers["beta"] + eps) + eps)
    ratios[:, :, 3] = np.log(band_powers["theta"] / (band_powers["alpha"] + eps) + eps)
    ratios[:, :, 4] = np.log(band_powers["delta"] / (band_powers["beta"] + eps) + eps)
    ratios[:, :, 5] = np.log(band_powers["delta"] / (band_powers["alpha"] + eps) + eps)
    return ratios.astype(np.float32)


def seedvig_to_fine_spectral(
    data: np.ndarray, sampling_rate: float, bin_hz: float = 2.0, eps: float = 1e-6,
) -> np.ndarray:
    """Finer spectral bins (e.g., 2Hz from 1-31Hz → 15 bins) per channel.

    Provides higher frequency resolution than the standard 5-band decomposition.
    """
    data = np.asarray(data, dtype=np.float32)
    n_samples, n_time, n_channels = data.shape
    centered = data - np.mean(data, axis=1, keepdims=True)
    window = np.hanning(n_time).astype(np.float32).reshape(1, n_time, 1)
    fft = np.fft.rfft(centered * window, axis=1)
    power = (np.abs(fft) ** 2).astype(np.float64) / max(n_time, 1)
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sampling_rate)

    band_edges = np.arange(1.0, 31.0 + bin_hz, bin_hz)
    n_bands = len(band_edges) - 1
    features = np.empty((n_samples, n_channels, n_bands), dtype=np.float32)
    for i in range(n_bands):
        lo, hi = band_edges[i], band_edges[i+1]
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            features[:, :, i] = np.log(np.mean(power[:, mask, :], axis=1) + eps).astype(np.float32)
        else:
            features[:, :, i] = np.log(eps)
    return features


def seedvig_to_temporal_stats(
    data: np.ndarray, bins: int, eps: float = 1e-6,
) -> np.ndarray:
    """Convert SEED-VIG epochs to compact temporal statistics."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"SEED-VIG data should be 3D, got {data.shape}")
    n_samples, n_time, n_channels = data.shape
    if bins < 1 or bins > n_time:
        raise ValueError(f"seedvig_de_bins must be in [1, {n_time}], got {bins}")
    segments = np.array_split(np.arange(n_time), bins)
    features = np.empty((n_samples, n_channels, bins * 4), dtype=np.float32)
    for bin_index, segment in enumerate(segments):
        segment_data = data[:, segment, :]
        base = bin_index * 4
        var = np.var(segment_data, axis=1)
        std = np.sqrt(var + eps)
        mean = np.mean(segment_data, axis=1)
        if segment_data.shape[1] > 1:
            line_length = np.mean(np.abs(np.diff(segment_data, axis=1)), axis=1)
        else:
            line_length = np.zeros((n_samples, n_channels), dtype=np.float32)
        features[:, :, base + 0] = np.log(var + eps)
        features[:, :, base + 1] = mean
        features[:, :, base + 2] = std
        features[:, :, base + 3] = line_length
    return features


def seedvig_to_enhanced_temporal(
    data: np.ndarray, bins: int, eps: float = 1e-6,
) -> np.ndarray:
    """Enhanced temporal statistics with more features per bin.

    Per bin: log_var, mean, std, line_length, hjorth_mobility, zero_crossing_rate
    """
    data = np.asarray(data, dtype=np.float32)
    n_samples, n_time, n_channels = data.shape
    segments = np.array_split(np.arange(n_time), bins)
    n_stats = 6
    features = np.empty((n_samples, n_channels, bins * n_stats), dtype=np.float32)
    for bi, seg in enumerate(segments):
        seg_data = data[:, seg, :]  # (n_samples, seg_len, n_channels)
        base = bi * n_stats
        var = np.var(seg_data, axis=1)
        std = np.sqrt(var + eps)
        mean = np.mean(seg_data, axis=1)
        # Line length
        if seg_data.shape[1] > 1:
            ll = np.mean(np.abs(np.diff(seg_data, axis=1)), axis=1)
            # Hjorth mobility = std(diff) / std(signal)
            diff_std = np.sqrt(np.var(np.diff(seg_data, axis=1), axis=1) + eps)
            mobility = diff_std / std
            # Zero-crossing rate
            signs = np.sign(seg_data - mean[:, np.newaxis, :])
            zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1) / 2.0
        else:
            ll = np.zeros((n_samples, n_channels), dtype=np.float32)
            mobility = np.zeros((n_samples, n_channels), dtype=np.float32)
            zcr = np.zeros((n_samples, n_channels), dtype=np.float32)
        features[:, :, base + 0] = np.log(var + eps)
        features[:, :, base + 1] = mean
        features[:, :, base + 2] = std
        features[:, :, base + 3] = ll
        features[:, :, base + 4] = mobility
        features[:, :, base + 5] = zcr
    return features.astype(np.float32)


def prepare_subject_data(data: np.ndarray, config: ClassifyConfig) -> np.ndarray:
    """Apply SEED-VIG feature compaction.

    Feature modes:
      - "hybrid_de": original (5 spectral bands + 5 temporal bins × 4 stats)
      - "enhanced": richer features with band ratios + finer spectral + enhanced temporal
      - "enhanced_v2": band ratios + fine spectral only (no temporal bins)
      - "spectral_de": only 5-band spectral DE
      - "temporal_stats": only temporal stats
      - "as_is": raw data
    """
    data = np.asarray(data, dtype=np.float32)
    mode = config.seedvig_feature_mode

    if mode == "hybrid_de":
        spectral = seedvig_to_spectral_de(
            data, sampling_rate=config.seedvig_sampling_rate, eps=config.feature_eps)
        temporal = seedvig_to_temporal_stats(
            data, bins=config.seedvig_de_bins, eps=config.feature_eps)
        return np.concatenate([spectral, temporal], axis=2).astype(np.float32)

    if mode == "enhanced":
        # Richer features: 5-band DE + band ratios + fine spectral + enhanced temporal
        spectral = seedvig_to_spectral_de(data, config.seedvig_sampling_rate, config.feature_eps)
        ratios = seedvig_to_band_ratios(data, config.seedvig_sampling_rate, config.feature_eps)
        fine_spec = seedvig_to_fine_spectral(data, config.seedvig_sampling_rate, bin_hz=2.0, eps=config.feature_eps)
        temporal = seedvig_to_enhanced_temporal(data, bins=max(4, config.seedvig_de_bins), eps=config.feature_eps)
        parts = [spectral, ratios, fine_spec, temporal]
        return np.concatenate(parts, axis=2).astype(np.float32)

    if mode == "enhanced_v2":
        # Band ratios + fine spectral (15 bins) + enhanced temporal (8 bins)
        ratios = seedvig_to_band_ratios(data, config.seedvig_sampling_rate, config.feature_eps)
        fine_spec = seedvig_to_fine_spectral(data, config.seedvig_sampling_rate, bin_hz=2.0, eps=config.feature_eps)
        temporal = seedvig_to_enhanced_temporal(data, bins=8, eps=config.feature_eps)
        return np.concatenate([ratios, fine_spec, temporal], axis=2).astype(np.float32)

    if mode == "spectral_de":
        return seedvig_to_spectral_de(
            data, sampling_rate=config.seedvig_sampling_rate, eps=config.feature_eps)

    if mode == "temporal_logvar":
        return seedvig_to_temporal_stats(
            data, bins=config.seedvig_de_bins, eps=config.feature_eps,
        )[:, :, :config.seedvig_de_bins]

    if mode == "temporal_stats":
        return seedvig_to_temporal_stats(
            data, bins=config.seedvig_de_bins, eps=config.feature_eps)

    if mode == "as_is":
        return data

    raise ValueError(f"Unknown seedvig_feature_mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_subjects(config: ClassifyConfig) -> dict[str, dict[str, Any]]:
    """Load SEED-VIG subjects and convert labels to classes."""
    subjects: dict[str, dict[str, Any]] = {}
    for subject_index, subject_dir in enumerate(find_subject_dirs(config.seedvig_root), start=1):
        subject_key = f"sub{subject_index}"
        subject = load_seedvig_subject(subject_dir, stack=True)
        subject["raw_data_shape"] = tuple(np.asarray(subject["data"]).shape)
        subject["raw_label"] = subject["label"].copy()  # keep original PERCLOS
        subject["data"] = prepare_subject_data(subject["data"], config)
        # Convert PERCLOS to class labels
        subject["label"] = perclos_to_class(subject["label"], num_classes=config.num_classes)
        subjects[subject_key] = subject

    print(
        f"Prepared SEED-VIG with {config.seedvig_feature_mode}: "
        f"raw epochs -> compact channel-stat features"
    )
    return subjects


def ordered_subject_keys(subjects: dict[str, dict[str, Any]]) -> list[str]:
    def key_fn(name: str) -> tuple[int, str]:
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 10**9, name)
    return sorted(subjects.keys(), key=key_fn)


@dataclass
class DatasetAudit:
    subjects: int
    num_classes: int
    class_counts: dict[int, int]
    class_names: dict[int, str]
    note: str


def audit_subjects(subjects: dict[str, dict[str, Any]], config: ClassifyConfig) -> DatasetAudit:
    all_labels = np.concatenate([sub["label"].flatten() for sub in subjects.values()])
    class_counts = {}
    for c in range(config.num_classes):
        class_counts[c] = int(np.sum(all_labels == c))
    return DatasetAudit(
        subjects=len(subjects),
        num_classes=config.num_classes,
        class_counts=class_counts,
        class_names={0: "low [0,0.35]", 1: "medium (0.35,0.7]", 2: "high (0.7,1.0]"},
        note=f"3-class fatigue classification on SEED-VIG with {config.seedvig_feature_mode} features.",
    )


def split_train_val_subject(
    data: np.ndarray, labels: np.ndarray, val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(labels)
    if n < 2:
        raise ValueError("Each subject must have at least two samples.")
    val_len = max(1, int(round(n * val_fraction)))
    split = max(1, n - val_len)
    if split >= n:
        split = n - 1
    return data[:split], labels[:split], data[split:], labels[split:]


def choose_validation_subjects(
    train_keys: list[str],
    subjects: dict[str, dict[str, Any]],
    config: ClassifyConfig,
) -> set[str]:
    if len(train_keys) <= 1:
        return set()
    val_count = (
        config.val_subject_count
        if config.val_subject_count > 0
        else max(1, int(round(len(train_keys) * config.val_fraction)))
    )
    val_count = min(val_count, len(train_keys) - 1)
    if val_count <= 0:
        return set()

    strategy = config.val_selection_strategy.lower()
    if strategy == "stratified_mean":
        # Use raw PERCLOS means for stratification (stored as raw_label)
        subject_means: list[tuple[str, float]] = []
        for key in train_keys:
            raw = subjects[key].get("raw_label", subjects[key]["label"])
            mean_label = float(np.nanmean(raw)) if raw.size else float("nan")
            if np.isfinite(mean_label):
                subject_means.append((key, mean_label))
        if len(subject_means) >= val_count:
            subject_means.sort(key=lambda item: (item[1], item[0]))
            if val_count == 1:
                candidate_indices = [len(subject_means) // 2]
            else:
                candidate_indices = [
                    int(round(index))
                    for index in np.linspace(0, len(subject_means) - 1, val_count)
                ]
            selected_indices: list[int] = []
            for index in candidate_indices:
                index = int(np.clip(index, 0, len(subject_means) - 1))
                if index not in selected_indices:
                    selected_indices.append(index)
            for index in range(len(subject_means)):
                if len(selected_indices) >= val_count:
                    break
                if index not in selected_indices:
                    selected_indices.append(index)
            return {subject_means[index][0] for index in selected_indices[:val_count]}

    if strategy != "random":
        raise ValueError("val_selection_strategy must be 'stratified_mean' or 'random'")

    rng = random.Random(config.seed)
    keys = list(train_keys)
    rng.shuffle(keys)
    return set(keys[:val_count])


def compute_train_stats(train_arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    sum_: np.ndarray | None = None
    sum_sq: np.ndarray | None = None
    for arr in train_arrays:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            continue
        arr64 = arr.astype(np.float64, copy=False)
        arr_sum = arr64.sum(axis=0, keepdims=True)
        arr_sum_sq = np.square(arr64).sum(axis=0, keepdims=True)
        if sum_ is None:
            sum_ = arr_sum
            sum_sq = arr_sum_sq
        else:
            sum_ += arr_sum
            sum_sq += arr_sum_sq
        count += arr.shape[0]
    if count == 0 or sum_ is None or sum_sq is None:
        raise ValueError("No training arrays supplied.")
    mean64 = sum_ / count
    var64 = np.maximum(sum_sq / count - np.square(mean64), 0.0)
    mean = mean64.astype(np.float32)
    std = np.sqrt(var64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((data.astype(np.float32) - mean) / std).astype(np.float32)


def compute_class_weights(
    train_labels: list[np.ndarray],
    num_classes: int,
    mode: str,
    manual_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    if mode == "none":
        return torch.ones(num_classes)
    if mode == "manual" and manual_weights is not None:
        return torch.tensor(manual_weights, dtype=torch.float32)
    if mode == "balanced":
        all_labels = np.concatenate([lbl.flatten() for lbl in train_labels])
        counts = np.bincount(all_labels.astype(np.int64), minlength=num_classes)
        counts = np.maximum(counts, 1)
        weights = len(all_labels) / (num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)
    raise ValueError(f"Unknown class_weight_mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    """Sliding-window sequence dataset preserving local temporal order."""

    def __init__(
        self,
        subject_arrays: list[np.ndarray],
        subject_labels: list[np.ndarray],
        sequence_length: int,
        stride: int,
    ) -> None:
        self.subject_arrays = [np.asarray(arr, dtype=np.float32) for arr in subject_arrays]
        self.subject_labels = [np.asarray(lbl, dtype=np.int64).flatten() for lbl in subject_labels]
        self.index: list[tuple[int, int]] = []
        self.sequence_length = sequence_length
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        for subject_index, (data, labels) in enumerate(zip(self.subject_arrays, self.subject_labels)):
            if data.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Data-label mismatch: {data.shape[0]} samples vs {labels.shape[0]} labels"
                )
            if data.shape[0] < sequence_length:
                continue
            for start in range(0, data.shape[0] - sequence_length + 1, stride):
                self.index.append((subject_index, start))

        if not self.index:
            raise ValueError("No sequence windows were created.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject_index, start = self.index[index]
        end = start + self.sequence_length
        x = self.subject_arrays[subject_index][start:end]
        y = self.subject_labels[subject_index][start:end]
        return torch.from_numpy(x), torch.from_numpy(y).long()


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_loaders_for_fold(
    subjects: dict[str, dict[str, Any]],
    test_key: str,
    config: ClassifyConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, ...], dict[str, Any]]:
    train_arrays: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    val_arrays: list[np.ndarray] = []
    val_labels: list[np.ndarray] = []
    test_arrays: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    subject_keys = ordered_subject_keys(subjects)
    train_candidate_keys = [key for key in subject_keys if key != test_key]
    validation_subjects = (
        choose_validation_subjects(train_candidate_keys, subjects, config)
        if config.val_strategy == "subject"
        else set()
    )

    for subject_key in subject_keys:
        data = np.asarray(subjects[subject_key]["data"], dtype=np.float32)
        labels = class_labels_to_unit_interval(subjects[subject_key]["label"], num_classes=config.num_classes)
        if data.ndim < 2:
            raise ValueError(f"{subject_key} data must be at least 2D, got {data.shape}")

        if subject_key == test_key:
            test_arrays.append(data)
            test_labels.append(labels)
        elif subject_key in validation_subjects:
            val_arrays.append(data)
            val_labels.append(labels)
        else:
            if config.val_strategy == "time":
                tr_x, tr_y, va_x, va_y = split_train_val_subject(
                    data, labels, val_fraction=config.val_fraction,
                )
                train_arrays.append(tr_x)
                train_labels.append(tr_y)
                val_arrays.append(va_x)
                val_labels.append(va_y)
            elif config.val_strategy == "subject":
                train_arrays.append(data)
                train_labels.append(labels)
            else:
                raise ValueError("val_strategy must be 'subject' or 'time'")

    mean, std = compute_train_stats(train_arrays)
    train_arrays = [normalize(arr, mean, std) for arr in train_arrays]
    val_arrays = [normalize(arr, mean, std) for arr in val_arrays]
    test_arrays = [normalize(arr, mean, std) for arr in test_arrays]

    train_dataset = SequenceDataset(
        train_arrays, train_labels,
        sequence_length=config.sequence_length, stride=config.train_stride,
    )
    val_dataset = SequenceDataset(
        val_arrays, val_labels,
        sequence_length=config.sequence_length, stride=config.eval_stride,
    )
    test_dataset = SequenceDataset(
        test_arrays, test_labels,
        sequence_length=config.sequence_length, stride=config.eval_stride,
    )

    # Weighted sampling for class imbalance
    if config.use_weighted_sampler:
        # Compute per-window class weights (mode of each window's labels)
        window_labels = []
        for subj_idx, start in train_dataset.index:
            y_win = train_dataset.subject_labels[subj_idx][start:start + train_dataset.sequence_length]
            window_labels.append(int(np.bincount(y_win, minlength=config.num_classes).argmax()))
        window_labels = np.array(window_labels)
        class_counts = np.bincount(window_labels, minlength=config.num_classes)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[window_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=config.num_workers, drop_last=False,
            pin_memory=config.pin_memory and config.device.startswith("cuda"),
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, drop_last=False,
            pin_memory=config.pin_memory and config.device.startswith("cuda"),
        )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=False,
        pin_memory=config.pin_memory and config.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=False,
        pin_memory=config.pin_memory and config.device.startswith("cuda"),
    )

    feature_shape = tuple(train_dataset[0][0].shape[1:])
    fold_info = {
        "test_subject": test_key,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "feature_shape": feature_shape,
        "normalization": {"mean_shape": tuple(mean.shape), "std_shape": tuple(std.shape)},
        "seedvig_feature_mode": config.seedvig_feature_mode,
        "val_strategy": config.val_strategy,
        "val_selection_strategy": config.val_selection_strategy,
        "validation_subjects": sorted(validation_subjects),
    }
    fold_info["runtime_arrays"] = {
        "mean": mean,
        "std": std,
        "val_arrays": val_arrays,
        "val_labels": val_labels,
        "train_arrays": train_arrays,
        "train_labels": train_labels,
        "test_arrays": test_arrays,
        "test_labels": test_labels,
    }
    return train_loader, val_loader, test_loader, feature_shape, fold_info


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def classification_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
) -> dict[str, float]:
    """Compute comprehensive classification metrics."""
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)

    # Accuracy
    accuracy = float(np.mean(pred == target))

    # Per-class metrics and confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[c, :].sum()),
        }

    # Macro averages
    macro_precision = float(np.mean([per_class[c]["precision"] for c in range(num_classes)]))
    macro_recall = float(np.mean([per_class[c]["recall"] for c in range(num_classes)]))
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in range(num_classes)]))

    # Weighted averages
    total_support = max(int(cm.sum()), 1)
    weighted_precision = float(sum(
        per_class[c]["precision"] * per_class[c]["support"] for c in range(num_classes)
    ) / total_support)
    weighted_recall = float(sum(
        per_class[c]["recall"] * per_class[c]["support"] for c in range(num_classes)
    ) / total_support)
    weighted_f1 = float(sum(
        per_class[c]["f1"] * per_class[c]["support"] for c in range(num_classes)
    ) / total_support)

    # Cohen's kappa
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / cm.sum()
    kappa = (accuracy - (expected.diagonal().sum() / cm.sum())) / (
        1.0 - (expected.diagonal().sum() / cm.sum())
    ) if cm.sum() > 0 and (1.0 - expected.diagonal().sum() / cm.sum()) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "kappa": float(kappa),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def validation_score(metrics: dict[str, Any], metric_name: str) -> float:
    """Return a minimization score for checkpoint selection."""
    if metric_name == "macro_f1":
        return -float(metrics.get("macro_f1", 0.0))
    if metric_name == "accuracy":
        return -float(metrics.get("accuracy", 0.0))
    if metric_name == "kappa":
        return -float(metrics.get("kappa", 0.0))
    value = float(metrics.get(metric_name, float("inf")))
    return value if not np.isfinite(value) else -value


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: ClassifyNTLDFE,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ClassifyConfig,
    scaler: Any | None = None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    amp_enabled = config.use_amp and device.type == "cuda"
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        # Mild Gaussian noise augmentation
        if config.aug_noise_std > 0:
            x = x + torch.randn_like(x) * config.aug_noise_std

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
            # Flatten per-time-step logits for per-step loss
            loss = criterion(
                outputs["logits"].reshape(-1, config.num_classes), y.reshape(-1),
            )
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def evaluate_window(model: ClassifyNTLDFE, loader: DataLoader, device: torch.device, num_classes: int = 3) -> dict[str, Any]:
    """Window-level evaluation (used for debugging/optimization tracking)."""
    model.eval()
    all_pred: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        outputs = model(x)
        pred = outputs["logits"].argmax(dim=-1).cpu().numpy()
        all_pred.append(pred)
        all_y.append(y.numpy())
    pred = np.concatenate(all_pred)
    target = np.concatenate(all_y)
    return classification_metrics(pred, target, num_classes=num_classes)


@torch.no_grad()
def predict_full_sequence(
    model: ClassifyNTLDFE,
    data: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    sequence_length: int,
    batch_size: int,
    prediction_stride: int = 1,
    num_classes: int = 3,
) -> dict[str, np.ndarray]:
    """Predict one subject with SOFT voting (probability averaging).

    Instead of hard-vote argmax, averages probability distributions across
    overlapping windows and then argmaxes the result. This preserves confidence
    information that hard voting loses.
    """
    model.eval()
    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).flatten()
    n_samples = data.shape[0]
    if n_samples < sequence_length:
        raise ValueError("Subject sequence shorter than sequence_length.")

    prediction_stride = max(1, int(prediction_stride))
    starts = list(range(0, n_samples - sequence_length + 1, prediction_stride))
    last_start = n_samples - sequence_length
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    prob_sum = np.zeros((n_samples, num_classes), dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.float64)

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start: batch_start + batch_size]
        batch = np.stack([data[start: start + sequence_length] for start in batch_starts])
        x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        outputs = model(x)
        logits = outputs["logits"].detach().cpu().numpy()  # (batch, seq_len, 3)

        # Softmax → probabilities per time step
        logits_shifted = logits - logits.max(axis=-1, keepdims=True)
        ex = np.exp(logits_shifted)
        probs = ex / ex.sum(axis=-1, keepdims=True)  # (batch, seq_len, 3)

        for local_index, start in enumerate(batch_starts):
            end = start + sequence_length
            for c in range(num_classes):
                prob_sum[start:end, c] += probs[local_index, :, c]
            counts[start:end] += 1.0

    valid = counts > 0
    # Average probabilities, then argmax
    avg_probs = np.zeros((n_samples, num_classes), dtype=np.float64)
    avg_probs[valid] = prob_sum[valid] / counts[valid, np.newaxis]
    pred = avg_probs.argmax(axis=-1).astype(np.int64)

    return {
        "pred": pred.astype(np.int64),
        "target": labels.astype(np.int64),
        "valid": valid,
    }


def evaluate_trajectories(
    model: ClassifyNTLDFE,
    arrays: list[np.ndarray],
    labels_list: list[np.ndarray],
    device: torch.device,
    config: ClassifyConfig,
    prediction_stride: int = 1,
) -> dict[str, Any]:
    """Evaluate full trajectories over one or more subjects."""
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    per_subject: list[dict[str, float]] = []

    for data, labels in zip(arrays, labels_list):
        trajectory = predict_full_sequence(
            model, data, labels,
            device=device,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            prediction_stride=prediction_stride,
            num_classes=config.num_classes,
        )
        valid = trajectory["valid"]
        subj_metrics = classification_metrics(
            trajectory["pred"][valid],
            trajectory["target"][valid],
            num_classes=config.num_classes,
        )
        per_subject.append(strip_arrays(subj_metrics))
        pred_parts.append(trajectory["pred"][valid])
        target_parts.append(trajectory["target"][valid])

    pred = np.concatenate(pred_parts) if pred_parts else np.array([], dtype=np.int64)
    target = np.concatenate(target_parts) if target_parts else np.array([], dtype=np.int64)
    if pred.size == 0:
        raise ValueError("No valid trajectory predictions were produced.")

    metrics = classification_metrics(pred, target, num_classes=config.num_classes)
    metrics["per_subject"] = per_subject
    return metrics


def strip_arrays(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove ndarray fields for JSON serialization; keep per_class and cm."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            out[key] = float(value) if np.isfinite(value) else None
        elif key in ("per_class", "confusion_matrix"):
            out[key] = value
    return out


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Fold training
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(
    subjects: dict[str, dict[str, Any]],
    test_key: str,
    config: ClassifyConfig,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(config.device)
    train_loader, val_loader, test_loader, feature_shape, fold_info = make_loaders_for_fold(
        subjects, test_key, config,
    )
    runtime_arrays = fold_info.pop("runtime_arrays")

    # Compute class weights for focal loss alpha
    class_weights = compute_class_weights(
        runtime_arrays["train_labels"],
        num_classes=config.num_classes,
        mode=config.class_weight_mode,
        manual_weights=config.class_weights,
    )

    model = ClassifyNTLDFE(config, feature_shape).to(device)

    # Focal Loss with class-balanced alpha
    focal_alpha = class_weights.to(device) if config.class_weight_mode != "none" else None
    criterion = FocalLoss(
        gamma=config.focal_gamma,
        alpha=focal_alpha,
        label_smoothing=config.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Cosine annealing with linear warmup
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    if config.use_scheduler:
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps),
            eta_min=config.learning_rate * 0.01,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[max(1, warmup_steps)],
        )
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    no_improve = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device, config, scaler=scaler,
        )
        val_metrics = evaluate_trajectories(
            model,
            runtime_arrays["val_arrays"],
            runtime_arrays["val_labels"],
            device,
            config,
            prediction_stride=config.selection_prediction_stride,
        )
        val_score = validation_score(val_metrics, config.validation_metric)

        # Step the per-step scheduler (cosine + warmup)
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train": train_losses,
            "val": strip_arrays(val_metrics),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        can_update_best = epoch >= config.min_epochs
        if can_update_best and val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        elif can_update_best:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{test_key}] epoch {epoch:03d} "
                f"train_loss={train_losses['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if can_update_best and no_improve >= config.patience:
            break

    if best_state is None:
        best_epoch = config.epochs
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Evaluate on test set
    test_window_metrics = evaluate_window(model, test_loader, device, num_classes=config.num_classes)
    test_metrics = evaluate_trajectories(
        model,
        runtime_arrays["test_arrays"],
        runtime_arrays["test_labels"],
        device,
        config,
    )

    fold_dir = output_dir / f"fold_{test_key}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    if config.save_predictions:
        trajectory = predict_full_sequence(
            model,
            runtime_arrays["test_arrays"][0],
            runtime_arrays["test_labels"][0],
            device=device,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            num_classes=config.num_classes,
        )
        np.savez_compressed(
            fold_dir / "trajectory_predictions.npz",
            pred=trajectory["pred"],
            target=trajectory["target"],
            valid=trajectory["valid"],
        )

    result = {
        "test_subject": test_key,
        "best_epoch": best_epoch,
        "fold_info": fold_info,
        "class_weights": class_weights.tolist(),
        "test": strip_arrays(test_metrics),
        "test_window_debug": strip_arrays(test_window_metrics),
        "best_val": strip_arrays(
            next((h["val"] for h in history if h["epoch"] == best_epoch), {})
        ),
        "history": history if config.save_history else [],
    }
    save_json(fold_dir / "metrics.json", result)

    print(
        f"[{test_key}] TEST acc={result['test']['accuracy']:.4f} "
        f"f1={result['test']['macro_f1']:.4f} "
        f"kappa={result['test']['kappa']:.4f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Summarisation
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std/min/max over folds for each scalar metric."""
    # Collect scalar metric names from the first fold's test dict
    sample = results[0]["test"]
    metric_names = [k for k in sample if isinstance(sample[k], (int, float)) and sample[k] is not None]
    summary: dict[str, Any] = {"folds": len(results), "metrics": {}}
    for name in metric_names:
        values = np.asarray([fold["test"].get(name, np.nan) for fold in results], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary["metrics"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    # Aggregate confusion matrix
    agg_cm = np.zeros((3, 3), dtype=np.int64)
    for fold in results:
        cm = fold["test"].get("confusion_matrix", [])
        if len(cm) == 3:
            agg_cm += np.asarray(cm, dtype=np.int64)
    summary["aggregate_confusion_matrix"] = agg_cm.tolist()
    return summary


def run_loso(config: ClassifyConfig = CFG) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir) / config.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(config))

    subjects = load_subjects(config)
    audit = audit_subjects(subjects, config)
    keys = ordered_subject_keys(subjects)
    if config.max_folds is not None:
        keys = keys[: config.max_folds]

    print(f"Dataset: {config.dataset_name} (classification)")
    print(f"Subjects loaded: {len(subjects)} | folds to run: {len(keys)}")
    print(f"Classes: {audit.num_classes}")
    for c in range(config.num_classes):
        print(f"  Class {c} ({audit.class_names[c]}): {audit.class_counts[c]} samples")
    print(f"Feature mode: {config.seedvig_feature_mode}")
    for key in ordered_subject_keys(subjects):
        data = np.asarray(subjects[key]["data"])
        labels = np.asarray(subjects[key]["label"])
        raw_shape = subjects[key].get("raw_data_shape")
        if raw_shape is not None:
            print(f"  {key}: raw={raw_shape} prepared={data.shape} label={labels.shape}")
        else:
            print(f"  {key}: data={data.shape} label={labels.shape}")

    results_by_key: dict[str, dict[str, Any]] = {}
    if config.resume_completed_folds:
        for test_key in keys:
            metrics_path = output_dir / f"fold_{test_key}" / "metrics.json"
            if not metrics_path.exists():
                continue
            with metrics_path.open("r", encoding="utf-8") as handle:
                saved_result = json.load(handle)
            if saved_result.get("test_subject") == test_key and "test" in saved_result:
                results_by_key[test_key] = saved_result
                print(f"[{test_key}] resume: loaded existing fold metrics")

    for test_key in keys:
        if test_key in results_by_key:
            continue
        print(f"\n=== LOSO fold: test={test_key} ===")
        result = train_fold(subjects, test_key, config, output_dir)
        results_by_key[test_key] = result
        if config.save_partial_results:
            partial_results = [
                results_by_key[key] for key in keys if key in results_by_key
            ]
            partial_summary = summarize_results(partial_results)
            save_json(
                output_dir / "partial_results.json",
                {"summary": partial_summary, "results": partial_results},
            )

    results = [results_by_key[key] for key in keys if key in results_by_key]
    summary = summarize_results(results)
    payload = {
        "summary": summary,
        "dataset_audit": asdict(audit),
        "results": results,
    }
    save_json(output_dir / "loso_results.json", payload)

    print("\n=== LOSO Classification Summary ===")
    for metric, values in summary["metrics"].items():
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print(f"\nAggregate Confusion Matrix:")
    cm = np.array(summary["aggregate_confusion_matrix"])
    print(f"         Pred0  Pred1  Pred2")
    for i, label in enumerate(["True0", "True1", "True2"]):
        print(f"  {label}  {cm[i,0]:5d}  {cm[i,1]:5d}  {cm[i,2]:5d}")
    return payload


if __name__ == "__main__":
    run_loso(CFG)
