"""Complete LOSO training pipeline for NT-LDFE.

Edit the Config dataclass below to change all parameters. This script does not
use terminal command-line arguments.

Supported loaders:
    - mydata_dataset.py: data shape per subject (samples, 31, 5)
    - seed_dataloader.py: data shape per subject (samples, 1600, 17)

Both are treated as ordered per-window feature matrices. Sliding windows are
created within each subject only, preserving temporal order and avoiding
cross-subject leakage.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mydata_dataset import load_my_fatigue
from nt_ldfe_model import NTLDFE, NTLDFEConfig, NTLDFELoss, NTLDFELossConfig
from seed_dataloader import find_subject_dirs, load_subject as load_seedvig_subject


@dataclass
class Config:
    """All user-editable experiment parameters."""

    dataset_name: str = "seedvig"  # "my_fatigue" or "seedvig"
    my_fatigue_root: str = "dataset/My_fatigue/My_fatigue"
    seedvig_root: str = "dataset/seedvig"
    output_dir: str = "outputs/nt_ldfe_loso_v2"

    seed: int = 42
    max_folds: int | None = None  # set small integer for a quick pilot
    val_fraction: float = 0.2
    val_subject_count: int = 3
    val_selection_strategy: str = "stratified_mean"  # "stratified_mean" or "random"
    val_strategy: str = "subject"  # "subject" or "time"
    sequence_length: int = 45
    train_stride: int = 5
    eval_stride: int = 45
    selection_prediction_stride: int = 5
    checkpoint_pool_size: int = 5

    # SEED-VIG loader returns samples shaped (1600, 17). Treating that whole
    # matrix as one feature is memory-heavy and keeps the GPU underfed. The
    # default converts each epoch to compact channel x spectral-DE bands,
    # giving fatigue-relevant frequency features without adding modalities.
    seedvig_feature_mode: str = "hybrid_de"  # "hybrid_de", "spectral_de", "temporal_stats", "temporal_logvar", or "as_is"
    seedvig_de_bins: int = 5
    seedvig_sampling_rate: float = 200.0
    feature_eps: float = 1e-6

    batch_size: int = 48
    epochs: int = 220
    min_epochs: int = 10
    patience: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0

    temporal_type: str = "gru"  # "gru" or "tcn"
    encoder_dim: int = 96
    encoder_layers: int = 2
    temporal_hidden_dim: int = 128
    temporal_layers: int = 2
    regression_hidden_dim: int = 96
    dropout: float = 0.1
    bidirectional: bool = True
    use_feature_attention: bool = True
    mu_activation: str = "tanh"

    lambda_nll: float = 0.15
    lambda_reg: float = 1.0
    lambda_corr: float = 0.08
    lambda_ccc: float = 0.25
    lambda_std: float = 0.20
    lambda_mean: float = 0.10
    beta_smooth: float = 1e-4
    beta_acc: float = 2e-4
    lambda_var: float = 1e-5
    train_label_smoothing_window: int = 3
    log_var_min: float = -4.0
    log_var_max: float = 1.0

    # Optional enhancement modules from final_methods.md.
    enable_soft_label: bool = False
    eta_soft: float = 0.1
    num_soft_levels: int = 5
    soft_tau: float = 0.1

    enable_ranking: bool = False
    gamma_rank: float = 0.05
    rank_margin: float = 0.05
    rank_delta: float = 0.1
    rank_gap: int = 4

    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = True
    use_scheduler: bool = True
    refit_on_train_val: bool = True
    refit_epochs: int = 5
    refit_learning_rate_scale: float = 0.35
    refit_patience: int = 0
    use_val_calibration: bool = True
    report_calibrated_as_primary: bool = True
    calibration_mode: str = "best"  # "none", "affine", "mean_shift", or "best"
    calibration_min_improvement: float = 1e-4
    calibration_requires_rmse_and_ccc: bool = True
    calibration_scale_bounds: tuple[float, float] = (0.5, 2.0)
    calibration_bias_bounds: tuple[float, float] = (-0.3, 0.3)
    calibration_ridge: float = 1e-2
    mean_shift_bounds: tuple[float, float] = (-0.35, 0.35)
    calibrate_log_var: bool = True
    conservative_log_var_calibration: bool = True
    log_var_offset_bounds: tuple[float, float] = (-1.5, 4.0)
    validation_metric: str = "rmse_pearson_ccc_r2_std"
    post_smooth_window: int = 1
    save_checkpoints: bool = False
    save_predictions: bool = False
    save_history: bool = True
    save_partial_results: bool = True
    resume_completed_folds: bool = True
    tda_epsilon: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetAudit:
    """Small audit block saved with each run to keep metric claims honest."""

    subjects: int
    label_min: float
    label_max: float
    label_mean: float
    label_std: float
    unique_rounded_labels: int
    is_effectively_binary: bool
    note: str


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def labels_to_unit_interval(labels: np.ndarray) -> np.ndarray:
    """Keep PERCLOS labels in [0, 1], accepting either [0, 1] or [0, 100]."""
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    finite = np.isfinite(labels)
    if not finite.all():
        raise ValueError("Labels contain NaN or Inf.")
    max_value = float(labels.max()) if labels.size else 1.0
    if max_value > 1.5:
        labels = labels / 100.0
    return np.clip(labels, 0.0, 1.0).astype(np.float32)


def seedvig_to_de_like(data: np.ndarray, bins: int, eps: float = 1e-6) -> np.ndarray:
    """Convert SEED-VIG raw epochs to compact DE-like log-variance features.

    The current seed_dataloader reads each .mat `eeg_data` as (1600, 17).
    A direct linear projection over 27,200 values per sample is slow and
    memory-heavy. This function splits the 1600 time points into `bins`
    contiguous segments and computes log variance per channel in each segment.

    Output shape:
        (samples, channels=17, bins)
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"SEED-VIG data should be 3D, got {data.shape}")
    n_samples, n_time, n_channels = data.shape
    if bins < 1 or bins > n_time:
        raise ValueError(f"seedvig_de_bins must be in [1, {n_time}], got {bins}")

    segments = np.array_split(np.arange(n_time), bins)
    features = np.empty((n_samples, n_channels, bins), dtype=np.float32)
    for bin_index, segment in enumerate(segments):
        segment_data = data[:, segment, :]
        features[:, :, bin_index] = np.log(np.var(segment_data, axis=1) + eps)
    return features


def seedvig_to_temporal_stats(data: np.ndarray, bins: int, eps: float = 1e-6) -> np.ndarray:
    """Convert SEED-VIG epochs to compact statistics from existing EEG only.

    For each channel and temporal segment, this keeps log variance as the
    DE-like core and appends mean, standard deviation, and line length. These
    are deterministic transforms of the available EEG sample, not new data.

    Output shape:
        (samples, channels=17, bins * 4)
    """
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


def seedvig_to_spectral_de(
    data: np.ndarray,
    sampling_rate: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert SEED-VIG epochs to channel x EEG-band DE-like features.

    This uses only the available EEG sample. Each epoch is transformed by an
    FFT, then band power is integrated for delta/theta/alpha/beta/gamma-like
    ranges and converted with log power, the same statistic family as DE.

    Output shape:
        (samples, channels=17, bands=5)
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"SEED-VIG data should be 3D, got {data.shape}")
    n_samples, n_time, n_channels = data.shape
    centered = data - np.mean(data, axis=1, keepdims=True)
    window = np.hanning(n_time).astype(np.float32).reshape(1, n_time, 1)
    fft = np.fft.rfft(centered * window, axis=1)
    power = (np.abs(fft) ** 2).astype(np.float64) / max(n_time, 1)
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sampling_rate)
    bands = [
        (1.0, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
        (30.0, 45.0),
    ]
    features = np.empty((n_samples, n_channels, len(bands)), dtype=np.float32)
    for band_index, (low, high) in enumerate(bands):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            raise ValueError(f"No FFT bins found for band {low}-{high} Hz")
        band_power = np.mean(power[:, mask, :], axis=1)
        features[:, :, band_index] = np.log(band_power + eps).astype(np.float32)
    return features


def prepare_subject_data(data: np.ndarray, config: Config) -> np.ndarray:
    """Apply dataset-specific feature compaction before splitting."""
    data = np.asarray(data, dtype=np.float32)
    if config.dataset_name == "seedvig" and config.seedvig_feature_mode != "as_is":
        if config.seedvig_feature_mode == "hybrid_de":
            spectral = seedvig_to_spectral_de(
                data,
                sampling_rate=config.seedvig_sampling_rate,
                eps=config.feature_eps,
            )
            temporal = seedvig_to_temporal_stats(
                data,
                bins=config.seedvig_de_bins,
                eps=config.feature_eps,
            )
            return np.concatenate([spectral, temporal], axis=2).astype(np.float32)
        if config.seedvig_feature_mode == "spectral_de":
            return seedvig_to_spectral_de(
                data,
                sampling_rate=config.seedvig_sampling_rate,
                eps=config.feature_eps,
            )
        if config.seedvig_feature_mode == "temporal_logvar":
            return seedvig_to_de_like(data, bins=config.seedvig_de_bins, eps=config.feature_eps)
        if config.seedvig_feature_mode == "temporal_stats":
            return seedvig_to_temporal_stats(
                data,
                bins=config.seedvig_de_bins,
                eps=config.feature_eps,
            )
        raise ValueError(f"Unknown seedvig_feature_mode: {config.seedvig_feature_mode}")
    return data


def load_subjects(config: Config) -> dict[str, dict[str, Any]]:
    if config.dataset_name == "my_fatigue":
        return load_my_fatigue(root=config.my_fatigue_root, stack=True, strict=False)
    if config.dataset_name == "seedvig":
        subjects: dict[str, dict[str, Any]] = {}
        for subject_index, subject_dir in enumerate(find_subject_dirs(config.seedvig_root), start=1):
            subject_key = f"sub{subject_index}"
            subject = load_seedvig_subject(subject_dir, stack=True)
            subject["raw_data_shape"] = tuple(np.asarray(subject["data"]).shape)
            if config.seedvig_feature_mode != "as_is":
                subject["data"] = prepare_subject_data(subject["data"], config)
            subjects[subject_key] = subject
        if config.seedvig_feature_mode != "as_is":
            print(
                f"Prepared SEED-VIG with {config.seedvig_feature_mode}: "
                f"raw epochs -> compact channel-stat features"
            )
        return subjects
    raise ValueError("dataset_name must be 'my_fatigue' or 'seedvig'")


def ordered_subject_keys(subjects: dict[str, dict[str, Any]]) -> list[str]:
    def key_fn(name: str) -> tuple[int, str]:
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 10**9, name)

    return sorted(subjects.keys(), key=key_fn)


def audit_subjects(subjects: dict[str, dict[str, Any]]) -> DatasetAudit:
    labels = [
        labels_to_unit_interval(subject["label"])
        for subject in subjects.values()
    ]
    all_labels = np.concatenate(labels) if labels else np.asarray([], dtype=np.float32)
    rounded_unique = int(np.unique(np.round(all_labels, 3)).size) if all_labels.size else 0
    is_binary = rounded_unique <= 2
    note = (
        "Labels look binary/discrete; report continuous-regression results as "
        "stage-score regression, not true continuous PERCLOS trajectory estimation."
        if is_binary
        else "Labels contain many values and are compatible with continuous PERCLOS regression."
    )
    return DatasetAudit(
        subjects=len(subjects),
        label_min=float(all_labels.min()) if all_labels.size else float("nan"),
        label_max=float(all_labels.max()) if all_labels.size else float("nan"),
        label_mean=float(all_labels.mean()) if all_labels.size else float("nan"),
        label_std=float(all_labels.std()) if all_labels.size else float("nan"),
        unique_rounded_labels=rounded_unique,
        is_effectively_binary=is_binary,
        note=note,
    )


def split_train_val_subject(
    data: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-forward split inside one training subject."""
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
    config: Config,
) -> set[str]:
    """Choose complete validation subjects from training subjects only."""
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
        subject_means: list[tuple[str, float]] = []
        for key in train_keys:
            labels = labels_to_unit_interval(subjects[key]["label"])
            mean_label = float(np.nanmean(labels)) if labels.size else float("nan")
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


def smooth_training_labels(labels: np.ndarray, window: int) -> np.ndarray:
    """Optional training-only label smoothing for noisy PERCLOS observations."""
    labels = labels_to_unit_interval(labels)
    if window <= 1 or labels.size < 3:
        return labels
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(labels, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_train_stats(train_arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise mean/std using training samples only."""
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
            assert sum_sq is not None
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
        self.subject_labels = [labels_to_unit_interval(labels) for labels in subject_labels]
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
            raise ValueError(
                "No sequence windows were created. Reduce sequence_length or check data."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject_index, start = self.index[index]
        end = start + self.sequence_length
        x = self.subject_arrays[subject_index][start:end]
        y = self.subject_labels[subject_index][start:end]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders_for_fold(
    subjects: dict[str, dict[str, Any]],
    test_key: str,
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, ...], dict[str, Any]]:
    train_arrays: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    val_arrays: list[np.ndarray] = []
    val_labels: list[np.ndarray] = []
    test_arrays: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    baseline_train_labels: list[np.ndarray] = []
    subject_keys = ordered_subject_keys(subjects)
    train_candidate_keys = [key for key in subject_keys if key != test_key]
    validation_subjects = (
        choose_validation_subjects(train_candidate_keys, subjects, config)
        if config.val_strategy == "subject"
        else set()
    )
    if config.val_strategy == "subject":
        validation_anchor_labels = [
            labels_to_unit_interval(subjects[key]["label"])
            for key in validation_subjects
        ]
        if validation_anchor_labels:
            val_train_label_mean = float(np.mean(np.concatenate(validation_anchor_labels)))
        else:
            val_train_label_mean = float("nan")
    else:
        val_train_label_mean = float("nan")

    for subject_key in subject_keys:
        data = np.asarray(subjects[subject_key]["data"], dtype=np.float32)
        labels = labels_to_unit_interval(subjects[subject_key]["label"])
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
                    data,
                    labels,
                    val_fraction=config.val_fraction,
                )
                train_arrays.append(tr_x)
                baseline_train_labels.append(labels_to_unit_interval(tr_y))
                train_labels.append(smooth_training_labels(tr_y, config.train_label_smoothing_window))
                val_arrays.append(va_x)
                val_labels.append(va_y)
            elif config.val_strategy == "subject":
                train_arrays.append(data)
                baseline_train_labels.append(labels_to_unit_interval(labels))
                train_labels.append(smooth_training_labels(labels, config.train_label_smoothing_window))
            else:
                raise ValueError("val_strategy must be 'subject' or 'time'")

    mean, std = compute_train_stats(train_arrays)
    train_label_mean = float(np.mean(np.concatenate(baseline_train_labels)))
    if not np.isfinite(val_train_label_mean):
        val_train_label_mean = train_label_mean
    train_arrays = [normalize(arr, mean, std) for arr in train_arrays]
    val_arrays = [normalize(arr, mean, std) for arr in val_arrays]
    test_arrays = [normalize(arr, mean, std) for arr in test_arrays]

    train_dataset = SequenceDataset(
        train_arrays,
        train_labels,
        sequence_length=config.sequence_length,
        stride=config.train_stride,
    )
    val_dataset = SequenceDataset(
        val_arrays,
        val_labels,
        sequence_length=config.sequence_length,
        stride=config.eval_stride,
    )
    test_dataset = SequenceDataset(
        test_arrays,
        test_labels,
        sequence_length=config.sequence_length,
        stride=config.eval_stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.pin_memory and config.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.pin_memory and config.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.pin_memory and config.device.startswith("cuda"),
    )

    feature_shape = tuple(train_dataset[0][0].shape[1:])
    fold_info = {
        "test_subject": test_key,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "feature_shape": feature_shape,
        "normalization": {
            "mean_shape": tuple(mean.shape),
            "std_shape": tuple(std.shape),
        },
        "seedvig_feature_mode": config.seedvig_feature_mode,
        "seedvig_sampling_rate": config.seedvig_sampling_rate,
        "val_strategy": config.val_strategy,
        "val_selection_strategy": config.val_selection_strategy,
        "validation_subjects": sorted(validation_subjects),
        "train_label_smoothing_window": config.train_label_smoothing_window,
        "selection_prediction_stride": config.selection_prediction_stride,
        "checkpoint_pool_size": config.checkpoint_pool_size,
        "min_epochs": config.min_epochs,
        "refit_on_train_val": config.refit_on_train_val,
        "refit_epochs": config.refit_epochs,
        "calibration_mode": config.calibration_mode,
        "calibrate_log_var": config.calibrate_log_var,
        "val_train_label_mean": val_train_label_mean,
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
        "train_label_mean": train_label_mean,
        "val_train_label_mean": val_train_label_mean,
    }
    return train_loader, val_loader, test_loader, feature_shape, fold_info


def build_model(feature_shape: tuple[int, ...], config: Config) -> NTLDFE:
    model_config = NTLDFEConfig(
        feature_shape=feature_shape,
        encoder_dim=config.encoder_dim,
        encoder_layers=config.encoder_layers,
        temporal_hidden_dim=config.temporal_hidden_dim,
        temporal_layers=config.temporal_layers,
        regression_hidden_dim=config.regression_hidden_dim,
        dropout=config.dropout,
        temporal_type=config.temporal_type,
        bidirectional=config.bidirectional if config.temporal_type.lower() == "gru" else False,
        use_feature_attention=config.use_feature_attention,
        mu_activation=config.mu_activation,
        num_soft_levels=config.num_soft_levels if config.enable_soft_label else 0,
        log_var_min=config.log_var_min,
        log_var_max=config.log_var_max,
    )
    return NTLDFE(model_config)


def build_loss(config: Config) -> NTLDFELoss:
    loss_config = NTLDFELossConfig(
        lambda_nll=config.lambda_nll,
        lambda_reg=config.lambda_reg,
        lambda_corr=config.lambda_corr,
        lambda_ccc=config.lambda_ccc,
        lambda_std=config.lambda_std,
        lambda_mean=config.lambda_mean,
        beta_smooth=config.beta_smooth,
        beta_acc=config.beta_acc,
        lambda_var=config.lambda_var,
        eta_soft=config.eta_soft if config.enable_soft_label else 0.0,
        gamma_rank=config.gamma_rank if config.enable_ranking else 0.0,
        num_soft_levels=config.num_soft_levels,
        soft_tau=config.soft_tau,
        rank_margin=config.rank_margin,
        rank_delta=config.rank_delta,
        rank_gap=config.rank_gap,
    )
    return NTLDFELoss(loss_config)


def concordance_corrcoef(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.size < 2:
        return float("nan")
    pred_mean = pred.mean()
    target_mean = target.mean()
    covariance = np.mean((pred - pred_mean) * (target - target_mean))
    pred_var = np.var(pred)
    target_var = np.var(target)
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    if denominator <= 1e-12:
        return float("nan")
    return float((2.0 * covariance) / denominator)


def safe_corr(pred: np.ndarray, target: np.ndarray, method: str) -> float:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.size < 3 or np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return float("nan")
    try:
        from scipy.stats import pearsonr, spearmanr

        if method == "pearson":
            return float(pearsonr(pred, target).statistic)
        if method == "spearman":
            return float(spearmanr(pred, target).statistic)
    except Exception:
        return float("nan")
    raise ValueError(method)


def sequence_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    log_var: np.ndarray,
    tda_epsilon: float = 1e-4,
) -> dict[str, float]:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    log_var = log_var.astype(np.float64)
    err = pred - target
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    target_var = float(np.var(target))
    r2 = float(1.0 - mse / target_var) if target_var > 1e-12 else float("nan")
    var = np.exp(log_var)
    nll_train = float(np.mean(0.5 * np.exp(-log_var) * err**2 + 0.5 * log_var))
    nll_gaussian = float(
        np.mean(0.5 * np.log(2.0 * np.pi * var) + 0.5 * err**2 / var)
    )
    sigma = np.sqrt(var)
    abs_err = np.abs(err)
    sigma_abs_err_corr = safe_corr(sigma, abs_err, "spearman")

    if pred.ndim == 2 and pred.shape[1] > 1:
        tv = float(np.mean(np.abs(np.diff(pred, axis=1))))
        if pred.shape[1] > 2:
            adv = float(np.mean(np.abs(pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2])))
        else:
            adv = 0.0
        pred_diff = np.diff(pred, axis=1)
        target_diff = np.diff(target, axis=1)
        valid = np.abs(target_diff) > tda_epsilon
        if np.any(valid):
            tda = float(np.mean(np.sign(pred_diff[valid]) == np.sign(target_diff[valid])))
        else:
            tda = float("nan")
    else:
        tv = 0.0
        adv = 0.0
        tda = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "pearson": safe_corr(pred, target, "pearson"),
        "spearman": safe_corr(pred, target, "spearman"),
        "ccc": concordance_corrcoef(pred, target),
        "nll_train_objective": nll_train,
        "nll_gaussian": nll_gaussian,
        "tv": tv,
        "adv": adv,
        "tda": tda,
        "sigma_abs_err_spearman": sigma_abs_err_corr,
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
        "pred_std": float(np.std(pred)),
        "target_std": float(np.std(target)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
    }


def moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Prediction-only smoothing for optional offline trajectory reporting."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < 3:
        return values.astype(np.float32)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def train_one_epoch(
    model: NTLDFE,
    criterion: NTLDFELoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
    scaler: Any | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, list[float]] = {
        "total": [],
        "nll": [],
        "reg": [],
        "corr": [],
        "ccc_loss": [],
        "std_loss": [],
        "mean": [],
        "smooth": [],
        "acc": [],
        "var": [],
        "soft": [],
        "rank": [],
    }
    amp_enabled = config.use_amp and device.type == "cuda"
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
            losses = criterion(outputs, y)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(losses["total"]).backward()
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

        for key in totals:
            totals[key].append(float(losses[key].detach().cpu()))

    return {key: float(np.mean(values)) for key, values in totals.items()}


def refit_on_train_val(
    model: NTLDFE,
    criterion: NTLDFELoss,
    runtime_arrays: dict[str, Any],
    config: Config,
    device: torch.device,
) -> list[dict[str, float]]:
    """Fine-tune on train+validation subjects after checkpoint selection.

    The LOSO test subject remains unseen. Validation labels are used only after
    model selection, matching a common final-fit protocol.
    """
    if not config.refit_on_train_val or config.refit_epochs <= 0:
        return []

    refit_arrays = list(runtime_arrays["train_arrays"]) + list(runtime_arrays["val_arrays"])
    refit_labels = list(runtime_arrays["train_labels"]) + [
        smooth_training_labels(labels, config.train_label_smoothing_window)
        for labels in runtime_arrays["val_labels"]
    ]
    refit_dataset = SequenceDataset(
        refit_arrays,
        refit_labels,
        sequence_length=config.sequence_length,
        stride=config.train_stride,
    )
    refit_loader = DataLoader(
        refit_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.pin_memory and config.device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate * config.refit_learning_rate_scale,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    for _ in range(config.refit_epochs):
        losses = train_one_epoch(
            model,
            criterion,
            refit_loader,
            optimizer,
            device,
            config,
            scaler=scaler,
        )
        history.append(losses)
    return history


@torch.no_grad()
def evaluate(model: NTLDFE, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_mu: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_log_var: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        outputs = model(x)
        all_mu.append(outputs["mu"].detach().cpu().numpy())
        all_log_var.append(outputs["log_var"].detach().cpu().numpy())
        all_y.append(y.numpy())

    pred = np.concatenate(all_mu, axis=0)
    target = np.concatenate(all_y, axis=0)
    log_var = np.concatenate(all_log_var, axis=0)
    metrics = sequence_metrics(pred, target, log_var)
    metrics["predictions"] = pred
    metrics["targets"] = target
    metrics["log_var"] = log_var
    return metrics


@torch.no_grad()
def predict_full_sequence(
    model: NTLDFE,
    data: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    sequence_length: int,
    batch_size: int,
    post_smooth_window: int = 1,
    prediction_stride: int = 1,
) -> dict[str, np.ndarray]:
    """Predict one subject as one continuous trajectory.

    This avoids repeated-window metric artifacts by averaging predictions from
    all windows covering each original time point.
    """
    model.eval()
    data = np.asarray(data, dtype=np.float32)
    labels = labels_to_unit_interval(labels)
    n_samples = data.shape[0]
    if n_samples < sequence_length:
        raise ValueError("Subject sequence shorter than sequence_length.")

    prediction_stride = max(1, int(prediction_stride))
    starts = list(range(0, n_samples - sequence_length + 1, prediction_stride))
    last_start = n_samples - sequence_length
    if starts[-1] != last_start:
        starts.append(last_start)
    sum_mu = np.zeros(n_samples, dtype=np.float64)
    sum_log_var = np.zeros(n_samples, dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.float64)

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        batch = np.stack([data[start : start + sequence_length] for start in batch_starts])
        x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        outputs = model(x)
        mu = outputs["mu"].detach().cpu().numpy()
        log_var = outputs["log_var"].detach().cpu().numpy()
        for local_index, start in enumerate(batch_starts):
            end = start + sequence_length
            sum_mu[start:end] += mu[local_index]
            sum_log_var[start:end] += log_var[local_index]
            counts[start:end] += 1.0

    valid = counts > 0
    pred = np.full(n_samples, np.nan, dtype=np.float64)
    log_var_out = np.full(n_samples, np.nan, dtype=np.float64)
    pred[valid] = sum_mu[valid] / counts[valid]
    log_var_out[valid] = sum_log_var[valid] / counts[valid]
    if post_smooth_window > 1 and np.any(valid):
        pred[valid] = moving_average_1d(pred[valid], post_smooth_window)
    return {
        "pred": pred.astype(np.float32),
        "target": labels.astype(np.float32),
        "log_var": log_var_out.astype(np.float32),
        "valid": valid,
    }


def trajectory_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    log_var: np.ndarray,
    valid: np.ndarray,
    tda_epsilon: float,
    train_label_mean: float | None = None,
) -> dict[str, float]:
    pred_v = pred[valid]
    target_v = target[valid]
    log_var_v = log_var[valid]
    metrics = sequence_metrics(
        pred_v.reshape(1, -1),
        target_v.reshape(1, -1),
        log_var_v.reshape(1, -1),
        tda_epsilon=tda_epsilon,
    )
    oracle_mean = np.full_like(target_v, np.mean(target_v), dtype=np.float64)
    oracle_mean_mae = float(np.mean(np.abs(oracle_mean - target_v)))
    oracle_mean_rmse = float(np.sqrt(np.mean((oracle_mean - target_v) ** 2)))
    metrics["diagnostic_oracle_subject_mean_mae"] = oracle_mean_mae
    metrics["diagnostic_oracle_subject_mean_rmse"] = oracle_mean_rmse
    if train_label_mean is not None and np.isfinite(train_label_mean):
        baseline = np.full_like(target_v, train_label_mean, dtype=np.float64)
        baseline_mae = float(np.mean(np.abs(baseline - target_v)))
        baseline_rmse = float(np.sqrt(np.mean((baseline - target_v) ** 2)))
        metrics["baseline_train_mean_mae"] = baseline_mae
        metrics["baseline_train_mean_rmse"] = baseline_rmse
        metrics["mae_vs_train_mean_baseline_delta"] = metrics["mae"] - baseline_mae
        metrics["rmse_vs_train_mean_baseline_delta"] = metrics["rmse"] - baseline_rmse
    target_diff = np.diff(target_v)
    if target_diff.size:
        previous = target_v[:-1]
        next_target = target_v[1:]
        previous_mae = float(np.mean(np.abs(previous - next_target)))
        previous_rmse = float(np.sqrt(np.mean((previous - next_target) ** 2)))
        metrics["diagnostic_previous_label_mae"] = previous_mae
        metrics["diagnostic_previous_label_rmse"] = previous_rmse
    return metrics


def evaluate_trajectories(
    model: NTLDFE,
    arrays: list[np.ndarray],
    labels_list: list[np.ndarray],
    device: torch.device,
    config: Config,
    calibration: tuple[float, float] | None = None,
    log_var_offset: float = 0.0,
    train_label_mean: float | None = None,
    prediction_stride: int = 1,
) -> dict[str, Any]:
    """Evaluate full trajectories over one or more subjects."""
    trajectories: list[dict[str, np.ndarray]] = []
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    log_var_parts: list[np.ndarray] = []
    per_subject: list[dict[str, float]] = []

    for data, labels in zip(arrays, labels_list):
        trajectory = predict_full_sequence(
            model,
            data,
            labels,
            device=device,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            post_smooth_window=config.post_smooth_window,
            prediction_stride=prediction_stride,
        )
        if log_var_offset != 0.0:
            trajectory["log_var"] = (
                trajectory["log_var"].astype(np.float64) + log_var_offset
            ).astype(np.float32)
        if calibration is not None:
            scale, bias = calibration
            calibrated = scale * trajectory["pred"].astype(np.float64) + bias
            trajectory["pred"] = np.clip(calibrated, 0.0, 1.0).astype(np.float32)
        metrics = trajectory_metrics(
            trajectory["pred"],
            trajectory["target"],
            trajectory["log_var"],
            trajectory["valid"],
            tda_epsilon=config.tda_epsilon,
            train_label_mean=train_label_mean,
        )
        per_subject.append(strip_arrays(metrics))
        valid = trajectory["valid"]
        pred_parts.append(trajectory["pred"][valid])
        target_parts.append(trajectory["target"][valid])
        log_var_parts.append(trajectory["log_var"][valid])
        trajectories.append(trajectory)

    pred = np.concatenate(pred_parts) if pred_parts else np.asarray([], dtype=np.float32)
    target = np.concatenate(target_parts) if target_parts else np.asarray([], dtype=np.float32)
    log_var = np.concatenate(log_var_parts) if log_var_parts else np.asarray([], dtype=np.float32)
    if pred.size == 0:
        raise ValueError("No valid trajectory predictions were produced.")
    valid = np.ones_like(pred, dtype=bool)
    metrics = trajectory_metrics(
        pred,
        target,
        log_var,
        valid,
        tda_epsilon=config.tda_epsilon,
        train_label_mean=train_label_mean,
    )
    metrics["per_subject"] = per_subject
    metrics["trajectories"] = trajectories
    return metrics


def fit_affine_calibration(
    pred: np.ndarray,
    target: np.ndarray,
    config: Config,
) -> tuple[float, float]:
    """Fit y ~= scale * pred + bias on validation data only."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    if pred.size < 3 or np.std(pred) < 1e-8:
        return 1.0, 0.0
    design = np.column_stack([pred, np.ones_like(pred)])
    ridge = config.calibration_ridge * np.eye(2, dtype=np.float64)
    ridge[1, 1] = 0.0
    solution = np.linalg.solve(design.T @ design + ridge, design.T @ target)
    scale, bias = solution
    scale = float(np.clip(scale, *config.calibration_scale_bounds))
    bias = float(np.clip(bias, *config.calibration_bias_bounds))
    return scale, bias


def fit_calibration_from_metrics(metrics: dict[str, Any], config: Config) -> tuple[float, float]:
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for trajectory in metrics.get("trajectories", []):
        valid = trajectory["valid"]
        pred_parts.append(trajectory["pred"][valid])
        target_parts.append(trajectory["target"][valid])
    if not pred_parts:
        return 1.0, 0.0
    return fit_affine_calibration(
        np.concatenate(pred_parts),
        np.concatenate(target_parts),
        config,
    )


def fit_mean_shift_from_metrics(metrics: dict[str, Any], config: Config) -> tuple[float, float]:
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for trajectory in metrics.get("trajectories", []):
        valid = trajectory["valid"]
        pred_parts.append(trajectory["pred"][valid])
        target_parts.append(trajectory["target"][valid])
    if not pred_parts:
        return 1.0, 0.0
    pred = np.concatenate(pred_parts).astype(np.float64)
    target = np.concatenate(target_parts).astype(np.float64)
    bias = float(np.mean(target) - np.mean(pred))
    bias = float(np.clip(bias, *config.mean_shift_bounds))
    return 1.0, bias


def fit_log_var_offset(metrics: dict[str, Any], config: Config) -> float:
    """Calibrate predicted variance using validation residuals only."""
    if not config.calibrate_log_var:
        return 0.0
    residual_var, current_var = estimate_residual_and_predicted_variance(metrics)
    if residual_var is None or current_var is None:
        return 0.0
    offset = float(np.log(residual_var / current_var))
    return float(np.clip(offset, *config.log_var_offset_bounds))


def estimate_residual_and_predicted_variance(
    metrics: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Return empirical squared error and mean predicted variance from trajectories."""
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    log_var_parts: list[np.ndarray] = []
    for trajectory in metrics.get("trajectories", []):
        valid = trajectory["valid"]
        pred_parts.append(trajectory["pred"][valid])
        target_parts.append(trajectory["target"][valid])
        log_var_parts.append(trajectory["log_var"][valid])
    if not pred_parts:
        return None, None
    pred = np.concatenate(pred_parts).astype(np.float64)
    target = np.concatenate(target_parts).astype(np.float64)
    log_var = np.concatenate(log_var_parts).astype(np.float64)
    residual_var = np.mean((target - pred) ** 2) + 1e-6
    current_var = np.mean(np.exp(log_var)) + 1e-6
    return float(residual_var), float(current_var)


def choose_log_var_offset(
    selected_val_metrics: dict[str, Any],
    raw_holdout_val_metrics: dict[str, Any],
    config: Config,
) -> float:
    """Use validation residuals to calibrate uncertainty without test labels.

    If the model is refit on train+validation, the refit validation residual can
    be unrealistically small because those subjects have become training data.
    The pre-refit holdout validation residual is a safer floor for sigma.
    """
    selected_offset = fit_log_var_offset(selected_val_metrics, config)
    if not config.conservative_log_var_calibration:
        return selected_offset
    selected_residual_var, selected_current_var = estimate_residual_and_predicted_variance(
        selected_val_metrics
    )
    holdout_residual_var, _ = estimate_residual_and_predicted_variance(
        raw_holdout_val_metrics
    )
    if selected_residual_var is None or selected_current_var is None:
        return selected_offset
    if holdout_residual_var is None:
        return selected_offset
    residual_floor = max(selected_residual_var, holdout_residual_var)
    offset = float(np.log(residual_floor / selected_current_var))
    return float(np.clip(offset, *config.log_var_offset_bounds))


def calibration_is_allowed(
    raw_val_metrics: dict[str, Any],
    calibrated_val_metrics: dict[str, Any],
    config: Config,
) -> bool:
    if not config.use_val_calibration:
        return False
    if config.calibration_requires_rmse_and_ccc:
        raw_rmse = float(raw_val_metrics.get("rmse", float("inf")))
        calibrated_rmse = float(calibrated_val_metrics.get("rmse", float("inf")))
        raw_ccc = float(raw_val_metrics.get("ccc", -float("inf")))
        calibrated_ccc = float(calibrated_val_metrics.get("ccc", -float("inf")))
        rmse_ok = calibrated_rmse + config.calibration_min_improvement < raw_rmse
        ccc_ok = calibrated_ccc + config.calibration_min_improvement >= raw_ccc
        if not (rmse_ok and ccc_ok):
            return False
    raw_score = validation_score(raw_val_metrics, config.validation_metric)
    calibrated_score = validation_score(calibrated_val_metrics, config.validation_metric)
    return calibrated_score + config.calibration_min_improvement < raw_score


def choose_calibration(
    raw_val_metrics: dict[str, Any],
    candidates: dict[str, tuple[tuple[float, float] | None, dict[str, Any]]],
    config: Config,
) -> tuple[str, tuple[float, float] | None, bool]:
    """Choose a validation-approved calibration candidate."""
    if not config.use_val_calibration or config.calibration_mode == "none":
        return "raw", None, False
    allowed_modes = {"affine", "mean_shift"} if config.calibration_mode == "best" else {config.calibration_mode}
    best_name = "raw"
    best_calibration: tuple[float, float] | None = None
    best_score = validation_score(raw_val_metrics, config.validation_metric)
    for name, (calibration, metrics) in candidates.items():
        if name not in allowed_modes or calibration is None:
            continue
        if not calibration_is_allowed(raw_val_metrics, metrics, config):
            continue
        score = validation_score(metrics, config.validation_metric)
        if score < best_score:
            best_name = name
            best_calibration = calibration
            best_score = score
    return best_name, best_calibration, best_calibration is not None


def strip_arrays(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value)
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def validation_score(metrics: dict[str, Any], metric_name: str) -> float:
    """Return a minimization score for checkpoint selection."""
    if metric_name == "rmse_ccc":
        rmse = float(metrics.get("rmse", float("inf")))
        ccc = float(metrics.get("ccc", 0.0))
        if not np.isfinite(rmse):
            return float("inf")
        return rmse - 0.15 * ccc
    if metric_name == "rmse_pearson_ccc":
        rmse = float(metrics.get("rmse", float("inf")))
        pearson = float(metrics.get("pearson", 0.0))
        ccc = float(metrics.get("ccc", 0.0))
        if not np.isfinite(rmse):
            return float("inf")
        return rmse - 0.08 * pearson - 0.12 * ccc
    if metric_name == "rmse_pearson_ccc_r2":
        rmse = float(metrics.get("rmse", float("inf")))
        pearson = float(metrics.get("pearson", 0.0))
        ccc = float(metrics.get("ccc", 0.0))
        r2 = float(metrics.get("r2", 0.0))
        if not np.isfinite(rmse):
            return float("inf")
        return rmse - 0.08 * pearson - 0.15 * ccc - 0.03 * r2
    if metric_name == "rmse_pearson_ccc_r2_std":
        rmse = float(metrics.get("rmse", float("inf")))
        pearson = float(metrics.get("pearson", 0.0))
        ccc = float(metrics.get("ccc", 0.0))
        r2 = float(metrics.get("r2", 0.0))
        pred_std = float(metrics.get("pred_std", 0.0))
        target_std = float(metrics.get("target_std", 0.0))
        std_gap = abs(pred_std - target_std)
        if not np.isfinite(rmse):
            return float("inf")
        return rmse - 0.08 * pearson - 0.18 * ccc - 0.05 * r2 + 0.03 * std_gap
    value = float(metrics.get(metric_name, float("inf")))
    if not np.isfinite(value):
        return float("inf")
    if metric_name in {"ccc", "pearson", "spearman", "tda"}:
        return -value
    return value


def train_fold(
    subjects: dict[str, dict[str, Any]],
    test_key: str,
    config: Config,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(config.device)
    train_loader, val_loader, test_loader, feature_shape, fold_info = make_loaders_for_fold(
        subjects, test_key, config
    )
    runtime_arrays = fold_info.pop("runtime_arrays")
    model = build_model(feature_shape, config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(3, config.patience // 3),
            min_lr=1e-5,
        )
        if config.use_scheduler
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    no_improve = 0
    history: list[dict[str, Any]] = []
    checkpoint_pool: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_losses = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            config,
            scaler=scaler,
        )
        val_window_metrics = evaluate(model, val_loader, device)
        val_metrics = evaluate_trajectories(
            model,
            runtime_arrays["val_arrays"],
            runtime_arrays["val_labels"],
            device,
            config,
            train_label_mean=runtime_arrays["train_label_mean"],
            prediction_stride=config.selection_prediction_stride,
        )
        val_score = validation_score(val_metrics, config.validation_metric)
        if scheduler is not None:
            scheduler.step(val_score)
        row = {
            "epoch": epoch,
            "train": train_losses,
            "val": strip_arrays(val_metrics),
            "val_window_debug": strip_arrays(val_window_metrics),
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
        if can_update_best and config.checkpoint_pool_size > 0:
            checkpoint_pool.append({
                "score": float(val_score),
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "approx_val": strip_arrays(val_metrics),
            })
            checkpoint_pool.sort(key=lambda item: item["score"])
            del checkpoint_pool[config.checkpoint_pool_size :]

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{test_key}] epoch {epoch:03d} "
                f"train_total={train_losses['total']:.4f} "
                f"val_mae={val_metrics['mae']:.4f} "
                f"val_ccc={val_metrics['ccc']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if can_update_best and no_improve >= config.patience:
            break

    if best_state is None:
        best_epoch = config.epochs
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        checkpoint_pool.append({
            "score": float("inf"),
            "epoch": best_epoch,
            "state": best_state,
            "approx_val": {},
        })

    selection_pool_summary: list[dict[str, Any]] = []
    best_full_score = float("inf")
    selected_pool_state = best_state
    selected_pool_epoch = best_epoch
    seen_epochs: set[int] = set()
    for candidate in checkpoint_pool:
        epoch_id = int(candidate["epoch"])
        if epoch_id in seen_epochs:
            continue
        seen_epochs.add(epoch_id)
        model.load_state_dict(candidate["state"])
        full_val_metrics = evaluate_trajectories(
            model,
            runtime_arrays["val_arrays"],
            runtime_arrays["val_labels"],
            device,
            config,
            train_label_mean=runtime_arrays["train_label_mean"],
        )
        full_score = validation_score(full_val_metrics, config.validation_metric)
        selection_pool_summary.append({
            "epoch": epoch_id,
            "approx_score": float(candidate["score"]),
            "full_score": float(full_score),
            "full_val": strip_arrays(full_val_metrics),
        })
        if full_score < best_full_score:
            best_full_score = full_score
            selected_pool_state = candidate["state"]
            selected_pool_epoch = epoch_id

    best_state = selected_pool_state
    best_epoch = selected_pool_epoch
    model.load_state_dict(best_state)

    best_val_metrics = evaluate_trajectories(
        model,
        runtime_arrays["val_arrays"],
        runtime_arrays["val_labels"],
        device,
        config,
        train_label_mean=runtime_arrays["train_label_mean"],
    )
    refit_history = refit_on_train_val(
        model,
        criterion,
        runtime_arrays,
        config,
        device,
    )
    refit_val_metrics = evaluate_trajectories(
        model,
        runtime_arrays["val_arrays"],
        runtime_arrays["val_labels"],
        device,
        config,
        train_label_mean=runtime_arrays["train_label_mean"],
    )
    affine_calibration = fit_calibration_from_metrics(refit_val_metrics, config)
    affine_val_metrics = evaluate_trajectories(
        model,
        runtime_arrays["val_arrays"],
        runtime_arrays["val_labels"],
        device,
        config,
        calibration=affine_calibration,
        train_label_mean=runtime_arrays["train_label_mean"],
    )
    mean_shift_calibration = fit_mean_shift_from_metrics(refit_val_metrics, config)
    mean_shift_val_metrics = evaluate_trajectories(
        model,
        runtime_arrays["val_arrays"],
        runtime_arrays["val_labels"],
        device,
        config,
        calibration=mean_shift_calibration,
        train_label_mean=runtime_arrays["train_label_mean"],
    )
    calibration_name, selected_calibration, calibration_used = choose_calibration(
        refit_val_metrics,
        {
            "affine": (affine_calibration, affine_val_metrics),
            "mean_shift": (mean_shift_calibration, mean_shift_val_metrics),
        },
        config,
    )
    calibration_for_test = selected_calibration if calibration_used else None
    selected_val_metrics_for_uncertainty = (
        refit_val_metrics
        if not calibration_used
        else affine_val_metrics
        if calibration_name == "affine"
        else mean_shift_val_metrics
    )
    log_var_offset = choose_log_var_offset(
        selected_val_metrics_for_uncertainty,
        best_val_metrics,
        config,
    )

    # Window-level metrics are useful for debugging the optimization objective,
    # but paper-facing metrics should be computed on the reconstructed full
    # subject trajectory to avoid overlapping-window artifacts.
    test_window_metrics = evaluate(model, test_loader, device)
    raw_test_metrics = evaluate_trajectories(
        model,
        runtime_arrays["test_arrays"],
        runtime_arrays["test_labels"],
        device,
        config,
        log_var_offset=log_var_offset,
        train_label_mean=runtime_arrays["train_label_mean"],
    )
    affine_test_metrics = evaluate_trajectories(
        model,
        runtime_arrays["test_arrays"],
        runtime_arrays["test_labels"],
        device,
        config,
        calibration=affine_calibration,
        log_var_offset=log_var_offset,
        train_label_mean=runtime_arrays["train_label_mean"],
    )
    mean_shift_test_metrics = evaluate_trajectories(
        model,
        runtime_arrays["test_arrays"],
        runtime_arrays["test_labels"],
        device,
        config,
        calibration=mean_shift_calibration,
        log_var_offset=log_var_offset,
        train_label_mean=runtime_arrays["train_label_mean"],
    )
    if calibration_name == "affine":
        calibrated_test_metrics = affine_test_metrics
    elif calibration_name == "mean_shift":
        calibrated_test_metrics = mean_shift_test_metrics
    else:
        calibrated_test_metrics = raw_test_metrics
    test_metrics = calibrated_test_metrics if calibration_used else raw_test_metrics
    primary_source = (
        calibration_name
        if calibration_used and config.report_calibrated_as_primary
        else "raw"
    )
    primary_test_metrics = (
        test_metrics
        if calibration_used and config.report_calibrated_as_primary
        else raw_test_metrics
    )
    fold_dir = output_dir / f"fold_{test_key}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if config.save_checkpoints:
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_shape": feature_shape,
                "config": asdict(config),
                "fold_info": fold_info,
                "best_epoch": best_epoch,
                "calibration": {
                    "mode": calibration_name,
                    "scale": selected_calibration[0] if selected_calibration else 1.0,
                    "bias": selected_calibration[1] if selected_calibration else 0.0,
                    "used": calibration_used,
                    "log_var_offset": log_var_offset,
                },
            },
            fold_dir / "checkpoint.pt",
        )
    if config.save_predictions:
        trajectory = primary_test_metrics["trajectories"][0]
        np.savez_compressed(
            fold_dir / "trajectory_predictions.npz",
            pred=trajectory["pred"],
            target=trajectory["target"],
            log_var=trajectory["log_var"],
            valid=trajectory["valid"],
        )
    result = {
        "test_subject": test_key,
        "best_epoch": best_epoch,
        "fold_info": fold_info,
        "calibration": {
            "mode": calibration_name,
            "scale": selected_calibration[0] if selected_calibration else 1.0,
            "bias": selected_calibration[1] if selected_calibration else 0.0,
            "affine_scale": affine_calibration[0],
            "affine_bias": affine_calibration[1],
            "mean_shift_bias": mean_shift_calibration[1],
            "used": calibration_used,
            "log_var_offset": log_var_offset,
            "selected_by": config.validation_metric,
            "conservative_log_var_calibration": config.conservative_log_var_calibration,
        },
        "best_val": strip_arrays(best_val_metrics),
        "refit_val": strip_arrays(refit_val_metrics),
        "val_affine": strip_arrays(affine_val_metrics),
        "val_mean_shift": strip_arrays(mean_shift_val_metrics),
        "test_primary_source": primary_source,
        "test": strip_arrays(primary_test_metrics),
        "test_calibrated": strip_arrays(calibrated_test_metrics),
        "test_affine": strip_arrays(affine_test_metrics),
        "test_mean_shift": strip_arrays(mean_shift_test_metrics),
        "test_raw": strip_arrays(raw_test_metrics),
        "test_window_debug": strip_arrays(test_window_metrics),
        "selection_pool": selection_pool_summary,
        "refit_history": refit_history,
        "history": history if config.save_history else [],
    }
    save_json(fold_dir / "metrics.json", result)
    print(
        f"[{test_key}] TEST mae={result['test']['mae']:.4f} "
        f"rmse={result['test']['rmse']:.4f} "
        f"ccc={result['test'].get('ccc', float('nan')):.4f} "
        f"source={result['test_primary_source']}"
    )
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(results[0]["test"].keys()) if results else []
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
    return summary


def run_loso(config: Config = CFG) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir) / config.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(config))

    subjects = load_subjects(config)
    audit = audit_subjects(subjects)
    keys = ordered_subject_keys(subjects)
    if config.max_folds is not None:
        keys = keys[: config.max_folds]

    print(f"Dataset: {config.dataset_name}")
    print(f"Subjects loaded: {len(subjects)} | folds to run: {len(keys)}")
    print(
        "Label audit: "
        f"min={audit.label_min:.4f} max={audit.label_max:.4f} "
        f"mean={audit.label_mean:.4f} std={audit.label_std:.4f} "
        f"unique@1e-3={audit.unique_rounded_labels}"
    )
    if audit.is_effectively_binary:
        print(f"Warning: {audit.note}")
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
                results_by_key[key]
                for key in keys
                if key in results_by_key
            ]
            partial_summary = summarize_results(partial_results)
            save_json(
                output_dir / "partial_results.json",
                {"summary": partial_summary, "results": partial_results},
            )

    results = [
        results_by_key[key]
        for key in keys
        if key in results_by_key
    ]
    summary = summarize_results(results)
    payload = {"summary": summary, "dataset_audit": asdict(audit), "results": results}
    save_json(output_dir / "loso_results.json", payload)
    print("\n=== LOSO summary ===")
    for metric, values in summary["metrics"].items():
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    return payload


if __name__ == "__main__":
    run_loso(CFG)
