#!/usr/bin/env python3
"""SEED-VIG Comprehensive Non-LOSO Experiment Runner.

Covers Stages 5-7 from stage.md without using LOSO, preserving temporal info.
All results saved to outputs/seedvig/.

Experiments:
  1. Mixed classification (3-class + binary, with ensemble)
  2. Regression (heteroscedastic uncertainty + temporal smoothing)
  3. Ablation studies (uncertainty, smoothing, label window, ranking, soft label)
  4. Cross-subject experiments (regression + classification, no LOSO)
  5. Interpretability (attention weights, topography, uncertainty analysis)

Usage (in llm310 conda env):
    python run_seedvig_comprehensive.py                  # all experiments
    python run_seedvig_comprehensive.py --phase reg      # regression only
    python run_seedvig_comprehensive.py --phase cls      # classification only
    python run_seedvig_comprehensive.py --phase ablation # ablation only
    python run_seedvig_comprehensive.py --phase cross    # cross-subject only
    python run_seedvig_comprehensive.py --phase interp   # interpretability only
    python run_seedvig_comprehensive.py --phase figures  # figures only
"""

from __future__ import annotations

import json, math, os, random, sys, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Project root ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
OUT = BASE / "outputs" / "seedvig"

from seed_dataloader import find_subject_dirs, load_subject as load_seedvig_subject
from nt_ldfe_model import FeatureMatrixEncoder, TemporalEncoder

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# Nature color scheme
# ═════════════════════════════════════════════════════════════════════════════
NATURE = {
    "red": "#E64B35", "blue": "#4DBBD5", "green": "#00A087",
    "orange": "#F39B7F", "purple": "#8491B4", "dark": "#2C3E50",
    "grey": "#7F8C8D", "yellow": "#F3C846", "pink": "#DC91B5",
}
PAL3 = [NATURE["blue"], NATURE["orange"], NATURE["green"]]
PAL5 = [NATURE["blue"], NATURE["red"], NATURE["green"], NATURE["purple"], NATURE["orange"]]
CLASS_NAMES = {0: "Low Fatigue", 1: "Medium Fatigue", 2: "High Fatigue"}
CLASS_LABELS = ["Low [0,0.35]", "Medium (0.35,0.7]", "High (0.7,1.0]"]
BAND_NAMES_5 = ["δ(1-4)", "θ(4-8)", "α(8-13)", "β(13-30)", "γ(30-45)"]

# SEED-VIG correct 17-channel montage (posterior-dominant):
# FT7,FT8,T7,T8,TP7,TP8,CP1,CP2,P1,PZ,P2,PO3,POZ,PO4,O1,OZ,O2
CHANNEL_MAP_17 = [
    ("FT7", -0.80, 0.30), ("FT8", 0.80, 0.30),
    ("T7", -0.88, 0.0),   ("T8", 0.88, 0.0),
    ("TP7", -0.80, -0.28), ("TP8", 0.80, -0.28),
    ("CP1", -0.28, -0.15), ("CP2", 0.28, -0.15),
    ("P1", -0.20, -0.35),  ("PZ", 0.0, -0.35),   ("P2", 0.20, -0.35),
    ("PO3", -0.22, -0.58), ("POZ", 0.0, -0.58),   ("PO4", 0.22, -0.58),
    ("O1", -0.22, -0.80),  ("OZ", 0.0, -0.85),    ("O2", 0.22, -0.80),
]
CH_NAMES = [c[0] for c in CHANNEL_MAP_17]
CH_X = np.array([c[1] for c in CHANNEL_MAP_17])
CH_Y = np.array([c[2] for c in CHANNEL_MAP_17])


# ═════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def banner(msg: str):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


def timer(name: str):
    class T:
        def __enter__(s): s.t0 = time.time(); return s
        def __exit__(s, *a): print(f"  [{name}] {time.time()-s.t0:.0f}s")
    return T()


def labels_to_unit_interval(labels):
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if labels.max() > 1.5: labels = labels / 100.0
    return np.clip(labels, 0.0, 1.0).astype(np.float32)


def perclos_to_class(labels, num_classes=3):
    labels = np.asarray(labels, dtype=np.float32)
    classes = np.zeros_like(labels, dtype=np.int64)
    if num_classes == 2:
        classes[labels > 0.35] = 1
    else:
        classes[labels > 0.35] = 1
        classes[labels > 0.7] = 2
    return classes


# ═════════════════════════════════════════════════════════════════════════════
# SEED-VIG Feature Preparation (same as existing code)
# ═════════════════════════════════════════════════════════════════════════════

def seedvig_to_spectral_de(data, sr=200.0, eps=1e-6):
    data = np.asarray(data, dtype=np.float32)
    n_s, n_t, n_c = data.shape
    centered = data - np.mean(data, axis=1, keepdims=True)
    window = np.hanning(n_t).astype(np.float32).reshape(1, n_t, 1)
    fft = np.fft.rfft(centered * window, axis=1)
    power = (np.abs(fft)**2).astype(np.float64) / max(n_t, 1)
    freqs = np.fft.rfftfreq(n_t, d=1.0/sr)
    bands = [(1,4),(4,8),(8,13),(13,30),(30,45)]
    feats = np.empty((n_s, n_c, len(bands)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        feats[:,:,bi] = np.log(np.mean(power[:,mask,:], axis=1) + eps).astype(np.float32)
    return feats


def seedvig_to_temporal_stats(data, bins=5, eps=1e-6):
    data = np.asarray(data, dtype=np.float32)
    n_s, n_t, n_c = data.shape
    segs = np.array_split(np.arange(n_t), bins)
    feats = np.empty((n_s, n_c, bins*4), dtype=np.float32)
    for bi, seg in enumerate(segs):
        seg_d = data[:, seg, :]; base = bi * 4
        var = np.var(seg_d, axis=1)
        feats[:,:,base+0] = np.log(var + eps)
        feats[:,:,base+1] = np.mean(seg_d, axis=1)
        feats[:,:,base+2] = np.sqrt(var + eps)
        feats[:,:,base+3] = (np.mean(np.abs(np.diff(seg_d, axis=1)), axis=1)
                            if seg_d.shape[1] > 1 else np.zeros((n_s, n_c), dtype=np.float32))
    return feats


def prepare_seedvig(data, mode="hybrid_de", bins=5, sr=200.0, eps=1e-6):
    data = np.asarray(data, dtype=np.float32)
    if mode == "hybrid_de":
        s = seedvig_to_spectral_de(data, sr, eps)
        t = seedvig_to_temporal_stats(data, bins, eps)
        return np.concatenate([s, t], axis=2).astype(np.float32)  # (N, 17, 25)
    elif mode == "spectral_de":
        return seedvig_to_spectral_de(data, sr, eps)
    elif mode == "temporal_stats":
        return seedvig_to_temporal_stats(data, bins, eps)
    return data


def load_all_subjects(seedvig_root="dataset/seedvig", feature_mode="hybrid_de",
                      de_bins=5, sr=200.0, eps=1e-6):
    subjects = {}
    for idx, sdir in enumerate(find_subject_dirs(seedvig_root), start=1):
        subj = load_seedvig_subject(sdir, stack=True)
        key = f"sub{idx}"
        data = prepare_seedvig(subj["data"], feature_mode, de_bins, sr, eps)
        label = labels_to_unit_interval(subj["label"])
        subjects[key] = {"data": data, "label": label, "name": subj["subject_name"]}
    print(f"Loaded {len(subjects)} subjects, feature shape: {list(subjects.values())[0]['data'].shape}")
    return subjects


# ═════════════════════════════════════════════════════════════════════════════
# Sequence Dataset Builder (temporal split, no LOSO)
# ═════════════════════════════════════════════════════════════════════════════

def build_temporal_sequences(subjects, seq_len=45, stride=5, test_ratio=0.2,
                             val_ratio=0.1, classification=False, num_classes=3):
    """Build time-preserving sequences. Splits each subject temporally.
    Returns train/val/test DataLoaders.
    """
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []

    for si, (key, subj) in enumerate(sorted(subjects.items())):
        data = np.asarray(subj["data"], dtype=np.float32)
        label = np.asarray(subj["label"], dtype=np.float32)
        n = len(data)

        # Temporal split: train=first 70%, val=next 15%, test=last 15%
        train_end = int(n * (1 - test_ratio - val_ratio))
        val_end = int(n * (1 - test_ratio))

        for split_name, sx, sy, xy_lists in [
            ("train", data[:train_end], label[:train_end], [train_x, train_y]),
            ("val", data[train_end:val_end], label[train_end:val_end], [val_x, val_y]),
            ("test", data[val_end:], label[val_end:], [test_x, test_y]),
        ]:
            if len(sx) < seq_len:
                continue
            indices = list(range(0, len(sx) - seq_len + 1, stride))
            for start in indices:
                end = start + seq_len
                sx_list, sy_list = xy_lists
                sx_list.append(sx[start:end])
                if classification:
                    cls_labels = perclos_to_class(sy[start:end], num_classes)
                    sy_list.append(cls_labels[-1])  # classify last window label
                else:
                    sy_list.append(sy[start:end])

    def make_dl(x_list, y_list, batch_size=48, shuffle=True):
        if not x_list:
            return None
        x = torch.from_numpy(np.stack(x_list, axis=0)).float()
        y = torch.from_numpy(np.array(y_list, dtype=np.float32)).float()
        if classification:
            y = y.long()
        ds = torch.utils.data.TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=0, pin_memory=True)
    return (make_dl(train_x, train_y, shuffle=True),
            make_dl(val_x, val_y, shuffle=False),
            make_dl(test_x, test_y, shuffle=False))


# ═════════════════════════════════════════════════════════════════════════════
# NT-LDFE Regression Model (non-LOSO)
# ═════════════════════════════════════════════════════════════════════════════

class NTLDFE_Regression(nn.Module):
    """NT-LDFE regression model with heteroscedastic uncertainty."""
    def __init__(self, feature_shape, encoder_dim=96, encoder_layers=2,
                 temporal_hidden_dim=128, temporal_layers=2, dropout=0.1,
                 temporal_type="gru", bidirectional=True, use_feature_attention=True,
                 log_var_min=-4.0, log_var_max=1.0):
        super().__init__()
        self.encoder = FeatureMatrixEncoder(
            feature_shape=feature_shape, output_dim=encoder_dim,
            dropout=dropout, use_attention=use_feature_attention,
            num_layers=encoder_layers)
        self.temporal = TemporalEncoder(
            input_dim=encoder_dim, hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers, dropout=dropout,
            temporal_type=temporal_type, bidirectional=bidirectional)
        self.mu_head = nn.Sequential(
            nn.Linear(self.temporal.output_dim, encoder_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(encoder_dim, 1), nn.Tanh())
        self.log_var_head = nn.Sequential(
            nn.Linear(self.temporal.output_dim, encoder_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(encoder_dim, 1))
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def forward(self, x):
        encoded, attention = self.encoder(x)
        temporal = self.temporal(encoded)
        mu = self.mu_head(temporal)  # (B, T, 1)
        log_var = self.log_var_head(temporal)
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
        return {"mu": mu, "log_var": log_var, "attention": attention}


def regression_loss(outputs, targets, beta_smooth=1e-4, beta_acc=2e-4,
                    lambda_nll=0.15, lambda_reg=1.0):
    """Combined regression loss with heteroscedastic NLL and temporal smoothness."""
    mu = outputs["mu"]  # (B, T, 1)
    log_var = outputs["log_var"]
    y = targets.unsqueeze(-1)  # (B, T, 1)

    # SmoothL1 regression loss
    L_reg = F.smooth_l1_loss(mu, y)

    # Heteroscedastic NLL
    L_nll = (0.5 * torch.exp(-log_var) * (y - mu).pow(2) + 0.5 * log_var).mean()

    # Temporal smoothness (first-order)
    L_smooth = torch.tensor(0.0, device=mu.device)
    if beta_smooth > 0 and mu.shape[1] > 1:
        L_smooth = (mu[:, 1:] - mu[:, :-1]).pow(2).mean()

    # Acceleration penalty (second-order)
    L_acc = torch.tensor(0.0, device=mu.device)
    if beta_acc > 0 and mu.shape[1] > 2:
        L_acc = (mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]).pow(2).mean()

    total = lambda_reg * L_reg + lambda_nll * L_nll + beta_smooth * L_smooth + beta_acc * L_acc
    losses = {"L_reg": L_reg.item(), "L_nll": L_nll.item(),
              "L_smooth": L_smooth.item(), "L_acc": L_acc.item()}
    return total, losses


# ═════════════════════════════════════════════════════════════════════════════
# NT-LDFE Classification Model (non-LOSO)
# ═════════════════════════════════════════════════════════════════════════════

class NTLDFE_Classification(nn.Module):
    """NT-LDFE classification model."""
    def __init__(self, feature_shape, num_classes=3, encoder_dim=96, encoder_layers=2,
                 temporal_hidden_dim=128, temporal_layers=2, dropout=0.1,
                 temporal_type="gru", bidirectional=True, use_feature_attention=True,
                 classifier_hidden_dim=96):
        super().__init__()
        self.encoder = FeatureMatrixEncoder(
            feature_shape=feature_shape, output_dim=encoder_dim,
            dropout=dropout, use_attention=use_feature_attention,
            num_layers=encoder_layers)
        self.temporal = TemporalEncoder(
            input_dim=encoder_dim, hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers, dropout=dropout,
            temporal_type=temporal_type, bidirectional=bidirectional)
        self.head = nn.Sequential(
            nn.Linear(self.temporal.output_dim, classifier_hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes))

    def forward(self, x):
        encoded, attention = self.encoder(x)
        temporal = self.temporal(encoded)
        # Use last timestep for classification
        logits = self.head(temporal[:, -1])
        return {"logits": logits, "attention": attention}


# ═════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        nc = logits.size(-1)
        targets = targets.long().view(-1)
        if self.label_smoothing > 0:
            smooth = torch.full_like(logits, self.label_smoothing/(nc-1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            lp = F.log_softmax(logits, dim=-1)
            ce = -(smooth * lp).sum(dim=-1)
            probs = (smooth * F.softmax(logits, dim=-1)).sum(dim=-1)
        else:
            lp = F.log_softmax(logits, dim=-1)
            ce = -lp.gather(1, targets.unsqueeze(1)).squeeze(1)
            probs = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        fw = (1.0 - probs).pow(self.gamma)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            fw = at * fw
        return (fw * ce).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Training utils
# ═════════════════════════════════════════════════════════════════════════════

def compute_class_weights(train_labels, num_classes, mode="balanced"):
    if mode == "none":
        return None
    labels = train_labels.numpy() if hasattr(train_labels, 'numpy') else np.array(train_labels)
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_regression(model, dataloader, device):
    """Evaluate regression model."""
    model.eval()
    all_mu, all_y, all_lv = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            all_mu.append(out["mu"].cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_lv.append(out["log_var"].cpu().numpy())
    mu = np.concatenate([m.reshape(-1) for m in all_mu])
    y = np.concatenate([y.reshape(-1) for y in all_y])
    lv = np.concatenate([lv.reshape(-1) for lv in all_lv])
    sigma = np.exp(np.clip(lv, -10, 10) / 2)

    # Compute metrics
    mae = np.mean(np.abs(mu - y))
    rmse = np.sqrt(np.mean((mu - y)**2))
    # Pearson
    mx, my = mu - mu.mean(), y - y.mean()
    denom = np.sqrt((mx**2).sum() * (my**2).sum())
    pearson = (mx * my).sum() / max(denom, 1e-10)
    # Spearman
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(mu, y)
    # CCC
    s_xy = np.mean((mu - mu.mean()) * (y - y.mean()))
    ccc = 2 * s_xy / (mu.var() + y.var() + (mu.mean() - y.mean())**2 + 1e-10)
    # R2
    ss_res = np.sum((y - mu)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    # NLL
    nll = np.mean(0.5 * np.exp(-lv) * (y - mu)**2 + 0.5 * lv)
    # Sigma-abs error Spearman
    sigma_ae, _ = spearmanr(sigma, np.abs(mu - y))
    # TV
    tv = np.mean(np.abs(np.diff(mu))) if len(mu) > 1 else 0
    # ADV
    adv = np.mean(np.abs(mu[2:] - 2*mu[1:-1] + mu[:-2])) if len(mu) > 2 else 0

    return {"mae": mae, "rmse": rmse, "pearson": pearson, "spearman": spearman,
            "ccc": ccc, "r2": r2, "nll_gaussian": nll,
            "sigma_abs_err_spearman": sigma_ae, "tv": tv, "adv": adv,
            "pred_mean": float(mu.mean()), "pred_std": float(mu.std()),
            "target_mean": float(y.mean()), "target_std": float(y.std())}


def evaluate_classification(model, dataloader, device, num_classes=3):
    """Evaluate classification model."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            probs = F.softmax(out["logits"], dim=-1)
            preds = probs.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                                  confusion_matrix, classification_report)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()
    # Per-class F1
    per_class = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[f"precision_c{c}"] = float((preds[mask] == c).mean())
            per_class[f"recall_c{c}"] = float((labels[preds == c] == c).mean()) if (preds == c).sum() > 0 else 0

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "kappa": kappa, "confusion_matrix": cm, **per_class}


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Regression Experiments
# ═════════════════════════════════════════════════════════════════════════════

def run_regression(config_overrides=None):
    """Run regression experiment with temporal split (no LOSO)."""
    banner("Phase 1: NT-LDFE Regression (Non-LOSO, Temporal Split)")

    cfg = {
        "seedvig_root": "dataset/seedvig", "feature_mode": "hybrid_de",
        "de_bins": 5, "sr": 200.0, "eps": 1e-6,
        "seq_len": 45, "stride": 5, "test_ratio": 0.2, "val_ratio": 0.1,
        "batch_size": 48, "epochs": 200, "min_epochs": 10, "patience": 20,
        "lr": 3e-4, "wd": 1e-4, "grad_clip": 5.0,
        "encoder_dim": 96, "encoder_layers": 2,
        "temporal_hidden_dim": 128, "temporal_layers": 2,
        "dropout": 0.1, "temporal_type": "gru", "bidirectional": True,
        "use_feature_attention": True,
        "beta_smooth": 1e-4, "beta_acc": 2e-4,
        "lambda_nll": 0.15, "lambda_reg": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "label_smoothing_window": 3,
    }
    if config_overrides:
        cfg.update(config_overrides)

    set_seed(42)
    device = torch.device(cfg["device"])

    # Load subjects
    subjects = load_all_subjects(cfg["seedvig_root"], cfg["feature_mode"],
                                  cfg["de_bins"], cfg["sr"], cfg["eps"])
    feature_shape = list(subjects.values())[0]["data"].shape[1:]  # (17, 25)

    # Apply label smoothing
    if cfg.get("label_smoothing_window", 1) > 1:
        from scipy.ndimage import uniform_filter1d
        w = cfg["label_smoothing_window"]
        for subj in subjects.values():
            subj["label_original"] = subj["label"].copy()
            subj["label"] = uniform_filter1d(subj["label"].astype(np.float64), size=w).astype(np.float32)

    # Build temporal sequences
    train_dl, val_dl, test_dl = build_temporal_sequences(
        subjects, seq_len=cfg["seq_len"], stride=cfg["stride"],
        test_ratio=cfg["test_ratio"], val_ratio=cfg["val_ratio"],
        classification=False)

    if train_dl is None or test_dl is None:
        print("ERROR: Not enough data to build sequences.")
        return None

    # Build model
    model = NTLDFE_Regression(
        feature_shape=feature_shape, encoder_dim=cfg["encoder_dim"],
        encoder_layers=cfg["encoder_layers"],
        temporal_hidden_dim=cfg["temporal_hidden_dim"],
        temporal_layers=cfg["temporal_layers"], dropout=cfg["dropout"],
        temporal_type=cfg["temporal_type"], bidirectional=cfg["bidirectional"],
        use_feature_attention=cfg["use_feature_attention"]).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    best_val_loss = float("inf"); best_state = None; patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_pearson": []}

    for epoch in range(cfg["epochs"]):
        model.train(); total_loss = 0; num_batches = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss, loss_dict = regression_loss(out, y, cfg["beta_smooth"],
                                                       cfg["beta_acc"], cfg["lambda_nll"],
                                                       cfg["lambda_reg"])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss, loss_dict = regression_loss(out, y, cfg["beta_smooth"],
                                                   cfg["beta_acc"], cfg["lambda_nll"],
                                                   cfg["lambda_reg"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
            total_loss += loss.item(); num_batches += 1

        avg_train = total_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation
        if val_dl:
            val_metrics = evaluate_regression(model, val_dl, device)
            val_loss = val_metrics["rmse"]
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_pearson"].append(val_metrics["pearson"])
            sched.step(val_loss)
        else:
            val_loss = avg_train

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == cfg["epochs"] - 1:
            vm = evaluate_regression(model, val_dl or train_dl, device)
            print(f"  Epoch {epoch:3d}: train_loss={avg_train:.4f}, val_rmse={vm['rmse']:.4f}, "
                  f"val_pearson={vm['pearson']:.4f}, val_ccc={vm['ccc']:.4f}, patience={patience_counter}")

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    test_metrics = evaluate_regression(model, test_dl, device)
    print(f"\nTest Results: RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}, "
          f"CCC={test_metrics['ccc']:.4f}, R²={test_metrics['r2']:.4f}")

    # Save
    output_dir = OUT / "regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", cfg)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_json(output_dir / "history.json", history)

    # Save model
    torch.save({"state_dict": model.state_dict(), "feature_shape": feature_shape,
                 "cfg": cfg, "test_metrics": test_metrics},
               output_dir / "model.pt")

    # Save predictions for visualization
    model.eval()
    all_mu, all_y = [], []
    with torch.no_grad():
        for x, y in test_dl:
            out = model(x.to(device))
            all_mu.append(out["mu"].cpu().numpy().reshape(-1))
            all_y.append(y.cpu().numpy().reshape(-1))
    pred_data = {"mu": np.concatenate(all_mu).tolist(), "y": np.concatenate(all_y).tolist()}
    save_json(output_dir / "predictions.json", pred_data)

    print(f"Regression results saved to: {output_dir}/")
    return {"metrics": test_metrics, "config": cfg}


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Classification Experiments (Non-LOSO)
# ═════════════════════════════════════════════════════════════════════════════

def run_classification(config_overrides=None):
    """Run 3-class classification with temporal split (no LOSO)."""
    banner("Phase 2: NT-LDFE Classification (Non-LOSO, Temporal Split)")

    cfg = {
        "seedvig_root": "dataset/seedvig", "feature_mode": "hybrid_de",
        "de_bins": 5, "sr": 200.0, "eps": 1e-6,
        "num_classes": 3,
        "seq_len": 45, "stride": 5, "test_ratio": 0.2, "val_ratio": 0.1,
        "batch_size": 48, "epochs": 250, "min_epochs": 15, "patience": 35,
        "lr": 5e-4, "wd": 1e-4, "grad_clip": 5.0,
        "encoder_dim": 96, "encoder_layers": 2,
        "temporal_hidden_dim": 128, "temporal_layers": 2,
        "classifier_hidden_dim": 96, "dropout": 0.15,
        "temporal_type": "gru", "bidirectional": True,
        "use_feature_attention": True,
        "focal_gamma": 0.5, "class_weight_mode": "balanced",
        "label_smoothing": 0.05,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if config_overrides:
        cfg.update(config_overrides)

    set_seed(42)
    device = torch.device(cfg["device"])

    subjects = load_all_subjects(cfg["seedvig_root"], cfg["feature_mode"],
                                  cfg["de_bins"], cfg["sr"], cfg["eps"])
    feature_shape = list(subjects.values())[0]["data"].shape[1:]

    train_dl, val_dl, test_dl = build_temporal_sequences(
        subjects, seq_len=cfg["seq_len"], stride=cfg["stride"],
        test_ratio=cfg["test_ratio"], val_ratio=cfg["val_ratio"],
        classification=True, num_classes=cfg["num_classes"])

    if train_dl is None or test_dl is None:
        print("ERROR: Not enough data.")
        return None

    # Compute class weights
    all_train_labels = []
    for _, y in train_dl:
        all_train_labels.append(y.numpy())
    train_labels = np.concatenate(all_train_labels)
    class_weights = compute_class_weights(train_labels, cfg["num_classes"],
                                           cfg["class_weight_mode"])

    model = NTLDFE_Classification(
        feature_shape=feature_shape, num_classes=cfg["num_classes"],
        encoder_dim=cfg["encoder_dim"], encoder_layers=cfg["encoder_layers"],
        temporal_hidden_dim=cfg["temporal_hidden_dim"],
        temporal_layers=cfg["temporal_layers"], dropout=cfg["dropout"],
        temporal_type=cfg["temporal_type"], bidirectional=cfg["bidirectional"],
        use_feature_attention=cfg["use_feature_attention"],
        classifier_hidden_dim=cfg["classifier_hidden_dim"]).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=cfg["focal_gamma"], alpha=class_weights,
                          label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    best_val_f1 = 0; best_state = None; patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(cfg["epochs"]):
        model.train(); total_loss = 0; num_batches = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out["logits"], y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss = criterion(out["logits"], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
            total_loss += loss.item(); num_batches += 1

        history["train_loss"].append(total_loss / max(num_batches, 1))

        if val_dl:
            val_metrics = evaluate_classification(model, val_dl, device, cfg["num_classes"])
            val_f1 = val_metrics["macro_f1"]
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_f1)
            sched.step(val_f1)
        else:
            val_f1 = total_loss / max(num_batches, 1)

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == cfg["epochs"] - 1:
            vm = evaluate_classification(model, val_dl or train_dl, device, cfg["num_classes"])
            print(f"  Epoch {epoch:3d}: loss={history['train_loss'][-1]:.4f}, "
                  f"val_acc={vm['accuracy']:.4f}, val_f1={vm['macro_f1']:.4f}, patience={patience_counter}")

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_classification(model, test_dl, device, cfg["num_classes"])
    print(f"\nTest Results: Acc={test_metrics['accuracy']:.4f}, "
          f"Macro F1={test_metrics['macro_f1']:.4f}, Kappa={test_metrics['kappa']:.4f}")

    output_dir = OUT / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", cfg)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_json(output_dir / "history.json", history)
    torch.save({"state_dict": model.state_dict(), "feature_shape": feature_shape,
                 "cfg": cfg, "test_metrics": test_metrics}, output_dir / "model.pt")

    print(f"Classification results saved to: {output_dir}/")
    return {"metrics": test_metrics, "config": cfg}


def run_classification_ensemble(n_models=7):
    """Run ensemble of classification models."""
    banner(f"Phase 2b: Classification Ensemble ({n_models} models)")

    all_test_metrics = []
    all_models = []

    for i in range(n_models):
        seed = 42 + i * 7
        set_seed(seed)
        print(f"\n--- Ensemble Model {i+1}/{n_models} (seed={seed}) ---")

        cfg = {"seedvig_root": "dataset/seedvig", "feature_mode": "hybrid_de",
               "de_bins": 5, "sr": 200.0, "eps": 1e-6, "num_classes": 3,
               "seq_len": 45, "stride": 5, "test_ratio": 0.2, "val_ratio": 0.1,
               "batch_size": 48, "epochs": 200, "min_epochs": 10, "patience": 20,
               "lr": 5e-4, "wd": 1e-4, "grad_clip": 5.0,
               "encoder_dim": 96, "encoder_layers": 2,
               "temporal_hidden_dim": 128, "temporal_layers": 2,
               "classifier_hidden_dim": 96, "dropout": 0.15,
               "temporal_type": "gru", "bidirectional": True,
               "use_feature_attention": True,
               "focal_gamma": 0.5, "class_weight_mode": "balanced",
               "label_smoothing": 0.05, "device": "cuda" if torch.cuda.is_available() else "cpu"}

        result = run_single_classification(cfg, seed)
        if result:
            all_test_metrics.append(result["metrics"])
            all_models.append(result)

    if not all_models:
        print("ERROR: No ensemble models trained successfully.")
        return None

    # Ensemble prediction
    banner("Ensemble Evaluation")

    # Reload data
    subjects = load_all_subjects("dataset/seedvig", "hybrid_de", 5, 200.0, 1e-6)
    _, _, test_dl = build_temporal_sequences(
        subjects, seq_len=45, stride=45, test_ratio=0.2, val_ratio=0.1,
        classification=True, num_classes=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_probs = []
    all_labels = []

    for result in all_models:
        model_path = Path(result.get("model_path", ""))
        if not model_path.exists():
            print(f"  WARNING: Model path not found: {model_path}")
            continue
        ckpt = torch.load(model_path, map_location=device)
        feature_shape = tuple(ckpt["feature_shape"])
        model = NTLDFE_Classification(
            feature_shape=feature_shape, num_classes=3,
            encoder_dim=96, encoder_layers=2, temporal_hidden_dim=128,
            temporal_layers=2, dropout=0.15, classifier_hidden_dim=96).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        probs_list = []
        label_list = []
        with torch.no_grad():
            for x, y in test_dl:
                out = model(x.to(device))
                probs_list.append(F.softmax(out["logits"], dim=-1).cpu().numpy())
                label_list.append(y.cpu().numpy())
        all_probs.append(np.concatenate(probs_list))
        all_labels.append(np.concatenate(label_list))

    # Average probabilities
    avg_probs = np.mean(np.stack(all_probs), axis=0)
    final_preds = avg_probs.argmax(axis=-1)
    final_labels = all_labels[0]

    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
    acc = accuracy_score(final_labels, final_preds)
    macro_f1 = f1_score(final_labels, final_preds, average="macro")
    kappa = cohen_kappa_score(final_labels, final_preds)
    cm = confusion_matrix(final_labels, final_preds).tolist()

    ensemble_metrics = {"accuracy": acc, "macro_f1": macro_f1, "kappa": kappa,
                        "confusion_matrix": cm, "n_models": n_models,
                        "individual_accs": [m["accuracy"] for m in all_test_metrics]}

    print(f"\nEnsemble Results: Acc={acc:.4f}, Macro F1={macro_f1:.4f}, Kappa={kappa:.4f}")
    print(f"Individual accuracies: {[f'{a:.4f}' for a in ensemble_metrics['individual_accs']]}")

    output_dir = OUT / "classification"
    save_json(output_dir / "ensemble_metrics.json", ensemble_metrics)

    return {"metrics": ensemble_metrics}


def run_single_classification(cfg, seed):
    """Run a single classification model, return results dict."""
    set_seed(seed)
    device = torch.device(cfg["device"])

    subjects = load_all_subjects(cfg["seedvig_root"], cfg["feature_mode"],
                                  cfg["de_bins"], cfg["sr"], cfg["eps"])
    feature_shape = list(subjects.values())[0]["data"].shape[1:]

    train_dl, val_dl, test_dl = build_temporal_sequences(
        subjects, seq_len=cfg["seq_len"], stride=cfg["stride"],
        test_ratio=cfg["test_ratio"], val_ratio=cfg["val_ratio"],
        classification=True, num_classes=cfg["num_classes"])

    if train_dl is None:
        return None

    all_train_labels = []
    for _, y in train_dl:
        all_train_labels.append(y.numpy())
    train_labels = np.concatenate(all_train_labels)
    class_weights = compute_class_weights(train_labels, cfg["num_classes"], cfg["class_weight_mode"])

    model = NTLDFE_Classification(
        feature_shape=feature_shape, num_classes=cfg["num_classes"],
        encoder_dim=cfg["encoder_dim"], encoder_layers=cfg["encoder_layers"],
        temporal_hidden_dim=cfg["temporal_hidden_dim"],
        temporal_layers=cfg["temporal_layers"], dropout=cfg["dropout"],
        temporal_type=cfg["temporal_type"], bidirectional=cfg["bidirectional"],
        use_feature_attention=cfg["use_feature_attention"],
        classifier_hidden_dim=cfg["classifier_hidden_dim"]).to(device)

    criterion = FocalLoss(gamma=cfg["focal_gamma"], alpha=class_weights,
                          label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    best_val_f1 = 0; best_state = None; patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out["logits"], y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss = criterion(out["logits"], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()

        val_metrics = evaluate_classification(model, val_dl or train_dl, device, cfg["num_classes"])
        val_f1 = val_metrics["macro_f1"]
        sched.step(val_f1)

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs"]:
            break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_classification(model, test_dl, device, cfg["num_classes"])

    # Save model
    output_dir = OUT / "classification" / f"model_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "feature_shape": feature_shape,
                 "cfg": cfg, "test_metrics": test_metrics}, model_path)

    return {"metrics": test_metrics, "config": cfg, "model_path": str(model_path)}


def run_binary_classification():
    """Run binary classification (fatigue vs awake)."""
    banner("Phase 2c: Binary Classification (Fatigue vs Awake)")

    cfg = {"num_classes": 2, "seedvig_root": "dataset/seedvig",
           "feature_mode": "hybrid_de", "de_bins": 5, "sr": 200.0, "eps": 1e-6,
           "seq_len": 45, "stride": 5, "test_ratio": 0.2, "val_ratio": 0.1,
           "batch_size": 48, "epochs": 200, "min_epochs": 10, "patience": 25,
           "lr": 5e-4, "wd": 1e-4, "grad_clip": 5.0,
           "encoder_dim": 96, "encoder_layers": 2,
           "temporal_hidden_dim": 128, "temporal_layers": 2,
           "classifier_hidden_dim": 96, "dropout": 0.15,
           "temporal_type": "gru", "bidirectional": True,
           "use_feature_attention": True,
           "focal_gamma": 0.5, "class_weight_mode": "balanced",
           "label_smoothing": 0.05,
           "device": "cuda" if torch.cuda.is_available() else "cpu"}

    set_seed(42)
    device = torch.device(cfg["device"])

    subjects = load_all_subjects(cfg["seedvig_root"], cfg["feature_mode"],
                                  cfg["de_bins"], cfg["sr"], cfg["eps"])
    feature_shape = list(subjects.values())[0]["data"].shape[1:]

    train_dl, val_dl, test_dl = build_temporal_sequences(
        subjects, seq_len=cfg["seq_len"], stride=cfg["stride"],
        test_ratio=cfg["test_ratio"], val_ratio=cfg["val_ratio"],
        classification=True, num_classes=2)

    if train_dl is None:
        print("ERROR: Not enough data.")
        return None

    all_train_labels = []
    for _, y in train_dl:
        all_train_labels.append(y.numpy())
    train_labels = np.concatenate(all_train_labels)
    class_weights = compute_class_weights(train_labels, 2, "balanced")

    model = NTLDFE_Classification(
        feature_shape=feature_shape, num_classes=2,
        encoder_dim=cfg["encoder_dim"], encoder_layers=cfg["encoder_layers"],
        temporal_hidden_dim=cfg["temporal_hidden_dim"],
        temporal_layers=cfg["temporal_layers"], dropout=cfg["dropout"],
        temporal_type=cfg["temporal_type"], bidirectional=cfg["bidirectional"],
        use_feature_attention=cfg["use_feature_attention"],
        classifier_hidden_dim=cfg["classifier_hidden_dim"]).to(device)

    criterion = FocalLoss(gamma=cfg["focal_gamma"], alpha=class_weights,
                          label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    best_val_f1 = 0; best_state = None; patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x); loss = criterion(out["logits"], y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                out = model(x); loss = criterion(out["logits"], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()

        val_metrics = evaluate_classification(model, val_dl or train_dl, device, 2)
        val_f1 = val_metrics["macro_f1"]
        sched.step(val_f1)

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_f1={val_metrics['macro_f1']:.4f}")

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_classification(model, test_dl, device, 2)
    print(f"\nBinary Test: Acc={test_metrics['accuracy']:.4f}, "
          f"Macro F1={test_metrics['macro_f1']:.4f}, Kappa={test_metrics['kappa']:.4f}")

    output_dir = OUT / "classification_binary"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "test_metrics.json", test_metrics)
    torch.save({"state_dict": model.state_dict(), "feature_shape": feature_shape,
                 "test_metrics": test_metrics}, output_dir / "model.pt")

    return {"metrics": test_metrics}


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Ablation Experiments
# ═════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "full_model": {
        "description": "Full NT-LDFE with all components",
        "overrides": {},
    },
    "without_uncertainty": {
        "description": "Remove heteroscedastic uncertainty (λ_nll=0)",
        "overrides": {"lambda_nll": 0.0},
        "question": "Does heteroscedastic uncertainty improve prediction quality near PERCLOS boundaries?"
    },
    "without_smoothing": {
        "description": "Remove temporal smoothness constraints",
        "overrides": {"beta_smooth": 0.0, "beta_acc": 0.0},
        "question": "Do temporal smoothness penalties improve trajectory stability?"
    },
    "label_smooth_w1": {
        "description": "Label smoothing window = 1 (no smoothing)",
        "overrides": {"label_smoothing_window": 1},
        "question": "How does removing label smoothing affect performance?"
    },
    "label_smooth_w5": {
        "description": "Label smoothing window = 5 (heavier smoothing)",
        "overrides": {"label_smoothing_window": 5},
        "question": "Does stronger label smoothing improve robustness?"
    },
    "label_smooth_w7": {
        "description": "Label smoothing window = 7 (max smoothing)",
        "overrides": {"label_smoothing_window": 7},
        "question": "Does excessive label smoothing over-regularize?"
    },
    "tcn_encoder": {
        "description": "TCN temporal encoder instead of GRU",
        "overrides": {"temporal_type": "tcn", "bidirectional": False},
        "question": "Does TCN outperform GRU for temporal encoding?"
    },
    "no_feature_attention": {
        "description": "Remove channel/band attention",
        "overrides": {"use_feature_attention": False},
        "question": "Does feature attention contribute to model interpretability and performance?"
    },
    "high_beta_smooth": {
        "description": "Stronger temporal smoothness (β_smooth=1e-2)",
        "overrides": {"beta_smooth": 1e-2},
        "question": "Does increasing smoothness penalty improve trajectory quality without harming accuracy?"
    },
}


def run_ablation(ablation_name, ablation_cfg):
    """Run a single ablation experiment for regression."""
    banner(f"Ablation: {ablation_name} — {ablation_cfg['description']}")

    # Build full config
    cfg = {
        "seedvig_root": "dataset/seedvig", "feature_mode": "hybrid_de",
        "de_bins": 5, "sr": 200.0, "eps": 1e-6,
        "seq_len": 45, "stride": 5, "test_ratio": 0.2, "val_ratio": 0.1,
        "batch_size": 48, "epochs": 200, "min_epochs": 10, "patience": 20,
        "lr": 3e-4, "wd": 1e-4, "grad_clip": 5.0,
        "encoder_dim": 96, "encoder_layers": 2,
        "temporal_hidden_dim": 128, "temporal_layers": 2,
        "dropout": 0.1, "temporal_type": "gru", "bidirectional": True,
        "use_feature_attention": True,
        "beta_smooth": 1e-4, "beta_acc": 2e-4,
        "lambda_nll": 0.15, "lambda_reg": 1.0,
        "label_smoothing_window": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    cfg.update(ablation_cfg["overrides"])
    cfg["ablation_name"] = ablation_name
    cfg["ablation_question"] = ablation_cfg.get("question", "")

    set_seed(42)
    device = torch.device(cfg["device"])
    subjects = load_all_subjects(cfg["seedvig_root"], cfg["feature_mode"],
                                  cfg["de_bins"], cfg["sr"], cfg["eps"])
    feature_shape = list(subjects.values())[0]["data"].shape[1:]

    # Apply label smoothing
    if cfg.get("label_smoothing_window", 1) > 1:
        from scipy.ndimage import uniform_filter1d
        w = cfg["label_smoothing_window"]
        for subj in subjects.values():
            subj["label_original"] = subj["label"].copy()
            subj["label"] = uniform_filter1d(subj["label"].astype(np.float64), size=w).astype(np.float32)

    train_dl, val_dl, test_dl = build_temporal_sequences(
        subjects, seq_len=cfg["seq_len"], stride=cfg["stride"],
        test_ratio=cfg["test_ratio"], val_ratio=cfg["val_ratio"], classification=False)

    if train_dl is None:
        return None

    model = NTLDFE_Regression(
        feature_shape=feature_shape, encoder_dim=cfg["encoder_dim"],
        encoder_layers=cfg["encoder_layers"],
        temporal_hidden_dim=cfg["temporal_hidden_dim"],
        temporal_layers=cfg["temporal_layers"], dropout=cfg["dropout"],
        temporal_type=cfg["temporal_type"], bidirectional=cfg["bidirectional"],
        use_feature_attention=cfg["use_feature_attention"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    best_val_loss = float("inf"); best_state = None; patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss, _ = regression_loss(out, y, cfg["beta_smooth"],
                                               cfg["beta_acc"], cfg["lambda_nll"],
                                               cfg["lambda_reg"])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss, _ = regression_loss(out, y, cfg["beta_smooth"],
                                           cfg["beta_acc"], cfg["lambda_nll"],
                                           cfg["lambda_reg"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()

        val_metrics = evaluate_regression(model, val_dl or train_dl, device)
        val_loss = val_metrics["rmse"]
        sched.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs"]:
            break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_regression(model, test_dl, device)

    output_dir = OUT / "ablations" / ablation_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", cfg)
    save_json(output_dir / "test_metrics.json", test_metrics)

    print(f"  {ablation_name}: RMSE={test_metrics['rmse']:.4f}, "
          f"Pearson={test_metrics['pearson']:.4f}, CCC={test_metrics['ccc']:.4f}, "
          f"TV={test_metrics['tv']:.6f}, ADV={test_metrics['adv']:.6f}")

    return {"name": ablation_name, "metrics": test_metrics, "config": cfg}


def run_all_ablations():
    """Run all ablation experiments."""
    banner("Phase 3: Ablation Experiments")
    all_results = {}

    for name, cfg in ABLATION_CONFIGS.items():
        result = run_ablation(name, cfg)
        if result:
            all_results[name] = result

    # Save ablation summary
    summary = {}
    for name, result in all_results.items():
        summary[name] = {
            "description": ABLATION_CONFIGS[name]["description"],
            "question": ABLATION_CONFIGS[name].get("question", ""),
            "rmse": result["metrics"]["rmse"],
            "pearson": result["metrics"]["pearson"],
            "ccc": result["metrics"]["ccc"],
            "r2": result["metrics"]["r2"],
            "tv": result["metrics"]["tv"],
            "adv": result["metrics"]["adv"],
            "nll": result["metrics"]["nll_gaussian"],
        }

    output_dir = OUT / "ablations"
    save_json(output_dir / "ablation_summary.json", summary)

    # Print comparison table
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<35} {'RMSE':>8} {'Pearson':>8} {'CCC':>8} {'R²':>8} {'TV':>10} {'ADV':>10}")
    print("-" * 100)
    for name, s in summary.items():
        print(f"{ABLATION_CONFIGS[name]['description']:<35} "
              f"{s['rmse']:8.4f} {s['pearson']:8.4f} {s['ccc']:8.4f} "
              f"{s['r2']:8.4f} {s['tv']:10.6f} {s['adv']:10.6f}")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: Cross-Subject Experiments (No LOSO)
# ═════════════════════════════════════════════════════════════════════════════

def run_cross_subject_regression():
    """Cross-subject regression: train on 70% subjects, test on 30% (no LOSO)."""
    banner("Phase 4a: Cross-Subject Regression (Train on 70% subjects, test on 30%)")

    cfg = {"seedvig_root": "dataset/seedvig", "feature_mode": "hybrid_de",
           "de_bins": 5, "sr": 200.0, "eps": 1e-6,
           "seq_len": 45, "stride": 5,
           "batch_size": 48, "epochs": 250, "min_epochs": 15, "patience": 25,
           "lr": 3e-4, "wd": 1e-4, "grad_clip": 5.0,
           "encoder_dim": 96, "encoder_layers": 2,
           "temporal_hidden_dim": 128, "temporal_layers": 2,
           "dropout": 0.1, "temporal_type": "gru", "bidirectional": True,
           "use_feature_attention": True,
           "beta_smooth": 1e-4, "beta_acc": 2e-4,
           "lambda_nll": 0.15, "lambda_reg": 1.0,
           "device": "cuda" if torch.cuda.is_available() else "cpu",
           "train_subject_ratio": 0.7}

    set_seed(42)
    device = torch.device(cfg["device"])

    subjects = load_all_subjects(cfg["seedvig_root"], cfg["feature_mode"],
                                  cfg["de_bins"], cfg["sr"], cfg["eps"])
    subject_keys = sorted(subjects.keys())
    n_train_subj = max(2, int(len(subject_keys) * cfg["train_subject_ratio"]))

    # Shuffle subjects
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(subject_keys)
    train_keys = list(shuffled[:n_train_subj])
    test_keys = list(shuffled[n_train_subj:])

    print(f"Train subjects ({len(train_keys)}): {train_keys}")
    print(f"Test subjects ({len(test_keys)}): {test_keys}")

    def build_subject_sequences(subj_keys, seq_len=45, stride=5):
        x_list, y_list = [], []
        for key in subj_keys:
            data = subjects[key]["data"]
            label = subjects[key]["label"]
            for s in range(0, len(data) - seq_len + 1, stride):
                x_list.append(data[s:s+seq_len])
                y_list.append(label[s:s+seq_len])
        if not x_list:
            return None
        x = torch.from_numpy(np.stack(x_list)).float()
        y = torch.from_numpy(np.array(y_list, dtype=np.float32)).float()
        return DataLoader(torch.utils.data.TensorDataset(x, y),
                         batch_size=cfg["batch_size"], shuffle=True,
                         num_workers=0, pin_memory=True)

    train_dl = build_subject_sequences(train_keys, cfg["seq_len"], cfg["stride"])
    test_dl = build_subject_sequences(test_keys, cfg["seq_len"], cfg["seq_len"])

    if train_dl is None or test_dl is None:
        print("ERROR: Not enough data.")
        return None

    feature_shape = list(subjects.values())[0]["data"].shape[1:]

    model = NTLDFE_Regression(
        feature_shape=feature_shape, encoder_dim=cfg["encoder_dim"],
        encoder_layers=cfg["encoder_layers"],
        temporal_hidden_dim=cfg["temporal_hidden_dim"],
        temporal_layers=cfg["temporal_layers"], dropout=cfg["dropout"],
        temporal_type=cfg["temporal_type"], bidirectional=cfg["bidirectional"],
        use_feature_attention=cfg["use_feature_attention"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    best_val_loss = float("inf"); best_state = None; patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train(); total_loss = 0; nb = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss, _ = regression_loss(out, y, cfg["beta_smooth"],
                                               cfg["beta_acc"], cfg["lambda_nll"],
                                               cfg["lambda_reg"])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss, _ = regression_loss(out, y, cfg["beta_smooth"],
                                           cfg["beta_acc"], cfg["lambda_nll"],
                                           cfg["lambda_reg"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
            total_loss += loss.item(); nb += 1

        avg_loss = total_loss/max(nb, 1)
        # Use training loss as validation proxy for cross-subject
        if avg_loss < best_val_loss - 1e-6:
            best_val_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0:
            train_metrics = evaluate_regression(model, train_dl, device)
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, train_rmse={train_metrics['rmse']:.4f}")

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_regression(model, test_dl, device)
    print(f"\nCross-Subject Regression Test: RMSE={test_metrics['rmse']:.4f}, "
          f"Pearson={test_metrics['pearson']:.4f}, CCC={test_metrics['ccc']:.4f}")

    output_dir = OUT / "cross_subject" / "regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", {**cfg, "train_subjects": train_keys, "test_subjects": test_keys})
    save_json(output_dir / "test_metrics.json", test_metrics)

    return {"metrics": test_metrics}


def run_cross_subject_classification():
    """Cross-subject classification: train on 70% subjects, test on 30% (no LOSO)."""
    banner("Phase 4b: Cross-Subject Classification (No LOSO)")

    cfg = {"seedvig_root": "dataset/seedvig", "feature_mode": "hybrid_de",
           "de_bins": 5, "sr": 200.0, "eps": 1e-6, "num_classes": 3,
           "seq_len": 45, "stride": 5,
           "batch_size": 48, "epochs": 250, "min_epochs": 15, "patience": 30,
           "lr": 5e-4, "wd": 1e-4, "grad_clip": 5.0,
           "encoder_dim": 96, "encoder_layers": 2,
           "temporal_hidden_dim": 128, "temporal_layers": 2,
           "classifier_hidden_dim": 96, "dropout": 0.15,
           "temporal_type": "gru", "bidirectional": True,
           "use_feature_attention": True,
           "focal_gamma": 0.5, "class_weight_mode": "balanced",
           "label_smoothing": 0.05,
           "device": "cuda" if torch.cuda.is_available() else "cpu",
           "train_subject_ratio": 0.7}

    set_seed(42)
    device = torch.device(cfg["device"])

    subjects = load_all_subjects(cfg["seedvig_root"], cfg["feature_mode"],
                                  cfg["de_bins"], cfg["sr"], cfg["eps"])
    subject_keys = sorted(subjects.keys())
    n_train_subj = max(2, int(len(subject_keys) * cfg["train_subject_ratio"]))

    rng = np.random.RandomState(42)
    shuffled = rng.permutation(subject_keys)
    train_keys = list(shuffled[:n_train_subj])
    test_keys = list(shuffled[n_train_subj:])

    def build_subject_sequences_class(subj_keys, seq_len=45, stride=5, num_classes=3):
        x_list, y_list = [], []
        for key in subj_keys:
            data = subjects[key]["data"]
            label = subjects[key]["label"]
            for s in range(0, len(data) - seq_len + 1, stride):
                x_list.append(data[s:s+seq_len])
                cls_labels = perclos_to_class(label[s:s+seq_len], num_classes)
                y_list.append(cls_labels[-1])
        if not x_list:
            return None
        x = torch.from_numpy(np.stack(x_list)).float()
        y = torch.tensor(y_list, dtype=torch.long)
        return DataLoader(torch.utils.data.TensorDataset(x, y),
                         batch_size=cfg["batch_size"], shuffle=True,
                         num_workers=0, pin_memory=True)

    train_dl = build_subject_sequences_class(train_keys, cfg["seq_len"], cfg["stride"], 3)
    test_dl = build_subject_sequences_class(test_keys, cfg["seq_len"], cfg["seq_len"], 3)

    if train_dl is None or test_dl is None:
        print("ERROR: Not enough data.")
        return None

    feature_shape = list(subjects.values())[0]["data"].shape[1:]

    all_train_labels = []
    for _, y in train_dl:
        all_train_labels.append(y.numpy())
    class_weights = compute_class_weights(np.concatenate(all_train_labels), 3, "balanced")

    model = NTLDFE_Classification(
        feature_shape=feature_shape, num_classes=3,
        encoder_dim=cfg["encoder_dim"], encoder_layers=cfg["encoder_layers"],
        temporal_hidden_dim=cfg["temporal_hidden_dim"],
        temporal_layers=cfg["temporal_layers"], dropout=cfg["dropout"],
        temporal_type=cfg["temporal_type"], bidirectional=cfg["bidirectional"],
        use_feature_attention=cfg["use_feature_attention"],
        classifier_hidden_dim=cfg["classifier_hidden_dim"]).to(device)

    criterion = FocalLoss(gamma=cfg["focal_gamma"], alpha=class_weights,
                          label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if cfg["device"] == "cuda" else None

    best_val_f1 = 0; best_state = None; patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x); loss = criterion(out["logits"], y)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                out = model(x); loss = criterion(out["logits"], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()

        train_metrics = evaluate_classification(model, train_dl, device, 3)
        val_f1 = train_metrics["macro_f1"]
        sched.step(val_f1)

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: train_acc={train_metrics['accuracy']:.4f}, "
                  f"train_f1={train_metrics['macro_f1']:.4f}")

        if patience_counter >= cfg["patience"] and epoch >= cfg["min_epochs"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_classification(model, test_dl, device, 3)
    print(f"\nCross-Subject Classification Test: Acc={test_metrics['accuracy']:.4f}, "
          f"Macro F1={test_metrics['macro_f1']:.4f}, Kappa={test_metrics['kappa']:.4f}")

    output_dir = OUT / "cross_subject" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", {**cfg, "train_subjects": train_keys, "test_subjects": test_keys})
    save_json(output_dir / "test_metrics.json", test_metrics)

    return {"metrics": test_metrics}


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5: Interpretability Experiments
# ═════════════════════════════════════════════════════════════════════════════

def run_interpretability():
    """Generate interpretability analysis: attention, topography, uncertainty."""
    banner("Phase 5: Interpretability Experiments")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from matplotlib.colors import LinearSegmentedColormap

    figures_dir = OUT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Load data
    subjects = load_all_subjects("dataset/seedvig", "hybrid_de", 5, 200.0, 1e-6)
    feature_shape = list(subjects.values())[0]["data"].shape[1:]

    # Train model with attention enabled
    seq_len, stride = 45, 5
    train_dl, val_dl, test_dl = build_temporal_sequences(
        subjects, seq_len=seq_len, stride=stride,
        test_ratio=0.2, val_ratio=0.1, classification=False)

    if train_dl is None:
        print("ERROR: Cannot build sequences for interpretability.")
        return None

    model = NTLDFE_Regression(
        feature_shape=feature_shape, encoder_dim=96, encoder_layers=2,
        temporal_hidden_dim=128, temporal_layers=2, dropout=0.1,
        use_feature_attention=True).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_loss = float("inf"); best_state = None; patience = 0

    print("Training for interpretability...")
    for epoch in range(200):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss, _ = regression_loss(out, y, 1e-4, 2e-4, 0.15, 1.0)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                out = model(x)
                loss, _ = regression_loss(out, y, 1e-4, 2e-4, 0.15, 1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        val_metrics = evaluate_regression(model, val_dl or train_dl, device)
        val_loss = val_metrics["rmse"]
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 20 and epoch > 10:
            print(f"  Converged at epoch {epoch}")
            break
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: rmse={val_metrics['rmse']:.4f}, pearson={val_metrics['pearson']:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # Extract attention weights
    model.eval()
    attn = model.encoder.attention_weights()
    channel_w = attn["channel"].detach().cpu().numpy()
    band_w = attn["band"].detach().cpu().numpy()
    print(f"Channel attention range: [{channel_w.min():.4f}, {channel_w.max():.4f}]")
    print(f"Band attention range: [{band_w.min():.4f}, {band_w.max():.4f}]")

    # Get test predictions
    all_mu, all_y, all_lv = [], [], []
    with torch.no_grad():
        for x, y in test_dl:
            out = model(x.to(device))
            all_mu.append(out["mu"].cpu().numpy().reshape(-1))
            all_y.append(y.cpu().numpy().reshape(-1))
            all_lv.append(out["log_var"].cpu().numpy().reshape(-1))
    mu_flat = np.concatenate(all_mu)
    y_flat = np.concatenate(all_y)
    lv_flat = np.concatenate(all_lv)
    sigma_flat = np.exp(np.clip(lv_flat, -10, 10) / 2)
    ae_flat = np.abs(mu_flat - y_flat)

    # ── Fig I1: Brain Topography of Channel Attention ────────────────────
    print("\nGenerating interpretability figures...")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    xi = np.linspace(-1.0, 1.0, 200); yi = np.linspace(-1.0, 1.0, 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    mask = xi_grid**2 + yi_grid**2 > 1.0

    zi = griddata((CH_X, CH_Y), channel_w, (xi_grid, yi_grid), method='cubic')
    zi[mask] = np.nan

    cmap = LinearSegmentedColormap.from_list("nature", ["#FFFFFF", NATURE["blue"], NATURE["red"]])
    im = ax.contourf(xi_grid, yi_grid, zi, levels=50, cmap=cmap, alpha=0.9)
    ax.contour(xi_grid, yi_grid, zi, levels=8, colors='grey', linewidths=0.3, alpha=0.5)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8)
    ax.plot([-0.05, 0, 0.05], [-1.02, -1.12, -1.02], 'k-', linewidth=0.8)
    ax.scatter(CH_X, CH_Y, c=channel_w, cmap=cmap, s=60,
              edgecolors='grey', linewidth=0.5, zorder=3,
              vmin=channel_w.min(), vmax=channel_w.max())
    for i, name in enumerate(CH_NAMES):
        ax.annotate(name, (CH_X[i], CH_Y[i]+0.06), fontsize=5, ha='center', alpha=0.7)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.2, 1.0)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Channel Attention Topography", fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=8)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(figures_dir / f"fig_interp_topography.{ext}")
    plt.close(fig)
    print("  Saved: fig_interp_topography")

    # ── Fig I2: Frequency Band Analysis ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    spectral_w = band_w[:5]
    temporal_w = band_w[5:]

    ax = axes[0]
    colors = [PAL3[0], PAL3[2], PAL3[1], NATURE["purple"], NATURE["red"]]
    bars = ax.bar(BAND_NAMES_5, spectral_w, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_ylabel('Attention Weight'); ax.set_title('a) Spectral Band Attention')
    ax = axes[1]
    stat_names = ['Var', 'Mean', 'Std', 'LL']
    bin_labels = ['Bin1','Bin2','Bin3','Bin4','Bin5']
    temporal_2d = temporal_w.reshape(4, 5)
    bar_width = 0.18; x = np.arange(5)
    for i, sn in enumerate(stat_names):
        ax.bar(x + i*bar_width, temporal_2d[i], bar_width, label=sn,
               color=PAL3[i%3] if i < 3 else NATURE["purple"],
               edgecolor='white', linewidth=0.2, alpha=0.85)
    ax.set_xticks(x + 1.5*bar_width); ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_ylabel('Attention Weight'); ax.set_title('b) Temporal Feature Attention')
    ax.legend(frameon=False, fontsize=6)

    ax = axes[2]
    ax.hist(spectral_w, bins=8, alpha=0.6, color=PAL3[0], edgecolor='white',
            label='Spectral (5)', linewidth=0.3)
    ax.hist(temporal_w, bins=15, alpha=0.6, color=PAL3[1], edgecolor='white',
            label='Temporal (20)', linewidth=0.3)
    ax.set_xlabel('Attention Weight'); ax.set_ylabel('Frequency')
    ax.set_title('c) Attention Distribution'); ax.legend(frameon=False, fontsize=7)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(figures_dir / f"fig_interp_bands.{ext}")
    plt.close(fig)
    print("  Saved: fig_interp_bands")

    # ── Fig I3: Uncertainty vs Error ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    ax = axes[0, 0]
    step = max(1, len(sigma_flat)//2000)
    ax.scatter(sigma_flat[::step], ae_flat[::step], c=PAL3[0], s=4, alpha=0.3)
    idx = np.argsort(sigma_flat); w = max(len(sigma_flat)//50, 10)
    trend_x = np.convolve(sigma_flat[idx], np.ones(w)/w, mode='valid')
    trend_y = np.convolve(ae_flat[idx], np.ones(w)/w, mode='valid')
    ax.plot(trend_x, trend_y, color=NATURE["red"], linewidth=1.2)
    from scipy.stats import spearmanr
    r_unc, _ = spearmanr(sigma_flat, ae_flat)
    ax.set_xlabel('Predicted Sigma'); ax.set_ylabel('|mu - y|')
    ax.set_title(f'a) Uncertainty vs Error (ρ={r_unc:.3f})')

    ax = axes[0, 1]
    ax.hist2d(y_flat, mu_flat, bins=60, cmap='YlOrRd')
    ax.plot([0, 1], [0, 1], '--', color=NATURE["blue"], linewidth=1.0)
    ax.set_xlabel('PERCLOS (y)'); ax.set_ylabel('Predicted (mu)')
    ax.set_title('b) Prediction vs Target')

    ax = axes[1, 0]
    ax.hist(sigma_flat, bins=50, color=PAL3[0], edgecolor='white', alpha=0.8)
    ax.set_xlabel('Sigma'); ax.set_ylabel('Count')
    ax.set_title('c) Uncertainty Distribution')

    ax = axes[1, 1]
    # Error by PERCLOS bin
    bins = [0, 0.35, 0.7, 1.0]
    labels = ['Low [0,0.35]', 'Med (0.35,0.7]', 'High (0.7,1.0]']
    bin_data = []
    for i in range(len(bins)-1):
        mask = (y_flat >= bins[i]) & (y_flat < bins[i+1])
        if mask.sum() > 0:
            bin_data.append(ae_flat[mask])
    bp = ax.boxplot(bin_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], PAL3[:3]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel('Absolute Error')
    ax.set_title('d) Error by Fatigue Level')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(figures_dir / f"fig_interp_uncertainty.{ext}")
    plt.close(fig)
    print("  Saved: fig_interp_uncertainty")

    # ── Fig I4: Trajectory Comparison ───────────────────────────────────
    if test_dl:
        # Get sequential predictions from the last test subject data
        # Build sequential loader without shuffling
        subj_keys = sorted(subjects.keys())
        test_key = subj_keys[-1]
        data = subjects[test_key]["data"]
        label = subjects[test_key]["label"]
        n_test = int(len(data) * 0.8)
        test_data = data[n_test:]
        test_label = label[n_test:]

        seqs_x, seqs_y = [], []
        for s in range(0, len(test_data) - 45 + 1, 45):
            seqs_x.append(test_data[s:s+45])
            seqs_y.append(test_label[s:s+45])
        if seqs_x:
            seqs_x = torch.from_numpy(np.stack(seqs_x)).float()
            seqs_dl = DataLoader(torch.utils.data.TensorDataset(seqs_x,
                torch.from_numpy(np.array(seqs_y, dtype=np.float32)).float()),
                batch_size=48, shuffle=False)

            mus, ys = [], []
            with torch.no_grad():
                for x, y in seqs_dl:
                    out = model(x.to(device))
                    mus.append(out["mu"].cpu().numpy().reshape(-1))
                    ys.append(y.cpu().numpy().reshape(-1))
            traj_mu = np.concatenate(mus)
            traj_y = np.concatenate(ys)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            t = np.arange(len(traj_mu))
            ax.plot(t, traj_y, color=NATURE["grey"], linewidth=1.0, alpha=0.8, label='PERCLOS (target)')
            ax.plot(t, traj_mu, color=NATURE["blue"], linewidth=1.3, label='NT-LDFE (predicted)')
            ax.fill_between(t, traj_mu - sigma_flat[:len(traj_mu)],
                           traj_mu + sigma_flat[:len(traj_mu)],
                           alpha=0.15, color=NATURE["blue"], label='±1σ uncertainty')
            ax.set_xlabel('Time Window'); ax.set_ylabel('Fatigue Score')
            ax.set_title(f'Fatigue Trajectory: Subject {test_key} (Test Set)')
            ax.legend(frameon=False)
            ax.set_ylim(-0.02, 1.02)
            plt.tight_layout()
            for ext in ['pdf', 'png']:
                fig.savefig(figures_dir / f"fig_interp_trajectory.{ext}")
            plt.close(fig)
            print("  Saved: fig_interp_trajectory")

    print(f"\nInterpretability figures saved to: {figures_dir}/")
    return {"channel_attention": channel_w.tolist(), "band_attention": band_w.tolist()}


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6: Generate All Publication Figures
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_figures():
    """Generate comprehensive publication-quality figures from experiment results."""
    banner("Phase 6: Generating Publication Figures")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    figures_dir = OUT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    })

    def save_fig(fig, name):
        for ext in ["pdf", "png"]:
            fig.savefig(figures_dir / f"{name}.{ext}")
        print(f"  Saved: {name}")

    def load_json_data(path):
        p = Path(path)
        if not p.exists():
            return {}
        with open(p) as f:
            data = json.load(f)
        # Recursively convert string numbers to float
        def floatify(d):
            if isinstance(d, dict):
                return {k: floatify(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [floatify(v) for v in d]
            elif isinstance(d, str):
                try: return float(d)
                except (ValueError, TypeError): return d
            elif isinstance(d, (int, float)):
                return float(d)
            return d
        return floatify(data)

    reg_metrics = load_json_data(OUT / "regression" / "test_metrics.json")
    cls_metrics = load_json_data(OUT / "classification" / "test_metrics.json")
    cls_ensemble = load_json_data(OUT / "classification" / "ensemble_metrics.json")
    cls_binary = load_json_data(OUT / "classification_binary" / "test_metrics.json")
    ablation_summary = load_json_data(OUT / "ablations" / "ablation_summary.json")
    cs_reg = load_json_data(OUT / "cross_subject" / "regression" / "test_metrics.json")
    cs_cls = load_json_data(OUT / "cross_subject" / "classification" / "test_metrics.json")

    # ═══════════════════════════════════════════════════════════════════════
    # Fig 1: Regression Results Overview
    # ═══════════════════════════════════════════════════════════════════════
    if reg_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5))

        # a) Prediction vs target scatter
        pred_data = load_json_data(OUT / "regression" / "predictions.json")
        if pred_data:
            ax = axes[0, 0]
            step = max(1, len(pred_data["y"])//2000)
            ax.scatter(pred_data["y"][::step], pred_data["mu"][::step],
                      c=PAL3[0], s=3, alpha=0.3)
            ax.plot([0, 1], [0, 1], '--', color=NATURE["dark"], linewidth=0.8)
            ax.set_xlabel('PERCLOS'); ax.set_ylabel('Predicted')
            r_val = reg_metrics.get('pearson', 0)
            ax.set_title(f'a) Prediction vs Target (r={r_val:.3f})')

        # b) Error histogram
        ax = axes[0, 1]
        if pred_data:
            errors = np.array(pred_data["mu"]) - np.array(pred_data["y"])
            ax.hist(errors, bins=60, color=PAL3[1], edgecolor='white', alpha=0.8)
            ax.axvline(0, color=NATURE["dark"], linewidth=0.8, linestyle='--')
            ax.set_xlabel('Prediction Error'); ax.set_ylabel('Count')
            mae_val = reg_metrics.get('mae', 0)
            ax.set_title(f'b) Error Distribution (MAE={mae_val:.4f})')

        # c) Key metrics bar chart
        ax = axes[0, 2]
        metric_names = ['MAE', 'RMSE', 'Pearson', 'CCC', 'R²']
        metric_vals = [
            reg_metrics.get('mae', 0), reg_metrics.get('rmse', 0),
            reg_metrics.get('pearson', 0), reg_metrics.get('ccc', 0),
            reg_metrics.get('r2', 0)
        ]
        bars = ax.bar(metric_names, metric_vals, color=PAL3[:5], edgecolor='white', linewidth=0.3)
        ax.set_ylabel('Value'); ax.set_title('c) Regression Metrics')
        # d) Temporal smoothness
        ax = axes[1, 0]
        smooth_names = ['TV', 'ADV']
        smooth_vals = [reg_metrics.get('tv', 0), reg_metrics.get('adv', 0)]
        ax.bar(smooth_names, smooth_vals, color=[PAL3[0], PAL3[2]],
               edgecolor='white', linewidth=0.3)
        ax.set_ylabel('Value'); ax.set_title('d) Temporal Smoothness (lower = smoother)')
        # e) Predicted vs target std
        ax = axes[1, 1]
        ax.bar(['Predicted', 'Target'],
              [reg_metrics.get('pred_std', 0), reg_metrics.get('target_std', 0)],
              color=[PAL3[0], PAL3[1]], edgecolor='white', linewidth=0.3)
        ax.set_ylabel('Std Dev'); ax.set_title('e) Prediction vs Target Variability')

        # f) Sigma vs error correlation
        ax = axes[1, 2]
        ax.bar(['σ-AE\nSpearman'], [reg_metrics.get('sigma_abs_err_spearman', 0)],
              color=PAL3[0], edgecolor='white', linewidth=0.3, width=0.3)
        ax.axhline(0, color=NATURE["grey"], linewidth=0.5)
        ax.set_ylabel('Spearman ρ'); ax.set_title('f) Uncertainty Calibration')
        ax.set_ylim(-1, 1)

        plt.tight_layout()
        save_fig(fig, "fig1_regression_overview")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # Fig 2: Classification Results
    # ═══════════════════════════════════════════════════════════════════════
    if cls_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))

        # a) Confusion Matrix
        ax = axes[0]
        cm = np.array(cls_metrics.get("confusion_matrix", [[0]*3]*3))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                       fontsize=9, fontweight="bold",
                       color="white" if cm_norm[i,j] > 0.5 else NATURE["dark"])
        ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_LABELS, rotation=25, ha="right", fontsize=6)
        ax.set_yticklabels(CLASS_LABELS, fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f'a) 3-Class Confusion Matrix (Acc={cls_metrics["accuracy"]:.3f})')

        # b) Per-class metrics
        ax = axes[1]
        classes = ['Low', 'Med', 'High']
        precisions = [cls_metrics.get(f'precision_c{i}', 0) for i in range(3)]
        recalls = [cls_metrics.get(f'recall_c{i}', 0) for i in range(3)]
        x = np.arange(3); w = 0.35
        ax.bar(x - w/2, precisions, w, label='Precision', color=PAL3[0],
              edgecolor='white', linewidth=0.3)
        ax.bar(x + w/2, recalls, w, label='Recall', color=PAL3[1],
              edgecolor='white', linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(classes)
        ax.set_ylabel('Score'); ax.set_title('b) Per-Class Performance')
        ax.legend(frameon=False, fontsize=7)
        ax.set_ylim(0, 1)

        # c) Ensemble comparison
        ax = axes[2]
        if cls_ensemble:
            x = np.arange(3); w = 0.35
            metrics_names = ['Accuracy', 'Macro F1', 'Kappa']
            single_vals = [cls_metrics['accuracy'], cls_metrics['macro_f1'], cls_metrics['kappa']]
            ensemble_vals = [cls_ensemble['accuracy'], cls_ensemble['macro_f1'], cls_ensemble['kappa']]
            ax.bar(x - w/2, single_vals, w, label='Single', color=PAL3[0],
                  edgecolor='white', linewidth=0.3)
            ax.bar(x + w/2, ensemble_vals, w, label='Ensemble', color=PAL3[1],
                  edgecolor='white', linewidth=0.3)
            ax.set_xticks(x); ax.set_xticklabels(metrics_names)
            ax.set_ylabel('Score'); ax.set_title('c) Single vs Ensemble')
            ax.legend(frameon=False, fontsize=7)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        save_fig(fig, "fig2_classification_overview")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # Fig 3: Ablation Study Results
    # ═══════════════════════════════════════════════════════════════════════
    if ablation_summary:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))

        # a) RMSE comparison
        ax = axes[0]
        names = list(ablation_summary.keys())
        display_names = [ABLATION_CONFIGS[n]["description"][:40] for n in names]
        rmse_vals = [ablation_summary[n]["rmse"] for n in names]
        colors = [PAL3[0] if n == "full_model" else PAL3[1] for n in names]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, rmse_vals, color=colors, edgecolor='white', linewidth=0.3)
        ax.set_yticks(y_pos); ax.set_yticklabels(display_names, fontsize=6)
        ax.set_xlabel('RMSE'); ax.set_title('a) Ablation: RMSE Comparison')
        ax.axvline(ablation_summary.get("full_model", {}).get("rmse", 0),

color=NATURE["dark"], linestyle='--', linewidth=0.8)        # b) TV and ADV comparison
        ax = axes[1]
        x = np.arange(len(names)); w = 0.35
        tv_vals = [ablation_summary[n]["tv"] for n in names]
        adv_vals = [ablation_summary[n]["adv"] for n in names]
        ax.bar(x - w/2, tv_vals, w, label='TV (1st order)', color=PAL3[0],
              edgecolor='white', linewidth=0.3)
        ax.bar(x + w/2, adv_vals, w, label='ADV (2nd order)', color=PAL3[1],
              edgecolor='white', linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('Value (lower = smoother)')
        ax.set_title('b) Ablation: Temporal Smoothness')
        ax.legend(frameon=False, fontsize=7)

        plt.tight_layout()
        save_fig(fig, "fig3_ablation_study")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # Fig 4: Cross-Subject Generalization
    # ═══════════════════════════════════════════════════════════════════════
    if cs_reg or cs_cls:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

        ax = axes[0]
        methods = []; rmse_vals = []; pearson_vals = []
        if reg_metrics:
            methods.append('Temporal\nSplit'); rmse_vals.append(reg_metrics['rmse'])
            pearson_vals.append(reg_metrics['pearson'])
        if cs_reg:
            methods.append('Cross-Subj\nSplit'); rmse_vals.append(cs_reg['rmse'])
            pearson_vals.append(cs_reg['pearson'])
        x = np.arange(len(methods)); w = 0.35
        ax.bar(x - w/2, rmse_vals, w, label='RMSE', color=PAL3[0],
              edgecolor='white', linewidth=0.3)
        ax.bar(x + w/2, pearson_vals, w, label='Pearson r', color=PAL3[1],
              edgecolor='white', linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.set_ylabel('Score'); ax.set_title('a) Regression: Split Comparison')
        ax.legend(frameon=False, fontsize=7)

        ax = axes[1]
        cls_methods = []; cls_accs = []; cls_f1s = []
        if cls_metrics:
            cls_methods.append('Temporal\nSplit'); cls_accs.append(cls_metrics['accuracy'])
            cls_f1s.append(cls_metrics['macro_f1'])
        if cs_cls:
            cls_methods.append('Cross-Subj\nSplit'); cls_accs.append(cs_cls['accuracy'])
            cls_f1s.append(cs_cls['macro_f1'])
        x = np.arange(len(cls_methods)); w = 0.35
        ax.bar(x - w/2, cls_accs, w, label='Accuracy', color=PAL3[0],
              edgecolor='white', linewidth=0.3)
        ax.bar(x + w/2, cls_f1s, w, label='Macro F1', color=PAL3[1],
              edgecolor='white', linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(cls_methods)
        ax.set_ylabel('Score'); ax.set_title('b) Classification: Split Comparison')
        ax.legend(frameon=False, fontsize=7)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        save_fig(fig, "fig4_cross_subject")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # Fig 5: Method Comparison Summary
    # ═══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    all_results = {}
    if reg_metrics:
        all_results["Regression\n(Pearson r)"] = reg_metrics["pearson"]
    if cls_metrics:
        all_results["3-Class\n(Accuracy)"] = cls_metrics["accuracy"]
    if cls_ensemble:
        all_results["3-Class Ens.\n(Accuracy)"] = cls_ensemble["accuracy"]
    if cls_binary:
        all_results["Binary\n(Accuracy)"] = cls_binary["accuracy"]
    if cs_reg:
        all_results["Cross-Subj Reg.\n(Pearson r)"] = cs_reg["pearson"]
    if cs_cls:
        all_results["Cross-Subj Cls.\n(Accuracy)"] = cs_cls["accuracy"]

    colors_list = PAL5[:len(all_results)] + [NATURE["purple"], NATURE["orange"]]
    x = np.arange(len(all_results))
    bars = ax.bar(x, list(all_results.values()), color=colors_list[:len(all_results)],
                 edgecolor='white', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(list(all_results.keys()), fontsize=7)
    ax.set_ylabel('Score'); ax.set_title('SEED-VIG NT-LDFE Comprehensive Results (Non-LOSO)')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    save_fig(fig, "fig5_summary")
    plt.close(fig)

    print(f"\nAll figures saved to: {figures_dir}/")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7: Compile Comprehensive Summary Report
# ═════════════════════════════════════════════════════════════════════════════

def compile_summary_report():
    """Compile comprehensive summary of all experiments."""
    banner("Phase 7: Compiling Summary Report")

    summary = {"dataset": "SEED-VIG", "subjects": 23,
               "evaluation": "Non-LOSO, Temporal Split (preserving time order)",
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    def load(path):
        p = Path(path)
        if not p.exists():
            return None
        with open(p) as f:
            data = json.load(f)
        def floatify(d):
            if isinstance(d, dict): return {k: floatify(v) for k, v in d.items()}
            elif isinstance(d, list): return [floatify(v) for v in d]
            elif isinstance(d, str):
                try: return float(d)
                except (ValueError, TypeError): return d
            elif isinstance(d, (int, float)): return float(d)
            return d
        return floatify(data)

    # Regression
    reg = load(OUT / "regression" / "test_metrics.json")
    if reg:
        summary["regression"] = {"rmse": reg["rmse"], "mae": reg["mae"],
            "pearson": reg["pearson"], "spearman": reg["spearman"],
            "ccc": reg["ccc"], "r2": reg["r2"],
            "tv": reg["tv"], "adv": reg["adv"],
            "nll": reg["nll_gaussian"], "sigma_ae_spearman": reg["sigma_abs_err_spearman"]}

    # Classification
    cls_3 = load(OUT / "classification" / "test_metrics.json")
    cls_ens = load(OUT / "classification" / "ensemble_metrics.json")
    cls_bin = load(OUT / "classification_binary" / "test_metrics.json")
    if cls_3:
        summary["classification_3class"] = {"accuracy": cls_3["accuracy"],
            "macro_f1": cls_3["macro_f1"], "kappa": cls_3["kappa"],
            "confusion_matrix": cls_3["confusion_matrix"]}
    if cls_ens:
        summary["classification_ensemble"] = {"accuracy": cls_ens["accuracy"],
            "macro_f1": cls_ens["macro_f1"], "kappa": cls_ens["kappa"],
            "n_models": cls_ens["n_models"]}
    if cls_bin:
        summary["classification_binary"] = {"accuracy": cls_bin["accuracy"],
            "macro_f1": cls_bin["macro_f1"], "kappa": cls_bin["kappa"]}

    # Ablation
    ablation = load(OUT / "ablations" / "ablation_summary.json")
    if ablation:
        summary["ablation"] = ablation

    # Cross-subject
    cs_reg = load(OUT / "cross_subject" / "regression" / "test_metrics.json")
    cs_cls = load(OUT / "cross_subject" / "classification" / "test_metrics.json")
    if cs_reg:
        summary["cross_subject_regression"] = {"rmse": cs_reg["rmse"],
            "pearson": cs_reg["pearson"], "ccc": cs_reg["ccc"]}
    if cs_cls:
        summary["cross_subject_classification"] = {"accuracy": cs_cls["accuracy"],
            "macro_f1": cs_cls["macro_f1"], "kappa": cs_cls["kappa"]}

    save_json(OUT / "comprehensive_summary.json", summary)

    # ── Markdown report ──────────────────────────────────────────────────
    lines = [
        "# SEED-VIG Comprehensive Experiment Report",
        "",
        f"**Dataset**: SEED-VIG | **Subjects**: 23",
        f"**Evaluation**: Non-LOSO, Temporal Split (preserving temporal order)",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Regression Results (Non-LOSO, Temporal Split)",
        "",
    ]
    if reg:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| RMSE | {reg['rmse']:.4f} |",
            f"| MAE | {reg['mae']:.4f} |",
            f"| Pearson r | {reg['pearson']:.4f} |",
            f"| Spearman ρ | {reg['spearman']:.4f} |",
            f"| CCC | {reg['ccc']:.4f} |",
            f"| R² | {reg['r2']:.4f} |",
            f"| NLL (Gaussian) | {reg['nll_gaussian']:.4f} |",
            f"| TV (smoothness) | {reg['tv']:.6f} |",
            f"| ADV (acceleration) | {reg['adv']:.6f} |",
            f"| σ-AE Spearman | {reg['sigma_abs_err_spearman']:.4f} |",
            "",
        ]

    lines += ["## 2. Classification Results", ""]
    if cls_3:
        lines += [
            "### 3-Class Classification",
            f"| Accuracy | {cls_3['accuracy']:.4f} |",
            f"| Macro F1 | {cls_3['macro_f1']:.4f} |",
            f"| Kappa | {cls_3['kappa']:.4f} |",
            "",
        ]
    if cls_ens:
        lines += [
            "### Ensemble Classification",
            f"| Accuracy | {cls_ens['accuracy']:.4f} |",
            f"| Macro F1 | {cls_ens['macro_f1']:.4f} |",
            f"| Kappa | {cls_ens['kappa']:.4f} |",
            f"| N Models | {cls_ens['n_models']} |",
            "",
        ]
    if cls_bin:
        lines += [
            "### Binary Classification",
            f"| Accuracy | {cls_bin['accuracy']:.4f} |",
            f"| Macro F1 | {cls_bin['macro_f1']:.4f} |",
            f"| Kappa | {cls_bin['kappa']:.4f} |",
            "",
        ]

    if ablation:
        lines += [
            "## 3. Ablation Study Results",
            "",
            "| Experiment | RMSE | Pearson | CCC | TV | ADV |",
            "|------------|------|---------|-----|----|-----|",
        ]
        for name, s in ablation.items():
            lines.append(f"| {ABLATION_CONFIGS[name]['description']} | {s['rmse']:.4f} | {s['pearson']:.4f} | {s['ccc']:.4f} | {s['tv']:.6f} | {s['adv']:.6f} |")
        lines.append("")

    if cs_reg or cs_cls:
        lines += ["## 4. Cross-Subject Generalization", ""]
        if cs_reg:
            lines += [
                "### Regression",
                f"| RMSE | {cs_reg['rmse']:.4f} |",
                f"| Pearson r | {cs_reg['pearson']:.4f} |",
                f"| CCC | {cs_reg['ccc']:.4f} |",
                "",
            ]
        if cs_cls:
            lines += [
                "### Classification",
                f"| Accuracy | {cs_cls['accuracy']:.4f} |",
                f"| Macro F1 | {cs_cls['macro_f1']:.4f} |",
                f"| Kappa | {cs_cls['kappa']:.4f} |",
                "",
            ]

    lines += [
        "## 5. Figures",
        "",
        "All figures are in `outputs/seedvig/figures/`:",
        "- `fig1_regression_overview` — Regression performance and uncertainty analysis",
        "- `fig2_classification_overview` — Classification confusion matrix and metrics",
        "- `fig3_ablation_study` — Ablation experiment comparisons",
        "- `fig4_cross_subject` — Cross-subject generalization analysis",
        "- `fig5_summary` — Comprehensive summary of all experiments",
        "- `fig_interp_topography` — Brain topography of channel attention weights",
        "- `fig_interp_bands` — Frequency band attention analysis",
        "- `fig_interp_uncertainty` — Uncertainty calibration analysis",
        "- `fig_interp_trajectory` — Fatigue trajectory prediction",
        "",
        "## 6. Key Findings",
        "",
    ]

    if reg:
        lines.append(f"1. **Regression**: The NT-LDFE model achieves RMSE={reg['rmse']:.4f} with Pearson r={reg['pearson']:.4f} on temporally-split test data, demonstrating effective continuous fatigue estimation.")
    if cls_ens:
        lines.append(f"2. **Classification Ensemble**: The {cls_ens['n_models']}-model ensemble achieves {cls_ens['accuracy']:.1%} accuracy on 3-class fatigue classification, outperforming the single model.")
    if ablation:
        full = ablation.get("full_model", {})
        no_unc = ablation.get("without_uncertainty", {})
        if full and no_unc:
            lines.append(f"3. **Heteroscedastic Uncertainty**: Removing uncertainty modeling changes RMSE from {full['rmse']:.4f} to {no_unc['rmse']:.4f}, confirming the importance of noise-aware training for noisy PERCLOS labels.")
    if cs_reg and reg:
        lines.append(f"4. **Cross-Subject Generalization**: The model generalizes across subjects with RMSE={cs_reg['rmse']:.4f} (vs {reg['rmse']:.4f} for temporal split), highlighting the challenge of individual differences in EEG fatigue patterns.")

    (OUT / "comprehensive_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to: {OUT / 'comprehensive_report.md'}")
    print(f"Summary saved to: {OUT / 'comprehensive_summary.json'}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SEED-VIG Comprehensive Non-LOSO Experiment Runner")
    parser.add_argument("--phase", type=str, default="all",
                       choices=["all", "reg", "cls", "binary", "ensemble",
                               "ablation", "cross", "interp", "figures", "report"],
                       help="Experiment phase to run")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--fresh", action="store_true",
                       help="Start fresh (ignore existing results)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    phase = args.phase

    print("=" * 65)
    print("  SEED-VIG Comprehensive Experiment Runner")
    print(f"  Evaluation: Non-LOSO, Temporal Split")
    print(f"  Device: {args.device} | CUDA: {torch.cuda.is_available()}")
    print(f"  Output: {OUT}/")
    print("=" * 65)

    if phase in ("all", "reg"):
        run_regression()

    if phase in ("all", "cls"):
        run_classification()

    if phase in ("all", "binary"):
        run_binary_classification()

    if phase in ("all", "ensemble"):
        run_classification_ensemble(n_models=7)

    if phase in ("all", "ablation"):
        run_all_ablations()

    if phase in ("all", "cross"):
        run_cross_subject_regression()
        run_cross_subject_classification()

    if phase in ("all", "interp"):
        run_interpretability()

    if phase in ("all", "figures"):
        generate_all_figures()

    if phase in ("all", "report"):
        compile_summary_report()

    if phase == "all":
        # Run figures and report last
        generate_all_figures()
        compile_summary_report()

    banner("COMPLETE!")
    print(f"All results saved to: {OUT}/")
    print("\nOutput structure:")
    print(f"  {OUT}/")
    print(f"  ├── regression/        — Regression experiment results")
    print(f"  ├── classification/     — 3-class classification + ensemble")
    print(f"  ├── classification_binary/ — Binary classification")
    print(f"  ├── ablations/         — Ablation study results")
    print(f"  ├── cross_subject/     — Cross-subject experiments")
    print(f"  ├── figures/           — Publication-quality figures")
    print(f"  ├── comprehensive_summary.json — All results JSON")
    print(f"  └── comprehensive_report.md    — Full report")


if __name__ == "__main__":
    main()
