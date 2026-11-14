# -*- coding: utf-8 -*-
"""Channel & band ablation importance for NT-LDFE interpretability.
Measures RMSE increase when each channel/band is zeroed out."""

import json, sys, warnings, numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE)); warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

NATURE = {"blue":"#4DBBD5","red":"#E64B35","green":"#00A087","orange":"#F39B7F",
          "purple":"#8491B4","dark":"#2C3E50","grey":"#7F8C8D"}
PAL = [NATURE["blue"],NATURE["red"],NATURE["green"],NATURE["orange"],NATURE["purple"]]

# Correct SEED-VIG 17-channel layout (top-down view, nose at top)
CHANNEL_MAP_17 = [
    ("FT7",-0.80,0.30),("FT8",0.80,0.30),  # Frontal-temporal
    ("T7",-0.88,0.0),("T8",0.88,0.0),       # Temporal
    ("TP7",-0.80,-0.28),("TP8",0.80,-0.28), # Temporal-parietal
    ("CP1",-0.28,-0.15),("CP2",0.28,-0.15), # Centro-parietal
    ("P1",-0.20,-0.35),("PZ",0.0,-0.35),("P2",0.20,-0.35), # Parietal
    ("PO3",-0.22,-0.58),("POZ",0.0,-0.58),("PO4",0.22,-0.58), # Parieto-occipital
    ("O1",-0.22,-0.80),("OZ",0.0,-0.85),("O2",0.22,-0.80), # Occipital
]
CH_NAMES = [c[0] for c in CHANNEL_MAP_17]
CH_X = np.array([c[1] for c in CHANNEL_MAP_17])
CH_Y = np.array([c[2] for c in CHANNEL_MAP_17])
BAND_NAMES = [u"δ(1-4)","θ(4-8)","α(8-13)","β(13-30)","γ(30-45)"]
STAT_NAMES = ["logVar","Mean","Std","LineLen"]

FIG_DIR = BASE / "outputs/seedvig/figures"
FIG_DIR2 = BASE / "paper/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR2.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":9,"axes.titlesize":10,"axes.labelsize":9,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
    "axes.linewidth":0.5,"axes.spines.top":False,"axes.spines.right":False,
})

def save_fig(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(FIG_DIR / f"{name}.{ext}")
        fig.savefig(FIG_DIR2 / f"{name}.{ext}")
    print(f"  Saved: {name}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Train model and compute ablation importance
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_ablate():
    from nt_ldfe_loso import Config, load_subjects, labels_to_unit_interval, compute_train_stats, normalize, ordered_subject_keys
    from nt_ldfe_model import NTLDFE, NTLDFEConfig

    cfg = Config()
    cfg.dataset_name = "seedvig"; cfg.seedvig_root = "dataset/seedvig"
    cfg.seedvig_feature_mode = "hybrid_de"; cfg.seedvig_de_bins = 5
    cfg.seedvig_sampling_rate = 200.0; cfg.sequence_length = 45; cfg.train_stride = 5
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    subjects = load_subjects(cfg); keys = ordered_subject_keys(subjects)
    print(f"Loaded {len(subjects)} subjects on {cfg.device}")

    # Train on 20 subjects, test on 3
    test_keys = set(keys[-3:])
    train_arrays, train_labels, test_arrays, test_labels = [], [], [], []
    for k in keys:
        d = np.asarray(subjects[k]["data"], dtype=np.float32)
        l = labels_to_unit_interval(subjects[k]["label"])
        (test_arrays if k in test_keys else train_arrays).append(d)
        (test_labels if k in test_keys else train_labels).append(l)

    mean, std = compute_train_stats(train_arrays)
    train_arrays = [normalize(a, mean, std) for a in train_arrays]
    test_arrays = [normalize(a, mean, std) for a in test_arrays]

    feature_shape = tuple(train_arrays[0].shape[1:])  # (17, 25)
    print(f"Feature shape: {feature_shape}")
    n_channels, n_bands = feature_shape

    model_cfg = NTLDFEConfig(
        feature_shape=feature_shape, encoder_dim=96, encoder_layers=2,
        temporal_hidden_dim=128, temporal_layers=2, dropout=0.1,
        bidirectional=True, use_feature_attention=True,
    )
    device = torch.device(cfg.device)

    class SeqDataset(Dataset):
        def __init__(self, arrays, labels, seq_len, stride):
            self.idx = []
            for si, (a, l) in enumerate(zip(arrays, labels)):
                for s in range(0, len(a)-seq_len+1, stride):
                    self.idx.append((si, s))
            self.arrays = [torch.from_numpy(a).float() for a in arrays]
            self.labels = [torch.from_numpy(l).float() for l in labels]
            self.seq_len = seq_len
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            si, s = self.idx[i]; e = s + self.seq_len
            return self.arrays[si][s:e], self.labels[si][s:e]

    # Train full model
    ds = SeqDataset(train_arrays, train_labels, 45, 5)
    dl = DataLoader(ds, batch_size=48, shuffle=True)

    model = NTLDFE(config=model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)

    print("Training full model (200 epochs)...")
    for epoch in range(200):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            out = model(x); mu, lv = out["mu"], out["log_var"]
            loss = (0.5 * torch.exp(-lv) * (y - mu).pow(2) + 0.5 * lv).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        if (epoch+1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}")

    # Baseline RMSE on test set
    def compute_rmse(m, arrays, labels):
        m.eval()
        test_ds = SeqDataset(arrays, labels, 45, 45)
        test_dl = DataLoader(test_ds, batch_size=48, shuffle=False)
        all_mu, all_y = [], []
        with torch.no_grad():
            for x, y in test_dl:
                out = m(x.to(device))
                all_mu.append(out["mu"].cpu().numpy())
                all_y.append(y.numpy())
        mu_f = np.concatenate([m.reshape(-1) for m in all_mu])
        y_f = np.concatenate([y.reshape(-1) for y in all_y])
        return np.sqrt(np.mean((mu_f - y_f)**2)), mu_f, y_f

    baseline_rmse, baseline_mu, baseline_y = compute_rmse(model, test_arrays, test_labels)
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # ── Channel ablation ──
    print("\nAblating channels...")
    channel_impact = []
    for ch in range(n_channels):
        # Zero out channel ch in test data
        ablated_arrays = []
        for a in test_arrays:
            a_copy = a.copy()
            a_copy[:, ch, :] = 0.0  # zero all bands for this channel
            ablated_arrays.append(a_copy)
        rmse_ch, _, _ = compute_rmse(model, ablated_arrays, test_labels)
        delta = rmse_ch - baseline_rmse
        channel_impact.append(float(delta))
        if ch < 5 or delta > 0.003:
            print(f"  Ch {ch:2d} ({CH_NAMES[ch]:4s}): RMSE={rmse_ch:.4f} delta={delta:+.4f}")

    # ── Band ablation ──
    print("\nAblating spectral bands...")
    band_impact = []
    for b in range(5):  # first 5 are spectral bands
        ablated_arrays = []
        for a in test_arrays:
            a_copy = a.copy()
            a_copy[:, :, b] = 0.0  # zero all channels for this band
            ablated_arrays.append(a_copy)
        rmse_b, _, _ = compute_rmse(model, ablated_arrays, test_labels)
        delta = rmse_b - baseline_rmse
        band_impact.append(float(delta))
        print(f"  Band {b} ({BAND_NAMES[b]}): RMSE={rmse_b:.4f} delta={delta:+.4f}")

    # ── Temporal feature ablation ──
    print("\nAblating temporal features...")
    stat_impact = [0.0] * 4  # logVar, Mean, Std, LineLen
    for s in range(4):
        ablated_arrays = []
        for a in test_arrays:
            a_copy = a.copy()
            for b in range(5):  # 5 temporal bins
                a_copy[:, :, 5 + b*4 + s] = 0.0
            ablated_arrays.append(a_copy)
        rmse_s, _, _ = compute_rmse(model, ablated_arrays, test_labels)
        delta = rmse_s - baseline_rmse
        stat_impact[s] = float(delta)
        print(f"  {STAT_NAMES[s]}: RMSE={rmse_s:.4f} delta={delta:+.4f}")

    return (np.array(channel_impact), np.array(band_impact), np.array(stat_impact),
            baseline_rmse, baseline_mu, baseline_y)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Brain Topography of Channel Importance
# ═══════════════════════════════════════════════════════════════════════════════

def plot_brain_importance(channel_impact, title, filename):
    """Plot channel ablation importance as MNE topomap."""
    from mne_topomap_utils import _get_seedvig_info, SEEDVIG_CH_NAMES
    import mne

    info = _get_seedvig_info()
    data = np.asarray(channel_impact, dtype=np.float64).ravel()

    fig, ax = plt.subplots(figsize=(5, 4.5))

    evoked = mne.EvokedArray(data[:, np.newaxis], info, tmin=0.0)
    vmax = max(abs(data.min()), abs(data.max()), 0.001)
    im, _ = mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(0, vmax),
        sensors=True,
        contours=0,
        sphere=(0, 0, 0, 0.105),
        names=SEEDVIG_CH_NAMES,
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("RMSE Increase (Delta)", fontsize=8)
    plt.tight_layout(); save_fig(fig, filename); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Band & Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def plot_band_importance(band_impact, stat_impact, filename):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    colors = [PAL[0], PAL[2], PAL[1], PAL[3], NATURE["red"]]
    bars = ax.bar(BAND_NAMES, band_impact, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_ylabel('RMSE Increase'); ax.set_title('a) Spectral Band Importance (Ablation)')

    ax = axes[1]
    bars = ax.bar(STAT_NAMES, stat_impact, color=PAL[:4], edgecolor='white', linewidth=0.3)

    ax.set_ylabel('RMSE Increase'); ax.set_title('b) Temporal Statistic Importance')
    plt.tight_layout(); save_fig(fig, filename); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Regional Summary
# ═══════════════════════════════════════════════════════════════════════════════

def plot_region_summary(channel_impact, filename):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    frontal_temporal = [0,1]; temporal = [2,3]; temporal_parietal = [4,5]
    centro_parietal = [6,7]; parietal = [8,9,10]; parieto_occipital = [11,12,13]; occipital = [14,15,16]
    region_delta = {
        'Frontal-Temporal': np.mean(channel_impact[frontal_temporal]),
        'Temporal': np.mean(channel_impact[temporal]),
        'Temporal-Parietal': np.mean(channel_impact[temporal_parietal]),
        'Centro-Parietal': np.mean(channel_impact[centro_parietal]),
        'Parietal': np.mean(channel_impact[parietal]),
        'Parieto-Occipital': np.mean(channel_impact[parieto_occipital]),
        'Occipital': np.mean(channel_impact[occipital]),
    }
    ax = axes[0]
    regions = list(region_delta.keys()); values = list(region_delta.values())
    colors = [PAL[0],PAL[1],PAL[2],PAL[3],NATURE["red"]]
    bars = ax.barh(regions, values, color=colors, edgecolor='white', linewidth=0.3)

    ax.set_xlabel('Mean RMSE Increase'); ax.set_title('a) Importance by Brain Region')
    ax = axes[1]
    sort_idx = np.argsort(channel_impact)[::-1]
    top5 = sort_idx[:5]; bot5 = sort_idx[-5:][::-1]
    names = [CH_NAMES[i] for i in list(top5)+list(bot5)]
    vals = list(channel_impact[top5]) + list(channel_impact[bot5])
    bar_colors = [PAL[0]]*5 + [NATURE["grey"]]*5
    bars = ax.barh(range(10)[::-1], vals[::-1], color=bar_colors[::-1], edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(10)[::-1]); ax.set_yticklabels(names[::-1])
    ax.set_xlabel('RMSE Increase'); ax.set_title('b) Top-5 vs Bottom-5 Channels')
    ax.axvline(x=0, color=NATURE["grey"], linewidth=0.5, linestyle='--')

    plt.tight_layout(); save_fig(fig, filename); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NT-LDFE Channel & Band Ablation Importance")
    print("=" * 60)

    channel_impact, band_impact, stat_impact, baseline_rmse, mu, y = train_and_ablate()

    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
    print(f"Channel impact range: [{channel_impact.min():.4f}, {channel_impact.max():.4f}]")
    print(f"Band impact range:   [{band_impact.min():.4f}, {band_impact.max():.4f}]")

    # Top channels
    top_ch = np.argsort(channel_impact)[::-1][:5]
    print(f"Top-5 channels: {[CH_NAMES[i] for i in top_ch]} ({[f'{channel_impact[i]:.4f}' for i in top_ch]})")
    print(f"Top-5 bands: {[BAND_NAMES[i] for i in np.argsort(band_impact)[::-1]]}")

    print("\nGenerating figures...")
    plot_brain_importance(channel_impact, "NT-LDFE Channel Ablation Importance", "fig_interp_brain_importance")
    plot_band_importance(band_impact, stat_impact, "fig_interp_feature_importance")
    plot_region_summary(channel_impact, "fig_interp_region_summary")

    # Save results
    frontal_temporal = [0, 1]
    occipital = [14, 15, 16]
    results = {
        "baseline_rmse": float(baseline_rmse),
        "channel_importance": {CH_NAMES[i]: float(channel_impact[i]) for i in range(17)},
        "band_importance": {BAND_NAMES[i]: float(band_impact[i]) for i in range(5)},
        "stat_importance": {STAT_NAMES[i]: float(stat_impact[i]) for i in range(4)},
        "top_channels": [CH_NAMES[i] for i in top_ch],
        "frontal_temporal_mean_delta": float(np.mean(channel_impact[frontal_temporal])),
        "occipital_mean_delta": float(np.mean(channel_impact[occipital])),
    }
    json.dump(results, open(FIG_DIR / "ablation_importance.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nResults: {FIG_DIR}/ablation_importance.json")
    print("Done! 3 interpretability figures generated.")
