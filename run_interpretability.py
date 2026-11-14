"""Interpretability experiments for NT-LDFE: brain topography, attention, uncertainty."""

import json, sys, warnings, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Nature colors ────────────────────────────────────────────────────────────
NATURE = {"blue":"#4DBBD5","red":"#E64B35","green":"#00A087","orange":"#F39B7F",
          "purple":"#8491B4","dark":"#2C3E50","grey":"#7F8C8D"}
PAL = [NATURE["blue"],NATURE["red"],NATURE["green"],NATURE["orange"],NATURE["purple"]]

# ── EEG 10-20 positions for SEED-VIG 17 channels ─────────────────────────────
# SEED-VIG correct montage: FT7,FT8,T7,T8,TP7,TP8,CP1,CP2,P1,PZ,P2,PO3,POZ,PO4,O1,OZ,O2
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

BAND_NAMES = ["δ(1-4)","θ(4-8)","α(8-13)","β(13-30)","γ(30-45)"]
ENHANCED_BANDS = ["δ","θ","α","β","γ"] + [f"R{i}" for i in range(6)] + [f"F{i}" for i in range(15)] + [f"T{i}" for i in range(48)]

FIG_DIR = BASE / "outputs/seedvig/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR2 = BASE / "paper/figures"
FIG_DIR2.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":9,"axes.titlesize":10,"axes.labelsize":9,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
    "axes.linewidth":0.5,"axes.spines.top":False,"axes.spines.right":False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Train model and extract attention
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_extract():
    """Train an NT-LDFE regression model on mixed subjects and extract attention."""
    from nt_ldfe_loso import Config, load_subjects, labels_to_unit_interval, \
        compute_train_stats, normalize, ordered_subject_keys
    from nt_ldfe_model import NTLDFE, NTLDFEConfig

    cfg = Config()
    cfg.dataset_name = "seedvig"
    cfg.seedvig_root = "dataset/seedvig"
    cfg.seedvig_feature_mode = "hybrid_de"
    cfg.seedvig_de_bins = 5
    cfg.seedvig_sampling_rate = 200.0
    cfg.sequence_length = 45
    cfg.train_stride = 5
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    subjects = load_subjects(cfg)
    keys = ordered_subject_keys(subjects)
    print(f"Loaded {len(subjects)} subjects on {cfg.device}")

    # Train on 20 subjects, hold out 3 for testing
    test_keys = set(keys[-3:])
    train_arrays, train_labels = [], []
    test_arrays, test_labels = [], []
    for k in keys:
        d = np.asarray(subjects[k]["data"], dtype=np.float32)
        l = labels_to_unit_interval(subjects[k]["label"])
        if k in test_keys:
            test_arrays.append(d); test_labels.append(l)
        else:
            train_arrays.append(d); train_labels.append(l)

    mean, std = compute_train_stats(train_arrays)
    train_arrays = [normalize(a, mean, std) for a in train_arrays]
    test_arrays = [normalize(a, mean, std) for a in test_arrays]

    # Create model
    feature_shape = tuple(train_arrays[0].shape[1:])  # (17, 25)
    model_cfg = NTLDFEConfig(
        feature_shape=feature_shape,
        encoder_dim=96, encoder_layers=2,
        temporal_hidden_dim=128, temporal_layers=2,
        dropout=0.1, bidirectional=True, use_feature_attention=True,
    )
    model = NTLDFE(config=model_cfg).to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Simple training loop
    from torch.utils.data import Dataset
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

    ds = SeqDataset(train_arrays, train_labels, cfg.sequence_length, cfg.train_stride)
    dl = DataLoader(ds, batch_size=48, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)
    device = torch.device(cfg.device)

    print("Training (200 epochs for meaningful attention)...")
    for epoch in range(200):
        model.train(); total_loss = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            mu, lv = out["mu"], out["log_var"]
            loss = (0.5 * torch.exp(-lv) * (y - mu).pow(2) + 0.5 * lv).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"  epoch {epoch+1:3d} loss={total_loss/len(dl):.4f}")

    # Extract attention weights
    model.eval()
    attn = model.encoder.attention_weights()
    channel_w = attn["channel"].detach().cpu().numpy()  # (17,)
    band_w = attn["band"].detach().cpu().numpy()        # (25,)
    print(f"Channel attention: min={channel_w.min():.4f} max={channel_w.max():.4f}")
    print(f"Band attention:    min={band_w.min():.4f} max={band_w.max():.4f}")

    # Compute uncertainty on test subjects
    test_seq = SeqDataset(test_arrays, test_labels, cfg.sequence_length, 45)
    test_dl = DataLoader(test_seq, batch_size=48, shuffle=False)
    all_mu, all_lv, all_y = [], [], []
    with torch.no_grad():
        for x, y in test_dl:
            out = model(x.to(device))
            all_mu.append(out["mu"].cpu().numpy())
            all_lv.append(out["log_var"].cpu().numpy())
            all_y.append(y.numpy())
    mu_flat = np.concatenate([m.reshape(-1) for m in all_mu])
    lv_flat = np.concatenate([lv.reshape(-1) for lv in all_lv])
    y_flat = np.concatenate([y.reshape(-1) for y in all_y])
    sigma_flat = np.exp(lv_flat / 2)
    ae_flat = np.abs(mu_flat - y_flat)

    return channel_w, band_w, mu_flat, y_flat, sigma_flat, ae_flat, model


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Brain Topography Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_brain_topography(channel_weights, title, filename):
    """Plot scalp topography of channel attention weights using MNE."""
    from mne_topomap_utils import _get_seedvig_info, SEEDVIG_CH_NAMES
    import mne

    info = _get_seedvig_info()
    data = np.asarray(channel_weights, dtype=np.float64).ravel()

    fig, ax = plt.subplots(figsize=(5, 4.5))

    evoked = mne.EvokedArray(data[:, np.newaxis], info, tmin=0.0)
    vabs = max(abs(data.min()), abs(data.max()), 0.01)
    im, _ = mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(0, vabs),
        sensors=True,
        contours=0,
        sphere=(0, 0, 0, 0.105),
        names=SEEDVIG_CH_NAMES,
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=8)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{filename}.{ext}")
        fig.savefig(FIG_DIR2 / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Frequency Band Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_frequency_analysis(band_weights, filename):
    """Plot frequency band attention and contribution."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    # Split the 25 band weights: first 5 are spectral DE, next 20 are temporal
    spectral_w = band_weights[:5]  # δ,θ,α,β,γ
    temporal_w = band_weights[5:]  # 5 bins x 4 stats = 20

    # (a) Spectral band attention
    ax = axes[0]
    colors = [PAL[0], PAL[2], PAL[1], PAL[3], NATURE["red"]]
    ax.bar(BAND_NAMES, spectral_w, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_ylabel('Attention Weight'); ax.set_title('a) Spectral Band Attention')

    # (b) Temporal feature attention
    ax = axes[1]
    # Group by statistic type: 5 bins, each with 4 stats
    stat_names = ['Var', 'Mean', 'Std', 'LL']
    bin_labels = ['Bin1','Bin2','Bin3','Bin4','Bin5']
    temporal_2d = temporal_w.reshape(4, 5)  # 4 stats x 5 bins

    import matplotlib.patches as mpatches
    bar_width = 0.18
    x = np.arange(5)
    for i, (sn, c) in enumerate(zip(stat_names, PAL[:4])):
        ax.bar(x + i*bar_width, temporal_2d[i], bar_width, label=sn, color=c, edgecolor='white', linewidth=0.2, alpha=0.85)
    ax.set_xticks(x + 1.5*bar_width); ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_ylabel('Attention Weight'); ax.set_title('b) Temporal Feature Attention')
    ax.legend(frameon=False, fontsize=6)

    # (c) Distribution comparison
    ax = axes[2]
    ax.hist(spectral_w, bins=8, alpha=0.6, color=PAL[0], edgecolor='white', label='Spectral (5)', linewidth=0.3)
    ax.hist(temporal_w, bins=15, alpha=0.6, color=PAL[1], edgecolor='white', label='Temporal (20)', linewidth=0.3)
    ax.set_xlabel('Attention Weight'); ax.set_ylabel('Frequency')
    ax.set_title('c) Attention Distribution'); ax.legend(frameon=False, fontsize=7)

    plt.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(FIG_DIR / f"{filename}.{ext}")
        fig.savefig(FIG_DIR2 / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Uncertainty vs PERCLOS Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_analysis(mu, y, sigma, ae, filename):
    """Analyze the relationship between predicted uncertainty and PERCLOS noise."""

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # (a) Uncertainty vs Absolute Error
    ax = axes[0, 0]
    ax.scatter(sigma[::10], ae[::10], c=PAL[0], s=4, alpha=0.3, edgecolors='none')
    ax.set_xlabel('Predicted Sigma'); ax.set_ylabel('Absolute Error |mu - y|')
    ax.set_title('a) Uncertainty vs Prediction Error')

    # Add trend line
    idx = np.argsort(sigma)
    window = max(len(sigma)//50, 10)
    trend_x = np.convolve(sigma[idx], np.ones(window)/window, mode='valid')
    trend_y = np.convolve(ae[idx], np.ones(window)/window, mode='valid')
    ax.plot(trend_x[::20], trend_y[::20], color=NATURE["red"], linewidth=1.5, alpha=0.8)

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, p = spearmanr(sigma, ae)
    ax.text(0.95, 0.95, f'Spearman rho={rho:.3f}\np={p:.3f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (b) Uncertainty distribution by fatigue level
    ax = axes[0, 1]
    low = y < 0.35; mid = (y >= 0.35) & (y < 0.70); high = y >= 0.70
    for label, mask, color in [("Low\n[0,0.35)", low, PAL[0]), ("Medium\n(0.35,0.7]", mid, PAL[1]), ("High\n(0.7,1.0]", high, NATURE["red"])]:
        ax.hist(sigma[mask][::5], bins=30, alpha=0.5, label=label, color=color, edgecolor='white', linewidth=0.2)
    ax.set_xlabel('Predicted Sigma'); ax.set_ylabel('Frequency')
    ax.set_title('b) Uncertainty by Fatigue Level'); ax.legend(frameon=False, fontsize=7)

    # (c) NT-LDFE trajectory vs PERCLOS
    ax = axes[1, 0]
    # Sort by time and show first 200 samples
    idx_sort = np.arange(min(200, len(mu)))
    ax.plot(idx_sort, y[idx_sort], color=NATURE["grey"], linewidth=0.8, alpha=0.6, label='PERCLOS')
    ax.plot(idx_sort, mu[idx_sort], color=PAL[0], linewidth=1.2, label='NT-LDFE')

    # Add uncertainty band
    ax.fill_between(idx_sort, mu[idx_sort]-sigma[idx_sort], mu[idx_sort]+sigma[idx_sort],
                     color=PAL[0], alpha=0.15, label='+/-1sigma')
    ax.set_xlabel('Time Window Index'); ax.set_ylabel('Fatigue Level')
    ax.set_title('c) NT-LDFE Trajectory vs PERCLOS'); ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 1.0)

    # (d) Key insight: where does uncertainty help?
    ax = axes[1, 1]
    # Sort by uncertainty and bin
    n_bins = 10
    sigma_bins = np.percentile(sigma, np.linspace(0, 100, n_bins+1))
    bin_mae = []
    for i in range(n_bins):
        mask = (sigma >= sigma_bins[i]) & (sigma < sigma_bins[i+1])
        if mask.sum() > 0:
            bin_mae.append(np.mean(ae[mask]))
        else:
            bin_mae.append(0)
    ax.bar(range(n_bins), bin_mae, color=PAL[1], edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Uncertainty Decile (low to high)'); ax.set_ylabel('Mean Absolute Error')
    ax.set_title('d) Error vs Uncertainty Decile')

    plt.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(FIG_DIR / f"{filename}.{ext}")
        fig.savefig(FIG_DIR2 / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Channel-Level Fatigue Pattern Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_channel_fatigue_patterns(channel_w, filename):
    """Show how attention maps to known EEG fatigue physiology."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) Channel attention by brain region
    # Group channels by region
    frontal = [0, 1, 2, 3, 4, 5]  # FP1,FP2,F7,F3,F4,F8
    temporal = [6, 9]              # T7,T8
    central = [7, 8, 16]           # C3,C4,Cz
    parietal = [10, 11, 12, 13]    # P7,P3,P4,P8
    occipital = [14, 15]           # O1,O2

    region_weights = {
        'Frontal': np.mean(channel_w[frontal]),
        'Temporal': np.mean(channel_w[temporal]),
        'Central': np.mean(channel_w[central]),
        'Parietal': np.mean(channel_w[parietal]),
        'Occipital': np.mean(channel_w[occipital]),
    }

    ax = axes[0]
    regions = list(region_weights.keys())
    values = list(region_weights.values())
    colors = [PAL[0], PAL[1], PAL[2], PAL[3], NATURE["red"]]
    ax.barh(regions, values, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Mean Attention Weight'); ax.set_title('a) Attention by Brain Region')

    # (b) Top 5 channels
    ax = axes[1]
    sorted_idx = np.argsort(channel_w)[::-1]
    top5_idx = sorted_idx[:5]
    top5_names = [CH_NAMES[i] for i in top5_idx]
    top5_vals = channel_w[top5_idx]
    ax.barh(range(5)[::-1], top5_vals[::-1], color=PAL[0], edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(top5_names[::-1])
    ax.set_xlabel('Attention Weight'); ax.set_title('b) Top-5 Channels by Attention')

    plt.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(FIG_DIR / f"{filename}.{ext}")
        fig.savefig(FIG_DIR2 / f"{filename}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NT-LDFE Interpretability Experiments")
    print("=" * 60)

    # Step 1: Train model and extract attention
    print("\n[1/5] Training model and extracting attention weights...")
    channel_w, band_w, mu, y, sigma, ae, model = train_and_extract()

    # Step 2: Brain topography
    print("\n[2/5] Generating brain topography maps...")
    plot_brain_topography(channel_w, "NT-LDFE Channel Attention Topography", "fig_interpret_brain_topo")
    plot_brain_topography(np.ones(17)/17, "Uniform Baseline (No Attention)", "fig_interpret_uniform_baseline")

    # Step 3: Frequency band analysis
    print("\n[3/5] Analyzing frequency band contributions...")
    plot_frequency_analysis(band_w, "fig_interpret_frequency")

    # Step 4: Uncertainty analysis
    print("\n[4/5] Analyzing uncertainty vs PERCLOS...")
    plot_uncertainty_analysis(mu, y, sigma, ae, "fig_interpret_uncertainty")

    # Step 5: Channel patterns
    print("\n[5/5] Analyzing channel-level fatigue patterns...")
    plot_channel_fatigue_patterns(channel_w, "fig_interpret_channels")

    # Save data for paper
    results = {
        "channel_attention": channel_w.tolist(),
        "channel_names": CH_NAMES,
        "band_attention": band_w[:5].tolist(),
        "band_names": BAND_NAMES,
        "frontal_attention": float(np.mean(channel_w[[0,1,2,3,4,5]])),
        "occipital_attention": float(np.mean(channel_w[[14,15]])),
        "central_attention": float(np.mean(channel_w[[7,8,16]])),
        "alpha_attention": float(band_w[2]),
        "theta_attention": float(band_w[1]),
        "beta_attention": float(band_w[3]),
    }
    with open(FIG_DIR / "interpretability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {FIG_DIR}/interpretability_results.json")
    print(f"Figures saved to: {FIG_DIR}/ and {FIG_DIR2}/")
    print("\n=== Interpretability experiments complete! ===")
