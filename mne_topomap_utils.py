#!/usr/bin/env python3
"""Shared MNE-based brain topography plotting utilities.

Supports:
  - My_fatigue: 31 channels via custom 31.locs file
  - SEED-VIG:   17 channels via standard_1020 montage subset

All functions produce publication-quality MNE topomaps.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# Channel definitions
CHANNEL_MAP_17_SEEDVIG = [
    ("FT7", -0.80, 0.30), ("FT8", 0.80, 0.30),
    ("T7", -0.88, 0.0),   ("T8", 0.88, 0.0),
    ("TP7", -0.80, -0.28), ("TP8", 0.80, -0.28),
    ("CP1", -0.28, -0.15), ("CP2", 0.28, -0.15),
    ("P1", -0.20, -0.35),  ("PZ", 0.0, -0.35),   ("P2", 0.20, -0.35),
    ("PO3", -0.22, -0.58), ("POZ", 0.0, -0.58),   ("PO4", 0.22, -0.58),
    ("O1", -0.22, -0.80),  ("OZ", 0.0, -0.85),    ("O2", 0.22, -0.80),
]
SEEDVIG_CH_NAMES = [c[0] for c in CHANNEL_MAP_17_SEEDVIG]

# My_fatigue 31-channel map from 31.locs: idx, angle(deg), radius, name
CHANNEL_31_MAP = [
    (1, 18, 0.5, "Fp1"), (2, 39, 0.333, "F3"), (3, 54, 0.5, "F7"),
    (4, 72, 0.628, "FT9"), (5, 68, 0.383, "FC5"), (6, 44, 0.172, "FC1"),
    (7, 90, 0.25, "C3"), (8, 90, 0.5, "T7"), (9, 108, 0.628, "TP9"),
    (10, 111, 0.383, "CP5"), (11, 136, 0.172, "CP1"), (12, -180, 0.25, "Pz"),
    (13, 141, 0.333, "P3"), (14, 126, 0.5, "P7"), (15, 162, 0.5, "O1"),
    (16, -180, 0.5, "Oz"), (17, -162, 0.5, "O2"), (18, -141, 0.333, "P4"),
    (19, -126, 0.5, "P8"), (20, -108, 0.628, "TP10"), (21, -111, 0.383, "CP6"),
    (22, -136, 0.172, "CP2"), (23, 0, 0, "Cz"), (24, -90, 0.25, "C4"),
    (25, -90, 0.5, "T8"), (26, -72, 0.628, "FT10"), (27, -69, 0.383, "FC6"),
    (28, -44, 0.172, "FC2"), (29, -39, 0.333, "F4"), (30, -54, 0.5, "F8"),
    (31, -18, 0.5, "Fp2"),
]
MYFATIGUE_CH_NAMES = [c[3] for c in CHANNEL_31_MAP]


def _polar_to_cart(angle_deg, radius):
    rad = np.deg2rad(angle_deg)
    return radius * np.sin(rad), radius * np.cos(rad)


# ── Cached MNE Info objects ─────────────────────────────────────────────────────

_INFO_CACHE: dict[str, object] = {}


def _get_seedvig_info():
    """Build MNE Info for SEED-VIG 17 channels using standard_1020 montage."""
    if "seedvig" in _INFO_CACHE:
        return _INFO_CACHE["seedvig"]

    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]

    # Map channel names to MNE positions, using case-insensitive matching
    ch_names_mne = []
    ch_pos_3d = []
    for ch in SEEDVIG_CH_NAMES:
        # Try exact match first, then lowercase-z variants
        if ch in pos:
            p = pos[ch]
        elif ch.replace("Z", "z") in pos:
            p = pos[ch.replace("Z", "z")]
        else:
            # Fallback: use manual coordinates scaled to head radius
            for name, x, y in CHANNEL_MAP_17_SEEDVIG:
                if name == ch:
                    p = np.array([x, y, 0.0])
                    break
            else:
                p = np.array([0.0, 0.0, 0.0])
        ch_names_mne.append(ch)
        ch_pos_3d.append(p)

    ch_pos_3d = np.array(ch_pos_3d)

    info = mne.create_info(
        ch_names=ch_names_mne,
        sfreq=200.0,
        ch_types="eeg",
    )

    # Create a DigMontage from the positions
    dig_ch_pos = {ch: pos_3d for ch, pos_3d in zip(ch_names_mne, ch_pos_3d)}
    # Also add fiducials for proper head outline
    dig_ch_pos["Nz"] = np.array([0.0, 0.105, 0.0])
    dig_ch_pos["LPA"] = np.array([-0.09, 0.0, 0.0])
    dig_ch_pos["RPA"] = np.array([0.09, 0.0, 0.0])

    montage_custom = mne.channels.make_dig_montage(
        ch_pos=dig_ch_pos,
        coord_frame="head",
    )
    info.set_montage(montage_custom)

    _INFO_CACHE["seedvig"] = info
    return info


def _get_myfatigue_info():
    """Build MNE Info for My_fatigue 31 channels from 31.locs."""
    if "myfatigue" in _INFO_CACHE:
        return _INFO_CACHE["myfatigue"]

    import mne

    ch_names = []
    ch_pos_3d = []
    for _, angle_deg, radius, name in CHANNEL_31_MAP:
        x, y = _polar_to_cart(angle_deg, radius)
        # Scale to MNE head radius (~0.095 m for the outermost electrodes)
        # The locs file uses radius 0.5 max, so scale by 0.19
        scale = 0.095 / 0.5  # max radius in locs = 0.628, in MNE head ~ 0.1
        ch_names.append(name)
        ch_pos_3d.append(np.array([x * scale, y * scale, 0.0]))

    ch_pos_3d = np.array(ch_pos_3d)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=200.0,
        ch_types="eeg",
    )

    dig_ch_pos = {ch: pos for ch, pos in zip(ch_names, ch_pos_3d)}
    dig_ch_pos["Nz"] = np.array([0.0, 0.105, 0.0])
    dig_ch_pos["LPA"] = np.array([-0.09, 0.0, 0.0])
    dig_ch_pos["RPA"] = np.array([0.09, 0.0, 0.0])

    montage_custom = mne.channels.make_dig_montage(
        ch_pos=dig_ch_pos,
        coord_frame="head",
    )
    info.set_montage(montage_custom)

    _INFO_CACHE["myfatigue"] = info
    return info


# ── Public API ──────────────────────────────────────────────────────────────────

def plot_mne_topomap(
    data: np.ndarray,
    dataset: str = "seedvig",
    axes=None,
    show: bool = False,
    cmap: str = "RdBu_r",
    vlim: Optional[tuple[float, float]] = None,
    sensor_size: int = 50,
    contours: int = 6,
    title: Optional[str] = None,
    names: Optional[list[str]] = None,
    show_names: bool = True,
    mask: Optional[np.ndarray] = None,
):
    """Plot a publication-quality MNE topomap.

    Args:
        data: 1D array of values per channel (17 for seedvig, 31 for myfatigue)
        dataset: "seedvig" or "myfatigue"
        axes: Optional matplotlib axes
        show: If True, call plt.show()
        cmap: Colormap name
        vlim: (vmin, vmax) tuple
        sensor_size: Size of sensor dots
        contours: Number of contour lines
        title: Plot title
        names: Channel names to display
        show_names: Whether to show channel name labels
        mask: Optional boolean mask for significance

    Returns:
        fig, im: The matplotlib figure and the image object
    """
    import matplotlib.pyplot as plt
    import mne

    if dataset == "seedvig":
        info = _get_seedvig_info()
        n_ch = 17
    elif dataset == "myfatigue":
        info = _get_myfatigue_info()
        n_ch = 31
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'seedvig' or 'myfatigue'.")

    data = np.asarray(data, dtype=np.float64).ravel()
    if len(data) != n_ch:
        raise ValueError(
            f"Expected {n_ch} channel values for {dataset}, got {len(data)}"
        )

    if axes is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        ax = axes
        fig = ax.figure

    if vlim is None:
        vabs = max(abs(data.min()), abs(data.max()))
        vlim = (-vabs, vabs)

    # Create evoked-like object for topomap
    evoked = mne.EvokedArray(
        data[:, np.newaxis],
        info,
        tmin=0.0,
    )

    # Use contours=0 to avoid MNE 1.2.3 QuadContourSet bug
    try:
        im, cn = mne.viz.plot_topomap(
            evoked.data[:, 0],
            evoked.info,
            axes=ax,
            show=False,
            cmap=cmap,
            vlim=vlim,
            sensors=True,
            contours=0,  # avoid QuadContourSet bug in MNE 1.2.3
            sphere=(0, 0, 0, 0.105),
            names=names if show_names else None,
            mask=mask,
            mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="k",
                             linewidth=1, markersize=8),
        )
    except AttributeError:
        # Fallback if contours=0 also fails
        im, cn = mne.viz.plot_topomap(
            evoked.data[:, 0],
            evoked.info,
            axes=ax,
            show=False,
            cmap=cmap,
            vlim=vlim,
            sensors=True,
            contours=6,
            sphere=(0, 0, 0, 0.105),
            names=names if show_names else None,
        )

    if title:
        ax.set_title(title)

    if show:
        plt.show()

    return fig, im


def plot_mne_topomap_with_cbar(
    data: np.ndarray,
    dataset: str = "seedvig",
    figsize: tuple = (5, 4.5),
    cmap: str = "RdBu_r",
    vlim: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    cbar_label: str = "",
    names: Optional[list[str]] = None,
):
    """Plot MNE topomap with integrated colorbar in a single figure.

    Returns:
        fig: matplotlib Figure
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data, dtype=np.float64).ravel()

    if vlim is None:
        vabs = max(abs(data.min()), abs(data.max()), 0.01)
        vlim = (-vabs, vabs)

    fig, ax = plt.subplots(figsize=figsize)

    # We handle the colorbar through MNE's built-in mechanism
    import mne

    if dataset == "seedvig":
        info = _get_seedvig_info()
        n_ch = 17
    elif dataset == "myfatigue":
        info = _get_myfatigue_info()
        n_ch = 31
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if len(data) != n_ch:
        raise ValueError(f"Expected {n_ch} values, got {len(data)}")

    evoked = mne.EvokedArray(data[:, np.newaxis], info, tmin=0.0)

    im, cn = mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=vlim,
        sensors=True,
        contours=0,
        sphere=(0, 0, 0, 0.105),
        names=names,
        mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="k",
                         linewidth=1, markersize=6),
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)

    if title:
        ax.set_title(title)

    return fig


def plot_dual_topomap(
    data1: np.ndarray,
    data2: np.ndarray,
    dataset: str = "seedvig",
    title1: str = "",
    title2: str = "",
    cmap: str = "RdBu_r",
    cbar_label: str = "",
    figsize: tuple = (10, 4.5),
    vlim: Optional[tuple[float, float]] = None,
    names: Optional[list[str]] = None,
):
    """Plot two MNE topomaps side by side.

    Args:
        data1, data2: 1D arrays of per-channel values
        dataset: "seedvig" or "myfatigue"
        title1, title2: subplot titles
        cmap: colormap name
        cbar_label: colorbar label
        figsize: figure size
        vlim: (vmin, vmax) tuple. If None, auto-computed as symmetric range.
        names: list of channel names to display
    """
    import matplotlib.pyplot as plt

    data1 = np.asarray(data1, dtype=np.float64).ravel()
    data2 = np.asarray(data2, dtype=np.float64).ravel()

    if dataset == "seedvig":
        info = _get_seedvig_info()
        n_ch = 17
    elif dataset == "myfatigue":
        info = _get_myfatigue_info()
        n_ch = 31
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if len(data1) != n_ch or len(data2) != n_ch:
        raise ValueError(f"Expected {n_ch} values each")

    if vlim is None:
        # Auto-compute: use data-driven range (supports asymmetric for positive data)
        all_data = np.concatenate([data1, data2])
        vmin, vmax = float(all_data.min()), float(all_data.max())
        # For data crossing zero, use symmetric; for all-positive, anchor at min
        if vmin >= 0:
            pad = (vmax - vmin) * 0.05 if vmax > vmin else 0.01
            vlim = (vmin - pad, vmax + pad)
        else:
            vabs = max(abs(vmin), abs(vmax), 0.01)
            pad = vabs * 0.05
            vlim = (-vabs - pad, vabs + pad)
    else:
        vlim = (float(vlim[0]), float(vlim[1]))

    import mne

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    evoked1 = mne.EvokedArray(data1[:, np.newaxis], info, tmin=0.0)
    evoked2 = mne.EvokedArray(data2[:, np.newaxis], info, tmin=0.0)

    im1, _ = mne.viz.plot_topomap(
        evoked1.data[:, 0], evoked1.info, axes=axes[0], show=False,
        cmap=cmap, vlim=vlim, sensors=True, contours=0, sphere=(0, 0, 0, 0.105),
        names=names,
    )
    axes[0].set_title(title1)

    im2, _ = mne.viz.plot_topomap(
        evoked2.data[:, 0], evoked2.info, axes=axes[1], show=False,
        cmap=cmap, vlim=vlim, sensors=True, contours=0, sphere=(0, 0, 0, 0.105),
        names=names,
    )
    axes[1].set_title(title2)

    cbar = plt.colorbar(im2, ax=axes, shrink=0.6, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)

    return fig


def get_channel_names(dataset: str) -> list[str]:
    """Return channel name list for the dataset."""
    if dataset == "seedvig":
        return list(SEEDVIG_CH_NAMES)
    elif dataset == "myfatigue":
        return list(MYFATIGUE_CH_NAMES)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_info(dataset: str):
    """Return MNE Info object for the dataset."""
    if dataset == "seedvig":
        return _get_seedvig_info()
    elif dataset == "myfatigue":
        return _get_myfatigue_info()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ═══════════════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Test SEED-VIG topomap
    seedvig_data = np.random.randn(17)
    fig = plot_mne_topomap_with_cbar(
        seedvig_data, dataset="seedvig",
        title="Test SEED-VIG 17ch Topomap (MNE)",
        cbar_label="Value",
    )
    fig.savefig("/tmp/test_seedvig_topomap.png", dpi=150)
    print("Saved test_seedvig_topomap.png")

    # Test My_fatigue topomap
    myfatigue_data = np.random.randn(31)
    fig = plot_mne_topomap_with_cbar(
        myfatigue_data, dataset="myfatigue",
        title="Test My_fatigue 31ch Topomap (MNE)",
        cbar_label="Value",
    )
    fig.savefig("/tmp/test_myfatigue_topomap.png", dpi=150)
    print("Saved test_myfatigue_topomap.png")

    print("MNE topomap utilities OK")
