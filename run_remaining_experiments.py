"""Run and package the remaining paper-facing NT-LDFE experiments.

This module keeps the original NT-LDFE training code intact and writes each
experiment family to its own output directory. It covers:

- Stage 5: proposed NT-LDFE main strict LOSO result.
- Stage 6: fold-level robustness, failure, calibration, and bin analyses.
- Stage 7: publication-oriented data tables and figures.

The deep NT-LDFE ablations are exposed as named configurations so they can be
run one at a time without editing nt_ldfe_loso.py.

The current campaign scope is proposed-method only. Classical/shallow baseline
comparison commands are intentionally disabled from the CLI so the experiment
frontier stays aligned with the paper question.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from nt_ldfe_loso import (
    Config,
    audit_subjects,
    compute_train_stats,
    labels_to_unit_interval,
    load_subjects,
    normalize,
    ordered_subject_keys,
    run_loso,
    save_json,
    sequence_metrics,
    summarize_results,
    train_fold,
)


MAIN_RESULT = Path("outputs/nt_ldfe_loso_v2/seedvig/loso_results.json")
PREVIOUS_RESULT = Path("outputs/nt_ldfe_loso/seedvig/loso_results.json")
ROOT = Path("outputs/remaining_experiments")
PROPOSED_METHOD_ONLY = ROOT / "proposed_method_only"


GPU_EXPERIMENT_COMMANDS = [
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation a7_without_uncertainty",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation a8_without_temporal_regularization",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation smoothing_window_1",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation smoothing_window_5",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation smoothing_window_7",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation ranking_gamma_0_01",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation ranking_gamma_0_05",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation soft_label_eta_0_1",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation temporal_tcn",
    "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py selected-predictions --subjects sub1 sub9 sub20 sub22",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cuda_environment() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    devices = []
    if available:
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            devices.append({
                "index": idx,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 3),
            })
    return {
        "torch_cuda_available": available,
        "torch_cuda_device_count": device_count,
        "devices": devices,
    }


def require_cuda() -> None:
    env = cuda_environment()
    if not env["torch_cuda_available"]:
        raise RuntimeError(
            "GPU-only experiment requested, but torch.cuda.is_available() is False. "
            "Per the current experiment scope, training was not started on CPU."
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_mean_std(payload: dict[str, Any], metric: str) -> tuple[float, float]:
    values = payload["summary"]["metrics"][metric]
    return float(values["mean"]), float(values["std"])


def rows_from_loso_payload(payload: dict[str, Any], metric_source: str = "test") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in payload["results"]:
        metrics = fold[metric_source]
        row = {
            "subject": fold["test_subject"],
            "best_epoch": fold.get("best_epoch"),
            "primary_source": fold.get("test_primary_source", ""),
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                row[key] = float(value)
        rows.append(row)
    rows.sort(key=lambda row: int("".join(ch for ch in row["subject"] if ch.isdigit())))
    return rows


def summarize_metric_rows(rows: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"folds": len(rows), "metrics": {}}
    for key in metric_keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            summary["metrics"][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def stage6_7_from_existing(
    result_path: Path = MAIN_RESULT,
    previous_path: Path = PREVIOUS_RESULT,
    output_dir: Path = ROOT / "stage6_stage7_existing_ntldfe",
) -> None:
    """Generate analysis tables and figures from completed NT-LDFE LOSO runs."""
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    figs_dir = ensure_dir(output_dir / "figures")
    payload = read_json(result_path)
    previous = read_json(previous_path) if previous_path.exists() else None

    test_rows = rows_from_loso_payload(payload, "test")
    raw_rows = rows_from_loso_payload(payload, "test_raw")
    calibrated_rows = rows_from_loso_payload(payload, "test_calibrated")
    write_csv(tables_dir / "per_subject_metrics.csv", test_rows)
    write_csv(tables_dir / "per_subject_raw_metrics.csv", raw_rows)
    write_csv(tables_dir / "per_subject_calibrated_metrics.csv", calibrated_rows)

    main_metrics = [
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "ccc",
        "r2",
        "nll_gaussian",
        "pred_std",
        "target_std",
        "tda",
        "tv",
        "adv",
        "sigma_abs_err_spearman",
    ]
    summary_rows: list[dict[str, Any]] = []
    for metric in main_metrics:
        if metric in payload["summary"]["metrics"]:
            m = payload["summary"]["metrics"][metric]
            row = {
                "metric": metric,
                "nt_ldfe_final_mean": m["mean"],
                "nt_ldfe_final_std": m["std"],
                "nt_ldfe_final_min": m["min"],
                "nt_ldfe_final_max": m["max"],
            }
            if previous and metric in previous["summary"]["metrics"]:
                p = previous["summary"]["metrics"][metric]
                row["previous_mean"] = p["mean"]
                row["previous_std"] = p["std"]
                row["delta_final_minus_previous"] = m["mean"] - p["mean"]
            summary_rows.append(row)
    write_csv(tables_dir / "summary_metrics.csv", summary_rows)

    checks = [
        ("rmse_le_0_20", "RMSE <= 0.20", lambda r: r["rmse"] <= 0.20),
        ("rmse_le_0_2385", "RMSE <= 0.2385", lambda r: r["rmse"] <= 0.2385),
        ("pearson_ge_0_61", "Pearson >= 0.61", lambda r: r["pearson"] >= 0.61),
        ("pearson_ge_0_71", "Pearson >= 0.71", lambda r: r["pearson"] >= 0.71),
        ("ccc_ge_0_50", "CCC >= 0.50", lambda r: r["ccc"] >= 0.50),
        ("r2_gt_0", "R2 > 0", lambda r: r["r2"] > 0),
    ]
    stability_rows = []
    for key, label, fn in checks:
        count = sum(bool(fn(row)) for row in test_rows)
        stability_rows.append({
            "check_id": key,
            "criterion": label,
            "count": count,
            "total": len(test_rows),
            "ratio": count / max(len(test_rows), 1),
        })
    write_csv(tables_dir / "fold_stability_checks.csv", stability_rows)

    failure_rows = []
    for row in test_rows:
        target_std = row.get("target_std", float("nan"))
        pred_std = row.get("pred_std", float("nan"))
        target_mean = row.get("target_mean", float("nan"))
        pred_mean = row.get("pred_mean", float("nan"))
        failure_rows.append({
            "subject": row["subject"],
            "rmse": row.get("rmse"),
            "pearson": row.get("pearson"),
            "ccc": row.get("ccc"),
            "r2": row.get("r2"),
            "mean_bias_pred_minus_target": pred_mean - target_mean,
            "std_gap_pred_minus_target": pred_std - target_std,
            "amplitude_ratio_pred_over_target": (
                pred_std / target_std if np.isfinite(target_std) and abs(target_std) > 1e-12 else np.nan
            ),
        })
    failure_rows.sort(key=lambda row: (row["rmse"], -abs(row["mean_bias_pred_minus_target"])), reverse=True)
    write_csv(tables_dir / "failure_mode_ranking.csv", failure_rows)

    bin_rows = []
    bins = [
        ("low_target_mean", 0.0, 0.35),
        ("mid_target_mean", 0.35, 0.70),
        ("high_target_mean", 0.70, 1.01),
    ]
    for name, low, high in bins:
        selected = [row for row in test_rows if low <= row.get("target_mean", -1) < high]
        if not selected:
            continue
        for metric in ["mae", "rmse", "pearson", "ccc", "r2"]:
            values = np.asarray([row.get(metric, np.nan) for row in selected], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                bin_rows.append({
                    "bin": name,
                    "target_mean_range": f"[{low:.2f}, {high:.2f})",
                    "metric": metric,
                    "folds": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                })
    write_csv(tables_dir / "fold_level_target_mean_bin_metrics.csv", bin_rows)

    calibration_rows = []
    for raw, cal in zip(raw_rows, calibrated_rows):
        row = {"subject": raw["subject"]}
        for metric in ["mae", "rmse", "pearson", "ccc", "r2", "nll_gaussian", "pred_std"]:
            row[f"raw_{metric}"] = raw.get(metric)
            row[f"calibrated_{metric}"] = cal.get(metric)
            if metric in raw and metric in cal:
                row[f"delta_{metric}"] = cal[metric] - raw[metric]
        calibration_rows.append(row)
    write_csv(tables_dir / "calibration_effects.csv", calibration_rows)

    generate_figures_from_tables(tables_dir, figs_dir)

    report = {
        "source_result": str(result_path),
        "previous_result": str(previous_path) if previous_path.exists() else None,
        "outputs": {
            "tables": str(tables_dir),
            "figures": str(figs_dir),
        },
        "headline": {
            "mae": payload["summary"]["metrics"]["mae"],
            "rmse": payload["summary"]["metrics"]["rmse"],
            "pearson": payload["summary"]["metrics"]["pearson"],
            "ccc": payload["summary"]["metrics"]["ccc"],
        },
        "stability": stability_rows,
        "top_failure_subjects": failure_rows[:6],
    }
    save_json(output_dir / "analysis_manifest.json", report)
    write_markdown_report(output_dir / "README.md", report, summary_rows)


def generate_figures_from_tables(tables_dir: Path, figs_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    ensure_dir(figs_dir)
    per_subject = pd.read_csv(tables_dir / "per_subject_metrics.csv")
    summary = pd.read_csv(tables_dir / "summary_metrics.csv")
    stability = pd.read_csv(tables_dir / "fold_stability_checks.csv")

    plt.style.use("default")
    colors = {
        "accent": "#7F8F84",
        "neutral": "#8A9199",
        "soft": "#D8D1C7",
        "warn": "#B88C8C",
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(per_subject))
    ax.bar(x, per_subject["rmse"], color=colors["soft"], edgecolor=colors["neutral"], linewidth=0.6)
    ax.axhline(0.20, color=colors["accent"], linestyle="--", linewidth=1.2, label="RMSE = 0.20")
    ax.axhline(0.2385, color=colors["warn"], linestyle=":", linewidth=1.2, label="RMSE = 0.2385")
    ax.set_xticks(x)
    ax.set_xticklabels(per_subject["subject"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("LOSO test subject")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#D8D1C7", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figs_dir / "loso_rmse_by_subject.png", dpi=220)
    fig.savefig(figs_dir / "loso_rmse_by_subject.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.scatter(per_subject["pearson"], per_subject["ccc"], s=44, color=colors["accent"], alpha=0.85)
    for _, row in per_subject.iterrows():
        if row["ccc"] < 0.45 or row["rmse"] > 0.24:
            ax.annotate(row["subject"], (row["pearson"], row["ccc"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Pearson")
    ax.set_ylabel("CCC")
    ax.grid(color="#D8D1C7", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figs_dir / "pearson_vs_ccc_failure_map.png", dpi=220)
    fig.savefig(figs_dir / "pearson_vs_ccc_failure_map.svg")
    plt.close(fig)

    metric_order = ["mae", "rmse", "ccc", "pearson", "spearman", "r2"]
    sub = summary[summary["metric"].isin(metric_order)].copy()
    sub["metric"] = pd.Categorical(sub["metric"], categories=metric_order, ordered=True)
    sub = sub.sort_values("metric")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.errorbar(
        np.arange(len(sub)),
        sub["nt_ldfe_final_mean"],
        yerr=sub["nt_ldfe_final_std"],
        fmt="o",
        color=colors["accent"],
        ecolor=colors["neutral"],
        capsize=3,
    )
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["metric"], rotation=30, ha="right")
    ax.set_ylabel("Mean ± SD")
    ax.grid(axis="y", color="#D8D1C7", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figs_dir / "main_metric_point_ranges.png", dpi=220)
    fig.savefig(figs_dir / "main_metric_point_ranges.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(stability["criterion"], stability["ratio"], color=colors["accent"], alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of LOSO folds")
    ax.grid(axis="x", color="#D8D1C7", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figs_dir / "fold_stability_summary.png", dpi=220)
    fig.savefig(figs_dir / "fold_stability_summary.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.scatter(
        per_subject["target_std"],
        per_subject["pred_std"],
        s=42,
        color=colors["neutral"],
        alpha=0.85,
    )
    lim = max(float(per_subject["target_std"].max()), float(per_subject["pred_std"].max())) * 1.08
    ax.plot([0, lim], [0, lim], color=colors["accent"], linestyle="--", linewidth=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Target trajectory std")
    ax.set_ylabel("Predicted trajectory std")
    ax.grid(color="#D8D1C7", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figs_dir / "prediction_amplitude_gap.png", dpi=220)
    fig.savefig(figs_dir / "prediction_amplitude_gap.svg")
    plt.close(fig)


def write_markdown_report(path: Path, report: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 6/7 Analysis Outputs",
        "",
        "This folder contains data tables and figures generated from the completed strict LOSO NT-LDFE run.",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for metric in ["mae", "rmse", "pearson", "spearman", "ccc", "r2", "nll_gaussian", "tda", "tv", "adv"]:
        for row in summary_rows:
            if row["metric"] == metric:
                lines.append(f"| {metric} | {row['nt_ldfe_final_mean']:.4f} | {row['nt_ldfe_final_std']:.4f} |")
                break
    lines.extend([
        "",
        "## Stability Checks",
        "",
        "| Criterion | Count | Ratio |",
        "|---|---:|---:|",
    ])
    for row in report["stability"]:
        lines.append(f"| {row['criterion']} | {row['count']}/{row['total']} | {row['ratio']:.3f} |")
    lines.extend([
        "",
        "## Top Failure Subjects",
        "",
        "| Subject | RMSE | Pearson | CCC | Mean bias | Std gap |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["top_failure_subjects"]:
        lines.append(
            f"| {row['subject']} | {row['rmse']:.4f} | {row['pearson']:.4f} | "
            f"{row['ccc']:.4f} | {row['mean_bias_pred_minus_target']:.4f} | "
            f"{row['std_gap_pred_minus_target']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_regressor(name: str, seed: int) -> Any:
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=10.0)
    if name == "huber":
        from sklearn.linear_model import HuberRegressor

        return HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=300)
    if name == "linear_svr":
        from sklearn.svm import LinearSVR

        return LinearSVR(C=1.0, epsilon=0.02, random_state=seed, max_iter=4000, dual="auto")
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=60,
            max_depth=10,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=90,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=seed,
        )
    raise ValueError(f"Unknown baseline: {name}")


def fit_predict_baseline(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "train_mean":
        pred = np.full(x_test.shape[0], float(np.mean(y_train)), dtype=np.float64)
        return pred, {"model": "train_mean", "status": "fit_not_required"}
    model = make_regressor(name, seed)
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test), dtype=np.float64)
    pred = np.clip(pred, 0.0, 1.0)
    meta = {"model": name}
    if hasattr(model, "n_iter_"):
        meta["n_iter"] = getattr(model, "n_iter_")
    return pred, meta


def run_classical_baselines(
    methods: list[str],
    output_dir: Path = ROOT / "stage5_classical_baselines",
) -> None:
    ensure_dir(output_dir)
    cfg = Config()
    cfg.output_dir = str(output_dir / "_unused_ntldfe")
    cfg.resume_completed_folds = False
    subjects = load_subjects(cfg)
    audit = audit_subjects(subjects)
    keys = ordered_subject_keys(subjects)
    all_results: dict[str, Any] = {
        "dataset_audit": asdict(audit),
        "methods": {},
    }

    for method in methods:
        method_dir = ensure_dir(output_dir / method / cfg.dataset_name)
        save_json(method_dir / "config.json", {"method": method, "seed": cfg.seed, "feature_mode": cfg.seedvig_feature_mode})
        rows: list[dict[str, Any]] = []
        fold_results: list[dict[str, Any]] = []
        for test_key in keys:
            metrics_path = method_dir / f"fold_{test_key}" / "metrics.json"
            if metrics_path.exists():
                saved = read_json(metrics_path)
                if saved.get("test_subject") == test_key and "test" in saved:
                    fold_results.append(saved)
                    row = {"subject": test_key}
                    row.update(saved["test"])
                    rows.append(row)
                    print(f"[{method}:{test_key}] resume existing fold")
                    continue
            train_arrays = []
            train_labels = []
            test_array = np.asarray(subjects[test_key]["data"], dtype=np.float32)
            test_label = labels_to_unit_interval(subjects[test_key]["label"])
            for key in keys:
                if key == test_key:
                    continue
                train_arrays.append(np.asarray(subjects[key]["data"], dtype=np.float32))
                train_labels.append(labels_to_unit_interval(subjects[key]["label"]))
            mean, std = compute_train_stats(train_arrays)
            x_train = np.concatenate([normalize(arr, mean, std).reshape(arr.shape[0], -1) for arr in train_arrays], axis=0)
            y_train = np.concatenate(train_labels, axis=0).astype(np.float64)
            x_test = normalize(test_array, mean, std).reshape(test_array.shape[0], -1)
            pred, model_meta = fit_predict_baseline(method, x_train, y_train, x_test, cfg.seed)
            train_resid_var = float(np.var(y_train - np.mean(y_train)))
            log_var = np.full_like(pred, np.log(max(train_resid_var, 1e-4)), dtype=np.float64)
            metrics = sequence_metrics(
                pred.reshape(1, -1),
                test_label.reshape(1, -1),
                log_var.reshape(1, -1),
                tda_epsilon=cfg.tda_epsilon,
            )
            train_mean = float(np.mean(y_train))
            baseline = np.full_like(test_label, train_mean, dtype=np.float64)
            metrics["baseline_train_mean_mae"] = float(np.mean(np.abs(baseline - test_label)))
            metrics["baseline_train_mean_rmse"] = float(np.sqrt(np.mean((baseline - test_label) ** 2)))
            metrics["mae_vs_train_mean_baseline_delta"] = metrics["mae"] - metrics["baseline_train_mean_mae"]
            metrics["rmse_vs_train_mean_baseline_delta"] = metrics["rmse"] - metrics["baseline_train_mean_rmse"]
            result = {
                "test_subject": test_key,
                "fold_info": {
                    "feature_shape": list(test_array.shape[1:]),
                    "train_samples": int(x_train.shape[0]),
                    "test_samples": int(x_test.shape[0]),
                    "normalization": "train_subjects_only",
                    "model_meta": model_meta,
                },
                "test_primary_source": "raw",
                "test": metrics,
            }
            fold_results.append(result)
            row = {"subject": test_key}
            row.update(metrics)
            rows.append(row)
            save_json(metrics_path, result)
            print(f"[{method}:{test_key}] mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} ccc={metrics['ccc']:.4f}")
        summary = summarize_results(fold_results)
        payload = {"summary": summary, "dataset_audit": asdict(audit), "results": fold_results}
        save_json(method_dir / "loso_results.json", payload)
        write_csv(method_dir / "per_subject_metrics.csv", rows)
        all_results["methods"][method] = {
            "path": str(method_dir / "loso_results.json"),
            "summary": summary,
        }
    save_json(output_dir / "classical_baseline_manifest.json", all_results)
    write_baseline_comparison(output_dir, methods)
    write_classical_baseline_report(output_dir)


def write_baseline_comparison(output_dir: Path, methods: list[str]) -> None:
    rows = []
    for method in methods:
        path = output_dir / method / "seedvig" / "loso_results.json"
        if not path.exists():
            continue
        payload = read_json(path)
        row = {"method": method}
        for metric in ["mae", "rmse", "pearson", "spearman", "ccc", "r2", "tda", "tv", "adv"]:
            if metric in payload["summary"]["metrics"]:
                row[f"{metric}_mean"] = payload["summary"]["metrics"][metric]["mean"]
                row[f"{metric}_std"] = payload["summary"]["metrics"][metric]["std"]
        rows.append(row)
    if MAIN_RESULT.exists():
        payload = read_json(MAIN_RESULT)
        row = {"method": "nt_ldfe_final"}
        for metric in ["mae", "rmse", "pearson", "spearman", "ccc", "r2", "tda", "tv", "adv"]:
            if metric in payload["summary"]["metrics"]:
                row[f"{metric}_mean"] = payload["summary"]["metrics"][metric]["mean"]
                row[f"{metric}_std"] = payload["summary"]["metrics"][metric]["std"]
        rows.append(row)
    write_csv(output_dir / "baseline_comparison.csv", rows)


def aggregate_existing_classical_baselines(
    output_dir: Path = ROOT / "stage5_classical_baselines",
) -> None:
    """Summarize any fully or partially completed classical baseline folders."""
    methods = []
    if output_dir.exists():
        for child in sorted(output_dir.iterdir()):
            if child.is_dir() and (child / "seedvig").exists():
                methods.append(child.name)
    write_baseline_comparison(output_dir, methods)
    write_classical_baseline_report(output_dir)


def write_classical_baseline_report(output_dir: Path) -> None:
    comparison_path = output_dir / "baseline_comparison.csv"
    if not comparison_path.exists():
        return
    rows: list[dict[str, str]] = []
    with comparison_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# Stage 5 Classical Baseline Results",
        "",
        "Each method is evaluated with strict LOSO and train-subject-only feature normalization. Slow or incomplete methods are kept in their own folders and are not hidden.",
        "",
        "| Method | MAE | RMSE | Pearson | Spearman | CCC | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def f(key: str) -> str:
            value = row.get(key, "")
            if value == "":
                return ""
            return f"{float(value):.4f}"

        lines.append(
            f"| {row['method']} | {f('mae_mean')} | {f('rmse_mean')} | "
            f"{f('pearson_mean')} | {f('spearman_mean')} | {f('ccc_mean')} | {f('r2_mean')} |"
        )
    lines.extend([
        "",
        "## Folder Status",
        "",
        "| Method | Completed folds | Result file |",
        "|---|---:|---|",
    ])
    for child in sorted(output_dir.iterdir()) if output_dir.exists() else []:
        if not child.is_dir() or not (child / "seedvig").exists():
            continue
        fold_count = len(list((child / "seedvig").glob("fold_*/metrics.json")))
        result = child / "seedvig" / "loso_results.json"
        lines.append(f"| {child.name} | {fold_count}/23 | {result.as_posix() if result.exists() else 'partial'} |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_deep_ablations(
    output_dir: Path = ROOT / "stage6_deep_ablations",
    report_dir: Path = PROPOSED_METHOD_ONLY,
) -> None:
    """Aggregate completed or partial NT-LDFE internal experiments."""
    ensure_dir(report_dir)
    rows: list[dict[str, Any]] = []
    if MAIN_RESULT.exists():
        payload = read_json(MAIN_RESULT)
        row = {
            "experiment": "full_nt_ldfe",
            "status": "completed",
            "folds": len(payload.get("results", [])),
            "result_file": str(MAIN_RESULT),
        }
        for metric in ["mae", "rmse", "pearson", "spearman", "ccc", "r2", "nll_gaussian", "tda", "tv", "adv"]:
            if metric in payload["summary"]["metrics"]:
                row[f"{metric}_mean"] = payload["summary"]["metrics"][metric]["mean"]
                row[f"{metric}_std"] = payload["summary"]["metrics"][metric]["std"]
        rows.append(row)

    for name in sorted(DEEP_ABLATIONS):
        run_dir = output_dir / name / "seedvig"
        result_path = run_dir / "loso_results.json"
        partial_path = run_dir / "partial_results.json"
        path = result_path if result_path.exists() else partial_path
        fold_files = list(run_dir.glob("fold_*/metrics.json")) if run_dir.exists() else []
        row = {
            "experiment": name,
            "status": "completed" if result_path.exists() else ("partial" if partial_path.exists() or fold_files else "pending"),
            "folds": 0,
            "result_file": str(path) if path.exists() else "",
        }
        if path.exists():
            payload = read_json(path)
            row["folds"] = len(payload.get("results", []))
            for metric in ["mae", "rmse", "pearson", "spearman", "ccc", "r2", "nll_gaussian", "tda", "tv", "adv"]:
                if metric in payload["summary"]["metrics"]:
                    row[f"{metric}_mean"] = payload["summary"]["metrics"][metric]["mean"]
                    row[f"{metric}_std"] = payload["summary"]["metrics"][metric]["std"]
        elif fold_files:
            row["folds"] = len(fold_files)
        rows.append(row)

    fieldnames = [
        "experiment",
        "status",
        "folds",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "pearson_mean",
        "pearson_std",
        "spearman_mean",
        "spearman_std",
        "ccc_mean",
        "ccc_std",
        "r2_mean",
        "r2_std",
        "nll_gaussian_mean",
        "nll_gaussian_std",
        "tda_mean",
        "tda_std",
        "tv_mean",
        "tv_std",
        "adv_mean",
        "adv_std",
        "result_file",
    ]
    write_csv(report_dir / "ntldfe_internal_experiment_summary.csv", rows, fieldnames=fieldnames)
    save_json(report_dir / "ntldfe_internal_experiment_summary.json", {"experiments": rows})

    lines = [
        "# NT-LDFE Internal Experiment Summary",
        "",
        "| Experiment | Status | Folds | MAE | RMSE | Pearson | CCC | TDA | TV | ADV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def f(key: str) -> str:
            value = row.get(key, "")
            if value == "":
                return ""
            return f"{float(value):.4f}"

        lines.append(
            f"| {row['experiment']} | {row['status']} | {row['folds']} | "
            f"{f('mae_mean')} | {f('rmse_mean')} | {f('pearson_mean')} | "
            f"{f('ccc_mean')} | {f('tda_mean')} | {f('tv_mean')} | {f('adv_mean')} |"
        )
    lines.append("")
    (report_dir / "ntldfe_internal_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


DEEP_ABLATIONS: dict[str, dict[str, Any]] = {
    "a7_without_uncertainty": {
        "lambda_nll": 0.0,
        "lambda_var": 0.0,
        "calibrate_log_var": False,
        "conservative_log_var_calibration": False,
    },
    "a8_without_temporal_regularization": {
        "beta_smooth": 0.0,
        "beta_acc": 0.0,
    },
    "smoothing_window_1": {"train_label_smoothing_window": 1},
    "smoothing_window_5": {"train_label_smoothing_window": 5},
    "smoothing_window_7": {"train_label_smoothing_window": 7},
    "ranking_gamma_0_01": {"enable_ranking": True, "gamma_rank": 0.01},
    "ranking_gamma_0_05": {"enable_ranking": True, "gamma_rank": 0.05},
    "soft_label_eta_0_1": {"enable_soft_label": True, "eta_soft": 0.1},
    "temporal_tcn": {"temporal_type": "tcn", "bidirectional": False},
}


def run_deep_ablation(name: str, max_folds: int | None = None, output_dir: Path = ROOT / "stage6_deep_ablations") -> None:
    if name not in DEEP_ABLATIONS:
        raise ValueError(f"Unknown deep ablation {name}. Available: {sorted(DEEP_ABLATIONS)}")
    require_cuda()
    cfg = Config()
    cfg.output_dir = str(output_dir / name)
    cfg.max_folds = max_folds
    cfg.save_predictions = False
    cfg.resume_completed_folds = True
    cfg.device = "cuda"
    cfg.use_amp = True
    for key, value in DEEP_ABLATIONS[name].items():
        setattr(cfg, key, value)
    ensure_dir(Path(cfg.output_dir))
    run_loso(cfg)


def run_selected_fold_predictions(
    subjects: list[str],
    output_dir: Path = ROOT / "stage7_selected_trajectory_predictions",
) -> None:
    require_cuda()
    cfg = Config()
    cfg.output_dir = str(output_dir)
    cfg.save_predictions = True
    cfg.resume_completed_folds = False
    cfg.device = "cuda"
    cfg.use_amp = True
    ensure_dir(output_dir / cfg.dataset_name)
    nt_subjects = load_subjects(cfg)
    audit = audit_subjects(nt_subjects)
    ordered = ordered_subject_keys(nt_subjects)
    selected = [s for s in subjects if s in nt_subjects]
    if not selected:
        raise ValueError(f"No selected subjects found. Available: {ordered}")
    results = []
    for subject in selected:
        print(f"=== Selected trajectory fold: {subject} ===")
        results.append(train_fold(nt_subjects, subject, cfg, output_dir / cfg.dataset_name))
    payload = {"summary": summarize_results(results), "dataset_audit": asdict(audit), "results": results}
    save_json(output_dir / cfg.dataset_name / "loso_results.json", payload)
    make_prediction_figures(output_dir / cfg.dataset_name, selected)


def make_prediction_figures(run_dir: Path, selected_subjects: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(run_dir / "figures")
    for subject in selected_subjects:
        npz_path = run_dir / f"fold_{subject}" / "trajectory_predictions.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        pred = np.asarray(data["pred"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        x = np.arange(target.size)[valid]
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        ax.plot(x, target[valid], color="#8A9199", linewidth=1.1, label="PERCLOS")
        ax.plot(x, pred[valid], color="#7F8F84", linewidth=1.3, label="NT-LDFE")
        ax.set_xlabel("Time window")
        ax.set_ylabel("Fatigue score")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#D8D1C7", linewidth=0.6, alpha=0.7)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{subject}_trajectory.png", dpi=220)
        fig.savefig(fig_dir / f"{subject}_trajectory.svg")
        plt.close(fig)


def copy_main_results_to_campaign() -> None:
    target = ensure_dir(ROOT / "reference_main_ntldfe" / "seedvig")
    if MAIN_RESULT.exists():
        shutil.copy2(MAIN_RESULT, target / "loso_results.json")
    cfg_path = MAIN_RESULT.parent / "config.json"
    if cfg_path.exists():
        shutil.copy2(cfg_path, target / "config.json")


def write_proposed_method_only_plan(output_dir: Path = PROPOSED_METHOD_ONLY) -> None:
    ensure_dir(output_dir)
    env = cuda_environment()
    queue = [
        {
            "exp_id": "E0_main_ntldfe_loso",
            "stage": "5",
            "question": "Can the proposed NT-LDFE estimate continuous PERCLOS trajectories under strict LOSO validation?",
            "status": "completed_from_existing_run" if MAIN_RESULT.exists() else "missing_reference_run",
            "output": str(ROOT / "reference_main_ntldfe" / "seedvig" / "loso_results.json"),
            "command": "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py copy-reference",
        },
        {
            "exp_id": "E1_existing_fold_analysis",
            "stage": "6/7",
            "question": "Which subjects, target ranges, and calibration effects explain the current NT-LDFE result?",
            "status": "completed_from_existing_run",
            "output": str(ROOT / "stage6_stage7_existing_ntldfe"),
            "command": "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py analyze-existing",
        },
    ]
    ablation_questions = {
        "a7_without_uncertainty": "Does heteroscedastic uncertainty improve noisy PERCLOS modeling and calibrated likelihood?",
        "a8_without_temporal_regularization": "Do smoothness and acceleration penalties improve trajectory stability without erasing correlation?",
        "smoothing_window_1": "How does removing label smoothing affect point accuracy and trajectory stability?",
        "smoothing_window_5": "Does stronger label smoothing improve robustness over the default window of 3?",
        "smoothing_window_7": "Does excessive label smoothing over-regularize subject trajectories?",
        "ranking_gamma_0_01": "Does a weak ranking prior improve trend consistency?",
        "ranking_gamma_0_05": "Does the default ranking strength improve or harm local dynamics?",
        "soft_label_eta_0_1": "Does the soft label distribution head help with fuzzy fatigue boundaries?",
        "temporal_tcn": "Does the TCN temporal encoder improve or trade off against the current GRU-based line?",
    }
    for name in DEEP_ABLATIONS:
        queue.append({
            "exp_id": name,
            "stage": "6",
            "question": ablation_questions[name],
            "status": "pending_gpu" if env["torch_cuda_available"] else "blocked_no_cuda_in_current_environment",
            "output": str(ROOT / "stage6_deep_ablations" / name),
            "command": f"conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py deep-ablation {name}",
        })
    queue.append({
        "exp_id": "E7_selected_trajectory_predictions",
        "stage": "7",
        "question": "Do representative and failure-case trajectories support the continuous latent-fatigue interpretation?",
        "status": "pending_gpu" if env["torch_cuda_available"] else "blocked_no_cuda_in_current_environment",
        "output": str(ROOT / "stage7_selected_trajectory_predictions"),
        "command": "conda run -n pytorchgpu python -X utf8 run_remaining_experiments.py selected-predictions --subjects sub1 sub9 sub20 sub22",
    })
    manifest = {
        "scope": "proposed_method_only",
        "gpu_required": True,
        "cuda_environment": env,
        "excluded_experiments": [
            {
                "family": "classical_or_external_baseline_comparison",
                "reason": "Excluded by the current user requirement; these results are not part of the continued evidence package.",
                "legacy_folder": str(ROOT / "stage5_classical_baselines"),
            }
        ],
        "queue": queue,
        "commands": GPU_EXPERIMENT_COMMANDS,
    }
    save_json(output_dir / "method_experiment_queue.json", manifest)

    lines = [
        "# Proposed-Method-Only Experiment Queue",
        "",
        "This campaign is restricted to experiments that test NT-LDFE and its internal mechanisms. Classical or external baseline comparisons are excluded from the active evidence package.",
        "",
        "## GPU Requirement",
        "",
        f"- GPU required: `True`",
        f"- Current `torch.cuda.is_available()`: `{env['torch_cuda_available']}`",
        f"- Current CUDA device count: `{env['torch_cuda_device_count']}`",
        "",
        "Training commands are guarded by `require_cuda()`. If CUDA is unavailable, the script exits before starting CPU training.",
        "",
        "## Completed From Existing NT-LDFE Run",
        "",
        f"- Main strict LOSO result: `{ROOT / 'reference_main_ntldfe' / 'seedvig' / 'loso_results.json'}`",
        f"- Stage 6/7 analysis outputs: `{ROOT / 'stage6_stage7_existing_ntldfe'}`",
        "",
        "## Pending GPU Runs",
        "",
        "| Experiment | Stage | Purpose | Command |",
        "|---|---:|---|---|",
    ]
    for item in queue:
        if item["status"].startswith("completed"):
            continue
        lines.append(f"| {item['exp_id']} | {item['stage']} | {item['question']} | `{item['command']}` |")
    lines.extend([
        "",
        "## Scope Boundary",
        "",
        f"- Legacy baseline folder, not used for the current manuscript package: `{ROOT / 'stage5_classical_baselines'}`",
        "- Do not include Ridge/SVR/RandomForest/MLP baseline tables in the continued experiments unless the scope is explicitly changed again.",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("analyze-existing", help="Generate Stage 6/7 tables and figures from completed NT-LDFE outputs.")

    p_deep = sub.add_parser("deep-ablation", help="Run one deep NT-LDFE ablation with the existing training pipeline.")
    p_deep.add_argument("name", choices=sorted(DEEP_ABLATIONS))
    p_deep.add_argument("--max-folds", type=int, default=None)

    p_pred = sub.add_parser("selected-predictions", help="Rerun selected folds and save trajectory predictions for Stage 7 figures.")
    p_pred.add_argument("--subjects", nargs="+", default=["sub1", "sub9", "sub20", "sub22"])

    sub.add_parser("copy-reference", help="Copy the final NT-LDFE reference result into the campaign folder.")
    sub.add_parser("method-plan", help="Write the proposed-method-only GPU experiment queue.")
    sub.add_parser("aggregate-method", help="Aggregate completed or partial NT-LDFE internal experiment outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "analyze-existing":
        stage6_7_from_existing()
    elif args.command == "deep-ablation":
        run_deep_ablation(args.name, max_folds=args.max_folds)
    elif args.command == "selected-predictions":
        run_selected_fold_predictions(args.subjects)
    elif args.command == "copy-reference":
        copy_main_results_to_campaign()
    elif args.command == "method-plan":
        write_proposed_method_only_plan()
    elif args.command == "aggregate-method":
        aggregate_deep_ablations()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
