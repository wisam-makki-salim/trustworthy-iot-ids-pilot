#!/usr/bin/env python3
"""Uncertainty analysis for the frozen G3 operating point."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
G3 = ROOT / "outputs" / "g3"
OUT = ROOT / "outputs" / "g4"
SEED = 20260920
REPLICATES = 10000


def derived_metrics(tn, fp, fn, tp):
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    fpr = fp / np.maximum(fp + tn, 1)
    tnr = tn / np.maximum(tn + fp, 1)
    balanced = (recall + tnr) / 2
    numerator = tp * tn - fp * fn
    denominator = np.sqrt(
        np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
    )
    mcc = numerator / denominator
    return {
        "precision_attack": precision,
        "recall_attack": recall,
        "f1_attack": f1,
        "false_positive_rate": fpr,
        "balanced_accuracy": balanced,
        "mcc": mcc,
    }


def summarize(values: np.ndarray, observed: float) -> dict:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "observed": float(observed),
        "bootstrap_median": float(np.median(values)),
        "ci_95_lower": float(low),
        "ci_95_upper": float(high),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((G3 / "g3_selection_results.json").read_text())
    rng = np.random.default_rng(SEED)
    summaries = {}
    draws_for_plot = {}
    for partition, source in [
        ("validation", selection["validation_selected_operating_points"][selection["selected_model"]]),
        ("test", selection["selected_test_result"]),
    ]:
        n_benign = source["tn"] + source["fp"]
        n_attack = source["tp"] + source["fn"]
        # Stratified nonparametric resampling of binary decisions is equivalent
        # to these independent binomial draws within each true class.
        fp = rng.binomial(n_benign, source["false_positive_rate"], REPLICATES)
        tp = rng.binomial(n_attack, source["recall_attack"], REPLICATES)
        tn = n_benign - fp
        fn = n_attack - tp
        boot = derived_metrics(tn, fp, fn, tp)
        observed = derived_metrics(
            np.array([source["tn"]]),
            np.array([source["fp"]]),
            np.array([source["fn"]]),
            np.array([source["tp"]]),
        )
        summaries[partition] = {
            name: summarize(values, observed[name][0]) for name, values in boot.items()
        }
        draws_for_plot[partition] = boot

    output = {
        "method": {
            "name": "conditional stratified bootstrap of frozen binary decisions",
            "implementation": "independent binomial resampling within benign and attack strata",
            "replicates": REPLICATES,
            "seed": SEED,
            "confidence_level": 0.95,
            "model": selection["selected_model"],
            "threshold": selection["selected_threshold"],
            "scope_note": "Intervals condition on the fitted model, selected threshold, and observed partition; they do not include training or model-selection variability.",
        },
        "intervals": summaries,
        "constraint_assessment": {
            "target_fpr": 0.01,
            "validation_ci_contains_target": bool(
                summaries["validation"]["false_positive_rate"]["ci_95_lower"]
                <= 0.01
                <= summaries["validation"]["false_positive_rate"]["ci_95_upper"]
            ),
            "test_ci_contains_target": bool(
                summaries["test"]["false_positive_rate"]["ci_95_lower"]
                <= 0.01
                <= summaries["test"]["false_positive_rate"]["ci_95_upper"]
            ),
        },
    }
    (OUT / "g4_uncertainty_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    rows = []
    for partition, metrics in summaries.items():
        for metric, values in metrics.items():
            rows.append({"partition": partition, "metric": metric, **values})
    pd.DataFrame(rows).to_csv(OUT / "g4_confidence_intervals.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    colors = {"validation": "#1f5a7a", "test": "#b45f06"}
    for partition in ("validation", "test"):
        fpr = summaries[partition]["false_positive_rate"]
        axes[0].errorbar(
            fpr["observed"] * 100,
            partition.title(),
            xerr=[
                [(fpr["observed"] - fpr["ci_95_lower"]) * 100],
                [(fpr["ci_95_upper"] - fpr["observed"]) * 100],
            ],
            fmt="o",
            capsize=4,
            color=colors[partition],
            markersize=7,
        )
    axes[0].axvline(1.0, linestyle="--", color="#444444", linewidth=1.2, label="1% target")
    axes[0].set_xlabel("False-positive rate (%)")
    axes[0].set_title("A. FPR uncertainty")
    axes[0].grid(axis="x", alpha=0.22)
    axes[0].legend(frameon=False)

    metric_names = ["recall_attack", "balanced_accuracy", "mcc"]
    display = ["Attack recall", "Balanced accuracy", "MCC"]
    y = np.arange(len(metric_names))
    offsets = {"validation": -0.10, "test": 0.10}
    for partition in ("validation", "test"):
        points = np.array([summaries[partition][name]["observed"] for name in metric_names])
        lows = np.array([summaries[partition][name]["ci_95_lower"] for name in metric_names])
        highs = np.array([summaries[partition][name]["ci_95_upper"] for name in metric_names])
        axes[1].errorbar(
            points,
            y + offsets[partition],
            xerr=np.vstack([points - lows, highs - points]),
            fmt="o",
            capsize=4,
            color=colors[partition],
            label=partition.title(),
        )
    axes[1].set_yticks(y, display)
    axes[1].set_xlim(0.76, 1.005)
    axes[1].set_xlabel("Metric value")
    axes[1].set_title("B. Performance stability")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("Uncertainty at the frozen Random Forest operating point", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "g4_uncertainty_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "g4_uncertainty_stability.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
