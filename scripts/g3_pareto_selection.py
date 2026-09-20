#!/usr/bin/env python3
"""Validation-only operating-point selection for the CICIoT2023 pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
G2 = ROOT / "outputs" / "g2"
OUT = ROOT / "outputs" / "g3"
FPR_LIMIT = 0.01
RECALL_TIE_TOLERANCE = 0.001


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    ap_attack = average_precision_score(y_true, probability)
    ap_benign = average_precision_score(1 - y_true, 1 - probability)
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_attack": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_attack": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_attack": float(f1_score(y_true, prediction, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision_attack": float(ap_attack),
        "average_precision_benign": float(ap_benign),
        "macro_average_precision": float((ap_attack + ap_benign) / 2),
    }


def select_within_model(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    fpr, recall, thresholds = roc_curve(y_true, probability, drop_intermediate=False)
    candidates = pd.DataFrame({"threshold": thresholds, "false_positive_rate": fpr, "recall_attack": recall})
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna()
    feasible = candidates[candidates["false_positive_rate"] <= FPR_LIMIT].copy()
    max_recall = feasible["recall_attack"].max()
    best = feasible[np.isclose(feasible["recall_attack"], max_recall)]
    # If recall is identical, minimize false alarms; if still tied, use the
    # higher threshold as the more conservative deterministic choice.
    best = best.sort_values(["false_positive_rate", "threshold"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), candidates


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(G2 / "g2_predictions.csv.gz")
    g2 = json.loads((G2 / "g2_results.json").read_text())
    model_sizes = {name: result["model_size_bytes"] for name, result in g2["models"].items()}

    validation_results = {}
    frontiers = []
    thresholds = {}
    for model in sorted(predictions["model"].unique()):
        frame = predictions[(predictions["model"] == model) & (predictions["partition"] == "validation")]
        y = frame["y_true"].to_numpy()
        p = frame["probability_attack"].to_numpy()
        threshold, frontier = select_within_model(y, p)
        thresholds[model] = threshold
        validation_results[model] = metrics(y, p, threshold)
        frontier.insert(0, "model", model)
        frontiers.append(frontier)

    # Global pre-specified rule: maximize validation recall under the FPR cap.
    # When recall differs by <=0.1 percentage points, prefer the smaller model.
    ranked = sorted(validation_results, key=lambda name: validation_results[name]["recall_attack"], reverse=True)
    leader = ranked[0]
    near_ties = [
        name
        for name in ranked
        if validation_results[leader]["recall_attack"] - validation_results[name]["recall_attack"]
        <= RECALL_TIE_TOLERANCE
    ]
    selected_model = min(near_ties, key=lambda name: model_sizes[name])
    selected_threshold = thresholds[selected_model]

    test_frame = predictions[
        (predictions["model"] == selected_model) & (predictions["partition"] == "test")
    ]
    test_result = metrics(
        test_frame["y_true"].to_numpy(),
        test_frame["probability_attack"].to_numpy(),
        selected_threshold,
    )

    result = {
        "selection_policy": {
            "data_used_for_selection": "validation only",
            "fpr_constraint": FPR_LIMIT,
            "primary_objective": "maximize attack recall subject to FPR constraint",
            "near_tie_rule": f"if recall difference <= {RECALL_TIE_TOLERANCE}, select smaller serialized model",
            "test_used_for_selection": False,
        },
        "validation_selected_operating_points": validation_results,
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selected_test_result": test_result,
        "model_costs_from_g2": {
            name: {
                "model_size_bytes": model_sizes[name],
                "test_median_ms_per_flow": g2["models"][name]["test_inference"]["median_ms_per_flow"],
            }
            for name in model_sizes
        },
    }
    (OUT / "g3_selection_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    frontier_table = pd.concat(frontiers, ignore_index=True)
    frontier_table.to_csv(OUT / "g3_validation_operating_points.csv.gz", index=False)

    rows = []
    for model, values in validation_results.items():
        rows.append({"model": model, "partition": "validation", "selected": model == selected_model, **values})
    rows.append({"model": selected_model, "partition": "test", "selected": True, **test_result})
    pd.DataFrame(rows).to_csv(OUT / "g3_selected_results_table.csv", index=False)

    # Publication-quality figure. Only validation curves influence selection.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    labels = {"logistic_regression": "Logistic regression", "random_forest": "Random forest"}
    colors = {"logistic_regression": "#1f5a7a", "random_forest": "#b45f06"}
    for model in sorted(validation_results):
        curve = frontier_table[frontier_table["model"] == model].sort_values("false_positive_rate")
        curve = curve[curve["false_positive_rate"] <= 0.05]
        axes[0].plot(
            curve["false_positive_rate"] * 100,
            curve["recall_attack"] * 100,
            label=labels[model],
            color=colors[model],
            linewidth=1.8,
        )
        point = validation_results[model]
        axes[0].scatter(
            point["false_positive_rate"] * 100,
            point["recall_attack"] * 100,
            color=colors[model],
            s=52,
            zorder=4,
        )
    axes[0].axvline(FPR_LIMIT * 100, color="#555555", linestyle="--", linewidth=1.2, label="FPR constraint")
    axes[0].set(xlabel="False-positive rate (%)", ylabel="Attack recall (%)", xlim=(-0.05, 5.0), ylim=(94, 100.05))
    axes[0].set_title("A. Validation operating region")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="lower right")

    for name in sorted(model_sizes):
        size_mb = model_sizes[name] / (1024 * 1024)
        latency_us = g2["models"][name]["test_inference"]["median_ms_per_flow"] * 1000
        axes[1].scatter(size_mb, latency_us, s=80, color=colors[name], label=labels[name])
        axes[1].annotate(
            labels[name],
            (size_mb, latency_us),
            xytext=(-6, 7) if name == "random_forest" else (6, 5),
            textcoords="offset points",
            fontsize=9,
            ha="right" if name == "random_forest" else "left",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Serialized model size (MB, log scale)")
    axes[1].set_ylabel("Median batch latency (µs/flow)")
    axes[1].set_title("B. Computational cost")
    axes[1].grid(alpha=0.22)
    fig.suptitle("Validation-constrained operating-point selection", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "g3_operating_point_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "g3_operating_point_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
