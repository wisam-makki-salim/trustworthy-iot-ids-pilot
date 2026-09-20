#!/usr/bin/env python3
"""Leakage-controlled binary baselines for the fixed CICIoT2023 shard."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260920
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ciciot2023" / "Merged01.csv"
OUT = ROOT / "outputs" / "g2"
MODELS = OUT / "models"
EXPECTED_SHA256 = "8b43d6552a8cafd3b0ca2cedf6464ca3fe644d7fc9bb5dfe906368f1542792fe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    if sha256(DATA) != EXPECTED_SHA256:
        raise RuntimeError("Input SHA-256 does not match the locked G1 artifact")
    df = pd.read_csv(DATA)
    labels = df.pop("Label").astype(str).str.strip().str.upper()
    y = labels.ne("BENIGN").astype("int8")
    X = df.replace([np.inf, -np.inf], np.nan)

    # Exact feature groups may never cross partitions. Conflicting groups are
    # removed from the primary analysis, then one representative is retained.
    hashes = pd.util.hash_pandas_object(X, index=False)
    group_label_count = pd.DataFrame({"hash": hashes, "y": y}).groupby("hash")["y"].nunique()
    conflicting = set(group_label_count[group_label_count > 1].index.tolist())
    keep = ~hashes.isin(conflicting)
    X, y, hashes = X.loc[keep].copy(), y.loc[keep].copy(), hashes.loc[keep]
    unique = ~hashes.duplicated(keep="first")
    X, y = X.loc[unique].reset_index(drop=True), y.loc[unique].reset_index(drop=True)

    audit = {
        "raw_rows": 712311,
        "conflicting_feature_hash_groups_removed": len(conflicting),
        "rows_after_conflict_removal_and_feature_deduplication": int(len(X)),
        "attack_rows_after_cleaning": int(y.sum()),
        "benign_rows_after_cleaning": int((y == 0).sum()),
    }
    return X, y, audit


def split_data(X: pd.DataFrame, y: pd.Series):
    X_devval, X_test, y_devval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_devval, y_devval, test_size=0.25, random_state=SEED, stratify=y_devval
    )
    return X_dev, X_val, X_test, y_dev, y_val, y_test


def metric_bundle(y_true: pd.Series, probability: np.ndarray, threshold: float = 0.5) -> dict:
    prediction = (probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
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
        "average_precision_attack": float(average_precision_score(y_true, probability)),
        "average_precision_benign": float(average_precision_score(1 - y_true, 1 - probability)),
        "macro_average_precision": float(
            (
                average_precision_score(y_true, probability)
                + average_precision_score(1 - y_true, 1 - probability)
            )
            / 2
        ),
    }


def timed_probability(model: Pipeline, X: pd.DataFrame, repeats: int = 5) -> tuple[np.ndarray, dict]:
    probability = model.predict_proba(X)[:, 1]
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(X)
        elapsed.append(time.perf_counter() - start)
    median_seconds = float(np.median(elapsed))
    return probability, {
        "repeats": repeats,
        "median_total_seconds": median_seconds,
        "median_ms_per_flow": median_seconds * 1000 / len(X),
        "throughput_flows_per_second": len(X) / median_seconds,
    }


def model_complexity(name: str, model: Pipeline) -> dict:
    estimator = model.named_steps["classifier"]
    if name == "logistic_regression":
        return {
            "nonzero_coefficients": int(np.count_nonzero(estimator.coef_)),
            "coefficient_count": int(estimator.coef_.size),
            "iterations": [int(value) for value in estimator.n_iter_],
        }
    return {
        "trees": len(estimator.estimators_),
        "total_nodes": int(sum(tree.tree_.node_count for tree in estimator.estimators_)),
        "max_observed_depth": int(max(tree.tree_.max_depth for tree in estimator.estimators_)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    X, y, cleaning = prepare_data()
    X_dev, X_val, X_test, y_dev, y_val, y_test = split_data(X, y)
    columns = list(X.columns)

    linear_preprocessor = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), columns)],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), columns)], remainder="drop"
    )
    models = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", linear_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }

    results = {
        "protocol": {
            "seed": SEED,
            "split": "60% development / 20% validation / 20% sealed test",
            "threshold": 0.5,
            "test_used_for_selection": False,
            "input_sha256": EXPECTED_SHA256,
        },
        "cleaning": cleaning,
        "split_counts": {
            "development": {"rows": len(X_dev), "attack": int(y_dev.sum()), "benign": int((y_dev == 0).sum())},
            "validation": {"rows": len(X_val), "attack": int(y_val.sum()), "benign": int((y_val == 0).sum())},
            "test": {"rows": len(X_test), "attack": int(y_test.sum()), "benign": int((y_test == 0).sum())},
        },
        "models": {},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported by operating system",
            "logical_cpu_count": os.cpu_count(),
            "timing_note": "Batch inference timing includes pipeline preprocessing and is environment-specific.",
        },
    }
    prediction_frames = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_dev, y_dev)
        fit_seconds = time.perf_counter() - start
        model_path = MODELS / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)

        val_probability, val_timing = timed_probability(model, X_val)
        test_probability, test_timing = timed_probability(model, X_test)
        results["models"][name] = {
            "fit_seconds": fit_seconds,
            "model_size_bytes": model_path.stat().st_size,
            "complexity": model_complexity(name, model),
            "validation_default_threshold": metric_bundle(y_val, val_probability),
            "test_default_threshold": metric_bundle(y_test, test_probability),
            "validation_inference": val_timing,
            "test_inference": test_timing,
        }
        prediction_frames.append(
            pd.DataFrame(
                {
                    "partition": "validation",
                    "row_id": X_val.index,
                    "y_true": y_val.to_numpy(),
                    "model": name,
                    "probability_attack": val_probability,
                }
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "partition": "test",
                    "row_id": X_test.index,
                    "y_true": y_test.to_numpy(),
                    "model": name,
                    "probability_attack": test_probability,
                }
            )
        )

    (OUT / "g2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "g2_predictions.csv.gz", index=False)

    rows = []
    for model_name, model_result in results["models"].items():
        for partition in ("validation", "test"):
            metrics = model_result[f"{partition}_default_threshold"]
            rows.append({"model": model_name, "partition": partition, **metrics})
    pd.DataFrame(rows).to_csv(OUT / "g2_results_table.csv", index=False)


if __name__ == "__main__":
    main()
