#!/usr/bin/env python3
"""Deterministic G1 audit for the fixed CICIoT2023 pilot shard."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ciciot2023" / "Merged01.csv"
OUT = ROOT / "outputs" / "g1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    features = [column for column in df.columns if column != "Label"]
    feature_frame = df[features]
    labels = df["Label"].astype(str).str.strip().str.upper()
    binary = labels.ne("BENIGN").astype("int8")

    # Exact-feature groups are the unit that must not cross data splits.
    feature_hash = pd.util.hash_pandas_object(feature_frame, index=False)
    conflict_counts = pd.DataFrame({"hash": feature_hash, "target": binary}).groupby("hash")[
        "target"
    ].nunique()

    numeric = feature_frame.select_dtypes(include=[np.number])
    report = {
        "scope": "CICIoT2023 official merged shard Merged01.csv",
        "file": {
            "name": DATA.name,
            "size_bytes": DATA.stat().st_size,
            "sha256": sha256(DATA),
        },
        "shape": {"rows": int(len(df)), "columns_including_label": int(df.shape[1])},
        "features": features,
        "feature_count": len(features),
        "label_column": "Label",
        "label_counts": {key: int(value) for key, value in labels.value_counts().items()},
        "binary_counts": {
            "attack": int(binary.sum()),
            "benign": int((binary == 0).sum()),
        },
        "attack_prevalence": float(binary.mean()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {
            key: int(value) for key, value in df.isna().sum().items() if value
        },
        "infinite_numeric_cells": int(np.isinf(numeric.to_numpy()).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "exact_duplicate_feature_rows": int(feature_frame.duplicated().sum()),
        "unique_feature_vectors": int(feature_hash.nunique()),
        "feature_vectors_with_conflicting_binary_labels": int((conflict_counts > 1).sum()),
        "constant_columns": [
            column for column, count in df.nunique(dropna=False).items() if count <= 1
        ],
        "dtype_counts": {str(key): int(value) for key, value in df.dtypes.astype(str).value_counts().items()},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (OUT / "g1_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
