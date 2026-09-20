#!/usr/bin/env python3
"""Build a compact notebook with outputs derived from the retained result files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(source: str, output: str, count: int) -> dict:
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": (output.rstrip() + "\n").splitlines(keepends=True),
            }
        ],
    }


def main() -> None:
    audit = json.loads((ROOT / "results/g1_audit.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(ROOT / "results/g2_results_table.csv")
    selected = pd.read_csv(ROOT / "results/g3_selected_results_table.csv")
    intervals = pd.read_csv(ROOT / "results/g4_confidence_intervals.csv")

    audit_table = pd.DataFrame(
        [
            ("Rows", audit["shape"]["rows"]),
            ("Predictors", audit["feature_count"]),
            ("Exact duplicate rows", audit["exact_duplicate_rows"]),
            ("Duplicate feature rows", audit["exact_duplicate_feature_rows"]),
            ("Conflicting feature groups", audit["feature_vectors_with_conflicting_binary_labels"]),
            ("Missing cells", audit["missing_cells"]),
            ("Infinite cells", audit["infinite_numeric_cells"]),
        ],
        columns=["Item", "Observed"],
    )
    baseline_view = baseline.loc[
        baseline["partition"].eq("test"),
        ["model", "recall_attack", "false_positive_rate", "mcc", "macro_average_precision"],
    ].copy()
    selected_view = selected.loc[
        selected["selected"].astype(str).str.lower().eq("true"),
        ["model", "partition", "threshold", "recall_attack", "false_positive_rate", "mcc"],
    ].copy()
    interval_view = intervals.loc[
        intervals["partition"].eq("test")
        & intervals["metric"].isin(["false_positive_rate", "recall_attack", "balanced_accuracy", "mcc"]),
        ["metric", "observed", "ci_95_lower", "ci_95_upper"],
    ].copy()

    cells = [
        markdown(
            "# False-Positive-Constrained IoT Intrusion Detection\n\n"
            "This notebook reviews the retained outputs of a reproducible pilot study on the "
            "CICIoT2023 dataset. The raw data are not redistributed. The full pipeline can be "
            "rerun with the scripts in this repository after placing the verified dataset shard "
            "at `data/ciciot2023/Merged01.csv`."
        ),
        markdown(
            "## Data source and input identity\n\n"
            "Official source: https://www.unb.ca/cic/datasets/iotdataset-2023.html  \n"
            "Article: https://doi.org/10.3390/s23135941  \n"
            "Input SHA-256: `8b43d6552a8cafd3b0ca2cedf6464ca3fe644d7fc9bb5dfe906368f1542792fe`"
        ),
        markdown("## Dataset audit"),
        code(
            "import json, pandas as pd\n"
            "audit = json.load(open('results/g1_audit.json'))\n"
            "# Compact audit table assembled from the retained JSON record.\n"
            "audit_table",
            audit_table.to_string(index=False),
            1,
        ),
        markdown(
            "The audit motivated feature-level deduplication before splitting. Groups with "
            "conflicting binary labels were excluded from the primary analysis."
        ),
        markdown("## Baselines at the default threshold"),
        code(
            "baseline = pd.read_csv('results/g2_results_table.csv')\n"
            "baseline.loc[baseline.partition.eq('test'), "
            "['model','recall_attack','false_positive_rate','mcc','macro_average_precision']]",
            baseline_view.to_string(index=False),
            2,
        ),
        markdown("## Validation-constrained selection"),
        code(
            "selected = pd.read_csv('results/g3_selected_results_table.csv')\n"
            "selected.loc[selected.selected, "
            "['model','partition','threshold','recall_attack','false_positive_rate','mcc']]",
            selected_view.to_string(index=False),
            3,
        ),
        markdown("![Validation selection and computational cost](figures/g3_operating_point_tradeoff.png)"),
        markdown("## Conditional uncertainty intervals"),
        code(
            "intervals = pd.read_csv('results/g4_confidence_intervals.csv')\n"
            "intervals.loc[(intervals.partition == 'test') & intervals.metric.isin(" 
            "['false_positive_rate','recall_attack','balanced_accuracy','mcc'])]",
            interval_view.to_string(index=False),
            4,
        ),
        markdown("![Conditional uncertainty intervals](figures/g4_uncertainty_stability.png)"),
        markdown(
            "## Interpretation\n\n"
            "The selected Random Forest reached 98.167% attack recall and 1.208% FPR on the "
            "held-out partition. Its validation FPR was below 1%, but the held-out point estimate "
            "was not. The intervals condition on the fitted model, selected threshold, and observed "
            "partitions; they do not include training or model-selection variability.\n\n"
            "A stronger follow-up should use temporal or device-group separation, an external "
            "dataset, repeated training runs, and confidence-aware threshold selection."
        ),
        markdown(
            "## Full rerun\n\n"
            "```bash\n"
            "python scripts/g1_audit.py\n"
            "python scripts/g2_baselines.py\n"
            "python scripts/g3_pareto_selection.py\n"
            "python scripts/g4_uncertainty.py\n"
            "```"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "Wisam_PhD_Research_Pilot.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
