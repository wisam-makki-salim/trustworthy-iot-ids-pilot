# False-Positive-Constrained IoT Intrusion Detection

A reproducible pilot study using one fixed shard of the CICIoT2023 dataset.

## Study question

Can an operating point chosen on validation data reduce false alarms while retaining high attack recall, and does that operating point continue to satisfy the same false-positive-rate constraint on held-out data?

## Dataset

- **Dataset:** CICIoT2023
- **Official source:** https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **Associated article:** https://doi.org/10.3390/s23135941
- **Input used:** `Merged01.csv`
- **Required SHA-256:** `8b43d6552a8cafd3b0ca2cedf6464ca3fe644d7fc9bb5dfe906368f1542792fe`

The raw dataset is not redistributed. Download `Merged01.csv` from the official source and place it at:

```text
data/ciciot2023/Merged01.csv
```

## Design

The analysis treats identical feature vectors as one sampling unit. Feature groups with conflicting binary labels are excluded, and one representative of each remaining feature vector is retained before a fixed 60/20/20 development-validation-test split is created. Preprocessing is fitted on development data only.

Two class-weighted baselines are compared: Logistic Regression and Random Forest. For each model, the validation threshold with the highest attack recall subject to `FPR <= 1%` is identified. If the best recalls differ by no more than 0.1 percentage points, the smaller fitted model is preferred. The selected model and threshold are then applied once to the held-out test partition.

## Main result

The rule selected Random Forest at a threshold of `0.5985298794`. On the held-out partition, attack recall was `98.167%` and the false-positive rate was `1.208%`. The conditional 95% bootstrap interval for the false-positive rate was `0.846% to 1.601%`. The validation constraint therefore did not remain satisfied as a point estimate on the held-out partition.

This result is specific to one randomly partitioned benchmark shard. It is not evidence of temporal robustness, deployment readiness, or cross-network generalization.

## Reproduce the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/g1_audit.py
python scripts/g2_baselines.py
python scripts/g3_pareto_selection.py
python scripts/g4_uncertainty.py
```

The scripts write full intermediate artifacts to `outputs/`. The `results/` directory contains the compact tables and JSON records retained with this repository. Open `Wisam_PhD_Research_Pilot.ipynb` for an executed review of those retained results.

## Timing caveat

Inference timings are included as measurements from the original execution environment. Hardware metadata were not retained for that run, so the absolute values should not be used for comparisons with other systems. The current baseline script records platform and processor information on subsequent runs.

## Repository contents

```text
scripts/                              Data audit, baselines, selection, uncertainty
results/                              Retained machine-readable results
figures/                              Figures in PNG and PDF formats
Wisam_PhD_Research_Pilot.ipynb        Executed results-review notebook
Wisam_Makki_Salim_Research_Pilot.pdf  Four-page research report
requirements.txt                      Pinned Python dependencies
CITATION.cff                          Citation metadata
```

## Scope

The implemented contribution is a constrained, validation-only operating-point rule with a model-size tie-breaker. It is not a new classifier or a full multi-objective optimizer. A subsequent study should use temporal or device-group separation, external validation, repeated training runs, and confidence-aware threshold selection under changing traffic conditions.

## Author

Wisam Makki Salim  
ORCID: https://orcid.org/0009-0000-6998-3912
