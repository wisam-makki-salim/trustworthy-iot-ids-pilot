# Trustworthy and Resilient Network Intrusion Detection

## A reproducible proof-of-execution pilot

**Researcher:** Wisam Makki Salim  
**Purpose:** German PhD supervisor outreach - research execution evidence, not a completed paper

This pilot evaluates binary IoT intrusion detection under explicit false-alarm and computational constraints. It uses one fixed official shard of CICIoT2023 and emphasizes provenance, leakage control, reproducibility, operational trade-offs, and uncertainty rather than an accuracy-only claim.

## Research question

Can validation-only operating-point selection produce a more operationally useful trade-off between attack recall, false-positive rate, inference cost, and model complexity than default-threshold intrusion-detection baselines?

## Data

- Dataset: CICIoT2023
- Official source: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Associated article: https://doi.org/10.3390/s23135941
- Pilot input: `Merged01.csv`
- Required SHA-256: `8b43d6552a8cafd3b0ca2cedf6464ca3fe644d7fc9bb5dfe906368f1542792fe`
- Raw data are not redistributed. Registration may be required at the official portal.

Place the downloaded file at:

```text
data/ciciot2023/Merged01.csv
```

## Core findings

The audit found 23.50% exact duplicate rows, 27.26% duplicate feature rows, 16 feature groups with conflicting binary labels, 22 missing cells, and 14 infinite values. The primary workflow removes conflicting groups, keeps one representative per feature vector, and creates a fixed 60/20/20 stratified split.

At the default threshold, Logistic Regression produced fewer false alarms while Random Forest detected more attacks and achieved higher MCC. Validation-only constrained selection chose Random Forest at threshold 0.5985298794. On held-out test data, attack recall was 98.167% and FPR was 1.208%. The 95% bootstrap interval for FPR was 0.846% to 1.601%, so the evidence does not establish that the operational FPR is reliably below the 1% target.

## Reproduce

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the stages:

```bash
python scripts/g1_audit.py
python scripts/g2_baselines.py
python scripts/g3_pareto_selection.py
python scripts/g4_uncertainty.py
```

Or open `Wisam_PhD_Research_Pilot.ipynb` and execute the staged cells.

## Reproducibility controls

- Fixed seed: `20260920`
- Locked input hash
- Pre-split feature deduplication
- Conflicting-label group exclusion
- Development-only preprocessing
- Validation-only threshold and model selection
- Frozen test application
- Saved machine-readable metrics and figures

## Boundaries

This is a proof-of-execution pilot. It does not claim a new classifier, state-of-the-art performance, temporal robustness, production readiness, or cross-network generalization. The natural PhD extension is confidence-bounded adaptive thresholding under distribution shift with external and temporally separated validation.

## Repository map

```text
scripts/                              Auditing, baselines, selection, uncertainty
results/                              Saved audit and evaluation outputs
figures/                              Publication-quality figures (PNG and PDF)
Wisam_PhD_Research_Pilot.ipynb        Staged reproducible notebook
Wisam_Makki_Salim_Research_Pilot.pdf  Four-page supervisor-facing report
SUPERVISOR_RELEVANCE.md               Supervisor-specific relevance blocks
```
