#!/usr/bin/env python3
"""Build the four-page research report from retained result files."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Wisam_Makki_Salim_Research_Pilot.pdf"
TEMP = ROOT / "tmp_report.pdf"
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2B6F92")
PALE = colors.HexColor("#EEF4F7")
MID = colors.HexColor("#B8C8D2")
TEXT = colors.HexColor("#25323B")


def load_font() -> str:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    bold_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    regular = next(path for path in candidates if path.exists())
    bold = next(path for path in bold_candidates if path.exists())
    pdfmetrics.registerFont(TTFont("ResearchSans", str(regular)))
    pdfmetrics.registerFont(TTFont("ResearchSans-Bold", str(bold)))
    return "ResearchSans"


FONT = load_font()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="ResearchSans-Bold", fontSize=24,
            leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName=FONT, fontSize=11.5,
            leading=16, textColor=BLUE, spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="ResearchSans-Bold", fontSize=14,
            leading=18, textColor=NAVY, spaceBefore=7, spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="ResearchSans-Bold", fontSize=10.5,
            leading=14, textColor=BLUE, spaceBefore=5, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=FONT, fontSize=9.15,
            leading=13.1, textColor=TEXT, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName=FONT, fontSize=7.6,
            leading=10.2, textColor=TEXT,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="ResearchSans-Bold", fontSize=10,
            leading=14.2, textColor=NAVY, borderColor=BLUE, borderWidth=0.8,
            borderPadding=8, backColor=PALE, spaceBefore=4, spaceAfter=9,
        ),
    }


S = styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def table(rows, widths, header=True, font_size=7.5):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "ResearchSans-Bold"),
        ]
        for row in range(1, len(rows)):
            if row % 2 == 0:
                commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    t.setStyle(TableStyle(commands))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(MID)
    canvas.setLineWidth(0.45)
    canvas.line(18 * mm, 16 * mm, width - 18 * mm, 16 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#60717D"))
    canvas.drawString(18 * mm, 10.5 * mm, "Wisam Makki Salim | IoT intrusion detection pilot | 2026")
    canvas.drawRightString(width - 18 * mm, 10.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    audit = json.loads((ROOT / "results/g1_audit.json").read_text())
    g2 = json.loads((ROOT / "results/g2_results.json").read_text())
    g3 = json.loads((ROOT / "results/g3_selection_results.json").read_text())
    g4 = json.loads((ROOT / "results/g4_uncertainty_results.json").read_text())
    lr = g2["models"]["logistic_regression"]["test_default_threshold"]
    rf = g2["models"]["random_forest"]["test_default_threshold"]
    selected = g3["selected_test_result"]
    intervals = g4["intervals"]["test"]

    doc = SimpleDocTemplate(
        str(TEMP), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=22 * mm,
        title="False-Positive-Constrained IoT Intrusion Detection",
        author="Wisam Makki Salim",
        subject="Reproducible pilot study using CICIoT2023",
    )
    story = []

    # Page 1
    story += [
        P("RESEARCH PILOT", "h2"),
        P("False-Positive-Constrained IoT Intrusion Detection", "title"),
        P("A reproducible pilot study using a fixed CICIoT2023 shard", "subtitle"),
        P("<b>Wisam Makki Salim</b><br/>Computer Engineering | Network Security and Resilient Systems<br/>Al-Iraqia University, Baghdad, Iraq | ORCID 0009-0000-6998-3912", "body"),
        Spacer(1, 6),
        P("Study aim", "h1"),
        P(
            "This study examines a practical question in binary IoT intrusion detection: whether an operating point selected to limit false alarms on validation data retains that behavior on unseen data. The emphasis is on data quality, leakage control, and the cost of changing the decision threshold, rather than on proposing a new classifier.",
        ),
        P("Research question", "h1"),
        P(
            "Can validation-based threshold selection retain high attack recall under a 1% false-positive-rate constraint, and does the chosen operating point continue to meet that constraint on a held-out partition?",
            "callout",
        ),
        P("Main finding", "h1"),
        P(
            "The selection rule chose a Random Forest threshold of <b>0.59853</b>. On the held-out partition, attack recall was <b>98.167%</b> and the false-positive rate was <b>1.208%</b>. The conditional 95% interval for the false-positive rate was <b>0.846% to 1.601%</b>. The point constraint observed during validation therefore did not remain satisfied on the held-out data.",
        ),
        P("Why the result matters", "h1"),
        P(
            "A threshold can appear operationally acceptable during model selection and still cross the same false-alarm limit when evaluated once on unseen data. For an adaptive detector, threshold choice should therefore account for uncertainty and distribution change, not only a validation point estimate.",
        ),
        P("Scope", "h1"),
        P(
            "The evidence is limited to one randomly partitioned benchmark shard. It does not establish temporal robustness, production readiness, or cross-network generalization.",
        ),
        PageBreak(),
    ]

    # Page 2
    audit_rows = [
        ["Audit item", "Observed", "Consequence for this study"],
        ["Exact duplicate rows", f'{audit["exact_duplicate_rows"]:,} (23.50%)', "Avoid splitting identical rows across partitions"],
        ["Duplicate feature rows", f'{audit["exact_duplicate_feature_rows"]:,} (27.26%)', "Use feature vectors as the sampling unit"],
        ["Conflicting binary-label groups", str(audit["feature_vectors_with_conflicting_binary_labels"]), "Exclude these groups from the primary analysis"],
        ["Attack prevalence", f'{audit["attack_prevalence"]*100:.3f}%', "Do not rely on accuracy or attack PR-AUC alone"],
        ["Missing / infinite cells", f'{audit["missing_cells"]} / {audit["infinite_numeric_cells"]}', "Convert infinities and fit imputation on development data"],
    ]
    baseline_rows = [
        ["Held-out metric", "Logistic Regression", "Random Forest"],
        ["Attack recall", f'{lr["recall_attack"]*100:.3f}%', f'{rf["recall_attack"]*100:.3f}%'],
        ["False-positive rate", f'{lr["false_positive_rate"]*100:.3f}%', f'{rf["false_positive_rate"]*100:.3f}%'],
        ["MCC", f'{lr["mcc"]:.3f}', f'{rf["mcc"]:.3f}'],
        ["Macro average precision", f'{lr["macro_average_precision"]:.3f}', f'{rf["macro_average_precision"]:.3f}'],
        ["Serialized size", "2.9 KB", "5.56 MB"],
        ["Median batch latency", "0.574 us/flow", "1.195 us/flow"],
    ]
    story += [
        P("Data and methods", "h1"),
        P(
            "CICIoT2023 was obtained from the Canadian Institute for Cybersecurity. The dataset documentation describes 105 IoT devices and 33 attacks in seven categories. This analysis uses <b>Merged01.csv</b> (712,311 rows and 39 predictors), identified by SHA-256 <font size='7'>8b43d6552a8cafd3b0ca2cedf6464ca3fe644d7fc9bb5dfe906368f1542792fe</font>. Raw data are not redistributed.",
        ),
        P("Data audit", "h1"),
        table(audit_rows, [46 * mm, 35 * mm, 92 * mm]),
        Spacer(1, 7),
        P("Analysis design", "h1"),
        P(
            "After binary label conversion, feature groups with conflicting labels were removed and one representative of each remaining feature vector was retained. The resulting 518,091 observations were split 60/20/20 into development, validation, and held-out partitions using seed 20260920. Imputation and scaling were fitted on development data only.",
        ),
        P(
            "The two class-weighted baselines were Logistic Regression and Random Forest. Within each model, the validation threshold with the highest attack recall under FPR <= 1% was retained. If model recalls differed by no more than 0.1 percentage points, the smaller fitted model was preferred. The selected model and threshold were then applied once to the held-out partition.",
        ),
        P("Default-threshold baselines", "h1"),
        table(baseline_rows, [57 * mm, 58 * mm, 58 * mm]),
        Spacer(1, 4),
        P(
            "The Random Forest recovered more attacks and achieved higher MCC, while Logistic Regression produced fewer false alarms and occupied substantially less storage. Timing values are retained as measurements from the original run; hardware metadata were not recorded, so the absolute values should not be compared across systems.",
            "small",
        ),
        PageBreak(),
    ]

    # Page 3
    ci_rows = [["Held-out metric", "Observed", "Conditional 95% interval"]]
    labels = [
        ("false_positive_rate", "False-positive rate", True),
        ("recall_attack", "Attack recall", True),
        ("balanced_accuracy", "Balanced accuracy", True),
        ("mcc", "MCC", False),
    ]
    for key, label, percent in labels:
        entry = intervals[key]
        if percent:
            observed = f'{entry["observed"]*100:.4f}%'
            interval = f'{entry["ci_95_lower"]*100:.4f}% to {entry["ci_95_upper"]*100:.4f}%'
        else:
            observed = f'{entry["observed"]:.4f}'
            interval = f'{entry["ci_95_lower"]:.4f} to {entry["ci_95_upper"]:.4f}'
        ci_rows.append([label, observed, interval])
    story += [
        P("Operating-point selection", "h1"),
        Image(str(ROOT / "figures/g3_operating_point_tradeoff.png"), width=170 * mm, height=66.5 * mm),
        P(
            "Random Forest was selected at threshold 0.59853 because its validation recall (98.228%) exceeded the feasible Logistic Regression point (97.266%) by more than the 0.1 percentage-point tie tolerance. On the held-out partition, the new threshold reduced Random Forest false positives from 61 to 40 and increased missed attacks from 1,614 to 1,839 relative to threshold 0.50.",
        ),
        P("Conditional uncertainty", "h1"),
        Image(str(ROOT / "figures/g4_uncertainty_stability.png"), width=170 * mm, height=66.1 * mm),
        table(ci_rows, [59 * mm, 43 * mm, 71 * mm]),
        Spacer(1, 4),
        P(
            "Intervals are conditional on the fitted Random Forest, selected threshold, and observed partition. They do not include variation from retraining, repartitioning, or model selection.",
            "small",
        ),
        PageBreak(),
    ]

    # Page 4
    story += [
        P("Interpretation", "h1"),
        P(
            "The study shows a concrete gap between satisfying an operating constraint during validation and retaining the same point estimate on held-out data. It does not show that Random Forest is universally preferable: Logistic Regression offered a much lower false-positive rate and a far smaller model at the default threshold. The appropriate choice depends on the cost assigned to missed attacks, false alarms, memory, and latency.",
        ),
        P("Limitations", "h1"),
        P(
            "The experiment uses one shard whose records had already been globally shuffled by the dataset producer; it cannot represent a future deployment period. Results come from one fixed random split and one training seed. The binary task also merges heterogeneous attack families into one class. Finally, the timing measurements lack retained hardware metadata and are reported only as run-specific observations.",
        ),
        P("Next study", "h1"),
        P(
            "A stronger evaluation should separate devices or time periods, repeat training across seeds, and include an external dataset. The decision rule can then be extended to use an upper confidence bound on false-positive rate and explicit compute limits. This would test whether adaptive threshold selection remains reliable when traffic composition and available resources change.",
        ),
        P("Reproducibility", "h1"),
        P(
            "The repository provides the data hash, audit and training scripts, fixed seed, pinned dependencies, preprocessing and split rules, retained result tables, figures, and a notebook with populated outputs. The scripts train and serialize both baselines when the verified dataset shard is placed in the documented location. Raw data are not redistributed.",
        ),
        P("References", "h1"),
        P(
            "[1] Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., and Ghorbani, A. A. (2023). CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment. <i>Sensors</i>, 23(13), 5941. https://doi.org/10.3390/s23135941",
            "small",
        ),
        Spacer(1, 4),
        P(
            "[2] Canadian Institute for Cybersecurity, University of New Brunswick. CIC IoT Dataset 2023. https://www.unb.ca/cic/datasets/iotdataset-2023.html",
            "small",
        ),
        Spacer(1, 12),
        P("<b>Researcher:</b> Wisam Makki Salim | wisam.m.salim@aliraqia.edu.iq", "body"),
    ]

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

    reader = PdfReader(str(TEMP))
    if len(reader.pages) != 4:
        raise RuntimeError(f"Expected 4 pages, produced {len(reader.pages)}")
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_metadata(
        {
            "/Title": "False-Positive-Constrained IoT Intrusion Detection",
            "/Author": "Wisam Makki Salim",
            "/Subject": "Reproducible pilot study using CICIoT2023",
            "/Creator": "Wisam Makki Salim",
            "/Producer": "",
        }
    )
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    TEMP.unlink()


if __name__ == "__main__":
    main()
