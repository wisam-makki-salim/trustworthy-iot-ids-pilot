#!/usr/bin/env python3
"""Build the two-page executive brief from retained result files."""

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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Wisam_Makki_Salim_Research_Pilot.pdf"
TEMP = ROOT / "tmp_report.pdf"
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2B6F92")
PALE = colors.HexColor("#EEF4F7")
MID = colors.HexColor("#B8C8D2")
TEXT = colors.HexColor("#25323B")
REPO_URL = "https://github.com/wisam-makki-salim/trustworthy-iot-ids-pilot"


def register_fonts() -> None:
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    pdfmetrics.registerFont(TTFont("ResearchSans", str(regular)))
    pdfmetrics.registerFont(TTFont("ResearchSans-Bold", str(bold)))


register_fonts()
base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle(
        "Title", parent=base["Title"], fontName="ResearchSans-Bold", fontSize=21,
        leading=25, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4,
    ),
    "subtitle": ParagraphStyle(
        "Subtitle", parent=base["Normal"], fontName="ResearchSans", fontSize=10.2,
        leading=13, textColor=BLUE, spaceAfter=8,
    ),
    "h1": ParagraphStyle(
        "H1", parent=base["Heading1"], fontName="ResearchSans-Bold", fontSize=12.2,
        leading=14.5, textColor=NAVY, spaceBefore=4, spaceAfter=3,
    ),
    "h2": ParagraphStyle(
        "H2", parent=base["Heading2"], fontName="ResearchSans-Bold", fontSize=9,
        leading=11, textColor=BLUE, spaceBefore=2, spaceAfter=2,
    ),
    "body": ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="ResearchSans", fontSize=8.15,
        leading=11.1, textColor=TEXT, spaceAfter=3.5,
    ),
    "small": ParagraphStyle(
        "Small", parent=base["BodyText"], fontName="ResearchSans", fontSize=6.65,
        leading=8.6, textColor=TEXT,
    ),
    "callout": ParagraphStyle(
        "Callout", parent=base["BodyText"], fontName="ResearchSans-Bold", fontSize=8.8,
        leading=12, textColor=NAVY, borderColor=BLUE, borderWidth=0.7,
        borderPadding=6, backColor=PALE, spaceBefore=2, spaceAfter=4,
    ),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def data_table(rows, widths, font_size=6.8):
    result = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "ResearchSans"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, MID),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "ResearchSans-Bold"),
    ]
    for row in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    result.setStyle(TableStyle(commands))
    return result


def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(MID)
    canvas.setLineWidth(0.4)
    canvas.line(16 * mm, 13.5 * mm, width - 16 * mm, 13.5 * mm)
    canvas.setFont("ResearchSans", 6.7)
    canvas.setFillColor(colors.HexColor("#60717D"))
    canvas.drawString(16 * mm, 8.5 * mm, "Wisam Makki Salim | IoT intrusion detection pilot | 2026")
    canvas.drawRightString(width - 16 * mm, 8.5 * mm, f"Page {doc.page} of 2")
    canvas.restoreState()


def main() -> None:
    audit = json.loads((ROOT / "results/g1_audit.json").read_text())
    g2 = json.loads((ROOT / "results/g2_results.json").read_text())
    g3 = json.loads((ROOT / "results/g3_selection_results.json").read_text())
    g4 = json.loads((ROOT / "results/g4_uncertainty_results.json").read_text())
    lr = g2["models"]["logistic_regression"]["test_default_threshold"]
    rf = g2["models"]["random_forest"]["test_default_threshold"]
    intervals = g4["intervals"]["test"]

    doc = SimpleDocTemplate(
        str(TEMP), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=13 * mm, bottomMargin=18 * mm,
        title="False-Positive-Constrained IoT Intrusion Detection",
        author="Wisam Makki Salim",
        subject="Two-page reproducible pilot brief using CICIoT2023",
    )

    audit_rows = [
        ["Audit item", "Observed", "Decision"],
        ["Exact duplicate rows", f'{audit["exact_duplicate_rows"]:,} (23.50%)', "Prevent identical rows from crossing partitions"],
        ["Duplicate feature rows", f'{audit["exact_duplicate_feature_rows"]:,} (27.26%)', "Treat feature vectors as the sampling unit"],
        ["Conflicting binary-label groups", str(audit["feature_vectors_with_conflicting_binary_labels"]), "Exclude from the primary analysis"],
        ["Attack prevalence", f'{audit["attack_prevalence"]*100:.3f}%', "Report class-sensitive metrics"],
        ["Missing / infinite cells", f'{audit["missing_cells"]} / {audit["infinite_numeric_cells"]}', "Fit imputation on development data only"],
    ]
    baseline_rows = [
        ["Held-out metric", "Logistic Regression", "Random Forest"],
        ["Attack recall", f'{lr["recall_attack"]*100:.3f}%', f'{rf["recall_attack"]*100:.3f}%'],
        ["False-positive rate", f'{lr["false_positive_rate"]*100:.3f}%', f'{rf["false_positive_rate"]*100:.3f}%'],
        ["MCC", f'{lr["mcc"]:.3f}', f'{rf["mcc"]:.3f}'],
        ["Macro average precision", f'{lr["macro_average_precision"]:.3f}', f'{rf["macro_average_precision"]:.3f}'],
        ["Serialized size", "2.9 KB", "5.56 MB"],
        ["Median batch latency", "0.574 µs/flow", "1.195 µs/flow"],
    ]
    labels = [
        ("false_positive_rate", "False-positive rate", True),
        ("recall_attack", "Attack recall", True),
        ("balanced_accuracy", "Balanced accuracy", True),
        ("mcc", "MCC", False),
    ]
    uncertainty_rows = [["Held-out metric", "Observed", "Conditional 95% interval"]]
    for key, label, percent in labels:
        entry = intervals[key]
        if percent:
            observed = f'{entry["observed"]*100:.4f}%'
            interval = f'{entry["ci_95_lower"]*100:.4f}% to {entry["ci_95_upper"]*100:.4f}%'
        else:
            observed = f'{entry["observed"]:.4f}'
            interval = f'{entry["ci_95_lower"]:.4f} to {entry["ci_95_upper"]:.4f}'
        uncertainty_rows.append([label, observed, interval])

    story = [
        # Page 1
        P("RESEARCH PILOT | TWO-PAGE BRIEF", "h2"),
        P("False-Positive-Constrained IoT Intrusion Detection", "title"),
        P("A reproducible pilot study using a fixed CICIoT2023 shard", "subtitle"),
        P("<b>Wisam Makki Salim</b> | Computer Engineering | Network Security and Resilient Systems<br/>Al-Iraqia University, Baghdad, Iraq | ORCID 0009-0000-6998-3912", "body"),
        P("Study aim", "h1"),
        P(
            "This study asks whether an operating point selected to limit false alarms on validation data retains that behavior on unseen traffic. The focus is not a new classifier; it is the reliability of a threshold chosen under an operational constraint.",
        ),
        P("Research question", "h1"),
        P(
            "Can validation-based threshold selection retain high attack recall under a 1% false-positive-rate constraint, and does the chosen point continue to meet that constraint on a held-out partition?",
            "callout",
        ),
        P("Main finding", "h1"),
        P(
            "The rule selected Random Forest at threshold <b>0.59853</b>. On the held-out partition, attack recall was <b>98.167%</b> and FPR was <b>1.208%</b>. The conditional 95% FPR interval was <b>0.846% to 1.601%</b>. The validation point constraint did not remain satisfied on the held-out data.",
        ),
        P("Data and methods", "h1"),
        P(
            "CICIoT2023 documents 105 IoT devices and 33 attacks in seven categories. This analysis uses <b>Merged01.csv</b> (712,311 rows; 39 predictors), fixed by SHA-256 <font size='6'>8b43d6552a8cafd3b0ca2cedf6464ca3fe644d7fc9bb5dfe906368f1542792fe</font>. Raw data are not redistributed.",
        ),
        P(
            "After binary label conversion, conflicting feature groups were removed and one representative of each remaining feature vector was retained. The resulting 518,091 observations were split 60/20/20 into development, validation, and held-out partitions using seed 20260920. Imputation and scaling were fitted on development data only.",
        ),
        P(
            "Logistic Regression and Random Forest were trained with class weighting. For each model, the validation threshold with the highest attack recall under FPR <= 1% was retained. If recalls differed by no more than 0.1 percentage points, the smaller fitted model was preferred. The selected configuration was then applied once to the held-out partition.",
        ),
        P("Data audit", "h1"),
        data_table(audit_rows, [47 * mm, 36 * mm, 95 * mm]),
        PageBreak(),

        # Page 2
        P("Baseline comparison", "h1"),
        data_table(baseline_rows, [60 * mm, 59 * mm, 59 * mm]),
        Spacer(1, 3),
        Image(str(ROOT / "figures/g3_operating_point_tradeoff.png"), width=158 * mm, height=63 * mm),
        P(
            "At threshold 0.59853, Random Forest validation recall was 98.228%. On held-out data, false positives fell from 61 to 40 relative to threshold 0.50, while missed attacks rose from 1,614 to 1,839.",
            "small",
        ),
        Image(str(ROOT / "figures/g4_uncertainty_stability.png"), width=158 * mm, height=62.5 * mm),
        data_table(uncertainty_rows, [61 * mm, 45 * mm, 72 * mm], font_size=6.6),
        Spacer(1, 4),
    ]

    left = P(
        "<b>Limitations.</b> The source shard was globally shuffled and cannot represent a future deployment period. Results come from one split and one training seed. The binary task merges heterogeneous attacks. Timing values are run-specific because hardware metadata were not retained.",
        "small",
    )
    right = P(
        "<b>Next study.</b> Use device- or time-separated evaluation, repeated training, an external dataset, and an upper confidence bound on FPR. This would test threshold adaptation under changing traffic and compute constraints.",
        "small",
    )
    lower = Table([[left, right]], colWidths=[88 * mm, 88 * mm], hAlign="LEFT")
    lower.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 5),
        ("LEFTPADDING", (1, 0), (1, 0), 5),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("LINEBEFORE", (1, 0), (1, 0), 0.35, MID),
    ]))
    story += [
        lower,
        Spacer(1, 4),
        P(
            f"<b>Code and results:</b> <link href='{REPO_URL}' color='#2B6F92'>{REPO_URL}</link>",
            "small",
        ),
        P(
            "<b>References:</b> [1] Neto, E. C. P. et al. (2023). CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment. <i>Sensors</i>, 23(13), 5941. https://doi.org/10.3390/s23135941. [2] Canadian Institute for Cybersecurity, University of New Brunswick. CIC IoT Dataset 2023. https://www.unb.ca/cic/datasets/iotdataset-2023.html",
            "small",
        ),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    reader = PdfReader(str(TEMP))
    if len(reader.pages) != 2:
        raise RuntimeError(f"Expected 2 pages, produced {len(reader.pages)}")
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_metadata({
        "/Title": "False-Positive-Constrained IoT Intrusion Detection",
        "/Author": "Wisam Makki Salim",
        "/Subject": "Two-page reproducible pilot brief using CICIoT2023",
        "/Creator": "Wisam Makki Salim",
        "/Producer": "",
    })
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    TEMP.unlink()


if __name__ == "__main__":
    main()
