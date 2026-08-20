"""
PsychroScan — report.py  (v3.0 — plain-language edition)
==========================================================
Generates a PDF + Markdown report from 09_predict_new_genome.py outputs.
Designed for biologists and researchers with no machine-learning background.

Key additions over v2.0:
  - Automatic enzyme-type detection from Protein_ID via UniProt REST API
  - Plain-language explanations replacing all ML jargon
  - Candidate table now shows enzyme type, not just probability
  - "What to do next" section with concrete lab steps

Usage:
    python src/report.py                        # all organisms in data/new_genomes/
    python src/report.py --organism NAME        # one organism
    python src/report.py --no-pdf               # Markdown only (faster)
    python src/report.py --no-uniprot           # skip API calls (offline mode)
"""

import os
import re
import time
import argparse
import requests
import pandas as pd
from datetime import date

# ── Directories ────────────────────────────────────────────────────────────────
GENOMES_DIR = os.path.join("data", "new_genomes")
REPORTS_DIR = os.path.join("results", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Colours (Premium Modern UI Palette) ────────────────────────────────────────
COLD   = "#0f2537" # Deep Corporate Navy
BLUE   = "#2b6cb0" # Accent Blue
LIGHT  = "#f8fafc" # Soft Background (Slate 50)
GREEN  = "#059669" # Success Emerald
WARN   = "#d97706" # Warning Amber
GREY   = "#64748b" # Clean Slate Grey
RED_C  = "#dc2626" # Error Crimson

# ── PPI constants ──────────────────────────────────────────────────────────────
RFLEX_MEAN = 2.55
RFLEX_SD   = 0.80
RFLEX_MIN  = RFLEX_MEAN - 2 * RFLEX_SD
RFLEX_MAX  = RFLEX_MEAN + 2 * RFLEX_SD

# ── Enzyme-type detection patterns (from Protein_ID string) ───────────────────
EC_PATTERNS = {
    "Lipase / Esterase":        re.compile(r"lipas|esterase|phospholipas", re.I),
    "Alpha-Amylase":            re.compile(r"amylas|glucosidas|glucoamylas", re.I),
    "Cellulase":                re.compile(r"cellulas|cellobiohydrolas|endoglucanas", re.I),
    "Serine Protease":          re.compile(r"proteas|peptidas|subtilis|trypsin|chymotrypsin", re.I),
    "Metalloprotease":          re.compile(r"metalloproteas|thermolysin|collagenase|neprilysin", re.I),
}

# Industrial application of each enzyme type (for plain-language summary)
ENZYME_APPLICATIONS = {
    "Lipase / Esterase":   "fat and oil processing at low temperatures (food, detergents, biodiesel)",
    "Alpha-Amylase":       "starch breakdown in cold baking, brewing, and textile processes",
    "Cellulase":           "cellulose degradation for cold biofuels and textile finishing",
    "Serine Protease":     "protein digestion at low temperatures (food processing, aquafeed)",
    "Metalloprotease":     "meat tenderisation and biomedical applications at cold temperatures",
    "Unknown":             "function not identified from sequence ID alone — check UniProt",
}


# ══════════════════════════════════════════════════════════════════════════════
#  ENZYME-TYPE ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def detect_enzyme_type_from_id(protein_id: str) -> str:
    """Detect enzyme type from Protein_ID string using keyword patterns."""
    for etype, pat in EC_PATTERNS.items():
        if pat.search(protein_id):
            return etype
    return "Unknown"


def annotate_from_uniprot(protein_ids: list, max_ids: int = 15) -> dict:
    """
    Fetch protein name and enzyme type from UniProt REST API.
    Returns dict: {protein_id: {"name": str, "enzyme_type": str}}
    Gracefully returns empty dict on timeout or API error.
    """
    results = {}
    ids_to_query = protein_ids[:max_ids]
    # Extract bare accession (format: tr|A0ABR0LQF3|A0ABR0LQF3_9PEZI → A0ABR0LQF3)
    acc_map = {}
    for pid in ids_to_query:
        parts = pid.split("|")
        acc = parts[1] if len(parts) >= 2 else parts[0]
        acc_map[acc] = pid

    if not acc_map:
        return results

    query = " OR ".join(f"accession:{acc}" for acc in acc_map)
    try:
        resp = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query":  query,
                "fields": "accession,protein_name,ec",
                "format": "json",
                "size":   len(acc_map),
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return results

        for entry in resp.json().get("results", []):
            acc    = entry.get("primaryAccession", "")
            pid    = acc_map.get(acc, acc)
            # Protein name
            pname  = ""
            try:
                pname = (entry["proteinDescription"]["recommendedName"]
                               ["fullName"]["value"])
            except (KeyError, TypeError):
                try:
                    pname = (entry["proteinDescription"]["submittedName"][0]
                                   ["fullName"]["value"])
                except (KeyError, TypeError, IndexError):
                    pname = ""
            # EC number → enzyme type
            ec_list = entry.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", [])
            if not ec_list:
                ec_list = []
            ec_str = ec_list[0].get("value", "") if ec_list else ""
            etype  = _ec_to_type(ec_str) or detect_enzyme_type_from_id(pid) or "Unknown"
            results[pid] = {"name": pname, "enzyme_type": etype}
    except Exception:
        pass

    return results


def _ec_to_type(ec: str) -> str:
    """Map EC number prefix to enzyme type label."""
    mapping = {
        "3.1.1": "Lipase / Esterase",
        "3.2.1.1": "Alpha-Amylase",
        "3.2.1.4": "Cellulase",
        "3.4.21": "Serine Protease",
        "3.4.24": "Metalloprotease",
    }
    for prefix, label in mapping.items():
        if ec.startswith(prefix):
            return label
    return ""


def enrich_top15(df: pd.DataFrame, use_uniprot: bool = True) -> pd.DataFrame:
    """
    Add 'enzyme_type' and 'protein_name' columns to the top-15 candidates.
    First tries UniProt API, falls back to keyword detection.
    """
    top15 = df.nlargest(15, "Cold_Probability").copy()
    top15["enzyme_type"]  = top15["Protein_ID"].apply(detect_enzyme_type_from_id)
    top15["protein_name"] = ""

    if use_uniprot:
        print("  Querying UniProt for enzyme annotations (top 15)...")
        anno = annotate_from_uniprot(top15["Protein_ID"].tolist())
        for pid, info in anno.items():
            mask = top15["Protein_ID"] == pid
            if mask.any():
                if info["enzyme_type"] and info["enzyme_type"] != "Unknown":
                    top15.loc[mask, "enzyme_type"] = info["enzyme_type"]
                if info["name"]:
                    top15.loc[mask, "protein_name"] = info["name"]

    return top15


# ══════════════════════════════════════════════════════════════════════════════
#  PPI CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ppi(df: pd.DataFrame) -> dict:
    probs     = df["Cold_Probability"]
    mu_top100 = probs.nlargest(100).mean()
    f_90      = (probs >= 0.90).mean()
    mean_prob = probs.mean()
    r_flex_raw  = mean_prob * (RFLEX_MAX - RFLEX_MIN) + RFLEX_MIN
    r_flex_norm = max(0.0, min(1.0, (r_flex_raw - RFLEX_MIN) / (RFLEX_MAX - RFLEX_MIN)))
    ids     = df["Protein_ID"].astype(str)
    matches = ids.apply(lambda pid: any(pat.search(pid) for pat in EC_PATTERNS.values()))
    s_ind   = matches.mean()
    ppi     = (0.40 * mu_top100 + 0.25 * f_90 + 0.15 * r_flex_norm + 0.20 * s_ind) * 100
    return {
        "ppi":         round(ppi, 2),
        "mu_top100":   round(mu_top100 * 100, 2),
        "f_90_pct":    round(f_90 * 100, 2),
        "n_above_90":  int((probs >= 0.90).sum()),
        "r_flex_norm": round(r_flex_norm, 4),
        "s_ind":       round(s_ind, 4),
        "n_total":     len(df),
    }


def interpret_ppi(ppi: float) -> tuple:
    if ppi > 50:
        return (
            "COLD-ACTIVE ORGANISM",
            "This organism shows strong signals of adaptation to cold environments.",
            GREEN, "❄️",
        )
    elif ppi > 30:
        return (
            "COLD-TOLERANT ORGANISM",
            "This organism can function in cold environments but prefers moderate temperatures.",
            WARN, "🌥️",
        )
    else:
        return (
            "WARM-ADAPTED ORGANISM",
            "This organism is best adapted to moderate or warm temperatures.",
            RED_C, "🌱",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PLAIN-LANGUAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plain_verdict_text(ppi: float, org: str, n_total: int, n_above_90: int) -> str:
    _, _, _, icon = interpret_ppi(ppi)
    org_clean = org.replace("_", " ")
    if ppi > 50:
        return (
            f"The analysis of {org_clean} ({n_total:,} proteins scanned) found "
            f"strong evidence that this organism produces enzymes that are naturally "
            f"active at cold temperatures (0–15 °C). "
            f"{n_above_90:,} proteins scored above the high-confidence threshold."
        )
    elif ppi > 30:
        return (
            f"The analysis of {org_clean} ({n_total:,} proteins scanned) found "
            f"moderate evidence of cold-temperature enzyme activity. "
            f"This organism is likely cold-tolerant rather than a strict psychrophile. "
            f"{n_above_90:,} proteins scored above the high-confidence threshold."
        )
    else:
        return (
            f"The analysis of {org_clean} ({n_total:,} proteins scanned) found "
            f"limited evidence of cold-temperature enzyme activity. "
            f"This organism appears to be primarily adapted to moderate temperatures."
        )


def enzyme_summary_text(top15: pd.DataFrame) -> str:
    """Build a plain-language summary of enzyme types found in top 15."""
    counts = top15["enzyme_type"].value_counts()
    known  = {k: v for k, v in counts.items() if k != "Unknown"}
    n_unk  = counts.get("Unknown", 0)

    if len(known) == 0 and n_unk > 0:
        return (
            f"The top 15 candidates could not be assigned to a specific industrial enzyme "
            f"family from their sequence identifier alone. This is common for organisms "
            f"whose proteomes are not well-annotated in public databases. "
            f"Use the UniProt links in the table below to determine each protein's function — "
            f"several may still be relevant depending on your application."
        )
    parts = []
    for etype, n in known.items():
        app = ENZYME_APPLICATIONS.get(etype, "")
        parts.append(f"{n} {etype}{'s' if n > 1 else ''} ({app})" if app else f"{n} {etype}")
    summary = "The top-ranked candidates include: " + "; ".join(parts) + "."
    if n_unk > 0:
        summary += f" {n_unk} candidate(s) could not be assigned to a specific family — check UniProt."
    return summary


def confidence_label(p: float) -> str:
    if p >= 0.95: return "Very high"
    if p >= 0.90: return "High"
    if p >= 0.80: return "Moderate"
    return "Low"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_results(organism: str) -> pd.DataFrame:
    path = os.path.join(GENOMES_DIR, f"{organism}_full_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results not found: {path}\n"
            f"Run first: python src/09_predict_new_genome.py"
        )
    df = pd.read_csv(path)
    df["Cold_Probability"] = pd.to_numeric(df["Cold_Probability"], errors="coerce")
    return df.dropna(subset=["Cold_Probability"])


# ══════════════════════════════════════════════════════════════════════════════
#  MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_markdown(organism: str, df: pd.DataFrame,
                   metrics: dict, top15: pd.DataFrame) -> str:
    ppi             = metrics["ppi"]
    verdict, _, _, icon = interpret_ppi(ppi)
    org_display     = organism.replace("_", " ")
    enz_summary     = enzyme_summary_text(top15)
    plain_text      = plain_verdict_text(
        ppi, organism, metrics["n_total"], metrics["n_above_90"])

    lines = [
        f"# Cold-Enzyme Analysis Report",
        f"## *{org_display}*",
        f"",
        f"**Date:** {date.today()}  |  "
        f"**Proteins analysed:** {metrics['n_total']:,}  |  "
        f"**Method:** PsychroScan v2.0",
        f"",
        f"---",
        f"",
        f"## {icon} Overall result: {verdict}",
        f"",
        f"{plain_text}",
        f"",
        f"> **Cold-adaptation score: {ppi:.0f} / 100** ",
        f"> Scale: 0–30 = warm-adapted · 31–50 = cold-tolerant · 51–100 = cold-active",
        f"",
        f"---",
        f"",
        f"## What enzymes were found?",
        f"",
        f"{enz_summary}",
        f"",
        f"---",
        f"",
        f"## Top 15 candidate enzymes",
        f"",
        f"These are the 15 proteins most likely to be active at cold temperatures "
        f"(0–15 °C). Search any **Protein ID** at "
        f"[uniprot.org/uniprotkb](https://www.uniprot.org/uniprotkb/) "
        f"to see its full biological description.",
        f"",
        f"| Rank | Protein ID | Enzyme type | Confidence |",
        f"|:---:|:---|:---|:---|",
    ]

    for i, (_, row) in enumerate(top15.iterrows(), 1):
        p      = row["Cold_Probability"]
        etype  = row.get("enzyme_type", "Unknown")
        pname  = row.get("protein_name", "")
        label  = confidence_label(p)
        pid    = row["Protein_ID"]
        # Show protein name if available, else just ID
        display_id = f"**{pid}**" + (f"<br>*{pname}*" if pname else "")
        lines.append(f"| {i} | {display_id} | {etype} | **{label}** ({p*100:.1f}%) |")

    lines += [
        f"",
        f"---",
        f"",
        f"## What to do next",
        f"",
        f"1. **Look up the top candidates on UniProt.** Copy any Protein ID above and "
        f"search at [uniprot.org/uniprotkb](https://www.uniprot.org/uniprotkb/). "
        f"This will show you the protein's known function, organism of origin, "
        f"and any published literature about it.",
        f"",
        f"2. **Prioritise candidates by enzyme type.** If you are looking for a specific "
        f"type of enzyme (e.g. a lipase for food processing, or a protease for aquafeed), "
        f"select the top-ranked candidates of that type from the table above.",
        f"",
        f"3. **Order synthetic gene synthesis or plan expression.** The top 3–5 candidates "
        f"are ready for heterologous expression in *E. coli* or yeast. "
        f"Once expressed, test enzymatic activity at 4 °C, 10 °C, and 25 °C to confirm "
        f"cold-active behaviour.",
        f"",
        f"4. **Check the biochemical distance dendrogram.** The PNG file generated alongside "
        f"this report shows where *{org_display}* sits relative to known psychrophiles "
        f"and mesophiles based on its full protein composition — not its evolutionary tree.",
        f"",
        f"---",
        f"",
        f"*Generated by PsychroScan v2.0 — "
        f"[github.com/CANOLIO/antarctic-fungi-ml]"
        f"(https://github.com/CANOLIO/antarctic-fungi-ml)* ",
        f"*Computational predictions require experimental validation.*",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _short_pid(pid: str) -> str:
    """Return the bare accession from a UniProt-style ID (tr|ACC|...) for display."""
    parts = pid.split("|")
    return parts[1] if len(parts) >= 2 else pid


def build_pdf(organism: str, df: pd.DataFrame,
              metrics: dict, top15: pd.DataFrame, out_path: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

    ppi                    = metrics["ppi"]
    verdict, _, v_color, icon = interpret_ppi(ppi)
    org_display            = organism.replace("_", " ")
    plain_text             = plain_verdict_text(
        ppi, organism, metrics["n_total"], metrics["n_above_90"])
    enz_summary            = enzyme_summary_text(top15)

    C_COLD  = colors.HexColor(COLD)
    C_BLUE  = colors.HexColor(BLUE)
    C_LIGHT = colors.HexColor(LIGHT)
    C_GREY  = colors.HexColor(GREY)
    C_VERD  = colors.HexColor(v_color)
    C_GREEN = colors.HexColor(GREEN)
    
    # UI Elements colors
    C_BORDER_LIGHT = colors.HexColor("#e2e8f0")
    C_TEXT_DARK    = colors.HexColor("#1e293b")
    C_TEXT_MUTED   = colors.HexColor("#475569")

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    # Modernized Typography Styles
    SECT  = S("h",  fontSize=13, fontName="Helvetica-Bold",
               textColor=C_COLD, spaceBefore=14, spaceAfter=6)
    BODY  = S("b",  fontSize=9.5,  fontName="Helvetica",
               textColor=C_TEXT_DARK,
               leading=15, alignment=TA_JUSTIFY)
    SMALL = S("sm", fontSize=8, fontName="Helvetica",
               textColor=C_TEXT_MUTED,
               leading=12)
    MONO  = S("m",  fontSize=8, fontName="Courier-Bold",
               textColor=C_BLUE, leading=12)
    VERD_S = S("v",  fontSize=14, fontName="Helvetica-Bold",
                textColor=C_VERD, alignment=TA_LEFT, spaceAfter=6)
    VERD_D = S("vd", fontSize=10,  fontName="Helvetica",
                textColor=C_TEXT_DARK,
                alignment=TA_JUSTIFY, leading=15)
    STEP   = S("st", fontSize=10, fontName="Helvetica-Bold",
                textColor=C_COLD, spaceBefore=8, spaceAfter=3)
    STEP_B = S("sb", fontSize=9.5, fontName="Helvetica",
                textColor=C_TEXT_DARK,
                leading=14, alignment=TA_JUSTIFY)

    # ── header / footer ───────────────────────────────────────────────────────
    def header_footer(canvas, doc):
        W, H = A4
        canvas.saveState()
        # Header Background
        canvas.setFillColor(C_COLD)
        canvas.rect(0, H - 4.0*cm, W, 4.0*cm, fill=1, stroke=0)
        
        # Subtle accent line at the bottom of header
        canvas.setFillColor(C_BLUE)
        canvas.rect(0, H - 4.2*cm, W, 0.2*cm, fill=1, stroke=0)
        
        # Header Text
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 22)
        canvas.drawString(2.0*cm, H - 1.8*cm, "PsychroScan")
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(colors.HexColor("#cbd5e1"))
        canvas.drawString(2.0*cm, H - 2.5*cm,
                          f"Cold-Enzyme Analysis Report   |   {org_display}")
        canvas.drawString(2.0*cm, H - 3.1*cm, f"Date: {date.today()}")
        
        # Modern Pill-shaped PPI badge
        ppi_col = (C_GREEN if ppi > 50
                   else colors.HexColor(WARN) if ppi > 30
                   else colors.HexColor(RED_C))
        canvas.setFillColor(ppi_col)
        canvas.roundRect(W - 5.5*cm, H - 3.2*cm, 3.5*cm, 1.4*cm,
                         8, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawCentredString(W - 3.75*cm, H - 2.45*cm, f"Score: {ppi:.0f}/100")
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(W - 3.75*cm, H - 2.95*cm, "Cold-adaptation index")
        
        # Footer
        canvas.setFillColor(colors.HexColor("#f1f5f9"))
        canvas.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
        canvas.setFillColor(C_GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(
            W / 2, 0.35*cm,
            "PsychroScan  |  Computational predictions require experimental validation  |  "
            "github.com/CANOLIO/antarctic-fungi-ml",
        )
        canvas.restoreState()

    doc  = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
        topMargin=5.0*cm, bottomMargin=2.0*cm,
    )
    W_IN  = A4[0] - 4.0*cm
    story = []

    # ── 1. Verdict box (Modern Callout Style) ─────────────────────────────────
    vt = Table(
        [[Paragraph(f"{icon}  {verdict}", VERD_S)],
         [Paragraph(plain_text, VERD_D)]],
        colWidths=[W_IN],
    )
    vt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("LINEBEFORE",    (0, 0), (0, -1), 4, C_VERD), # Thick left accent line
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT), # Thin subtle border
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
    ]))
    story.append(vt)
    story.append(Spacer(1, 0.6*cm))

    # ── 2. Cold-adaptation score bar ──────────────────────────────────────────
    story.append(Paragraph("Cold-Adaptation Score", SECT))

    score_data = [[
        Paragraph(f"<b>{ppi:.0f}</b> / 100", S("sc", fontSize=24,
                  fontName="Helvetica-Bold", textColor=C_VERD,
                  alignment=TA_CENTER)),
        Paragraph(
            f"Scale: <b>0–30</b> = warm-adapted organism  ·  "
            f"<b>31–50</b> = cold-tolerant  ·  "
            f"<b>51–100</b> = cold-active / psychrophile",
            S("sc2", fontSize=9, fontName="Helvetica",
              textColor=C_TEXT_MUTED,
              leading=14, alignment=TA_LEFT)),
    ]]
    st = Table(score_data, colWidths=[W_IN * 0.22, W_IN * 0.78])
    st.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.5*cm))

    # ── 3. Enzyme types found ─────────────────────────────────────────────────
    story.append(Paragraph("Enzyme Types Identified in Top Candidates", SECT))
    story.append(Paragraph(enz_summary, BODY))
    story.append(Spacer(1, 0.25*cm))

    # Enzyme type breakdown mini-table (Clean Design)
    counts = top15["enzyme_type"].value_counts()
    known_counts = {k: v for k, v in counts.items() if k != "Unknown"}
    if len(known_counts) > 0:
        counts = pd.Series(known_counts)
        enz_rows = [[
            Paragraph("<b>Enzyme type</b>", BODY),
            Paragraph("<b>Count in top 15</b>", BODY),
            Paragraph("<b>Industrial application</b>", BODY),
        ]]
        for etype, n in counts.items():
            app = ENZYME_APPLICATIONS.get(etype, "—")
            enz_rows.append([
                Paragraph(etype, BODY),
                Paragraph(str(n), BODY),
                Paragraph(app, SMALL),
            ])
        et = Table(enz_rows, colWidths=[W_IN*0.25, W_IN*0.18, W_IN*0.57])
        et_style = [
            ("BACKGROUND",    (0, 0), (-1, 0), C_COLD),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("LINEBELOW",     (0, 0), (-1, 0), 2, C_BLUE), # Accent line under header
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]
        # Alternating row colors with subtle lines
        for i in range(1, len(enz_rows)):
            bg = colors.HexColor(LIGHT) if i % 2 == 1 else colors.white
            et_style.append(("BACKGROUND", (0, i), (-1, i), bg))
            et_style.append(("LINEBELOW", (0, i), (-1, i), 0.5, C_BORDER_LIGHT))
        et.setStyle(TableStyle(et_style))
        story.append(et)
    story.append(Spacer(1, 0.6*cm))

    # ── 4. Top 15 candidates ──────────────────────────────────────────────────
    story.append(Paragraph("Top 15 Candidate Enzymes", SECT))
    story.append(Paragraph(
        "Ranked by cold-activity confidence. Search any Protein ID at "
        "<b>uniprot.org/uniprotkb</b> to view its biological description.",
        SMALL))
    story.append(Spacer(1, 0.3*cm))

    top_rows = [[
        Paragraph("<b>Rank</b>", BODY),
        Paragraph("<b>Protein ID</b>", BODY),
        Paragraph("<b>Enzyme type</b>", BODY),
        Paragraph("<b>Protein name</b>", BODY),
        Paragraph("<b>Confidence</b>", BODY),
    ]]
    for i, (_, row) in enumerate(top15.iterrows(), 1):
        p      = row["Cold_Probability"]
        etype  = row.get("enzyme_type", "Unknown")
        pname  = row.get("protein_name", "—") or "—"
        label  = confidence_label(p)
        conf_color = (colors.HexColor(GREEN)  if p >= 0.95 else
                      colors.HexColor("#047857") if p >= 0.90 else
                      colors.HexColor(WARN))
        top_rows.append([
            Paragraph(str(i), BODY),
            Paragraph(_short_pid(row["Protein_ID"]), MONO),
            Paragraph(etype, BODY),
            Paragraph(pname, SMALL),
            Paragraph(f"<b>{label}</b><br/>{p*100:.1f}%",
                      S(f"conf{i}", fontSize=8.5, fontName="Helvetica",
                        textColor=conf_color, leading=12, alignment=TA_CENTER)),
        ])

    tt = Table(top_rows,
               colWidths=[W_IN*0.08, W_IN*0.26, W_IN*0.22,
                          W_IN*0.26, W_IN*0.18])
    ts = [
        ("BACKGROUND",    (0, 0), (-1, 0), C_COLD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("LINEBELOW",     (0, 0), (-1, 0), 2, C_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    # Clean alternating rows without vertical borders
    for i in range(1, len(top_rows)):
        bg = colors.HexColor(LIGHT) if i % 2 == 1 else colors.white
        ts.append(("BACKGROUND", (0, i), (-1, i), bg))
        ts.append(("LINEBELOW", (0, i), (-1, i), 0.5, C_BORDER_LIGHT))
    tt.setStyle(TableStyle(ts))
    story.append(tt)
    story.append(Spacer(1, 0.6*cm))

    # ── 5. What to do next ────────────────────────────────────────────────────
    story.append(Paragraph("What to Do Next", SECT))

    steps = [
        ("Step 1 — Look up each candidate on UniProt",
         f"Go to uniprot.org/uniprotkb and search the Protein ID. "
         f"You will find the protein's known function, organism, and any published studies."),
        ("Step 2 — Select candidates by enzyme type",
         f"If you need a specific enzyme type (e.g. a lipase for food processing, "
         f"a protease for aquafeed), choose the top-ranked candidates of that type "
         f"from the table above."),
        ("Step 3 — Express and test",
         f"The top 3–5 candidates are ready for heterologous expression in E. coli or yeast. "
         f"Once expressed, measure enzymatic activity at 4 °C, 10 °C, and 25 °C "
         f"to confirm cold-active behaviour."),
        ("Step 4 — Review the dendrogram",
         f"The PNG file generated alongside this report shows where {org_display} "
         f"sits relative to known cold-adapted and warm-adapted organisms — "
         f"based on protein composition, not evolutionary history."),
    ]
    for title, body in steps:
        story.append(Paragraph(title, STEP))
        story.append(Paragraph(body, STEP_B))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Computational predictions require experimental validation. "
        "This report does not constitute proof of enzymatic activity at cold temperatures.",
        SMALL,
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"  PDF  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR + CLI
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(organism: str, make_pdf: bool = True,
                    use_uniprot: bool = True):
    print(f"\n  Organism : {organism}")
    df      = load_results(organism)
    metrics = compute_ppi(df)
    top15   = enrich_top15(df, use_uniprot=use_uniprot)

    print(f"  PPI      : {metrics['ppi']:.1f} / 100  — {interpret_ppi(metrics['ppi'])[0]}")
    print(f"  Proteins : {metrics['n_total']:,}  |  "
          f"High-confidence: {metrics['n_above_90']:,} ({metrics['f_90_pct']:.1f}%)")

    md_path = os.path.join(REPORTS_DIR, f"{organism}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(organism, df, metrics, top15))
    print(f"  MD   → {md_path}")

    if make_pdf:
        pdf_path = os.path.join(REPORTS_DIR, f"{organism}_report.pdf")
        build_pdf(organism, df, metrics, top15, pdf_path)


def main():
    parser = argparse.ArgumentParser(description="PsychroScan — Report Generator v3.0")
    parser.add_argument("--organism",   type=str)
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--no-pdf",     action="store_true")
    parser.add_argument("--no-uniprot", action="store_true",
                        help="Skip UniProt API calls (offline / faster)")
    args       = parser.parse_args()
    make_pdf   = not args.no_pdf
    use_uniprot = not args.no_uniprot

    print("\n" + "=" * 55)
    print("  PsychroScan — Report Generator v3.0")
    print("=" * 55)

    csvs = sorted(
        f.replace("_full_results.csv", "")
        for f in os.listdir(GENOMES_DIR)
        if f.endswith("_full_results.csv")
    )

    if args.organism:
        generate_report(args.organism, make_pdf, use_uniprot)
    else:
        if not csvs:
            print("  No results found.")
            print("  Run: python src/09_predict_new_genome.py")
            return
        for org in csvs:
            generate_report(org, make_pdf, use_uniprot)

    print(f"\n  Reports saved → {REPORTS_DIR}/\n")


if __name__ == "__main__":
    main()