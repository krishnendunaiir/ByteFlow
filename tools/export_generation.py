"""Create downloadable PDF, DOCX, PPTX, XLSX and CSV exports for ByteFlow.

All exports are built in memory. The module never writes uploaded dataset rows or
report files to disk.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
from typing import Any, Iterable

from docx import Document
from docx.shared import Inches as DocxInches, Pt as DocxPt, RGBColor as DocxRGBColor
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportLabImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as ReportLabTable,
    TableStyle,
)
from xml.sax.saxutils import escape


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _report_text(result: dict[str, Any]) -> str:
    text = result.get("report_markdown")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("There is no completed report to export.")
    return text


def _blocks(markdown: str) -> Iterable[tuple[str, Any]]:
    """Yield simple Markdown blocks: heading, paragraph, bullet or table."""
    lines = markdown.splitlines()
    paragraph: list[str] = []
    i = 0

    def flush_paragraph():
        if paragraph:
            value = " ".join(part.strip() for part in paragraph).strip()
            paragraph.clear()
            if value:
                return value
        return None

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            value = flush_paragraph()
            if value:
                yield "paragraph", value
            i += 1
            continue

        if line.startswith("|"):
            value = flush_paragraph()
            if value:
                yield "paragraph", value
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip()
                cells = [cell.strip() for cell in raw.strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
                    table_lines.append(cells)
                i += 1
            if table_lines:
                width = max(len(row) for row in table_lines)
                yield "table", [row + [""] * (width - len(row)) for row in table_lines]
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            value = flush_paragraph()
            if value:
                yield "paragraph", value
            yield "heading", (len(heading.group(1)), heading.group(2))
            i += 1
            continue

        if line.startswith("- ") or line.startswith("* "):
            value = flush_paragraph()
            if value:
                yield "paragraph", value
            yield "bullet", line[2:].strip()
            i += 1
            continue

        paragraph.append(line)
        i += 1

    value = flush_paragraph()
    if value:
        yield "paragraph", value


def _plain(text: Any) -> str:
    value = str(text)
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = re.sub(r"(?<!\*)\*(.*?)\*(?!\*)", r"\1", value)
    value = re.sub(r"`(.*?)`", r"\1", value)
    return html.unescape(value).strip()


def _docx_rich(paragraph, text: str) -> None:
    """Add basic **bold** and `code` formatting to a Word paragraph."""
    pattern = re.compile(r"(\*\*.+?\*\*|`.+?`)")
    pos = 0
    for match in pattern.finditer(str(text)):
        if match.start() > pos:
            paragraph.add_run(html.unescape(str(text)[pos:match.start()]))
        token = match.group(0)
        run = paragraph.add_run(token[2:-2] if token.startswith("**") else token[1:-1])
        run.bold = token.startswith("**")
        if token.startswith("`"):
            run.font.name = "Consolas"
        pos = match.end()
    if pos < len(str(text)):
        paragraph.add_run(html.unescape(str(text)[pos:]))


def _safe_chart_images(chart_images: list[tuple[str, bytes]] | None) -> list[tuple[str, bytes]]:
    """Accept only non-empty in-memory PNG/JPEG bytes supplied by the caller."""
    safe: list[tuple[str, bytes]] = []
    for title, data in chart_images or []:
        if isinstance(data, bytes) and data[:8] == b"\x89PNG\r\n\x1a\n":
            safe.append((str(title)[:120], data))
        elif isinstance(data, bytes) and data[:3] == b"\xff\xd8\xff":
            safe.append((str(title)[:120], data))
    return safe


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item if isinstance(item, dict) else {"value": item} for item in value]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)[:30000]
    if isinstance(value, str):
        text = value[:30000]
        # Prevent spreadsheet formula injection from data-derived text.
        if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + text
        return text
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)[:30000]


def _safe_sheet_title(value: str) -> str:
    title = re.sub(r"[\\/*?:\[\]]", "_", value).strip() or "Sheet"
    return title[:31]


def _flatten(prefix: str, value: Any, output: list[tuple[str, Any]], limit: int = 5000) -> None:
    if len(output) >= limit:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, output, limit)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _flatten(f"{prefix}[{index}]", item, output, limit)
    else:
        output.append((prefix, value))


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def build_pdf_bytes(
    result: dict[str, Any], chart_images: list[tuple[str, bytes]] | None = None
) -> bytes:
    """Build a landscape PDF report, including any validated chart images."""
    markdown = _report_text(result)
    output = io.BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(
        output, pagesize=page, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title="ByteFlow Analysis Report",
        author="ByteFlow",
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ByteFlowTitle", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, textColor=colors.HexColor("#17324D"),
        alignment=TA_CENTER, spaceAfter=14,
    )
    h1 = ParagraphStyle("ByteFlowH1", parent=base["Heading1"], textColor=colors.HexColor("#176B87"),
                        spaceBefore=12, spaceAfter=7, keepWithNext=True)
    h2 = ParagraphStyle("ByteFlowH2", parent=base["Heading2"], textColor=colors.HexColor("#24566D"),
                        spaceBefore=8, spaceAfter=5, keepWithNext=True)
    body = ParagraphStyle("ByteFlowBody", parent=base["BodyText"], fontSize=8.5, leading=12, spaceAfter=5)
    bullet = ParagraphStyle("ByteFlowBullet", parent=body, leftIndent=14, firstLineIndent=-8, spaceAfter=3)
    story: list[Any] = [Paragraph("ByteFlow Analysis Report", title_style)]

    for kind, value in _blocks(markdown):
        if kind == "heading":
            level, text = value
            clean = escape(_plain(text))
            story.append(Paragraph(clean, h1 if level == 1 else h2))
        elif kind == "paragraph":
            clean = escape(str(value))
            clean = re.sub(r"&lt;(.*?)&gt;", r"&lt;\1&gt;", clean)
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean)
            clean = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", clean)
            story.append(Paragraph(clean, body))
        elif kind == "bullet":
            clean = escape(str(value))
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean)
            story.append(Paragraph("&#8226; " + clean, bullet))
        elif kind == "table":
            data = [[Paragraph(escape(_plain(cell)), body) for cell in row] for row in value]
            table = ReportLabTable(data, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 7)])

    images = _safe_chart_images(chart_images)
    if images:
        story.append(PageBreak())
        story.append(Paragraph("Generated visualizations", h1))
        for title, data in images:
            story.append(Paragraph(escape(title), h2))
            image = ReportLabImage(io.BytesIO(data))
            image._restrictSize(245 * mm, 145 * mm)
            story.extend([image, Spacer(1, 8)])

    doc.build(story)
    return output.getvalue()


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def build_docx_bytes(
    result: dict[str, Any], chart_images: list[tuple[str, bytes]] | None = None
) -> bytes:
    """Build an editable Word report with headings, tables and optional charts."""
    markdown = _report_text(result)
    doc = Document()
    section = doc.sections[0]
    section.top_margin = DocxInches(0.65)
    section.bottom_margin = DocxInches(0.65)
    section.left_margin = DocxInches(0.75)
    section.right_margin = DocxInches(0.75)
    title = doc.add_heading("ByteFlow Analysis Report", 0)
    title.alignment = 1

    for kind, value in _blocks(markdown):
        if kind == "heading":
            level, text = value
            doc.add_heading(_plain(text), level=min(level + 1, 3))
        elif kind == "paragraph":
            _docx_rich(doc.add_paragraph(), value)
        elif kind == "bullet":
            _docx_rich(doc.add_paragraph(style="List Bullet"), value)
        elif kind == "table":
            rows = value
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.style = "Light Shading Accent 1"
            for row_index, row in enumerate(rows):
                cells = table.add_row().cells
                for col_index, cell_text in enumerate(row):
                    cells[col_index].text = _plain(cell_text)
                    if row_index == 0:
                        for run in cells[col_index].paragraphs[0].runs:
                            run.bold = True
            doc.add_paragraph()

    images = _safe_chart_images(chart_images)
    if images:
        doc.add_page_break()
        doc.add_heading("Generated visualizations", level=1)
        for title_text, data in images:
            doc.add_heading(title_text, level=2)
            doc.add_picture(io.BytesIO(data), width=DocxInches(6.2))

    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


# ---------------------------------------------------------------------------
# PowerPoint
# ---------------------------------------------------------------------------


def _add_slide_text(slide, title: str, bullets: list[str]) -> None:
    slide.shapes.title.text = title[:110]
    body = slide.placeholders[1].text_frame
    body.clear()
    body.word_wrap = True
    for index, text in enumerate(bullets[:8]):
        paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
        paragraph.text = _plain(text)[:420]
        paragraph.level = 0
        paragraph.font.size = Pt(20)


def build_pptx_bytes(
    result: dict[str, Any], chart_images: list[tuple[str, bytes]] | None = None
) -> bytes:
    """Build a concise presentation from report sections, followed by chart slides."""
    markdown = _report_text(result)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = "ByteFlow Analysis Report"
    title_slide.placeholders[1].text = "Automated data analysis and findings"

    sections: list[tuple[str, list[str]]] = []
    current_title = "Executive summary"
    current_bullets: list[str] = []
    for kind, value in _blocks(markdown):
        if kind == "heading":
            level, heading = value
            if level == 1:
                if current_bullets:
                    sections.append((current_title, current_bullets))
                current_title, current_bullets = _plain(heading), []
        elif kind == "paragraph":
            current_bullets.append(value)
        elif kind == "bullet":
            current_bullets.append(value)
        elif kind == "table":
            for row in value[1:7]:
                current_bullets.append(" | ".join(_plain(cell) for cell in row))
    if current_bullets:
        sections.append((current_title, current_bullets))

    for section_title, bullets in sections:
        chunks = [bullets[i:i + 7] for i in range(0, len(bullets), 7)] or [[]]
        for chunk_index, chunk in enumerate(chunks, start=1):
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            heading = section_title if len(chunks) == 1 else f"{section_title} ({chunk_index})"
            _add_slide_text(slide, heading, chunk)

    for chart_title, data in _safe_chart_images(chart_images):
        slide = prs.slides.add_slide(blank)
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.5))
        title_box.text_frame.text = chart_title[:110]
        title_box.text_frame.paragraphs[0].font.size = Pt(24)
        picture = slide.shapes.add_picture(io.BytesIO(data), Inches(0.8), Inches(1.0), width=Inches(11.7))

    output = io.BytesIO()
    prs.save(output)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def _write_records(workbook: Workbook, title: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    ws = workbook.create_sheet(_safe_sheet_title(title))
    headers: list[str] = []
    for record in records:
        for key in record:
            if key not in headers:
                headers.append(str(key))
    ws.append(headers)
    for record in records:
        ws.append([_cell(record.get(header)) for header in headers])
    _style_sheet(ws)


def _style_sheet(ws) -> None:
    header_fill = PatternFill("solid", fgColor="17324D")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for column_cells in ws.columns:
        length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
        ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max(length + 2, 12), 48)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def build_xlsx_bytes(result: dict[str, Any]) -> bytes:
    """Build an Excel workbook with summary, quality, statistical and ML sheets."""
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    upload = result.get("upload_info") or {}
    summary_rows = [
        ["Field", "Value"],
        ["Status", result.get("status", "")],
        ["Question", result.get("question", "")],
        ["Dataset", upload.get("display_name", upload.get("file_name", ""))],
        ["Rows analysed", (result.get("analysis_results") or {}).get("n_rows_analysed", "")],
        ["Columns", (result.get("profile") or {}).get("n_columns", "")],
    ]
    summary_data = result.get("report_markdown", "")
    match = re.search(r"## 1\. Executive summary\s+(.*?)(?:\n## |\Z)", summary_data, re.S)
    if match:
        summary_rows.append(["Executive summary", _plain(match.group(1).strip())[:3000]])
    for row in summary_rows:
        summary.append([_cell(value) for value in row])
    _style_sheet(summary)

    quality = result.get("quality") or {}
    missing = quality.get("missing") or {}
    _write_records(workbook, "Missing by column", _records(missing.get("per_column")))
    outliers = quality.get("outliers") or {}
    _write_records(workbook, "Outliers", _records(outliers.get("by_column")))
    _write_records(workbook, "Suspicious values", _records(quality.get("suspicious")))
    _write_records(workbook, "Cleaning log", _records(quality.get("cleaning_log")))
    _write_records(workbook, "Recommendations", _records(quality.get("recommendations")))

    stat_records: list[dict[str, Any]] = []
    for step in (result.get("analysis_results") or {}).get("steps", []):
        _flatten(step.get("result", {}), "", []) if False else None
        flat: list[tuple[str, Any]] = []
        _flatten("", step.get("result", {}), flat, limit=800)
        if not flat:
            stat_records.append({"action": step.get("action"), "status": step.get("status"),
                                 "detail": step.get("reason", "")})
        else:
            for metric, value in flat:
                stat_records.append({"action": step.get("action"), "status": step.get("status"),
                                     "metric": metric, "value": _cell(value)})
    _write_records(workbook, "Statistics", stat_records)

    ml = result.get("ml_results") or {}
    ml_records: list[dict[str, Any]] = []
    for metric, value in (ml.get("test_metrics") or {}).items():
        ml_records.append({"category": "test metric", "name": metric, "value": value})
    for model, values in (ml.get("cv_results") or {}).items():
        for metric, value in values.items():
            ml_records.append({"category": f"CV: {model}", "name": metric, "value": value})
    for item in ml.get("permutation_importance", []) or []:
        ml_records.append({"category": "permutation importance", **item})
    for cluster, size in (ml.get("cluster_sizes") or {}).items():
        ml_records.append({"category": "cluster size", "name": cluster, "value": size})
    _write_records(workbook, "Machine learning", ml_records)

    workflow_records = [{"type": "progress", "message": item} for item in result.get("progress", [])]
    workflow_records.extend({"type": "notice", "message": item} for item in result.get("errors", []))
    _write_records(workbook, "Workflow", workflow_records)

    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def build_csv_bytes(result: dict[str, Any]) -> bytes:
    """Build a portable CSV of flattened computed results and report metadata."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)

    def write_row(values: list[Any]) -> None:
        # CSV files are commonly opened in Excel; neutralise formula-like text.
        writer.writerow([_cell(value) for value in values])

    write_row(["section", "metric", "value"])
    write_row(["analysis", "status", result.get("status", "")])
    write_row(["analysis", "question", result.get("question", "")])
    upload = result.get("upload_info") or {}
    write_row(["dataset", "file", upload.get("display_name", upload.get("file_name", ""))])
    for step in (result.get("analysis_results") or {}).get("steps", []):
        flat: list[tuple[str, Any]] = []
        _flatten("", step.get("result", {}), flat)
        if not flat:
            write_row([step.get("action", "statistics"), "status", step.get("reason", step.get("status", ""))])
        for metric, value in flat:
            write_row([step.get("action", "statistics"), metric, value])
    quality = result.get("quality") or {}
    for metric in ("total_missing_cells", "rows_with_any_missing", "pct_cells_missing"):
        if metric in (quality.get("missing") or {}):
            write_row(["data quality", metric, quality["missing"][metric]])
    for metric in ("n_duplicate_rows", "pct_duplicate_rows"):
        if metric in (quality.get("duplicates") or {}):
            write_row(["data quality", metric, quality["duplicates"][metric]])
    for category in ("test_metrics", "cv_results"):
        _flatten(category, (result.get("ml_results") or {}).get(category, {}), []) if False else None
        flat = []
        _flatten(category, (result.get("ml_results") or {}).get(category, {}), flat)
        for metric, value in flat:
            write_row(["machine learning", metric, value])
    return output.getvalue().encode("utf-8-sig")
