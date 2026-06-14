"""
Section-aware PDF extraction pipeline.

Produces:
  1. text_chunks  – one Document per logical section (split on real headings
                    detected by font size, never across section boundaries).
  2. image_docs   – one Document per detected chart region or meaningful
                    embedded image, rendered as high-res PNG.

All configuration defaults are declared as constants on top of the file.
The public entry point accepts keyword-only arguments to override any of these,
packaging them into a single `PDFConfig` object to keep helper function signatures
clean and simple.
"""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Configurable Parameters (Constants)
# ──────────────────────────────────────────────────────────────────────

# ── Font-size thresholds for header classification ──
PAGE_TITLE_MIN_PT = 20.0
SECTION_HEADER_MIN_PT = 10.0
SUB_HEADER_MIN_PT = 9.0

# ── Chart detection tunables ──
CLUSTER_GAP_PT = 20.0
MIN_CHART_AREA = 100.0 * 100.0
MIN_CHART_ELEMENTS = 5  # min elements form a chart , exclude single dividing line as chart
CHART_TEXT_MAX_PT = 6.5  # axis labels like 4% on y-axis are part of chart
HEADER_LIMIT_PT = 150.0
FOOTER_LIMIT_PT = 750.0
CHART_RENDER_DPI = 100

# ── Embedded-image tunables ──
MIN_IMAGE_AREA = 100.0 * 100.0
SKIP_DUPLICATE_IMAGES = True

# ── Text-chunk safety split ──
MAX_CHUNK_TOKENS = 2000
FALLBACK_CHUNK_OVERLAP = 100

# ── Globally ignored bounding boxes (e.g. for page headers/logos) ──
# Format: list of (x0, y0, x1, y1) tuples
EXCLUDE_BBOXES = [
    (265.0, -10.0, 565.0, 110.0)  # Excludes the repeated ICICI logo banner on all pages
]

# ── Default paths ──
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = BASE_DIR / "static" / "icici-15-20.pdf"

# ── PDF metadata for temporal filtering ──
# Change this string when ingesting a new month's PDF.
MONTH_YEAR = "may_2026"


# ──────────────────────────────────────────────────────────────────────
#  Data Structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PDFConfig:
    """Consolidated configuration object passed to all internal helpers."""
    page_title_min_pt: float = PAGE_TITLE_MIN_PT
    section_header_min_pt: float = SECTION_HEADER_MIN_PT
    sub_header_min_pt: float = SUB_HEADER_MIN_PT
    cluster_gap_pt: float = CLUSTER_GAP_PT
    min_chart_area: float = MIN_CHART_AREA
    min_chart_elements: int = MIN_CHART_ELEMENTS
    chart_text_max_pt: float = CHART_TEXT_MAX_PT
    header_limit_pt: float = HEADER_LIMIT_PT
    footer_limit_pt: float = FOOTER_LIMIT_PT
    chart_render_dpi: int = CHART_RENDER_DPI
    min_image_area: float = MIN_IMAGE_AREA
    skip_duplicate_images: bool = SKIP_DUPLICATE_IMAGES
    max_chunk_tokens: int = MAX_CHUNK_TOKENS
    fallback_chunk_overlap: int = FALLBACK_CHUNK_OVERLAP
    exclude_bboxes: List[Tuple[float, float, float, float]] = field(
        default_factory=lambda: list(EXCLUDE_BBOXES)
    )
    month_year: str = MONTH_YEAR


@dataclass
class TextBlock:
    """A single text block parsed from fitz with classification metadata."""
    text: str
    bbox: Tuple[float, float, float, float]
    page_num: int
    max_font_size: float
    role: str = "body"  # "page_title" | "section_header" | "sub_header" | "body"


@dataclass
class ChartRegion:
    """A detected chart / visual region on a page."""
    bbox: fitz.Rect
    page_num: int
    num_paths: int
    region_type: str = "chart"  # "chart" | "image"


# ──────────────────────────────────────────────────────────────────────
#  Public entry-point
# ──────────────────────────────────────────────────────────────────────
async def load_and_chunk_pdf(
    pdf_path: str | Path | None = None,
    *,
    page_title_min_pt: float = PAGE_TITLE_MIN_PT,
    section_header_min_pt: float = SECTION_HEADER_MIN_PT,
    sub_header_min_pt: float = SUB_HEADER_MIN_PT,
    cluster_gap_pt: float = CLUSTER_GAP_PT,
    min_chart_area: float = MIN_CHART_AREA,
    min_chart_elements: int = MIN_CHART_ELEMENTS,
    chart_text_max_pt: float = CHART_TEXT_MAX_PT,
    header_limit_pt: float = HEADER_LIMIT_PT,
    footer_limit_pt: float = FOOTER_LIMIT_PT,
    chart_render_dpi: int = CHART_RENDER_DPI,
    min_image_area: float = MIN_IMAGE_AREA,
    skip_duplicate_images: bool = SKIP_DUPLICATE_IMAGES,
    max_chunk_tokens: int = MAX_CHUNK_TOKENS,
    fallback_chunk_overlap: int = FALLBACK_CHUNK_OVERLAP,
    exclude_bboxes: List[Tuple[float, float, float, float]] | None = None,
    month_year: str = MONTH_YEAR,
) -> Tuple[List[Document], List[Document], List[Document]]:
    """
    Extract section-aware text chunks, structured table documents, and
    chart/image documents from a PDF.

    Returns a 3-tuple: (text_chunks, table_docs, image_docs)

    All thresholds can be dynamically customized via argument overrides.
    The ``month_year`` argument stamps temporal metadata on every chunk
    (e.g. ``"may_2026"``); change it when ingesting a new month's PDF.
    """
    resolved_path = Path(pdf_path) if pdf_path else DEFAULT_PDF_PATH
    if not resolved_path.is_file():
        raise FileNotFoundError(f"PDF not found at {resolved_path}")

    # Build single thread-safe configuration object
    cfg = PDFConfig(
        page_title_min_pt=page_title_min_pt,
        section_header_min_pt=section_header_min_pt,
        sub_header_min_pt=sub_header_min_pt,
        cluster_gap_pt=cluster_gap_pt,
        min_chart_area=min_chart_area,
        min_chart_elements=min_chart_elements,
        chart_text_max_pt=chart_text_max_pt,
        header_limit_pt=header_limit_pt,
        footer_limit_pt=footer_limit_pt,
        chart_render_dpi=chart_render_dpi,
        min_image_area=min_image_area,
        skip_duplicate_images=skip_duplicate_images,
        max_chunk_tokens=max_chunk_tokens,
        fallback_chunk_overlap=fallback_chunk_overlap,
        exclude_bboxes=exclude_bboxes if exclude_bboxes is not None else list(EXCLUDE_BBOXES),
        month_year=month_year,
    )

    # Heavy I/O → run in a worker thread
    text_chunks, table_docs, image_docs = await asyncio.to_thread(
        _extract_all,
        resolved_path,
        cfg,
    )
    return text_chunks, table_docs, image_docs


# ══════════════════════════════════════════════════════════════════════
#  Core (sync) implementation – runs inside asyncio.to_thread
# ══════════════════════════════════════════════════════════════════════

def _extract_all(
    pdf_path: Path,
    cfg: PDFConfig,
) -> Tuple[List[Document], List[Document], List[Document]]:
    """Synchronous workhorse that does all fitz I/O.

    Returns a 3-tuple: (text_chunks, table_docs, image_docs)
    """

    doc = fitz.open(str(pdf_path))
    all_text_blocks: List[TextBlock] = []
    all_chart_regions: List[ChartRegion] = []
    seen_image_xrefs: set[int] = set()
    image_docs: List[Document] = []
    all_table_docs: List[Document] = []

    # Track fund name across pages (portfolio overflow pages inherit the previous fund)
    current_fund_name: str = ""

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ── 1a. Detect chart regions from vector drawings ──
        chart_regions = _detect_chart_regions(page, page_num, cfg)
        all_chart_regions.extend(chart_regions)

        # ── 1b. Extract structured tables from fund card pages ──
        table_docs, detected_fund_name, table_bboxes = _extract_tables(
            page, page_num, pdf_path, cfg, current_fund_name
        )
        if detected_fund_name:
            current_fund_name = detected_fund_name
        all_table_docs.extend(table_docs)

        # ── 1b-post. Remove chart regions that are mostly covered by table areas ──
        # Fund card pages have many vector elements (table borders, boxes) that
        # the chart detector clusters as a single large region. Filter those out.
        chart_regions = _filter_chart_regions_by_tables(chart_regions, table_bboxes)

        # ── 1c. Extract portfolio holdings from text blocks ──
        portfolio_docs = _extract_portfolio_section(
            page, page_num, pdf_path, cfg, current_fund_name
        )
        all_table_docs.extend(portfolio_docs)

        # ── 1d. Extract scheme details ──
        scheme_docs = _extract_scheme_details(
            page, page_num, pdf_path, cfg, current_fund_name
        )
        all_table_docs.extend(scheme_docs)

        # ── 1e. Extract quantitative indicators ──
        quant_docs = _extract_quantitative_indicators(
            page, page_num, pdf_path, cfg, current_fund_name
        )
        all_table_docs.extend(quant_docs)

        # ── 2. Classify text blocks, filtering out chart-interior text
        #       and already-extracted table regions ──
        exclusion_bboxes = [cr.bbox for cr in chart_regions] + table_bboxes
        text_blocks = _classify_text_blocks(
            page, page_num,
            chart_bboxes=exclusion_bboxes,
            cfg=cfg,
        )
        # Sort text blocks: primarily by y0 (top-to-bottom), secondarily by x0 (left-to-right)
        text_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        all_text_blocks.extend(text_blocks)

        # ── 3. Render chart regions — text-rich regions become text docs, true charts become images ──
        for idx, cr in enumerate(chart_regions):
            region_doc = _render_region_to_document(page, cr, idx, pdf_path, cfg, current_fund_name)
            if region_doc is None:
                continue
            if region_doc.metadata.get("modality") == "text":
                # Summary box extracted as text — route to structured table path
                all_table_docs.append(region_doc)
            else:
                image_docs.append(region_doc)

        # ── 4. Extract meaningful embedded raster images ──
        page_image_docs = _extract_embedded_images(
            doc, page, page_num, pdf_path,
            seen_xrefs=seen_image_xrefs,
            chart_bboxes=[cr.bbox for cr in chart_regions],
            cfg=cfg,
        )
        image_docs.extend(page_image_docs)

    doc.close()

    # ── 5. Accumulate text blocks into section-aware chunks ──
    text_chunks = _build_section_chunks(all_text_blocks, pdf_path, cfg)

    # ── 6. Stamp month_year on ALL chunks ──
    for chunk in text_chunks + all_table_docs + image_docs:
        chunk.metadata["month_year"] = cfg.month_year

    logger.info(
        f"Extraction complete: {len(text_chunks)} text chunks, "
        f"{len(all_table_docs)} table docs, "
        f"{len(image_docs)} image documents from {pdf_path.name}"
    )
    return text_chunks, all_table_docs, image_docs


# ──────────────────────────────────────────────────────────────────────
#  1. Structured table & section extraction (fund card pages)
# ──────────────────────────────────────────────────────────────────────

# Keywords that identify the low-value Riskometer table — we skip it.
_RISKOMETER_KEYWORDS = {"riskometer", "risk", "benchmark", "moderate", "low", "high", "very high"}

# Regex to detect portfolio rows: "Company Name  9.25%" or "Company  Rating  9.25%"
_PORTFOLIO_ROW_RE = re.compile(r"^(.+?)\s{2,}(\d+\.\d+%)\s*$")
# Broader pattern for lines that include a rating field between name and %
_PORTFOLIO_ROW_RATING_RE = re.compile(r"^(.+?)\s{2,}(\w+)\s{2,}(\d+\.\d+%)\s*$")


def _get_page_fund_name(page: fitz.Page, cfg: PDFConfig) -> str:
    """
    Extract the fund name from the page title text block (largest font on page,
    typically 17pt on fund card pages).  Returns empty string if not found.
    """
    raw = page.get_text("dict")
    best_size = 0.0
    best_text = ""
    for block in raw.get("blocks", []):
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                # Fund names are typically between 13–22pt
                if 13.0 <= span["size"] <= 22.0 and span["text"].strip():
                    if span["size"] > best_size:
                        best_size = span["size"]
                        best_text = span["text"].strip()
    return best_text


def _is_riskometer_table(table_text: str) -> bool:
    """Return True if the extracted table text looks like the boilerplate Riskometer."""
    lower = table_text.lower()
    hits = sum(1 for kw in _RISKOMETER_KEYWORDS if kw in lower)
    return hits >= 3


def _extract_tables(
    page: fitz.Page,
    page_num: int,
    pdf_path: Path,
    cfg: PDFConfig,
    current_fund_name: str,
) -> Tuple[List[Document], str, List[fitz.Rect]]:
    """
    Use ``page.find_tables()`` to extract structured tables from fund card pages.

    Returns:
        table_docs      – list of Documents (one per valid table)
        detected_fund   – fund name found on this page (empty str if none)
        table_bboxes    – bounding boxes of all detected tables (for exclusion)
    """
    table_docs: List[Document] = []
    table_bboxes: List[fitz.Rect] = []

    detected_fund = _get_page_fund_name(page, cfg)
    fund_name = detected_fund or current_fund_name

    try:
        tabs = page.find_tables()
    except Exception as e:
        logger.debug(f"Page {page_num}: find_tables() failed: {e}")
        return table_docs, detected_fund, table_bboxes

    if not tabs or not tabs.tables:
        return table_docs, detected_fund, table_bboxes

    for tab in tabs.tables:
        # Record bbox for exclusion from text classification and chart detection
        table_bboxes.append(fitz.Rect(tab.bbox))

        # Extract raw cell data
        try:
            data = tab.extract()
        except Exception as e:
            logger.debug(f"Page {page_num}: table.extract() failed: {e}")
            continue

        if not data:
            continue

        # Flatten all text for heuristic checks
        flat_text = " ".join(
            str(cell) for row in data for cell in row if cell is not None
        )

        # Skip Riskometer boilerplate
        if _is_riskometer_table(flat_text):
            continue

        # Determine number of rows / cols
        num_rows = len(data)
        num_cols = max(len(row) for row in data) if data else 0

        if num_rows < 4 or num_cols < 4:
            logger.debug(
                f"Page {page_num}: Skipping small table ({num_rows}r × {num_cols}c)"
            )
            continue

        # Normalise cells: strip whitespace, flatten inner newlines, replace None
        cleaned: List[List[str]] = []
        for row in data:
            cleaned.append([
                re.sub(r"\s+", " ", str(c)).strip() if c is not None else ""
                for c in row
            ])

        # Detect Returns table by keywords in first two rows
        header_text = " ".join(
            cell for row in cleaned[:2] for cell in row
        ).lower()
        is_returns_table = any(
            kw in header_text
            for kw in ("particulars", "cagr", "1 year", "3 year", "since inception")
        )

        if is_returns_table:
            # The returns table header spans 3 visual rows in the PDF, causing
            # find_tables() to produce garbled merged-cell content for column labels.
            # We bypass the extracted headers and use hardcoded ICICI-standard labels.
            _RETURNS_COLS = [
                "Particulars",
                "1Y CAGR(%)", "1Y Value(Rs.10k)",
                "3Y CAGR(%)", "3Y Value(Rs.10k)",
                "5Y CAGR(%)", "5Y Value(Rs.10k)",
                "SI CAGR(%)", "SI Value",
            ]

            # Data rows start after the two header rows (period labels + sub-labels).
            # Skip any trailing rows that look like NAV-only rows (first cell starts
            # with "NAV") or are entirely empty.
            data_rows = [
                row for row in (cleaned[2:] if len(cleaned) > 2 else cleaned[1:])
                if row and any(c for c in row)
                and not re.match(r"^nav\b", (row[0] or "").lower())
            ]

            # Truncate each row to the number of known columns
            n_cols = len(_RETURNS_COLS)
            lines = [f"Fund: {fund_name}", "Returns Table", ""]
            lines.append("| " + " | ".join(_RETURNS_COLS) + " |")
            lines.append("| " + " | ".join(["---"] * n_cols) + " |")
            for row in data_rows:
                # Pad or truncate row to exactly n_cols
                padded = (row + [""] * n_cols)[:n_cols]
                lines.append("| " + " | ".join(padded) + " |")
            page_content = "\n".join(lines)
            table_type = "returns"
        else:
            # Generic structured table
            md_lines = []
            for i, row in enumerate(cleaned):
                md_lines.append("| " + " | ".join(row) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(row)) + " |")
            page_content = "\n".join(md_lines)
            table_type = "generic"

        doc = Document(
            page_content=page_content,
            metadata={
                "source": str(pdf_path),
                "source_file": pdf_path.name,
                "page_num": page_num,
                "modality": "text",
                "table_type": table_type,
                "fund_name": fund_name,
            },
        )
        table_docs.append(doc)
        logger.debug(
            f"Page {page_num}: Extracted {table_type!r} table "
            f"({num_rows}r × {num_cols}c) for fund '{fund_name}'"
        )

    return table_docs, detected_fund, table_bboxes



def _extract_portfolio_section(
    page: fitz.Page,
    page_num: int,
    pdf_path: Path,
    cfg: PDFConfig,
    fund_name: str,
) -> List[Document]:
    """
    Parse Portfolio holdings using word-level extraction with y-coordinate grouping.

    Fund card pages use a two-column layout where company names (left column) and
    their percentages (right column) appear in *separate* text blocks at the same
    visual y-position.  We use ``get_text("words")`` and group words by approximate
    y-coordinate (±Y_TOL pt) so left-column and right-column content on the same
    visual row get reconstructed into one line before regex matching.
    """
    Y_TOL = 1.5  # tight tolerance so adjacent column headers at similar y don't merge

    # ── Collect all words with position ──
    words = page.get_text("words")  # list of (x0,y0,x1,y1, word, block_no, line_no, word_no)
    if not words:
        return []

    # ── Group words into visual rows by y-midpoint ──
    rows: dict[int, List[Tuple[float, str]]] = {}  # bucket_key -> [(x0, word), ...]
    for w in words:
        x0, y0, x1, y1, word, *_ = w
        y_mid = (y0 + y1) / 2
        # Find existing bucket within tolerance
        bucket = None
        for key in rows:
            if abs(key / 10.0 - y_mid) <= Y_TOL:
                bucket = key
                break
        if bucket is None:
            bucket = int(y_mid * 10)
        rows.setdefault(bucket, []).append((x0, word))

    # ── Reconstruct visual lines sorted by y then x ──
    visual_lines: List[Tuple[float, str]] = []  # (y_bucket, line_text)
    for key in sorted(rows):
        word_list = sorted(rows[key], key=lambda t: t[0])  # sort left→right by x
        line_text = " ".join(w for _, w in word_list).strip()
        visual_lines.append((key / 10.0, line_text))

    # ── Find portfolio section start ──
    # Match specifically "Portfolio as on" to avoid matching "Quantitative" section
    portfolio_start_idx = None
    for i, (_, text) in enumerate(visual_lines):
        if re.search(r"portfolio\s+as\s+on", text, re.IGNORECASE):
            portfolio_start_idx = i
            break

    if portfolio_start_idx is None:
        return []

    header_line = visual_lines[portfolio_start_idx][1]
    holdings: List[str] = []

    # Regex patterns applied to full reconstructed visual rows
    pct_re = re.compile(r"(\d+\.\d+)%")
    holding_re = re.compile(r"^(.+?)\s+(\d+\.\d+)%\s*$")

    for _, text in visual_lines[portfolio_start_idx + 1:]:
        text = text.strip()
        if not text:
            continue
        # Stop at next major section
        if re.search(r"(scheme details|quantitative indicator|riskometer|disclaimer)", text, re.IGNORECASE):
            break

        m = holding_re.match(text)
        if m:
            name = m.group(1).strip().lstrip("•· ")  # strip bullet chars
            pct = m.group(2) + "%"
            holdings.append(f"- {name}: {pct}")
        elif pct_re.search(text):
            # Category / sub-total line, e.g. "Equity Shares 97.69%"
            clean = text.lstrip("•· ")
            holdings.append(f"  {clean}")

    if not holdings:
        return []

    prefix = f"Fund: {fund_name}\n" if fund_name else ""
    page_content = f"{prefix}{header_line}\n\n" + "\n".join(holdings)

    doc = Document(
        page_content=page_content,
        metadata={
            "source": str(pdf_path),
            "source_file": pdf_path.name,
            "page_num": page_num,
            "modality": "text",
            "table_type": "portfolio",
            "fund_name": fund_name,
        },
    )
    logger.debug(
        f"Page {page_num}: Extracted portfolio section "
        f"({len(holdings)} holdings) for fund '{fund_name}'"
    )
    return [doc]

# ──────────────────────────────────────────────────────────────────────
#  2. Chart / drawing-cluster detection
# ──────────────────────────────────────────────────────────────────────


def _filter_chart_regions_by_tables(
    chart_regions: List[ChartRegion],
    table_bboxes: List[fitz.Rect],
    overlap_threshold: float = 0.5,
) -> List[ChartRegion]:
    """
    Remove chart regions that are predominantly made of table borders.

    A chart region is dropped if EITHER:
    - More than ``overlap_threshold`` of the chart's OWN area is covered by a table, OR
    - More than ``overlap_threshold`` of any table's area falls inside the chart region
      (i.e., the chart bbox encloses a table — its vector drawings are the table borders).
    """
    if not table_bboxes:
        return chart_regions

    filtered: List[ChartRegion] = []
    for cr in chart_regions:
        cr_area = cr.bbox.width * cr.bbox.height
        if cr_area <= 0:
            continue
        should_drop = False
        for tb in table_bboxes:
            tb_area = tb.width * tb.height
            inter = cr.bbox & tb
            if inter.is_empty:
                continue
            inter_area = inter.width * inter.height
            if inter_area / cr_area > overlap_threshold:
                should_drop = True
                break
            if tb_area > 0 and inter_area / tb_area > overlap_threshold:
                should_drop = True
                break
        if not should_drop:
            filtered.append(cr)
        else:
            logger.debug(f"Dropping chart region bbox={cr.bbox} (page {cr.page_num}): table overlap")
    return filtered




def _extract_scheme_details(
    page: fitz.Page,
    page_num: int,
    pdf_path: Path,
    cfg: PDFConfig,
    fund_name: str,
) -> List[Document]:

    """
    Parse the Scheme Details key-value section between the Returns table and Portfolio.
    Detects the section by the header \"Scheme Details\".
    """
    raw = page.get_text("dict")
    lines_with_pos: List[Tuple[float, float, str]] = []
    for block in raw.get("blocks", []):
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            parts = [sp["text"] for sp in line["spans"]]
            text = " ".join(parts).strip()
            if text:
                lines_with_pos.append((line["bbox"][1], line["bbox"][0], text))

    lines_with_pos.sort(key=lambda t: (t[0], t[1]))

    # Find "Scheme Details" section header
    start_idx = None
    for i, (_, _, text) in enumerate(lines_with_pos):
        if re.search(r"scheme\s+details", text, re.IGNORECASE):
            start_idx = i
            break

    if start_idx is None:
        return []

    kv_lines: List[str] = []
    for _, _, text in lines_with_pos[start_idx + 1:]:
        # Stop at next major section
        if re.search(r"(portfolio|quantitative indicator|riskometer)", text, re.IGNORECASE):
            break
        kv_lines.append(text)

    if not kv_lines:
        return []

    prefix = f"Fund: {fund_name}\nScheme Details\n\n" if fund_name else "Scheme Details\n\n"
    page_content = prefix + "\n".join(kv_lines)

    doc = Document(
        page_content=page_content,
        metadata={
            "source": str(pdf_path),
            "source_file": pdf_path.name,
            "page_num": page_num,
            "modality": "text",
            "table_type": "scheme_details",
            "fund_name": fund_name,
        },
    )
    return [doc]


def _extract_quantitative_indicators(
    page: fitz.Page,
    page_num: int,
    pdf_path: Path,
    cfg: PDFConfig,
    fund_name: str,
) -> List[Document]:
    """
    Extract Quantitative Indicators using targeted field-name search.

    Instead of capturing all text between two headers (which bleeds in portfolio
    data on two-column pages), we scan the entire page for the 5 known indicator
    field names and capture only their values.
    """
    # Known indicator labels and their regex patterns (value follows the label on same/next line)
    INDICATOR_PATTERNS = [
        ("Average Dividend Yield",     re.compile(r"Average\s+Dividend\s+Yield\s*[:\-]?\s*([\d.]+)", re.IGNORECASE)),
        ("Portfolio Turnover Ratio",   re.compile(r"Portfolio\s+Turnover\s+Ratio\s*[:\-]?\s*([\d.]+\s*(?:times?|x)?)", re.IGNORECASE)),
        ("Standard Deviation",         re.compile(r"Std(?:andard)?\s+Dev(?:iation)?\s*[:\-]?\s*([\d.]+%?)", re.IGNORECASE)),
        ("Sharpe Ratio",               re.compile(r"Sharpe\s+Ratio\s*[:\-]?\s*(-?[\d.]+)", re.IGNORECASE)),
        ("Portfolio Beta",             re.compile(r"Portfolio\s+Beta\s*[:\-]?\s*([\d.]+)", re.IGNORECASE)),
    ]

    # Get full page text (single string) — easiest to search across line boundaries
    page_text = page.get_text("text")

    # Verify the section exists on this page at all
    if not re.search(r"quantitative\s+indicator", page_text, re.IGNORECASE):
        return []

    found: List[str] = []
    for label, pattern in INDICATOR_PATTERNS:
        m = pattern.search(page_text)
        if m:
            found.append(f"{label}: {m.group(1).strip()}")

    if not found:
        return []

    prefix = f"Fund: {fund_name}\nQuantitative Indicators\n\n" if fund_name else "Quantitative Indicators\n\n"
    page_content = prefix + "\n".join(found)

    doc = Document(
        page_content=page_content,
        metadata={
            "source": str(pdf_path),
            "source_file": pdf_path.name,
            "page_num": page_num,
            "modality": "text",
            "table_type": "quant_indicators",
            "fund_name": fund_name,
        },
    )
    logger.debug(f"Page {page_num}: Extracted {len(found)} quantitative indicators for fund '{fund_name}'")
    return [doc]



# ──────────────────────────────────────────────────────────────────────
#  2. Chart / drawing-cluster detection
# ──────────────────────────────────────────────────────────────────────


def _filter_chart_regions_by_tables(
    chart_regions: List[ChartRegion],
    table_bboxes: List[fitz.Rect],
    overlap_threshold: float = 0.5,
) -> List[ChartRegion]:
    """
    Remove chart regions that are predominantly made of table borders.

    A chart region is dropped if EITHER:
    - More than ``overlap_threshold`` of the chart's OWN area is covered by a table, OR
    - More than ``overlap_threshold`` of any table's area is covered by the chart region
      (i.e., the chart fully encloses a table — it is the table's border drawing).
    """
    if not table_bboxes:
        return chart_regions

    filtered: List[ChartRegion] = []
    for cr in chart_regions:
        cr_area = cr.bbox.width * cr.bbox.height
        if cr_area <= 0:
            continue
        should_drop = False
        for tb in table_bboxes:
            tb_area = tb.width * tb.height
            inter = cr.bbox & tb
            if inter.is_empty:
                continue
            inter_area = inter.width * inter.height
            # Drop if most of the chart is inside a table (chart is a table decoration)
            if inter_area / cr_area > overlap_threshold:
                should_drop = True
                break
            # Drop if the chart contains most of a table (chart bbox wraps a table)
            if tb_area > 0 and inter_area / tb_area > overlap_threshold:
                should_drop = True
                break
        if not should_drop:
            filtered.append(cr)
        else:
            logger.debug(
                f"Dropping chart region bbox={cr.bbox} (page {cr.page_num}): "
                f"overlaps with table region"
            )
    return filtered


def _detect_chart_regions(
    page: fitz.Page,
    page_num: int,
    cfg: PDFConfig,
) -> List[ChartRegion]:
    """
    Cluster vector-drawing paths and small text blocks by spatial proximity.
    Filters out background/decorative drawings and page-spanning divider lines,
    then aggregates remaining drawings and small text labels into chart bounding boxes.
    """
    drawings = page.get_drawings()
    page_w = page.rect.width
    page_h = page.rect.height

    candidates: List[fitz.Rect] = []

    # 1. Collect filtered drawing rects
    for d in drawings:
        r = fitz.Rect(d["rect"])
        if r.is_empty or r.is_infinite:
            continue
        
        # Filter out header and footer
        if r.y1 < cfg.header_limit_pt or r.y0 > cfg.footer_limit_pt:
            continue
            
        # Filter out thin page-spanning horizontal separator lines
        if r.width > page_w * 0.6 and r.height < 5.0:
            continue
            
        # Filter out thin page-spanning vertical separator lines
        if r.height > page_h * 0.6 and r.width < 5.0:
            continue
            
        # Filter out excluded bboxes
        if _is_excluded(r, cfg):
            continue

        candidates.append(r)

    # 2. Collect small text blocks (font size < CHART_TEXT_MAX_PT)
    raw = page.get_text("dict")
    for b in raw.get("blocks", []):
        if b["type"] != 0:
            continue
            
        max_size = 0.0
        text_parts = []
        for line in b["lines"]:
            for span in line["spans"]:
                max_size = max(max_size, span["size"])
                text_parts.append(span["text"])
        text = " ".join(text_parts).strip()
        if not text:
            continue
            
        r = fitz.Rect(b["bbox"])
        # Filter out header and footer
        if r.y1 < cfg.header_limit_pt or r.y0 > cfg.footer_limit_pt:
            continue
            
        # Filter out excluded bboxes
        if _is_excluded(r, cfg):
            continue

        if max_size < cfg.chart_text_max_pt:
            candidates.append(r)

    if not candidates:
        return []

    # Greedy spatial clustering
    clusters = _cluster_rects(candidates, gap=cfg.cluster_gap_pt)

    regions: List[ChartRegion] = []
    for cluster_indices in clusters:
        merged = fitz.Rect()
        for ci in cluster_indices:
            merged |= candidates[ci]  # union

        # Clamp to page bounds
        merged = merged & page.rect

        area = merged.width * merged.height
        if area < cfg.min_chart_area or len(cluster_indices) < cfg.min_chart_elements:
            continue

        regions.append(ChartRegion(
            bbox=merged,
            page_num=page_num,
            num_paths=len(cluster_indices),
            region_type="chart",
        ))

    logger.debug(
        f"Page {page_num}: {len(candidates)} candidates → "
        f"{len(regions)} chart region(s)"
    )
    return regions


def _cluster_rects(
    rects: List[fitz.Rect],
    gap: float,
) -> List[List[int]]:
    """
    Union-find style greedy clustering. Two rects belong to the same
    cluster if their expanded versions (inflated by ``gap`` on each side)
    intersect.
    """
    n = len(rects)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build inflated rects for intersection tests
    inflated = []
    for r in rects:
        inflated.append(fitz.Rect(
            r.x0 - gap, r.y0 - gap,
            r.x1 + gap, r.y1 + gap,
        ))

    # Pairwise intersection
    for i in range(n):
        for j in range(i + 1, n):
            if not (inflated[i] & inflated[j]).is_empty:
                union(i, j)

    # Group by root
    from collections import defaultdict
    groups: dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ──────────────────────────────────────────────────────────────────────
#  2. Text-block classification
# ──────────────────────────────────────────────────────────────────────

def _classify_text_blocks(
    page: fitz.Page,
    page_num: int,
    *,
    chart_bboxes: List[fitz.Rect],
    cfg: PDFConfig,
) -> List[TextBlock]:
    """
    Walk every text block on the page, splitting it into sub-blocks if font size/role changes.
    - Classify by font-size → role.
    - Exclude blocks whose centre falls inside a chart bounding box
      AND whose font size is smaller than SUB_HEADER_MIN_PT (to keep section headers
      that overlap with title background boxes).
    """
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks: List[TextBlock] = []

    for block in raw.get("blocks", []):
        if block["type"] != 0:  # skip image blocks in dict output
            continue

        # Split block into sub-blocks by font-size role
        sub_blocks = []
        current_role = None
        current_lines = []
        current_max_size = 0.0
        current_bbox = None
        
        for line in block["lines"]:
            line_max_size = 0.0
            line_parts = []
            for span in line["spans"]:
                line_max_size = max(line_max_size, span["size"])
                line_parts.append(span["text"])
            
            line_text = " ".join(line_parts).strip()
            if not line_text:
                continue
                
            # Classify line role
            if line_max_size >= cfg.page_title_min_pt:
                line_role = "page_title"
            elif line_max_size >= cfg.section_header_min_pt:
                line_role = "section_header"
            elif line_max_size >= cfg.sub_header_min_pt:
                line_role = "sub_header"
            else:
                line_role = "body"
                
            if current_role is None:
                current_role = line_role
                current_lines = [line_text]
                current_max_size = line_max_size
                current_bbox = list(line["bbox"])
            elif line_role == current_role:
                current_lines.append(line_text)
                current_max_size = max(current_max_size, line_max_size)
                l_box = line["bbox"]
                current_bbox[0] = min(current_bbox[0], l_box[0])
                current_bbox[1] = min(current_bbox[1], l_box[1])
                current_bbox[2] = max(current_bbox[2], l_box[2])
                current_bbox[3] = max(current_bbox[3], l_box[3])
            else:
                # Flush previous sub-block
                sub_blocks.append((current_role, " ".join(current_lines), current_max_size, tuple(current_bbox)))
                current_role = line_role
                current_lines = [line_text]
                current_max_size = line_max_size
                current_bbox = list(line["bbox"])
                
        if current_role is not None:
            sub_blocks.append((current_role, " ".join(current_lines), current_max_size, tuple(current_bbox)))

        # Process classified sub-blocks
        for role, text, max_size, bbox in sub_blocks:
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            # ── Skip if inside a chart region AND it's a small label ──
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            is_inside_chart = any(_point_in_rect(cx, cy, cr) for cr in chart_bboxes)
            if is_inside_chart and max_size < cfg.sub_header_min_pt:
                continue

            # ── Skip if inside EXCLUDE_BBOXES ──
            if _is_excluded(fitz.Rect(bbox), cfg):
                continue

            blocks.append(TextBlock(
                text=text,
                bbox=bbox,
                page_num=page_num,
                max_font_size=max_size,
                role=role,
            ))

    return blocks


def _point_in_rect(x: float, y: float, rect: fitz.Rect) -> bool:
    """Check whether point (x, y) falls inside a fitz.Rect."""
    return rect.x0 <= x <= rect.x1 and rect.y0 <= y <= rect.y1


def _is_excluded(rect: fitz.Rect, cfg: PDFConfig) -> bool:
    """Check if the rect center falls inside any of the EXCLUDE_BBOXES."""
    if not cfg.exclude_bboxes:
        return False
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    for eb in cfg.exclude_bboxes:
        if isinstance(eb, (tuple, list)):
            x0, y0, x1, y1 = eb
        else:
            x0, y0, x1, y1 = eb.x0, eb.y0, eb.x1, eb.y1
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
#  3. Section-aware chunk builder
# ──────────────────────────────────────────────────────────────────────

def _build_section_chunks(
    blocks: List[TextBlock],
    pdf_path: Path,
    cfg: PDFConfig,
) -> List[Document]:
    """
    Walk text blocks in document order. Every time a section_header or
    sub_header appears, flush the accumulated body text as a chunk, then
    start a new one. Page titles are recorded but don't create chunks
    themselves.
    """
    chunks: List[Document] = []
    current_page_title = ""
    current_section = ""
    current_body_lines: List[str] = []
    current_pages: set[int] = set()

    def _flush() -> None:
        """Flush accumulated body text into one or more Documents."""
        nonlocal current_body_lines, current_pages
        if not current_body_lines:
            return

        body = "\n".join(current_body_lines).strip()
        if not body:
            current_body_lines = []
            current_pages = set()
            return

        # Skip chunks whose body is column-header boilerplate with no payload data.
        # Two cases:
        #  (a) Completely digit-free (pure labels, no values).
        #  (b) Matches portfolio column-header pattern: a date line + label line only.
        _no_digits = not re.search(r"\d", body)
        _col_header_only = bool(re.fullmatch(
            r"Portfolio as on .+?\n+Company/Issuer.+?",
            body.strip(), re.DOTALL | re.IGNORECASE,
        ))
        if _no_digits or _col_header_only:
            current_body_lines = []
            current_pages = set()
            return

        page_list = sorted(current_pages)
        meta_base = {
            "page_title": current_page_title,
            "section_title": current_section,
            "page_num": page_list[0] if len(page_list) == 1 else page_list,
            "source": str(pdf_path),
            "source_file": pdf_path.name,
            "modality": "text",
        }

        # Approximate token count (~4 chars/token)
        approx_tokens = len(body) / 4
        if approx_tokens <= cfg.max_chunk_tokens:
            doc = Document(page_content=body, metadata={**meta_base})
            chunks.append(doc)
        else:
            # Fallback split within the section
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.max_chunk_tokens * 4,  # chars
                chunk_overlap=cfg.fallback_chunk_overlap * 4,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            sub_docs = splitter.create_documents(
                [body],
                metadatas=[meta_base],
            )
            for sd in sub_docs:
                sd.metadata["split_within_section"] = True
            chunks.extend(sub_docs)
            logger.info(
                f"Section '{current_section}' exceeded {cfg.max_chunk_tokens} "
                f"tokens → split into {len(sub_docs)} sub-chunks"
            )

        current_body_lines = []
        current_pages = set()

    for block in blocks:
        if block.role == "page_title":
            current_page_title = block.text
            continue

        if block.role in ("section_header", "sub_header"):
            _flush()
            current_section = block.text
            continue

        # body text — skip bare page-number lines (e.g. "15", "20")
        stripped = block.text.strip()
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        current_body_lines.append(block.text)
        current_pages.add(block.page_num)

    # Final flush
    _flush()

    # Assign sequential chunk_id
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    logger.info(f"Produced {len(chunks)} section-aware text chunks.")
    return chunks


# ──────────────────────────────────────────────────────────────────────
#  4. Chart-region rendering
# ──────────────────────────────────────────────────────────────────────

def _render_region_to_document(
    page: fitz.Page,
    region: ChartRegion,
    index: int,
    pdf_path: Path,
    cfg: PDFConfig,
    fund_name: str = "",
) -> Document | None:
    """
    Convert a detected vector-drawing region into a Document.

    Strategy (text-first):
    1. Try to extract selectable text from the region's bounding box.
       If the region contains substantial text (> 60 chars after stripping),
       it is a *text-based visual element* (e.g. "Top 5 Holdings" summary box)
       and we return a text Document so it lands in the text embedding pathway.
    2. If text extraction is poor, fall back to rasterising the region as a
       base64 PNG (genuine charts with axis labels, bar plots, etc.).

    Returns None if the clip rect is degenerate.
    """
    clip = region.bbox & page.rect  # clamp to page
    if clip.is_empty:
        return None

    # ── Step 1: attempt text extraction from the region ──
    clipped_text = page.get_text("text", clip=clip).strip()
    # Collapse excessive whitespace for a fair length check
    clipped_text_clean = re.sub(r"\s+", " ", clipped_text)
    if len(clipped_text_clean) > 60:
        # Rich text region (summary box, holdings table, etc.) — store as text
        prefix = f"Fund: {fund_name}\n" if fund_name else ""
        return Document(
            page_content=prefix + clipped_text,
            metadata={
                "page_num": region.page_num,
                "source": str(pdf_path),
                "source_file": pdf_path.name,
                "modality": "text",
                "table_type": "summary_box",
                "fund_name": fund_name,
                "bbox": [round(region.bbox.x0, 1), round(region.bbox.y0, 1),
                         round(region.bbox.x1, 1), round(region.bbox.y1, 1)],
                "region_type": "summary_box",
            },
        )

    # ── Step 2: fall back to rasterising as PNG (genuine visual chart) ──
    # Add small padding so chart borders aren't cut off
    padding = 5
    clip = fitz.Rect(
        max(clip.x0 - padding, page.rect.x0),
        max(clip.y0 - padding, page.rect.y0),
        min(clip.x1 + padding, page.rect.x1),
        min(clip.y1 + padding, page.rect.y1),
    )

    pix = page.get_pixmap(clip=clip, dpi=cfg.chart_render_dpi)
    raw_bytes = pix.tobytes("png")
    b64 = base64.b64encode(raw_bytes).decode("utf-8")

    return Document(
        page_content="",
        metadata={
            "base64_image": b64,
            "mime_type": "image/png",
            "page_num": region.page_num,
            "image_index": index,
            "bbox": [round(region.bbox.x0, 1), round(region.bbox.y0, 1),
                     round(region.bbox.x1, 1), round(region.bbox.y1, 1)],
            "region_type": region.region_type,
            "num_paths": region.num_paths,
            "source": str(pdf_path),
            "modality": "image",
        },
    )



# ──────────────────────────────────────────────────────────────────────
#  5. Embedded raster-image extraction
# ──────────────────────────────────────────────────────────────────────

def _extract_embedded_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    pdf_path: Path,
    *,
    seen_xrefs: set[int],
    chart_bboxes: List[fitz.Rect],
    cfg: PDFConfig,
) -> List[Document]:
    """
    Extract embedded raster images from the page, skipping:
    - images below ``min_image_area``
    - duplicate xrefs (e.g. repeated logo/banner)
    - images whose on-page rect falls inside an already-detected chart
      region (to avoid double-counting)
    - images in header/footer regions, or excluded bounding boxes
    """
    image_docs: List[Document] = []
    img_list = page.get_images(full=True)

    for img_index, img_info in enumerate(img_list):
        xref = img_info[0]

        if cfg.skip_duplicate_images and xref in seen_xrefs:
            continue

        try:
            base_image = doc.extract_image(xref)
        except Exception as e:
            logger.warning(
                f"Failed to extract image xref={xref} on page {page_num}: {e}"
            )
            continue

        w, h = base_image["width"], base_image["height"]
        if w * h < cfg.min_image_area:
            continue

        img_rects = page.get_image_rects(img_info)
        
        # Check exclusion criteria on the placement rect
        if img_rects:
            r = img_rects[0]
            # Filter out header and footer
            if r.y1 < cfg.header_limit_pt or r.y0 > cfg.footer_limit_pt:
                continue
            
            # Filter out excluded bboxes
            if _is_excluded(r, cfg):
                continue
                
            # Filter out images inside chart regions
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if any(_point_in_rect(cx, cy, cr) for cr in chart_bboxes):
                continue
        else:
            # If we don't know where the image is on the page, skip it to be safe
            continue

        # Mark as seen
        seen_xrefs.add(xref)

        raw_bytes = base_image["image"]
        ext = base_image.get("ext", "png").lower()
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        b64 = base64.b64encode(raw_bytes).decode("utf-8")

        r = img_rects[0]
        bbox_list = [round(r.x0, 1), round(r.y0, 1),
                     round(r.x1, 1), round(r.y1, 1)]

        image_docs.append(Document(
            page_content="",
            metadata={
                "base64_image": b64,
                "mime_type": mime,
                "page_num": page_num,
                "image_index": img_index,
                "bbox": bbox_list,
                "region_type": "image",
                "source": str(pdf_path),
                "modality": "image",
            },
        ))

    return image_docs