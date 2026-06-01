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
CHART_RENDER_DPI = 60

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
DEFAULT_PDF_PATH = BASE_DIR / "static" / "icici.pdf"


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
) -> Tuple[List[Document], List[Document]]:
    """
    Extract section-aware text chunks **and** chart/image documents from a PDF.

    All thresholds can be dynamically customized via argument overrides.
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
    )

    # Heavy I/O → run in a worker thread
    text_chunks, image_docs = await asyncio.to_thread(
        _extract_all,
        resolved_path,
        cfg,
    )
    return text_chunks, image_docs


# ══════════════════════════════════════════════════════════════════════
#  Core (sync) implementation – runs inside asyncio.to_thread
# ══════════════════════════════════════════════════════════════════════

def _extract_all(
    pdf_path: Path,
    cfg: PDFConfig,
) -> Tuple[List[Document], List[Document]]:
    """Synchronous workhorse that does all fitz I/O."""

    doc = fitz.open(str(pdf_path))
    all_text_blocks: List[TextBlock] = []
    all_chart_regions: List[ChartRegion] = []
    seen_image_xrefs: set[int] = set()
    image_docs: List[Document] = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ── 1. Detect chart regions from vector drawings ──
        chart_regions = _detect_chart_regions(page, page_num, cfg)
        all_chart_regions.extend(chart_regions)

        # ── 2. Classify text blocks, filtering out chart-interior text ──
        text_blocks = _classify_text_blocks(
            page, page_num,
            chart_bboxes=[cr.bbox for cr in chart_regions],
            cfg=cfg,
        )
        # Sort text blocks: primarily by y0 (top-to-bottom), secondarily by x0 (left-to-right)
        text_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        all_text_blocks.extend(text_blocks)

        # ── 3. Render chart regions as PNG images ──
        for idx, cr in enumerate(chart_regions):
            img_doc = _render_region_to_document(page, cr, idx, pdf_path, cfg)
            if img_doc:
                image_docs.append(img_doc)

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

    logger.info(
        f"Extraction complete: {len(text_chunks)} text chunks, "
        f"{len(image_docs)} image documents from {pdf_path.name}"
    )
    return text_chunks, image_docs


# ──────────────────────────────────────────────────────────────────────
#  1. Chart / drawing-cluster detection
# ──────────────────────────────────────────────────────────────────────

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

        # body text
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
) -> Document | None:
    """
    Rasterise a chart bounding-box to a high-res PNG and wrap in a
    Document. Returns None if the clip rect is degenerate.
    """
    clip = region.bbox & page.rect  # clamp to page
    if clip.is_empty:
        return None

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