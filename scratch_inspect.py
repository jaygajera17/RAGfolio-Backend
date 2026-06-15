"""Verify extraction logic of load_and_chunk_pdf."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import asyncio
from app.rag.extract import load_and_chunk_pdf

async def main():
    pdf_path = "static/fund-factsheet-for-may-2026-5-50.pdf"
    text_chunks, table_docs, image_docs = await load_and_chunk_pdf(pdf_path)

    print(f"\n{'='*70}")
    print(f"  TEXT CHUNKS: {len(text_chunks)}")
    print(f"{'='*70}")
    for idx, chunk in enumerate(text_chunks):
        meta = chunk.metadata
        print(f"\nChunk {idx:2d} | Page: {meta.get('page_num')} | Section: {meta.get('section_title')}")
        print(f"  Length: {len(chunk.page_content)} chars (~{len(chunk.page_content)//4} tokens)")
        print(f"  Content: {chunk.page_content[:200]}...")

    print(f"\n{'='*70}")
    print(f"  IMAGE/CHART DOCUMENTS: {len(image_docs)}")
    print(f"{'='*70}")
    for idx, img in enumerate(image_docs):
        meta = img.metadata
        b64_len = len(meta.get('base64_image', ''))
        print(f"\nImage {idx:2d} | Page: {meta.get('page_num')} | Type: {meta.get('region_type')} | Bbox: {meta.get('bbox')}")
        print(f"  Base64 length: {b64_len} chars")

    # Save chart images to disk for visual inspection
    import base64
    from pathlib import Path
    out_dir = Path("static/extracted_charts")
    out_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(image_docs):
        meta = img.metadata
        if meta.get("region_type") == "chart":
            fname = out_dir / f"page{meta['page_num']}_chart_{idx}.png"
            raw = base64.b64decode(meta["base64_image"])
            fname.write_bytes(raw)
            print(f"  Saved: {fname}")

if __name__ == "__main__":
    asyncio.run(main())
