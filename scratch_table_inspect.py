"""Deeper inspection: portfolio table structure on all pages."""
import fitz

pdf_path = "static/fund-factsheet-for-may-2026-15-20.pdf"
doc = fitz.open(pdf_path)

for page_num in range(len(doc)):
    page = doc[page_num]
    tables = page.find_tables()
    print(f"\n{'='*80}")
    print(f"PAGE {page_num} — {len(tables.tables)} table(s) detected by find_tables()")
    
    for i, t in enumerate(tables.tables):
        print(f"\n  Table {i}: {t.row_count}x{t.col_count}, bbox={[round(v,1) for v in t.bbox]}")
        # Show first 3 rows of data
        data = t.extract()
        for r_idx, row in enumerate(data[:5]):
            cleaned = [str(c)[:40] if c else '(None)' for c in row]
            print(f"    Row {r_idx}: {cleaned}")
        if len(data) > 5:
            print(f"    ... ({len(data)-5} more rows)")
    
    # Count text blocks vs table area overlap
    text_blocks = page.get_text("dict")["blocks"]
    text_count = sum(1 for b in text_blocks if b["type"] == 0)
    print(f"  Text blocks: {text_count}")
    
    # Check for the Portfolio heading
    full_text = page.get_text()
    if "Portfolio" in full_text:
        print("  ** Contains 'Portfolio' section **")
    if "Returns of" in full_text:
        print("  ** Contains 'Returns' table **")
    if "Scheme Details" in full_text:
        print("  ** Contains 'Scheme Details' section **")
    if "Quantitative" in full_text:
        print("  ** Contains 'Quantitative Indicators' **")
    if "Benchmark" in full_text:
        print("  ** Contains 'Benchmark' section **")

doc.close()
