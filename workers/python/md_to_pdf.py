# -*- coding: utf-8 -*-
"""Convert the report markdown files to PDF (landscape, small font for wide
tables) using markdown -> HTML -> xhtml2pdf (pure-Python, no system deps)."""
import sys, os
import markdown
from xhtml2pdf import pisa

OUT = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'

REPORTS = [
    'report_comparacao_global.md',
    'report_este_trabalho.md',
    'appendix_split_comparison.md',
    'literature_comparison_certified.md',
    'literature_master_census.md',
    'literature_gaps_extracted.md',
    'methods.md',
]

CSS = """
@page { size: A4 landscape; margin: 1.2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 7pt; color: #111; }
h1 { font-size: 13pt; color: #1F4E4A; border-bottom: 1px solid #1F4E4A; }
h2 { font-size: 10pt; color: #1F4E4A; margin-top: 10px; }
h3 { font-size: 8.5pt; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; }
th, td { border: 0.5pt solid #999; padding: 2px 3px; font-size: 5.6pt;
         vertical-align: top; }
th { background: #E8F0EF; font-weight: bold; }
code { font-family: Courier, monospace; font-size: 6pt; background: #f2f2f2; }
blockquote { color: #444; font-size: 6.5pt; border-left: 2pt solid #ccc;
             padding-left: 6px; }
"""

def convert(md_name):
    src = f'{OUT}/{md_name}'
    if not os.path.exists(src):
        print(f'  [skip] {md_name} (not found)'); return False
    with open(src, encoding='utf-8') as f:
        text = f.read()
    html_body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])
    html = f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{html_body}</body></html>'
    dst = f'{OUT}/{os.path.splitext(md_name)[0]}.pdf'
    with open(dst, 'wb') as out:
        result = pisa.CreatePDF(html, dest=out, encoding='utf-8')
    ok = not result.err
    print(f'  [{"ok" if ok else "ERR"}] {md_name} -> {os.path.basename(dst)} '
          f'({os.path.getsize(dst)//1024} KB)' if ok else f'  [ERR] {md_name}')
    return ok

if __name__ == '__main__':
    print('Converting reports to PDF...')
    n = sum(convert(r) for r in REPORTS)
    print(f'Done: {n}/{len(REPORTS)} PDFs created.')
