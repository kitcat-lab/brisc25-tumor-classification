# -*- coding: utf-8 -*-
"""Convert artigo_esboco.md -> artigo_esboco.docx with headings, tables, bold,
and the 7 article figures embedded (each with its caption). Uses python-docx."""
import re, os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'
FIG = f'{BASE}/figuras_artigo'
md = open(f'{BASE}/artigo_esboco.md', encoding='utf-8').read().split('\n')

doc = Document()
doc.styles['Normal'].font.name = 'Calibri'
doc.styles['Normal'].font.size = Pt(11)

def add_inline(par, text):
    # bold **..**, code `..`
    for chunk in re.split(r'(\*\*.+?\*\*|`.+?`)', text):
        if not chunk: continue
        if chunk.startswith('**') and chunk.endswith('**'):
            r = par.add_run(chunk[2:-2]); r.bold = True
        elif chunk.startswith('`') and chunk.endswith('`'):
            r = par.add_run(chunk[1:-1]); r.font.name = 'Consolas'; r.font.size = Pt(9.5)
        else:
            par.add_run(chunk)

def flush_table(rows):
    if len(rows) < 2: return
    cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    header, body = cells[0], [r for r in cells[2:]]  # skip separator row
    t = doc.add_table(rows=1, cols=len(header)); t.style = 'Light Grid Accent 1'
    for j, h in enumerate(header):
        t.rows[0].cells[j].paragraphs[0].add_run(h).bold = True
    for r in body:
        rc = t.add_row().cells
        for j in range(len(header)):
            add_inline(rc[j].paragraphs[0], r[j] if j < len(r) else '')
    doc.add_paragraph()

i = 0; tbuf = []
while i < len(md):
    line = md[i]
    if line.strip().startswith('|'):
        tbuf.append(line); i += 1; continue
    elif tbuf:
        flush_table(tbuf); tbuf = []
    s = line.rstrip()
    if s.startswith('# '):
        h = doc.add_heading(s[2:], level=0)
    elif s.startswith('## '):
        doc.add_heading(s[3:], level=1)
    elif s.startswith('### '):
        doc.add_heading(s[4:], level=2)
    elif s.startswith('> '):
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3)
        r = p.add_run(); add_inline(p, s[2:])
        for run in p.runs: run.italic = True
    elif s.strip() == '---':
        pass
    elif s.strip():
        add_inline(doc.add_paragraph(), s)
    i += 1
if tbuf: flush_table(tbuf)

# --- embed figures ---
doc.add_page_break()
doc.add_heading('Figures', level=1)
caps = {}
if os.path.exists(f'{FIG}/figure_captions.md'):
    for para in open(f'{FIG}/figure_captions.md', encoding='utf-8').read().split('\n\n'):
        m = re.match(r'Figure (\d+)\.', para.strip())
        if m: caps[m.group(1)] = para.strip()
figs = ['fig1_workflow_schematic','fig2_grand_comparison','fig3_softmax_vs_hybrid',
        'fig4_leakage_delta','fig5_classical_factorial_split','fig6_shap','fig7_mcnemar',
        'fig8_literature_positioning']
for k, f in enumerate(figs, 1):
    png = f'{FIG}/{f}.png'
    if os.path.exists(png):
        doc.add_picture(png, width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp = doc.add_paragraph(); add_inline(cp, caps.get(str(k), f'Figure {k}.'))
        for run in cp.runs: run.font.size = Pt(9.5)
        doc.add_paragraph()

out = f'{BASE}/artigo_esboco.docx'
doc.save(out)
print('saved', out, round(os.path.getsize(out)/1024,1), 'KB')
