# -*- coding: utf-8 -*-
"""Stage 1 of the literature deep-dive pipeline (grounded, verbatim).

For every local PDF in literature/:
  1. Extract full text with pypdf -> cached to literature/_txt/<name>.txt
  2. Detect the DOI in the text and match it to _all_citations.csv (author/year/title)
  3. Copy VERBATIM the paragraphs under section headings of interest
     (Limitations / Future Work / Challenges / Conclusion) into a per-paper record.

Nothing is summarised or invented here: the gaps file contains literal excerpts
from the PDFs. A human (or a later careful pass) turns these into prose.

Outputs:
  literature/_txt/*.txt                      full-text cache
  literature/_index.csv                      pdf -> doi/author/year/title + extract status
  brisc_gui/literature_gaps_extracted.md     verbatim Future-Work/Limitations/Conclusion
"""
import os, re, csv, io, sys

LIT = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/literature'
TXT = f'{LIT}/_txt'
OUT = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'
os.makedirs(TXT, exist_ok=True)

# --- census for DOI -> metadata ---
cites = list(csv.DictReader(io.open(f'{LIT}/_all_citations.csv', encoding='utf-8', errors='replace')))
def norm_doi(d): return (d or '').strip().lower().replace('https://doi.org/', '').rstrip('.')
meta_by_doi = {}
for r in cites:
    d = norm_doi(r.get('doi',''))
    if d: meta_by_doi[d] = r

DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', re.I)

# Image-type vocabulary — modality, MR sequence/weighting, acquisition plane.
# Only terms that literally appear in the text are reported (grounded, no guess).
IMG_TERMS = [
    # sequence / weighting
    ('T1-CE', re.compile(r'\b(contrast[- ]enhanced\s+T1|T1[- ]?CE|T1c\b|T1[- ]?weighted\s+contrast|CE[- ]?T1|post[- ]?contrast\s+T1)', re.I)),
    ('T1', re.compile(r'\bT1[- ]?weighted|\bT1\b|\bT1w\b', re.I)),
    ('T2', re.compile(r'\bT2[- ]?weighted|\bT2\b|\bT2w\b', re.I)),
    ('FLAIR', re.compile(r'\bFLAIR\b', re.I)),
    ('DWI/ADC', re.compile(r'\bDWI\b|diffusion[- ]weighted|\bADC\b', re.I)),
    ('SWI/PD', re.compile(r'\bSWI\b|proton[- ]density', re.I)),
    # modality
    ('MRI', re.compile(r'\bMRI\b|magnetic\s+resonance', re.I)),
    ('CT', re.compile(r'\bCT\s+(scan|image|slice)|computed\s+tomography', re.I)),
    ('PET', re.compile(r'\bPET\b|positron\s+emission', re.I)),
    ('US', re.compile(r'\bultrasound\b|\bultrasonograph', re.I)),
    # plane
    ('axial', re.compile(r'\baxial\b', re.I)),
    ('coronal', re.compile(r'\bcoronal\b', re.I)),
    ('sagittal', re.compile(r'\bsagittal\b', re.I)),
]

def detect_image_type(text):
    """Return the image-type terms that literally occur in the text."""
    t = text or ''
    return [label for label, rx in IMG_TERMS if rx.search(t)]
# headings we want to copy verbatim
SECTIONS = [
    ('future_work', re.compile(r'\b(future\s+(work|direction|scope|research)|future\s+studies)\b', re.I)),
    ('limitations', re.compile(r'\b(limitation[s]?)\b', re.I)),
    ('challenges',  re.compile(r'\b(challenge[s]?|open\s+problem[s]?|open\s+challenge[s]?)\b', re.I)),
    ('conclusion',  re.compile(r'\b(conclusion[s]?|concluding\s+remarks)\b', re.I)),
]

def extract_text(pdf_path):
    from pypdf import PdfReader
    try:
        reader = PdfReader(pdf_path)
        parts = []
        for pg in reader.pages:
            try:
                parts.append(pg.extract_text() or '')
            except Exception:
                parts.append('')
        return '\n'.join(parts)
    except Exception as e:
        return f'__EXTRACT_ERROR__ {e}'

def find_doi(text):
    for m in DOI_RE.finditer(text[:5000]):  # DOIs usually near the top
        d = norm_doi(m.group(0))
        if d in meta_by_doi:
            return d
    m = DOI_RE.search(text)
    return norm_doi(m.group(0)) if m else ''

# Title index for fallback matching: normalized first ~50 chars -> census row.
def _norm_title(t):
    return re.sub(r'[^a-z0-9 ]', '', (t or '').lower()).strip()
title_index = []
for r in cites:
    nt = _norm_title(r.get('title',''))
    if len(nt) >= 25:
        title_index.append((nt, r))

def match_by_title(text):
    """Match a PDF to a census row if a census title appears in the PDF text."""
    head = _norm_title(text[:4000])
    head = re.sub(r'\s+', ' ', head)
    for nt, r in title_index:
        probe = nt[:50]
        if probe and probe in head:
            return r
    return None

def grab_sections(text):
    """Return {label: excerpt} for the first strong occurrence of each heading.
    Excerpt = up to 1400 chars following a line that looks like a heading."""
    out = {}
    lines = text.split('\n')
    for i, line in enumerate(lines):
        s = line.strip()
        if not (3 <= len(s) <= 80):  # headings are short lines
            continue
        for label, rx in SECTIONS:
            if label in out:
                continue
            # heading-like: matches keyword and is short / titlecase / numbered
            if rx.search(s) and (s.lower().startswith(('future','limitation','challenge','conclusion'))
                                 or re.match(r'^\d+(\.\d+)*\.?\s', s) or s.isupper() or s.istitle()):
                chunk = '\n'.join(lines[i:i+40])
                chunk = re.sub(r'\s+', ' ', chunk).strip()[:1400]
                out[label] = chunk
    return out

def main():
    pdfs = sorted(f for f in os.listdir(LIT) if f.lower().endswith('.pdf'))
    index_rows = []
    gaps = io.open(f'{OUT}/literature_gaps_extracted.md', 'w', encoding='utf-8')
    gaps.write('# Verbatim excerpts: Future Work / Limitations / Challenges / Conclusion\n\n')
    gaps.write('> Copied literally from the local PDFs (pypdf). Nothing summarised. '
               'Papers with no matching heading, or image-only PDFs, are omitted here '
               'but listed in `_index.csv`.\n\n')
    n_ok = n_txt = n_gaps = 0
    for k, name in enumerate(pdfs, 1):
        cache = f'{TXT}/{name}.txt'
        if os.path.exists(cache) and os.path.getsize(cache) > 0:
            text = io.open(cache, encoding='utf-8', errors='replace').read()
        else:
            text = extract_text(f'{LIT}/{name}')
            io.open(cache, 'w', encoding='utf-8').write(text)
        err = text.startswith('__EXTRACT_ERROR__')
        nchars = 0 if err else len(text)
        if not err and nchars > 200: n_txt += 1
        doi = '' if err else find_doi(text)
        m = meta_by_doi.get(doi, {})
        if not m and not err:                       # fallback: match by title
            m = match_by_title(text) or {}
            if m and not doi:
                doi = norm_doi(m.get('doi',''))
        author = (m.get('authors','') or '').split(';')[0].strip()
        year = m.get('year',''); title = (m.get('title','') or '')[:80]
        secs = {} if (err or nchars < 200) else grab_sections(text)
        img_type = [] if (err or nchars < 200) else detect_image_type(text)
        index_rows.append({'pdf': name, 'chars': nchars, 'doi': doi,
                           'author': author, 'year': year, 'title': title,
                           'image_type': ';'.join(img_type),
                           'sections_found': ';'.join(secs.keys()),
                           'status': 'error' if err else ('image_or_empty' if nchars < 200 else 'ok')})
        if not err and nchars >= 200: n_ok += 1
        if secs:
            n_gaps += 1
            gaps.write(f'## {author or name} {("("+year+")") if year else ""}\n')
            gaps.write(f'*{title}*  \nPDF: `{name}`  DOI: {doi or "n/d"}  '
                       f'Image type: {"; ".join(img_type) or "n/d"}\n\n')
            for label in ('future_work','limitations','challenges','conclusion'):
                if label in secs:
                    gaps.write(f'**{label.replace("_"," ").title()}:** {secs[label]}\n\n')
            gaps.write('---\n\n')
        if k % 20 == 0:
            print(f'  processed {k}/{len(pdfs)}  (text-ok={n_ok}, gaps={n_gaps})', flush=True)
    gaps.close()
    with io.open(f'{LIT}/_index.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pdf','status','chars','doi','author','year','title','image_type','sections_found'])
        w.writeheader(); w.writerows(index_rows)
    print(f'DONE. pdfs={len(pdfs)} text_ok={n_ok} image_or_empty={n_txt<len(pdfs)} '
          f'papers_with_gap_sections={n_gaps}')
    print(f'  -> {LIT}/_txt/  (text cache)')
    print(f'  -> {LIT}/_index.csv')
    print(f'  -> {OUT}/literature_gaps_extracted.md')

if __name__ == '__main__':
    main()
