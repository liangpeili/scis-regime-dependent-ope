#!/usr/bin/env python3
"""Convert the P1 Markdown draft to LaTeX fragments.

Reads ``paper/main-draft.md`` and ``paper/tables/table*-*.md`` and writes
``abstract.tex``, ``body.tex`` and ``tables/tableN.tex`` into the output
directory (default: ``paper/latex``).  The generated .tex files are the
committed artifacts; this script is the reproducible source of them.

Run ``code/latex_postprocess.py`` afterwards: it applies the hand-fixed
display-math and path-break patches and the escaping safety net.

Usage:
    python3 code/latex_convert.py [--out DIR]
    python3 code/latex_postprocess.py [--dir DIR]
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPECIAL = {
    '&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_',
    '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}', '<': r'\textless{}', '>': r'\textgreater{}',
    '|': r'\textbar{}',
}
UNICODE = {
    '–': '--', '—': '---', '×': r'$\times$',
    '§': r'\S', '“': "``", '”': "''",
    'ü': r'\"u', '′': r"$'$",
    '→': r'$\to$', 'γ': r'$\gamma$', 'τ': r'$\tau$',
    '≥': r'$\ge$', 'κ̂': r'$\hat\kappa$', '−': '-',
}


def protect_math(s):
    """Replace $...$ spans with placeholders and return (text, store)."""
    store = []

    def sub(m):
        store.append(m.group(0).replace('\\\\', '\\'))
        return f'@@MATH{len(store)-1}@@'

    # Do not touch $$ display math here; caller handles those.
    return re.sub(r'(?<!\\)\$(?!\$)(?:\\.|[^$\n\\])*?(?<!\\)\$(?!\$)', sub, s), store


def protect_code(s):
    store = []

    def sub(m):
        body = m.group(1)
        body = body.replace('\\', r'\textbackslash{}')
        for ch, rep in SPECIAL.items():
            body = body.replace(ch, rep)
        store.append(r'\texttt{' + body + '}')
        return f'@@CODE{len(store)-1}@@'

    return re.sub(r'`([^`]*)`', sub, s), store


def escape_text(s):
    s = s.replace('κ̂', r'$\hat\kappa$').replace('γ̂', r'$\hat\gamma$')
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        # The Markdown source already carries LaTeX escapes such as \% and
        # \_ (the draft is written with the PDF in mind).  Preserve them
        # instead of escaping the following character a second time, which
        # would turn them into a line break + comment/subscript.
        if ch == '\\' and i + 1 < len(s) and s[i + 1] in SPECIAL:
            out.append(ch + s[i + 1])
            i += 2
            continue
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        else:
            out.append(ch)
        i += 1
    return ''.join(out)


def inline(s):
    s, maths = protect_math(s)
    s, codes = protect_code(s)
    s = escape_text(s)
    # ranges split across source lines: "0.89-- 0.93" -> "0.89--0.93"
    s = re.sub(r'(?<!-)--(?!-)\s+', '--', s)
    # bold (may span newlines)
    s = re.sub(r'\*\*(.+?)\*\*', lambda m: r'\textbf{' + m.group(1) + '}', s,
               flags=re.S)
    # restore
    for i, v in enumerate(codes):
        s = s.replace(f'@@CODE{i}@@', v)
    for i, v in enumerate(maths):
        s = s.replace(f'@@MATH{i}@@', v)
    return s


def table_block(lines, caption, label, width=r'\linewidth'):
    """Convert one markdown pipe table to a tabular."""
    rows = []
    aligns = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            for c in cells:
                if c.endswith(':') and c.startswith(':'):
                    aligns.append('c')
                elif c.endswith(':'):
                    aligns.append('r')
                else:
                    aligns.append('l')
        else:
            rows.append(cells)
    ncol = max(len(r) for r in rows)
    aligns += ['l'] * (ncol - len(aligns))
    body = []
    for i, r in enumerate(rows):
        r += [''] * (ncol - len(r))
        cells = ' & '.join(inline(c) for c in r)
        body.append(cells + r' \\')
        if i == 0:
            body.append(r'\midrule')
    if label == '7':
        aligns = [
            r'>{\raggedright\arraybackslash}p{0.21\linewidth}',
            r'>{\raggedright\arraybackslash}p{0.52\linewidth}',
            r'>{\raggedright\arraybackslash}p{0.21\linewidth}',
        ]
        return (r'\begin{tabular}{@{}' + ''.join(aligns) + r'@{}}' + '\n'
                + r'\toprule' + '\n' + '\n'.join(body) + '\n'
                + r'\bottomrule' + '\n' + r'\end{tabular}')
    return (r'\begin{adjustbox}{max width=' + width + r'}' + '\n'
            + r'\begin{tabular}{@{}' + ''.join(aligns) + r'@{}}' + '\n'
            + r'\toprule' + '\n' + '\n'.join(body) + '\n'
            + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n'
            + r'\end{adjustbox}')


def inline_table_block(lines):
    """Markdown pipe table embedded in running text.

    Cells wrap in p-columns at \\footnotesize (8 pt, matching the float
    tables).  The l-column + adjustbox variant shrinks sentence cells to
    ~3 pt in a narrow measure.  Column alignment follows the markdown
    separator row (``---:`` right-aligns).
    """
    rows = []
    aligns = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            for c in cells:
                if c.startswith(':') and c.endswith(':'):
                    aligns.append('c')
                elif c.endswith(':'):
                    aligns.append('r')
                else:
                    aligns.append('l')
            continue
        rows.append(cells)
    ncol = max(len(r) for r in rows)
    aligns += ['l'] * (ncol - len(aligns))
    body = []
    for i, r in enumerate(rows):
        r += [''] * (ncol - len(r))
        body.append(' & '.join(inline(c) for c in r) + r' \\')
        if i == 0:
            body.append(r'\midrule')
    # Inline (in-text) tabulars at the restored IEEEtran defaults
    # (\\footnotesize 8 pt, default \\tabcolsep) must fit a 252 pt column.
    # Two layout-only facts drive this (measured on IEEEtran journal):
    #   * the table starts a new paragraph, so \\parindent (10 pt) would
    #     otherwise indent a full-width table into the margin; \\noindent
    #     removes it (these are full-column width tables, not indented text);
    #   * with @{} at the outer edges and one small explicit @{\\hspace{}}
    #     gap between the p-columns, the inter-column 2\\tabcolsep is replaced,
    #     so the natural width is sum(p) + (ncol-1)*gap.  No \\tabcolsep
    #     override, no \\sloppy, no font / leading / row-density change.
    # p-columns are floored at their longest unbreakable word so a single
    # long word (e.g. EXTENDED, identification) does not stick out.
    clean = re.compile(r'\\[a-zA-Z]+|[${}~^_]')

    def char_pts(w):
        # Times at 8 pt: uppercase glyphs are wider than lowercase.
        return sum(5.4 if ch.isupper() else (4.2 if ch.isdigit() else 3.8)
                   for ch in w)

    def word_need(col):
        best = 0.0
        for c in col:
            for w in re.split(r'[^A-Za-z0-9]+', clean.sub(' ', c)):
                if w:
                    best = max(best, char_pts(w))
        return best / 252.0 + 0.008

    strips = [max(len(clean.sub('', c)) for c in col) + 1
              for col in zip(*rows)]
    total = sum(strips)
    floors = [word_need(col) for col in zip(*rows)]
    gap = r'@{\hspace{0.3em}}'
    # With \\noindent the table may use the whole \\linewidth: budget the
    # p-columns + inter-column gaps to stay under it with a small margin.
    budget = 1.0 - (ncol - 1) * (0.3 * 8 / 252) - 0.015
    fr = [0.9 * w / total for w in strips]
    # Enforce the longest-word floors, taking any deficit from the widest
    # columns (never dropping a donor below its own floor).
    for i in range(ncol):
        if fr[i] < floors[i]:
            deficit = floors[i] - fr[i]
            fr[i] = floors[i]
            for j in sorted((j for j in range(ncol) if j != i),
                            key=lambda j: -fr[j]):
                take = min(deficit, max(0.0, fr[j] - floors[j]))
                fr[j] -= take
                deficit -= take
                if deficit <= 0:
                    break
    s = sum(fr)
    if s > budget:                      # never exceed \\linewidth
        fr = [f * budget / s for f in fr]
    fr = [round(f, 3) for f in fr]
    flush = {'l': r'\raggedright', 'r': r'\raggedleft', 'c': r'\centering'}
    cols = gap.join('>{' + flush[a] + r'\arraybackslash}p{'
                    + f'{f:.3f}' + r'\linewidth}' for a, f in zip(aligns, fr))
    return (r'\noindent{\footnotesize' + '\n'
            + r'\begin{tabular}{@{}' + cols + r'@{}}' + '\n'
            + r'\toprule' + '\n' + '\n'.join(body) + '\n'
            + r'\bottomrule' + '\n'
            + r'\end{tabular}}')


def convert_table_file(path):
    text = path.read_text()
    m = re.match(r'# Table (\d+) — (.+)', text)
    num, title = m.group(1), inline(m.group(2).strip())
    rest = text[m.end():]
    blocks = re.split(r'\n## ', rest)
    wide = num in {'4', '5', '6', '8', '9', '10', '11'}
    env = r'\begin{table*}[!htbp]' if wide else r'\begin{table}[t]'
    # Only table6 is a main-text (IEEEtran, two-column) table; every other
    # generated table lives in the supplement, which is typeset with the
    # neurips_2025 / article class.  The "restore IEEEtran defaults" step must
    # not leak its \footnotesize into the supplement: in the accepted 11-06
    # baseline every supplement table was \scriptsize, and the blanket
    # enlargement is exactly what made the 5-panel B2 float (table9) exceed
    # the double-column float height and collide with the p32 folio.  Setting
    # \scriptsize here is a *restore* of the accepted supplement typography,
    # not a new shrink: the 15 pp main budget is met by relocating two wide
    # tables, not by fonts, and table6 keeps \footnotesize.  No number,
    # negative result, boundary or note is changed.
    fsize = r'\footnotesize' if num == '6' else r'\scriptsize'
    out = [env, r'\centering', fsize]
    # Labels retain the source-file mapping while IEEEtran supplies Roman table
    # numbers in the main text and the appendix supplies its own B-prefixed ones.
    label = {'12': '7', '7': '8', '8': 'b1', '9': 'b2', '11': 'b3', '10': 'b4'}.get(num, num)
    out += [r'\caption{' + title + '}', r'\label{tab:' + label + '}']
    first_block = next((block.strip() for block in blocks if block.strip()), '')
    if first_block.startswith('Panel'):
        out.append(r'\vspace{0.5em}')
    panel_spacing = num in {'4', '5'}
    panel_seen = False
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith('Panel'):
            head, _, body = block.partition('\n')
            if panel_spacing and panel_seen:
                out.append(r'\vspace{0.4em}')
            out.append(r'\par\textbf{' + inline(head.strip()) + '}')
            if panel_spacing:
                out.append(r'\vspace{0.1em}')
            out.append('')
            panel_seen = True
        else:
            body = block
        # Split into pipe-table runs and prose runs.
        chunks = []
        cur = []
        kind = None
        for line in body.splitlines():
            is_pipe = line.strip().startswith('|')
            if is_pipe:
                if kind != 'pipe' and cur:
                    chunks.append(('prose', cur)); cur = []
                kind = 'pipe'; cur.append(line)
            else:
                if kind == 'pipe' and cur:
                    chunks.append(('pipe', cur)); cur = []
                kind = 'prose'; cur.append(line)
        if cur:
            chunks.append((kind, cur))
        for kind, lines in chunks:
            if kind == 'pipe':
                out.append(table_block(lines, title, num))
                out.append(r'\vspace{0.5em}')
            else:
                para = ' '.join(l.strip() for l in lines if l.strip())
                if not para:
                    continue
                # All prose (Notes and panel explanations) is boxed in a
                # raggedright minipage at the float's font (fsize): without the
                # leading \par the text joins the adjustbox line and wraps
                # around the tabular, and notes used to outsize the table body.
                # The minipage is what keeps each block flush-left under
                # \centering.
                out.append(r'\par\noindent\begin{minipage}{\linewidth}\raggedright{' + fsize + ' ' + inline(para) + r'}\end{minipage}\par')
                out.append('')
    out += [r'\end{table*}' if wide else r'\end{table}']
    return '\n'.join(out)


def parse_body(md):
    body = md.split('## Abstract', 1)[1].split('## Open items', 1)[0]
    body = re.sub(r'\s*\[AUDIT: A\d+\]', '', body)
    # Abstract
    abstract = body.split('---', 1)[0].strip()
    rest = body.split('---', 1)[1]
    lines = rest.splitlines()
    out = []
    i = 0
    n = len(lines)

    def flush_para(buf):
        if not buf:
            return
        para = ' '.join(x.strip() for x in buf).strip()
        if para:
            out.append(inline(para))
            out.append('')

    while i < n:
        line = lines[i]
        s = line.strip()
        if not s or s == '---':
            i += 1
            continue
        if s.startswith('## '):
            if re.match(r'^##\s+9\b', s):
                out.append(r'\input{sec9}')
                out.append('')
                i += 1
                while i < n and not re.match(r'^##\s+10\b', lines[i].strip()):
                    i += 1
                continue
            title = s[3:].strip()
            title = re.sub(r'^\d+\s+', '', title)
            sec_label = {'Conclusion': 'sec:conclusion', 'Identification': 'sec:identification', 'Results': 'sec:results'}.get(title, '')
            out.append(r'\section{' + inline(title) + '}' + (r'\label{' + sec_label + '}' if sec_label else ''))
            out.append('')
            i += 1
            continue
        if s.startswith('### '):
            title = s[4:].strip()
            title = re.sub(r'^\d+(\.\d+)?\s+', '', title)
            out.append(r'\subsection{' + inline(title) + '}')
            out.append('')
            i += 1
            continue
        if s.startswith('[Figure'):
            buf = [line]
            while ']' not in '\n'.join(buf) or buf[-1].strip().endswith('about here:'):
                i += 1
                if i >= n:
                    break
                buf.append(lines[i])
            block = ' '.join(x.strip() for x in buf)
            m = re.match(r'\[Figure (\d+) about here:\s*`([^`]+)`\.\s*Caption:\s*(.*)\]$',
                         block, re.S)
            if not m:
                raise SystemExit('bad figure block: ' + block[:200])
            fnum, fpath, cap = m.groups()
            fname = Path(fpath).name
            out += [r'\begin{figure}[t]', r'\centering',
                    r'\includegraphics[width=\linewidth]{../figures/' + fname + '}',
                    r'\caption{' + inline(cap.strip()) + '}',
                    r'\label{fig:' + fnum + '}', r'\end{figure}', '']
            i += 1
            continue
        if s.startswith('[Table'):
            m = re.match(r'\[Table (\d+)(?: \(display Table \d+\))? about here:\s*`([^`]+)`\.\]', s)
            if not m:
                raise SystemExit('bad table block: ' + s)
            tnum = m.group(1)
            out.append(r'\input{tables/table' + tnum + '}')
            out.append('')
            i += 1
            continue
        if s.startswith('$$'):
            buf = [line]
            if s == '$$':
                i += 1
                while i < n and lines[i].strip() != '$$':
                    buf.append(lines[i]); i += 1
                buf.append(lines[i] if i < n else '$$')
            block = '\n'.join(buf)
            block = block.strip()
            block = block[2:-2] if block.startswith('$$') and block.endswith('$$') else block
            out += [r'\[', block.strip(), r'\]', '']
            i += 1
            continue
        if s.startswith('|'):
            buf = []
            while i < n and lines[i].strip().startswith('|'):
                buf.append(lines[i]); i += 1
            out.append(inline_table_block(buf))
            out.append('')
            continue
        if re.match(r'^[-*]\s+', s):
            out.append(r'\begin{itemize}')
            while i < n and re.match(r'^\s*[-*]\s+', lines[i].strip()):
                item = re.sub(r'^\s*[-*]\s+', '', lines[i])
                j = i + 1
                while j < n and lines[j].strip() and not re.match(r'^\s*[-*]\s+', lines[j]) and not lines[j].strip().startswith(('##', '|', '[', '$$')):
                    item += ' ' + lines[j].strip(); j += 1
                out.append(r'\item ' + inline(item))
                i = j
            out.append(r'\end{itemize}')
            out.append('')
            continue
        if re.match(r'^\d+\.\s+', s):
            out.append(r'\begin{enumerate}')
            while i < n:
                while i < n and not lines[i].strip():
                    i += 1
                if i >= n or not re.match(r'^\s*\d+\.\s+', lines[i].strip()):
                    break
                item = re.sub(r'^\s*\d+\.\s+', '', lines[i])
                j = i + 1
                while j < n and lines[j].strip() and not re.match(r'^\s*\d+\.\s+', lines[j]) and not lines[j].strip().startswith(('##', '|', '[', '$$')):
                    item += ' ' + lines[j].strip(); j += 1
                out.append(r'\item ' + inline(item))
                i = j
            out.append(r'\end{enumerate}')
            out.append('')
            continue
        # paragraph
        buf = [line]
        i += 1
        while i < n:
            t = lines[i].strip()
            if not t or t == '---' or t.startswith(('## ', '### ', '[Figure', '[Table', '$$', '|')) or re.match(r'^([-*]|\d+\.)\s+', t):
                break
            buf.append(lines[i]); i += 1
        flush_para(buf)

    return abstract, out


def fix_refs(text):
    """Replace draft float numbers with \\ref to the LaTeX labels."""
    reps = [
        (r'Figures 1 and 3--6', r'Figures~\\ref{fig:1} and \\ref{fig:3}--\\ref{fig:6}'),
        (r'Tables 3--7', r'Tables~\\ref{tab:3}--\\ref{tab:7}'),
        (r'Tables 3--4', r'Tables~\\ref{tab:3}--\\ref{tab:4}'),
        (r'Tables 4--8', r'Tables~\\ref{tab:4}--\\ref{tab:8}'),
        (r'Tables 8--10', r'Tables~\\ref{tab:8}--\\ref{tab:10}'),
        (r'Table 11', r'Table~\\ref{tab:11}'),
        (r'Table 10', r'Table~\\ref{tab:10}'),
        (r'Table 9', r'Table~\\ref{tab:9}'),
        (r'Table 8', r'Table~\\ref{tab:8}'),
        (r'Table 7', r'Table~\\ref{tab:7}'),
        (r'Table 5', r'Table~\\ref{tab:5}'),
        (r'Table 4', r'Table~\\ref{tab:4}'),
        (r'Figure 6', r'Figure~\\ref{fig:6}'),
        (r'Figure 5', r'Figure~\\ref{fig:5}'),
        (r'Figure 4', r'Figure~\\ref{fig:4}'),
        (r'Figure 3', r'Figure~\\ref{fig:3}'),
        (r'Figure 1', r'Figure~\\ref{fig:1}'),
    ]
    for old, new in reps:
        text = text.replace(old, new)
    return text.replace('\\\\ref{', '\\ref{')


def fix_notation(text):
    """Disambiguate the stress jump scale from the coverage rate gamma."""
    text = text.replace(r'\hat\gamma', r'\hat\gamma_{\mathrm{st}}')
    text = text.replace(r'\gamma=4', r'\gamma_{\mathrm{st}}=4')
    text = text.replace(r'\gamma=0', r'\gamma_{\mathrm{st}}=0')
    text = text.replace(r'$\gamma$-free', r'$\gamma_{\mathrm{st}}$-free')
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(ROOT / 'paper/latex'),
                    help='output directory (default: paper/latex)')
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'tables').mkdir(parents=True, exist_ok=True)

    md = (ROOT / 'paper/main-draft.md').read_text()
    abstract, body = parse_body(md)
    (out / 'abstract.tex').write_text(inline(' '.join(x.strip() for x in abstract.splitlines())) + '\n')
    (out / 'body.tex').write_text(fix_notation(fix_refs('\n'.join(body))) + '\n')
    for f in sorted((ROOT / 'paper/tables').glob('table*-*.md')):
        num = re.match(r'table(\d+)-', f.name).group(1)
        (out / 'tables' / f'table{num}.tex').write_text(fix_notation(fix_refs(convert_table_file(f))) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
