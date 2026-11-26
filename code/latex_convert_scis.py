#!/usr/bin/env python3
"""Convert the SCIS Markdown draft to LaTeX fragments.

The TNNLS source and generated files are left untouched. Run
``code/latex_postprocess_scis.py`` afterwards.
"""
import argparse
import re
from pathlib import Path

from latex_convert import ROOT, convert_table_file, fix_notation, fix_refs, inline, parse_body
from scis_related_work import convert_related_work


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--source', default=str(ROOT / 'paper/main-scis.md'))
    ap.add_argument('--out', default=str(ROOT / 'paper/latex/scis'))
    ap.add_argument('--figure-root', default='../../figures/scis')
    ap.add_argument('--figure-width', default='0.55\\linewidth')
    args = ap.parse_args()

    out = Path(args.out)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    md = Path(args.source).read_text()
    keyword_match = re.search(r'(?m)^\*\*Keywords:\*\*\s*(.+)$', md)
    if not keyword_match:
        raise SystemExit('SCIS source is missing its **Keywords:** line')
    keywords = [item.strip() for item in keyword_match.group(1).split(',')]
    if not 3 <= len(keywords) <= 5 or any(not item for item in keywords):
        raise SystemExit('SCIS keywords must contain 3--5 nonempty items')
    main_tex = out / 'main-scis.tex'
    if main_tex.exists():
        latex_match = re.search(r'\\keywords\{([^}]*)\}', main_tex.read_text())
        if latex_match:
            latex_keywords = [item.strip() for item in latex_match.group(1).split(',')]
            if keywords != latex_keywords:
                raise SystemExit('SCIS source/LaTeX keyword mismatch')
    md = re.sub(r'(?m)^\*\*Keywords:\*\*.*\n?', '', md)
    md = re.sub(r'\s*\[AUDIT: [^\]]+\]', '', md)
    abstract, body = parse_body(md)
    (out / 'abstract.tex').write_text(
        inline(' '.join(x.strip() for x in abstract.splitlines())) + '\n')
    body_text = fix_notation(fix_refs('\n'.join(body)))
    # Inline table titles use "Table N <title>" (no trailing period) so their
    # punctuation matches the \caption-based main figures and supplement tables.
    # The expected-block counts below are kept in sync with that format; they
    # are never removed to bypass a failed conversion.
    body_text = body_text.replace(
        r'\textbf{Table~\ref{tab:4} Three independent failure axes and their remedies}',
        r'\textbf{Table 4 Three independent failure axes and their remedies}',
    )
    body_text, caption_count = re.subn(
        r'\\textbf\{(Table [1-4] [^\n]+)\}\n\n\\noindent\{\\footnotesize',
        lambda m: '\\vspace{4pt}\n\\noindent{\\footnotesize\\textbf{' + m.group(1) + '}\\par\n',
        body_text,
    )
    if caption_count != 4:
        raise SystemExit(f'SCIS expected 4 inline table captions, found {caption_count}')
    body_text, joined_count = re.subn(
        r'\\noindent\{\\footnotesize(\\textbf\{Table [1-4] .*?\\end\{tabular\})\}',
        lambda m: '\\noindent\\begin{minipage}{\\linewidth}\\footnotesize'
                  + m.group(1) + '\\end{minipage}',
        body_text,
        flags=re.S,
    )
    if joined_count != 4:
        raise SystemExit(f'SCIS expected 4 captioned inline tables, found {joined_count}')
    # Keep the run-in heading of the numbered limitations list with the first
    # item that follows it (otherwise it orphans at the bottom of the page).
    body_text, needspace_count = re.subn(
        r'(\\textbf\{2\. Estimation-side uncertainty and calibration scope\.\})',
        r'\\needspace{3\\baselineskip}\n\1',
        body_text,
    )
    if needspace_count != 1:
        raise SystemExit(f'SCIS expected 1 orphan-heading anchor, found {needspace_count}')
    # Table 3's explanatory paragraph is a table note, not body text: give it
    # an explicit Notes label and the table's own size so the hierarchy (and
    # the font-size inversion against the 7 pt table) is resolved. Numbers,
    # caveats and negative results are untouched.
    body_text, notes_count = re.subn(
        r"Here 95\\% denotes the bound's confidence level; ([^\n]+)\n\n",
        lambda m: r"\noindent{\footnotesize\textbf{Notes.} 95\% denotes the bound's confidence level; "
                  + m.group(1) + r"\par}" + "\n\n",
        body_text,
    )
    if notes_count != 1:
        raise SystemExit(f'SCIS expected 1 Table 3 note, found {notes_count}')
    # SCIS figures prefer in-place placement; reduces float-page underfull
    # vboxes.  TNNLS generator (latex_convert.py) is deliberately unchanged.
    body_text = body_text.replace(r'\begin{figure}[t]', r'\begin{figure}[!ht]')
    body_text = body_text.replace('../figures/', args.figure_root.rstrip('/') + '/')
    body_text = body_text.replace('width=\\linewidth', f'width={args.figure_width}')
    body_text = body_text.replace(r'Figure~\ref{fig:1} summarizes the setting',
                                  r'Figure~\ref{fig:0} summarizes the setting')
    body_text = body_text.replace('\\input{sec9}', '')
    (out / 'body.tex').write_text(body_text + '\n')
    for f in sorted((ROOT / 'paper/tables').glob('table*-*.md')):
        num = f.name.split('-', 1)[0].removeprefix('table')
        content = fix_notation(fix_refs(convert_table_file(f)))
        if num == '6':
            content = content.replace(r'\label{tab:6}', r'\label{tab:stress-scis}')
        if num == '7':
            content = content.replace(r'\S7.4, Table~\ref{tab:5}',
                                      r'\S7.5, Table~\ref{tab:5}')
            content = content.replace(r'\S5.2; \S\S7.1--7.3, \S7.7;',
                                      r'\S5.2; \S\S7.1--7.4;')
            content = content.replace(r'\S\S7.3--7.5, \S7.7',
                                      r'\S\S7.3--7.6')
        if num == '9':
            content = content.replace('placebo of Table 6.',
                                      'placebo of Appendix Table B10.')
        (out / 'tables' / f'table{num}.tex').write_text(content + '\n')
    # F-D2S-02 (2026-11-26): Table B3 Panel B has nine long headers; wrap
    # them so the fit-to-width table keeps its body text at the 6 pt floor.
    # Header-only change: no data, number, caption or note is touched.
    t11 = out / 'tables' / 'table11.tex'
    t11_lines = t11.read_text().splitlines()
    new_header = (
        r"Symbol & \makecell{weekend $\tau$\\median (h)} & "
        r"\makecell{weekend $\tau$\\p90 (h)} & "
        r"\makecell{holiday $\tau$\\median (h)} & "
        r"\makecell{holiday $\tau$\\p90 (h)} & "
        r"\makecell{weekend gap /\\$\sigma_h\sqrt{\tau}$} & "
        r"\makecell{holiday gap /\\$\sigma_h\sqrt{\tau}$} & "
        r"\makecell{weekend gap /\\$g(\gamma_{\mathrm{st}}=4)$} & "
        r"\makecell{holiday gap /\\$g(\gamma_{\mathrm{st}}=4)$} \\"
    )
    replaced = False
    for i, line in enumerate(t11_lines):
        if line.startswith('Symbol & weekend') and 'holiday gap' in line:
            t11_lines[i] = new_header
            replaced = True
            break
    if not replaced:
        raise SystemExit('table11 Panel B header not found for F-D2S-02 wrap')
    t11.write_text('\n'.join(t11_lines) + '\n')
    # Supplement tables should prefer in-place placement so a small table is
    # not pushed onto a sparse float-only page.  This only relaxes the float
    # specifier; no caption, number, note, or cell content is touched.
    for table in (out / 'tables').glob('table*.tex'):
        text = table.read_text()
        text = text.replace(r'\begin{table}[t]', r'\begin{table}[!ht]')
        text = text.replace(r'\begin{table*}[t]', r'\begin{table*}[!ht]')
        table.write_text(text)
    convert_related_work(ROOT / 'paper/related-work-scis.md', out / 'sec9.tex')
    print('wrote', out)


if __name__ == '__main__':
    main()
