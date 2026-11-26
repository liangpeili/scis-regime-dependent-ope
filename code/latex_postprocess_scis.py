#!/usr/bin/env python3
"""Apply shared display patches and SCIS numeric-citation conversion."""
import argparse
from pathlib import Path

from latex_postprocess import OPTIONAL_REPS, PROTOCOL_REPS, REPS, TEXT_REPS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', default=str(ROOT / 'paper/latex/scis'))
    args = ap.parse_args()
    p = Path(args.dir) / 'body.tex'
    s = p.read_text()

    for a, b in REPS + PROTOCOL_REPS:
        if a in s:
            s = s.replace(a, b)
        elif b in s:
            continue
        else:
            raise SystemExit('postprocess-scis: expected block not found: ' + a[:60])
    for a, b in OPTIONAL_REPS:
        s = s.replace(a, b)
    for a, b in TEXT_REPS:
        s = s.replace(a, b)

    s = s.replace(r'\citet{', r'\cite{').replace(r'\citep{', r'\cite{')
    for a, b in ((r'\\%', r'\%'), (r'\\_', r'\_'), (r'\\&', r'\&'),
                 (r'\\#', r'\#'), (r'\\$', r'\$'), (r'\\{', r'\{'),
                 (r'\\}', r'\}')):
        s = s.replace(a, b)
    p.write_text(s)
    print('postprocessed', p)

    sec9 = p.parent / 'sec9.tex'
    if sec9.exists():
        related = sec9.read_text()
        related = related.replace(r'\citet{', r'\cite{').replace(r'\citep{', r'\cite{')
        for a, b in ((r'\\%', r'\%'), (r'\\_', r'\_'), (r'\\&', r'\&'),
                     (r'\\#', r'\#'), (r'\\$', r'\$'), (r'\\{', r'\{'),
                     (r'\\}', r'\}')):
            related = related.replace(a, b)
        sec9.write_text(related)
        print('postprocessed', sec9)

    for appendix in p.parent.glob('appendix-*.tex'):
        text = appendix.read_text()
        text = text.replace(r'\citealp{', r'\cite{')
        text = text.replace(r'\citet{', r'\cite{').replace(r'\citep{', r'\cite{')
        appendix.write_text(text)
        print('postprocessed', appendix)

    for table in (p.parent / 'tables').glob('*.tex'):
        text = table.read_text()
        text = text.replace(r'\citealp{', r'\cite{')
        text = text.replace(r'\citet{', r'\cite{').replace(r'\citep{', r'\cite{')
        table.write_text(text)
        print('postprocessed', table)


if __name__ == '__main__':
    main()
