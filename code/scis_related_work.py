"""Convert the compact SCIS related-work Markdown to sec9.tex."""
import re
from pathlib import Path

from latex_convert import inline


def convert_related_work(source: Path, output: Path) -> None:
    def convert_text(text):
        citations = []

        def protect(match):
            citations.append(match.group(0))
            return f'@@CITE{len(citations) - 1}@@'

        text = re.sub(r'\\cite\{[^}]+\}', protect, text)
        text = inline(text)
        for i, citation in enumerate(citations):
            text = text.replace(f'@@CITE{i}@@', citation)
        return text

    lines = source.read_text().splitlines()
    out = []
    paragraph = []

    def flush():
        nonlocal paragraph
        if paragraph:
            out.append(convert_text(' '.join(paragraph)))
            out.append('')
            paragraph = []

    for line in lines:
        text = line.strip()
        if not text or text == '---':
            flush()
            continue
        if text.startswith('## '):
            flush()
            out.append(r'\section{' + convert_text(text[3:].split(' ', 1)[1]) + '}')
            out.append('')
        elif text.startswith('### '):
            flush()
            out.append(r'\subsection{' + convert_text(text[4:].split(' ', 1)[1]) + '}')
            out.append('')
        else:
            paragraph.append(text)
    flush()
    output.write_text('\n'.join(out) + '\n')
