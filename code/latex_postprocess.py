#!/usr/bin/env python3
"""Reapply the hand-fixed display-math and path-break patches to body.tex.

Run after ``code/latex_convert.py``.  The patches make four wide display
equations fit the IEEE two-column measure and add breakpoints to long
``\\texttt`` paths; the escaping safety net undoes any doubled ``\\%`` / ``\\_``
that would otherwise comment out the rest of a line or turn into a subscript.

The display-math replacements are idempotent: re-running on an already patched
body.tex skips them instead of failing.

Usage:
    python3 code/latex_convert.py
    python3 code/latex_postprocess.py [--dir paper/latex]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPS = [
(r"""\[
\log S_{t+1}=\log S_t+\kappa(\mu_{\log}-\log S_t)+\sigma\xi_t+J_t,
\qquad
J_t\sim N(0,(3\sigma)^2)\ \text{w.p. }0.005,\ \text{else }0,
\]""",
r"""\[
\begin{aligned}
\log S_{t+1}&=\log S_t+\kappa(\mu_{\log}-\log S_t)+\sigma\xi_t+J_t,\\
J_t&\sim N(0,(3\sigma)^2)\ \text{w.p. }0.005,\ \text{else }0,
\end{aligned}
\]"""),
(r"""\[
V^\pi(R_{\mathrm{eval}})=\mathbb E_{\theta,\pi}\Bigl[\sum_{t:r_t\in R_{\mathrm{eval}}}\gamma^t\rho(X_t,A_t,X_{t+1})\Bigr],
\qquad \theta=(T,h,\nu,\rho),
\]""",
r"""\[
\begin{aligned}
V^\pi(R_{\mathrm{eval}})&=\mathbb E_{\theta,\pi}\Bigl[\sum_{t:r_t\in R_{\mathrm{eval}}}\gamma^t\rho(X_t,A_t,X_{t+1})\Bigr],\\
\theta&=(T,h,\nu,\rho),
\end{aligned}
\]"""),
(r"""\[
\kappa=\sup_x|F_{\mathrm{cal}}(x)-F_{\mathrm{test}}(x)|,
\qquad
\hat\kappa=\sup_x|\hat F_{\mathrm{cal}}(x)-\hat F_{\mathrm{test}}(x)|.
\]""",
r"""\[
\kappa=\sup_x|F_{\mathrm{cal}}(x)-F_{\mathrm{test}}(x)|,
\quad
\hat\kappa=\sup_x|\hat F_{\mathrm{cal}}(x)-\hat F_{\mathrm{test}}(x)|.
\]"""),
(r"""\[
F_{\mathrm{test}}(\hat q)\ge\max\Bigl(0,\frac{k}{n}-\varepsilon_{\mathrm{cal}}-\hat\kappa-\varepsilon_\kappa\Bigr),
\quad
\varepsilon_{\mathrm{cal}}=\sqrt{\frac{\ln(2/\delta_{\mathrm{cal}})}{2n}},
\quad
\varepsilon_\kappa=\sqrt{\frac{\ln(4/\delta_\kappa)}{2n}}+\sqrt{\frac{\ln(4/\delta_\kappa)}{2m}},
\]""",
r"""\[
\begin{aligned}
F_{\mathrm{test}}(\hat q)&\ge\max\Bigl(0,\frac{k}{n}-\varepsilon_{\mathrm{cal}}-\hat\kappa-\varepsilon_\kappa\Bigr),\\
\varepsilon_{\mathrm{cal}}&=\sqrt{\frac{\ln(2/\delta_{\mathrm{cal}})}{2n}},\\
\varepsilon_\kappa&=\sqrt{\frac{\ln(4/\delta_\kappa)}{2n}}+\sqrt{\frac{\ln(4/\delta_\kappa)}{2m}},
\end{aligned}
\]"""),
]

# Long-path breakpoints are cosmetic: the patched path may legitimately be
# absent after a text revision.
OPTIONAL_REPS = [
(r'\texttt{results/semi\_synthetic\_calibration.json}',
 r'\texttt{results/\allowbreak semi\_synthetic\_calibration.json}'),
(r'\texttt{code/semi\_synthetic\_calibration.py}',
 r'\texttt{code/\allowbreak semi\_synthetic\_calibration.py}'),
]

# Author-year text in the Markdown draft is routed through natbib for IEEE.
TEXT_REPS = [
('(Shi et al., 2022; Uehara et al., 2023; Zhang and Jiang, 2024, 2025; Luo et al., 2026)', r'\citep{shi2022,uehara2023,zhang2024,zhang2025,luo2026}'),
('(Shi 2022; Uehara 2023; Zhang--Jiang 2024/25)', r'\citep{shi2022,uehara2023,zhang2024,zhang2025,luo2026}'),
('(Barber et al., 2021; Jin and Ren, 2024; Lakkaraju et al., 2017)', r'\citep{barber2021,jin2024jomi,lakkaraju2017}'),
('Barber et al. (2023)', r'\citet{barber2023}'),
('(Zhang and Jiang, 2024, 2025; Luo et al., 2026)', r'\citep{zhang2024,zhang2025,luo2026}'),
('(Barber et al., 2023)', r'\citep{barber2023}'),
('(Vovk et al., 2005; Lei et al., 2018)', r'\citep{vovk2005,lei2018}'),
('(Barber et al., 2021; Lakkaraju et al., 2017)', r'\citep{barber2021,lakkaraju2017}'),
]

# Protocol 1 is set in \small to keep the boxed reporting template economical
# in the two-column measure.
PROTOCOL_REPS = [
(r'\textbf{Protocol 1: Regime-aware OPE coverage audit.} \textbf{Input.}',
 r'\small'
 '\n'
 r'\noindent\textbf{Protocol 1: Regime-aware OPE coverage audit.}'
 '\n'
 r'\textbf{Input.}'),
(r'\textbf{Stop rules.} Stop and report',
 r'\textit{Stop rules.} Stop and report'),
('Do not repair a failed level by pooling.\n\nProtocol 1 makes',
 'Do not repair a failed level by pooling.\n\\normalsize\n\nProtocol 1 makes'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', default=str(ROOT / 'paper/latex'),
                    help='directory holding body.tex (default: paper/latex)')
    args = ap.parse_args()
    p = Path(args.dir) / 'body.tex'
    s = p.read_text()

    for a, b in REPS + PROTOCOL_REPS:
        if a in s:
            s = s.replace(a, b)
        elif b in s:
            continue  # already patched
        else:
            raise SystemExit('postprocess: expected block not found: ' + a[:60])

    for a, b in OPTIONAL_REPS:
        s = s.replace(a, b)

    for a, b in TEXT_REPS:
        s = s.replace(a, b)

    # Safety net: the Markdown source already uses LaTeX escapes.  The converter
    # must not double them (a doubled percent comments out the rest of the line;
    # a doubled underscore becomes a line break plus a subscript).
    for a, b in ((r'\\%', r'\%'), (r'\\_', r'\_'), (r'\\&', r'\&'),
                 (r'\\#', r'\#'), (r'\\$', r'\$'), (r'\\{', r'\{'),
                 (r'\\}', r'\}')):
        s = s.replace(a, b)
    p.write_text(s)
    print('postprocessed', p)


if __name__ == '__main__':
    main()
