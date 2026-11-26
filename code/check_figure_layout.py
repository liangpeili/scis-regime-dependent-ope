#!/usr/bin/env python3
"""Layout checker for the P1 figure PDFs.

Type, font-size and page-count checks are not enough: a figure can pass all
of them and still have text sitting on top of curves, markers, error bars or
reference lines.  This checker walks every text span in each figure PDF and
reports

  * span-span overlaps (axis-aligned boxes, 25% of the smaller box),
  * spans whose box leaves the page (would be clipped or truncated),
  * text that overprints vector ink (curves, scatter markers, error bars,
    nominal lines) — detected by redacting all text, re-rendering the
    vector-only layer, and measuring dark ink inside each text box, and
  * ink touching the page border (pixel-level clipping),
  * the font size that text will have once the figure is included at
    ``width=\\linewidth`` (252 pt in the IEEEtran two-column measure).

Known limitation: adjacent tick labels that merely abut (no box overlap and
no vector ink between them) are not flagged; ``fig7``'s "Known"/"Fitted" pair
of that kind was fixed by rotating the ticks and must be checked by eye.

Usage:
    python3 code/check_figure_layout.py [--display-pt 252] [figures...]

Defaults to every PDF in paper/figures/.  Exit code 1 if any problem is
found, so it can gate a rebuild.

2026-11-24: added TEXT-OUTSIDE-FRAME (text must stay inside its frame) and
--self-test (positive/negative controls).  Every alarm must be closed in the
project's figure-alarm-register.md; see docs/submission-preflight-checklist.md §10.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"

# Overlap counts only when the shared area is both >25% of the smaller span
# and >2 pt^2: rotated tick labels have generous axis-aligned boxes that can
# graze each other without the glyphs touching.
AREA_MIN = 2.0
FRAC_MIN = 0.25
# Truncation tolerance in pt: pymupdf's font box is a hair wider than ink.
EDGE_TOL_PT = 1.5
# Vector-ink detection: pixels darker than this (per-channel min) count as
# ink.  GRID (#e1e0d9, min=217) stays above the threshold so grid lines do
# not trigger; data colours, markers, error bars, the muted nominal line
# (#898781) and the axis-coloured reference diagonal (#c3c2b7) do.
INK_MIN = 190
# Dark pixels inside a text box that mean a line/mark is under the text.
INK_PIX_MIN = 8
# Render resolution for the raster checks.
RASTER_DPI = 200
# A text line may overhang its frame by at most this many pt.  P1 v0.6 Figure 1
# overhung its central frame by 11 pt on each side, which this detector catches.
FRAME_TOL_PT = 1.0


def spans_of(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                t = s["text"].strip()
                if not t:
                    continue
                x0, y0, x1, y1 = s["bbox"]
                out.append([x0, y0, x1, y1, s["size"], t])
    return out


def overlap(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy


def vector_layer_ink(page, spans):
    """Render the page with all text redacted; return the ink bitmap."""
    for s in spans:
        page.add_redact_annot(pymupdf.Rect(s[0], s[1], s[2], s[3]))
    page.apply_redactions(
        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
    )
    pix = page.get_pixmap(dpi=RASTER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return (img[:, :, :3].min(axis=2) < INK_MIN), pix.width, pix.height


def border_ink(page):
    pix = page.get_pixmap(dpi=RASTER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    ink = (img[:, :, :3] < 250).any(axis=2)
    return (
        bool(ink[:2, :].any()) or bool(ink[-2:, :].any())
        or bool(ink[:, :2].any()) or bool(ink[:, -2:].any())
    )


def frames_of(page):
    """Rounded concept-figure boxes (Matplotlib FancyBboxPatch), not axes clips.

    A real frame is a fill+stroke path with a border colour AND Bezier corner
    items ('c').  Matplotlib axes/clip rectangles are white, line-only ('l')
    polygons with ~19 points and often negative coordinates; treating those as
    frames produced ~100 false alarms on the data figures (2026-11-24).
    """
    W, H = page.rect.width, page.rect.height
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 40 or r.height < 15:
            continue
        if r.width > 0.9 * W or r.height > 0.9 * H:
            continue
        if r.x0 < 0 or r.y0 < 0 or r.x1 > W or r.y1 > H:
            continue   # clip rects of axes/legends sit (partly) off-page
        if d.get("type") != "fs" or d.get("color") is None:
            continue
        kinds = {it[0] for it in d.get("items", [])}
        if "c" not in kinds:            # rounded corners -> real frame
            continue
        out.append(r)
    return out


def text_lines(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            r = pymupdf.Rect(line["bbox"])
            text = "".join(sp["text"] for sp in line["spans"]).strip()
            if text:
                out.append((r, text))
    return out


def frame_overflows(page, tol=FRAME_TOL_PT, frames=None, lines=None):
    """Report text lines whose box leaves the frame that contains them.

    A line belongs to a frame when their intersection covers >=25% of the
    smaller box.  Data figures have no frames, so this is a no-op for them.
    Any overhang > tol on any side is a real defect (P1 v0.6 Figure 1
    overhung its central frame by 11 pt on each side).
    `frames`/`lines` can be injected for the --self-test controls.
    """
    problems = []
    frames = frames_of(page) if frames is None else frames
    lines = text_lines(page) if lines is None else lines
    if not frames:
        return problems
    for box in frames:
        for r, text in lines:
            inter = box & r
            if inter.is_empty:
                continue
            if inter.get_area() <= 0.25 * min(r.get_area(), box.get_area()):
                continue
            dl, dr = box.x0 - r.x0, r.x1 - box.x1
            dt, db = box.y0 - r.y0, r.y1 - box.y1
            if max(dl, dr, dt, db) > tol:
                problems.append((box, text, dl, dr, dt, db))
    return problems


def check(path: Path, display_pt: float) -> int:
    doc = pymupdf.open(path)
    page = doc[0]
    W, H = page.rect.width, page.rect.height
    spans = spans_of(page)
    frame_problems = frame_overflows(page)   # before redactions mutate the page
    problems = 0

    print(f"\n=== {path.name}  ({W:.0f} x {H:.0f} pt, {len(spans)} text spans) ===")

    # 1) text-text overlaps
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            area = overlap(a, b)
            if area <= AREA_MIN:
                continue
            smaller = min(
                (a[2] - a[0]) * (a[3] - a[1]),
                (b[2] - b[0]) * (b[3] - b[1]),
            )
            if smaller > 0 and area / smaller >= FRAC_MIN:
                problems += 1
                print(f"  OVERLAP ({area:.1f} pt^2): {a[5][:28]!r} <> {b[5][:28]!r}")

    # 2) truncation: span box outside the page
    for s in spans:
        if (s[0] < -EDGE_TOL_PT or s[1] < -EDGE_TOL_PT
                or s[2] > W + EDGE_TOL_PT or s[3] > H + EDGE_TOL_PT):
            problems += 1
            print(f"  OUTSIDE-PAGE: {s[5][:30]!r} bbox=({s[0]:.0f},{s[1]:.0f},{s[2]:.0f},{s[3]:.0f})")

    # 3) text over vector ink (curves, markers, error bars, nominal lines)
    ink, iw, ih = vector_layer_ink(page, spans)
    sx, sy = iw / W, ih / H
    for s in spans:
        x0 = max(0, int(s[0] * sx) + 2)
        y0 = max(0, int(s[1] * sy) + 2)
        x1 = min(iw, int(s[2] * sx) - 2)
        y1 = min(ih, int(s[3] * sy) - 2)
        if x1 <= x0 or y1 <= y0:
            continue
        n = int(ink[y0:y1, x0:x1].sum())
        if n > INK_PIX_MIN:
            problems += 1
            print(f"  TEXT-ON-INK ({n} px): {s[5][:32]!r} bbox=({s[0]:.0f},{s[1]:.0f},{s[2]:.0f},{s[3]:.0f})")

    # 4) pixel border clipping
    if border_ink(page):
        problems += 1
        print("  BORDER-INK: artwork touches the page edge (clipping?)")

    # 6) text must stay inside its frame (concept figures)
    for box, text, dl, dr, dt, db in frame_problems:
        over = max(dl, dr, dt, db)
        problems += 1
        print(f"  TEXT-OUTSIDE-FRAME ({over:.1f} pt): {text[:32]!r} "
              f"box=({box.x0:.0f},{box.y0:.0f},{box.x1:.0f},{box.y1:.0f}) "
              f"overhang L/R/T/B=({dl:.1f},{dr:.1f},{dt:.1f},{db:.1f})")

    # 5) font size after inclusion at width=\linewidth
    scale = display_pt / W
    sizes = {}
    for s in spans:
        sizes[round(s[4], 1)] = sizes.get(round(s[4], 1), 0) + len(s[5])
    real = [sz for sz in sizes if sz >= 0.75 * max(sizes)]
    if real:
        mn = min(real) * scale
        mx = max(sizes) * scale
        flag = "  ** SMALL **" if mn < 6.0 else ""
        print(f"  final text size: {mn:.2f}-{mx:.2f} pt (display scale {scale:.2f}){flag}")
        if mn < 6.0:
            problems += 1

    print(f"  problems: {problems}")
    return problems


def self_test() -> int:
    """Positive/negative controls for both the frame *recogniser* and the
    overflow *geometry*, plus a check that OUTSIDE-PAGE still fires."""
    ok = True
    print("=== check_figure_layout.py --self-test ===")
    box = pymupdf.Rect(20, 20, 120, 60)

    # 1) geometry positive: a 180pt-wide line inside a 100pt frame
    pos = [(pymupdf.Rect(10, 35, 190, 48), "X" * 30)]
    n = len(frame_overflows(None, frames=[box], lines=pos))
    print(f"  [{'PASS' if n >= 1 else 'FAIL'}] positive control: frame overflow detected ({n})")
    ok &= n >= 1

    # 2) geometry negative: short text centred inside the frame
    neg = [(pymupdf.Rect(95, 35, 110, 48), "OK")]
    n = len(frame_overflows(None, frames=[box], lines=neg))
    print(f"  [{'PASS' if n == 0 else 'FAIL'}] negative control: in-frame text clean ({n})")
    ok &= n == 0

    # 3) recogniser: real concept figure has frames, a data figure has none
    f0 = ROOT / "paper" / "figures" / "scis" / "fig0-observation-boundary.pdf"
    f3 = ROOT / "paper" / "figures" / "scis" / "fig3-l0-coverage.pdf"
    if f0.exists() and f3.exists():
        n0 = len(frames_of(pymupdf.open(f0)[0]))
        n3 = len(frames_of(pymupdf.open(f3)[0]))
        good = n0 >= 1 and n3 == 0
        print(f"  [{'PASS' if good else 'FAIL'}] recogniser: concept figure {n0} frames, data figure {n3} frames")
        ok &= good

    # 4) detector still fires on off-page text
    doc = pymupdf.open()
    page = doc.new_page(width=100, height=50)
    page.insert_text((95, 25), "Y" * 20, fontsize=11)
    outside = [sp for sp in spans_of(page) if sp[2] > 100 + EDGE_TOL_PT or sp[0] < -EDGE_TOL_PT]
    print(f"  [{'PASS' if outside else 'FAIL'}] detector self-check: OUTSIDE-PAGE fires off-page")
    ok &= bool(outside)
    print(f"=== self-test {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


def font_audit(files: list[Path]) -> int:
    """Print the (font, size) matrix and combinations unique to one figure.

    Cross-figure font/size consistency is a submission gate (§10.9): a combo
    appearing in only one delivered data figure is a finding unless registered
    as a written exception in figure-alarm-register.md.
    """
    per: dict[str, collections.Counter] = {}
    for f in files:
        page = pymupdf.open(f)[0]
        c: collections.Counter = collections.Counter()
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for sp in line["spans"]:
                    if sp["text"].strip():
                        c[(sp["font"], round(sp["size"], 1))] += 1
        per[f.name] = c
    print("=== font audit ===")
    uniques = []
    for name, c in per.items():
        print(f"  {name}")
        for (font, size), n in sorted(c.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"      {font:28s} {size:4.1f} pt  n={n}")
        for k in c:
            if sum(k in cc for cc in per.values()) == 1:
                uniques.append(f"{name}: {k[0]} {k[1]}pt")
    if uniques:
        print("  UNIQUE combos (must be empty or registered as written exceptions):")
        for u in uniques:
            print(f"      {u}")
    else:
        print("  no figure-unique (font,size) combos")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figures", nargs="*", type=Path)
    ap.add_argument("--display-pt", type=float, default=252.0)
    ap.add_argument("--self-test", action="store_true",
                    help="run positive/negative controls instead of checking figures")
    ap.add_argument("--font-audit", action="store_true",
                    help="print per-figure (font,size) matrix and cross-figure unique combos")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.font_audit:
        return font_audit(args.figures or sorted(FIGDIR.glob("*.pdf")))
    files = args.figures or sorted(FIGDIR.glob("*.pdf"))
    total = 0
    for f in files:
        total += check(f, args.display_pt)
    print(f"\nTOTAL problems: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
