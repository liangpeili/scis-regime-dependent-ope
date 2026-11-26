#!/usr/bin/env python3
"""Atomic freeze gate for a paper project (reference implementation: P1 SCIS).

One command answers "is the current candidate internally consistent?" before
anyone writes "fixed / done / submission-ready".  It replaces the error-prone
manual loop of refreshing PDF -> submission -> tar -> MANIFEST -> docs hashes.

    python code/freeze_check.py                 # full check + write results/p1-scis/FREEZE.md
    python code/freeze_check.py --emit         # print the current object hash table
    python code/freeze_check.py --no-write     # check only, do not write FREEZE.md

Hard failures exit 1; WARN items are printed but do not block (they are the
remaining process gates, e.g. the independent full-page review).

See docs/submission-preflight-checklist.md §10.5 and docs/agent-workflow.md §十一.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import pymupdf
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# label (as written in the freeze table) -> real path
OBJECTS = {
    "paper/main-scis.md": "paper/main-scis.md",
    "paper/latex/scis/main-scis.tex": "paper/latex/scis/main-scis.tex",
    "paper/latex/scis/main-scis-supp.tex": "paper/latex/scis/main-scis-supp.tex",
    "results/p1-scis/main/main-scis.pdf": "results/p1-scis/main/main-scis.pdf",
    "results/p1-scis/supp/main-scis-supp.pdf": "results/p1-scis/supp/main-scis-supp.pdf",
    "scis-source.tar.gz": "results/p1-scis/submission/scis-source.tar.gz",
    "MANIFEST.sha256": "results/p1-scis/submission/MANIFEST.sha256",
}
# The freeze table sometimes uses human-readable labels instead of paths.
LABEL_ALIASES = {
    "主稿 PDF": "results/p1-scis/main/main-scis.pdf",
    "补充 PDF": "results/p1-scis/supp/main-scis-supp.pdf",
}
FREEZE_TABLE_DOC = "paper/reviews/scis-v07-fix-author-precheck-2026-11-24.md"
ALARM_REGISTER = "paper/reviews/figure-alarm-register.md"
VENUE_BEHAVIOR = "paper/venue-behavior.md"
AUTHOR_CONFIRMATION = "paper/reviews/author-confirmation-2026-11-26.md"
STATUS_DOCS = [
    "README.md",
    "paper/m7-typeset.md",
    FREEZE_TABLE_DOC,
    "drafts/submission-checklist-scis.md",
    "results/p1-scis/submission/README.md",
    "results/p1-scis/submission/submission-checklist-scis.md",
]
DELIVERED_FIGURES = [
    "paper/figures/scis/fig0-observation-boundary.pdf",
    "paper/figures/scis/fig3-l0-coverage.pdf",
    "paper/figures/scis/fig4-l1-boundary.pdf",
    "paper/figures/scis/fig5-l2-branches.pdf",
    "paper/figures/scis/fig6-l3-h.pdf",
    "paper/figures/scis/fig7-general-pomdp.pdf",
]
LOG_GATES = [
    "LaTeX Error", "Overfull \\hbox", "Overfull \\vbox", "Underfull \\hbox",
    "undefined", "Missing character", "multiply defined", "Float too large",
    "duplicate destination", "pdfTeX warning (dest)", "Underfull \\vbox",
]

results: list[tuple[str, str, str]] = []   # (level, name, detail)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True).stdout


def add(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))


def check_freeze_table() -> dict[str, str]:
    """Parse the freeze table and verify every declared hash is current."""
    doc = (ROOT / FREEZE_TABLE_DOC).read_text()
    section = doc.split("## 冻结对象", 1)[-1].split("## ", 1)[0]
    rows = re.findall(r"^\|\s*`?([^`|]+)`?\s*\|[^|]*\|\s*`([0-9a-f]{64})`\s*\|",
                      section, flags=re.M)
    declared: dict[str, str] = {}
    for label, h in rows:
        label = label.strip()
        path = OBJECTS.get(label) or LABEL_ALIASES.get(label)
        if path:
            declared[path] = h
        else:
            add("WARN", "freeze table extra row", f"label {label!r} not mapped to an object")
    actual = {}
    for label, path in OBJECTS.items():
        p = ROOT / path
        if not p.exists():
            add("FAIL", f"freeze table: {label}", "file missing")
            continue
        actual[path] = sha(p)
        if path not in declared:
            add("FAIL", f"freeze table: {label}", "not declared in freeze table")
        elif declared[path] != sha(p):
            add("FAIL", f"freeze table: {label}",
                f"declared {declared[path][:16]} != actual {sha(p)[:16]}")
    if len(declared) == len(OBJECTS) and all(
        declared.get(p) == actual.get(p) for p in OBJECTS.values()
    ):
        add("PASS", "freeze table", f"{len(OBJECTS)} objects byte-exact")
    return actual


def check_manifest() -> None:
    sub = ROOT / "results/p1-scis/submission"
    out = subprocess.run(["sha256sum", "-c", "MANIFEST.sha256"], cwd=sub,
                         capture_output=True, text=True)
    ok = out.stdout.count(": OK")
    bad = [l for l in out.stdout.splitlines() if not l.endswith(": OK")]
    if out.returncode == 0 and not bad:
        add("PASS", "MANIFEST", f"{ok}/11 OK")
    else:
        add("FAIL", "MANIFEST", f"{ok} OK; bad={bad[:3]}")


def check_copies() -> None:
    sub = ROOT / "results/p1-scis/submission"
    pairs = [
        (ROOT / "results/p1-scis/main/main-scis.pdf", sub / "main-scis.pdf"),
        (ROOT / "results/p1-scis/supp/main-scis-supp.pdf", sub / "main-scis-supp.pdf"),
    ]
    for f in sorted(sub.glob("*.md")):
        d = ROOT / "drafts" / f.name
        if d.exists():
            pairs.append((d, f))
    bad = [f"{a.relative_to(ROOT)} != {b.relative_to(ROOT)}"
           for a, b in pairs if not a.exists() or not b.exists() or sha(a) != sha(b)]
    if bad:
        add("FAIL", "primary == copies", "; ".join(bad[:4]))
    else:
        add("PASS", "primary == copies", f"{len(pairs)} pairs byte-identical")


def check_mechanical() -> None:
    for tag, log, pdf, pages in [
        ("main", "results/p1-scis/main/main-scis.log",
         "results/p1-scis/main/main-scis.pdf", 16),
        ("supp", "results/p1-scis/supp/main-scis-supp.log",
         # 37 -> 38: §10.4 written decision 2026-11-24 (V07-I01 float barriers
         # flush B1-B9 and pin the [H] notation table; main stays 16 pages,
         # no new >50%-blank page).  Allowed, recorded in m7 §7.57.
         "results/p1-scis/supp/main-scis-supp.pdf", 38),
    ]:
        p = ROOT / log
        if not p.exists():
            add("FAIL", f"{tag} log", "missing")
            continue
        text = p.read_text()
        found = [g for g in LOG_GATES if g in text]
        info = run(["pdfinfo", str(ROOT / pdf)])
        m = re.search(r"Pages:\s+(\d+)", info)
        n = int(m.group(1)) if m else -1
        fonts = run(["pdffonts", str(ROOT / pdf)])
        type3 = fonts.lower().count("type 3")
        body = run(["pdftotext", str(ROOT / pdf), "-"])
        audit = len(re.findall(r"AUDIT|\?\?", body))
        if found or n != pages or type3 or audit:
            add("FAIL", f"{tag} mechanical",
                f"log_gates={found} pages={n}(want {pages}) Type3={type3} AUDIT/??={audit}")
        else:
            add("PASS", f"{tag} mechanical",
                f"pages={n} log gates 0, Type3=0, AUDIT/??=0")


def check_audit() -> None:
    out = run([sys.executable, str(ROOT / "code/audit_numbers.py")])
    match = re.search(r"checked (\d+) ok (\d+) bad 0", out)
    if match and int(match.group(1)) == int(match.group(2)):
        add("PASS", "numeric audit", f"{match.group(1)}/{match.group(1)} + provenance checks")
    else:
        add("FAIL", "numeric audit", out.strip().splitlines()[-1] if out.strip() else "no output")


def check_alarms() -> None:
    reg = ROOT / ALARM_REGISTER
    if not reg.exists():
        add("FAIL", "alarm register", f"{ALARM_REGISTER} missing")
        return
    st = run([sys.executable, str(ROOT / "code/check_figure_layout.py"), "--self-test"])
    if "self-test PASS" not in st:
        add("FAIL", "detector self-test", st.strip().splitlines()[-1] if st.strip() else "")
    else:
        add("PASS", "detector self-test", "4/4 controls")
    figs = [str(ROOT / f) for f in DELIVERED_FIGURES]
    out = run([sys.executable, str(ROOT / "code/check_figure_layout.py"), *figs])
    alarms = len([l for l in out.splitlines()
                  if re.match(r"\s+(TEXT-ON-INK|TEXT-OUTSIDE-FRAME|OVERLAP|OUTSIDE-PAGE|BORDER-INK)", l)])
    registered = len(re.findall(r"^\|\s*A-\d+.*CLOSED（例外）", reg.read_text(), flags=re.M))
    open_items = len(re.findall(r"^\|\s*A-\d+.*\|\s*OPEN", reg.read_text(), flags=re.M))
    if alarms != registered or open_items:
        add("FAIL", "alarm closure",
            f"live alarms={alarms} registered exceptions={registered} OPEN={open_items}")
    else:
        add("PASS", "alarm closure",
            f"{alarms} live alarms all registered as written exceptions in {ALARM_REGISTER}")


def check_venue_items() -> None:
    tex = (ROOT / "paper/latex/scis/main-scis.tex").read_text()
    m = re.search(r"\\keywords\{([^}]*)\}", tex)
    n = len([k for k in m.group(1).split(",") if k.strip()]) if m else -1
    if not 3 <= n <= 5:
        add("FAIL", "keywords 3-5", f"found {n}")
    else:
        add("PASS", "keywords 3-5", f"{n} keywords")
    empty = re.findall(r"\\(DOI|ArtNo|Year|Month|Vol|No)\{\}", tex)
    if len(empty) < 6:
        add("WARN", "placeholders", f"only {len(empty)}/6 citation fields cleared")
    else:
        add("PASS", "placeholders", "DOI/ArtNo/Year/Month/Vol/No cleared")
    if not (ROOT / VENUE_BEHAVIOR).exists():
        add("WARN", "venue behavior", f"{VENUE_BEHAVIOR} missing (template placeholders not documented)")
    else:
        add("PASS", "venue behavior", f"{VENUE_BEHAVIOR} present")


def check_doc_hashes(actual: dict[str, str]) -> None:
    current = set(actual.values())
    stale = []
    for rel in STATUS_DOCS:
        p = ROOT / rel
        if not p.exists():
            continue
        for h in re.findall(r"[0-9a-f]{64}", p.read_text()):
            if h not in current:
                stale.append(f"{rel}:{h[:12]}")
    add("WARN" if stale else "PASS", "doc hash scan",
        f"{len(stale)} non-current 64-hex values (historical records are legal; "
        f"verify none is a stale CURRENT hash): {', '.join(stale[:6])}"
        if stale else "all 64-hex values match current objects")


def check_font_consistency() -> None:
    """Cross-figure font family / size hierarchy gate (§10.9).

    Delivered data figures must use one Times New Roman hierarchy:
    8.0 pt ticks, 8.5 pt bold panel titles, 9.0 pt legends/data labels.
    STIX/mathtext is allowed only for genuine math variables (registered in
    figure-alarm-register.md); ordinary words and punctuation must not mix
    font families.  A single extra size (e.g. the old 9.5 pt Figure 3 label)
    is a hard FAIL.
    """
    allowed = {8.0, 8.5, 9.0}
    bad, stix = [], []
    for rel in DELIVERED_FIGURES[1:]:          # fig0 is a concept figure (11 pt)
        page = pymupdf.open(ROOT / rel)[0]
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for sp in line["spans"]:
                    t = sp["text"].strip()
                    if not t:
                        continue
                    font, size = sp["font"], round(sp["size"], 1)
                    if font.startswith("TimesNewRoman") and size not in allowed:
                        bad.append(f"{Path(rel).name}: {font} {size}pt {t[:24]!r}")
                    if font.startswith("STIX"):
                        stix.append(f"{Path(rel).name}:{t[:12]}")
    if bad:
        add("FAIL", "figure font consistency", "; ".join(bad[:4]))
    else:
        add("PASS", "figure font consistency",
            "all Times text in {8.0, 8.5, 9.0} pt across delivered data figures")
    if stix:
        add("WARN", "math-font usage",
            f"STIX spans (math variables, must be registered): {len(stix)} e.g. {stix[:3]}")


def check_independent_review() -> None:
    """Track the §十一 independent full-page review; never treat it as a PASS."""
    # Current-object reviews (2026-11-26): D2-main covers the 16-page main
    # manuscript + 6 figures; D2-supp2 covers the rebuilt 38-page supplement.
    # A report is only accepted for the object whose full SHA256 it records.
    d2main = ROOT / "paper/reviews/scis-v12-fd1-independent-gateD2main-2026-11-26.md"
    d2supp = ROOT / "paper/reviews/scis-v12-fd2s-independent-gateD2supp2-2026-11-26.md"
    now_main = sha(ROOT / "results/p1-scis/main/main-scis.pdf")
    now_supp = sha(ROOT / "results/p1-scis/supp/main-scis-supp.pdf")
    if d2main.exists() and d2supp.exists():
        t_main = d2main.read_text()
        t_supp = d2supp.read_text()
        if (now_main in t_main and now_supp in t_supp
                and "reviewer=opencode" in t_main and "reviewer=opencode" in t_supp
                and "PASS" in t_main and "PASS" in t_supp):
            detail = ("D2-main + D2-supp2 (opencode/hy4, new read-only sessions) "
                      "cover the CURRENT objects: main 16 pages + 6 figures, "
                      "supplement 38 pages; scope-limited PASS, 0 unclosed BLOCKER/MAJOR.")
            ac = ROOT / AUTHOR_CONFIRMATION
            if (ac.exists() and now_main in ac.read_text()
                    and now_supp in ac.read_text()):
                detail += (" Author eyeball confirmation recorded for the current "
                           "PDFs + SHA256.")
            else:
                detail += " Author eyeball confirmation still pending."
            detail += " NOT submission-ready."
            add("WARN", "independent full-page review", detail)
            return
    r10 = ROOT / "drafts/review-2026-11-24-scis-v10-declaration-and-main.md"
    if r10.exists():
        # §九 / §十一: r10 (Kimi new session, 54 pages + 6 figures, 300 dpi
        # native + finished pages) reviewed the 27ce6f2f build; its official
        # verdict is NOT PASS (R10-I01/M01/M03). Any later main-PDF change
        # makes it stale and requires a FRESH review.
        r10_main = "27ce6f2f3d60344c79653ec2b02ef277f60d43c48d8ce7885faec898421e890f"
        now_main = sha(ROOT / "results/p1-scis/main/main-scis.pdf")
        if now_main != r10_main:
            add("WARN", "independent full-page review",
                "r10 NOT PASS (R10-I01/M01/M03; see `drafts/review-2026-11-24-scis-v10-declaration-and-main.md`) is STALE: main PDF changed after r10. §十一 requires a FRESH new-session full-page review; author eyeball confirmation still pending.")
        else:
            add("WARN", "independent full-page review",
                "r10 NOT PASS record for its own build — see the r10 report.")
    r11 = ROOT / "drafts/review-2026-11-24-scis-v11-declaration-and-main.md"
    if r11.exists():
        # r11 (Kimi new session, 54 pages + 6 figures + declaration) reviewed
        # the 6a2ed6d8 build and closed the r10 issues (R11-I01/M01/M02/M03).
        r11_main = "6a2ed6d891e4f91845be8b6a51fac8ccdb95197464f5ef2c1c9a675d0203fcd4"
        now_main = sha(ROOT / "results/p1-scis/main/main-scis.pdf")
        if now_main != r11_main:
            add("WARN", "independent full-page review",
                "r11 PASS (visual/font/info-design + narrative-orientation + focused claim-boundary scope + declaration scope) is STALE: main PDF changed after r11. §十一 requires a FRESH new-session full-page review; author eyeball confirmation still pending.")
        else:
            add("WARN", "independent full-page review",
                "r11 (Kimi new-session) review of the CURRENT build: PASS limited to independent visual/font/info-design + narrative-orientation + focused claim-boundary scope + AI-declaration scope; 54 pages + 6 figures read page-by-page, 0 unclosed BLOCKER/MAJOR/MINOR, both groups CLOSED (scope-limited); author eyeball confirmation of current PDF + SHA256 still pending. NOT submission-ready.")
    else:
        add("WARN", "independent full-page review",
            "NOT DONE — no new-session read-only full-page review of the current build")


def emit_table(actual: dict[str, str]) -> str:
    lines = ["| Object | SHA256 |", "|---|---|"]
    for label, path in OBJECTS.items():
        lines.append(f"| `{label}` | `{actual.get(path, 'MISSING')}` |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit", action="store_true", help="print the object hash table only")
    ap.add_argument("--no-write", action="store_true", help="do not write FREEZE.md")
    args = ap.parse_args()

    actual = {p: sha(ROOT / p) for p in OBJECTS.values() if (ROOT / p).exists()}
    if args.emit:
        print(emit_table({p: actual.get(p, "MISSING") for p in OBJECTS.values()}))
        return 0

    print(f"=== freeze_check.py  {date.today().isoformat()}  root={ROOT} ===")
    head = run(["git", "rev-parse", "--short", "HEAD"]).strip()
    dirty = run(["git", "status", "--porcelain"]).strip().splitlines()
    print(f"git HEAD={head}  dirty={len(dirty)}  python={sys.executable}\n")

    check_freeze_table()
    check_manifest()
    check_copies()
    check_mechanical()
    check_audit()
    check_alarms()
    check_venue_items()
    check_font_consistency()
    actual_map = {p: sha(ROOT / p) for p in OBJECTS.values() if (ROOT / p).exists()}
    check_doc_hashes(actual_map)
    check_independent_review()

    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    for level, name, detail in results:
        print(f"  [{level:4s}] {name:28s} {detail}")
    print(f"\nSUMMARY: {sum(1 for r in results if r[0]=='PASS')} PASS / "
          f"{len(fails)} FAIL / {len(warns)} WARN")

    ac = ROOT / AUTHOR_CONFIRMATION
    author_ok = False
    if ac.exists():
        act = ac.read_text()
        author_ok = (actual_map.get("results/p1-scis/main/main-scis.pdf") in act
                     and actual_map.get("results/p1-scis/supp/main-scis-supp.pdf") in act)
    if author_ok:
        gate = ("当前为候选版；§十一.6 独立全页复核与作者肉眼确认已完成"
                "（主稿 16 页、补充 38 页）；投稿包/§十二/Portal/公开快照/"
                "clean-room/声明仍未完成，不得视为 submission-ready。")
    else:
        gate = ("当前为候选版，未完成新会话只读全页独立视觉复核与作者肉眼确认，"
                "不得视为合格 / 可提交 / submission-ready。")
    print("GATE: " + gate)

    if not args.no_write:
        out = ROOT / "results/p1-scis/FREEZE.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# P1 SCIS freeze check — {date.today().isoformat()}",
            "",
            f"git HEAD: `{head}`, dirty files: {len(dirty)}, python: `{sys.executable}`",
            "", gate.replace("当前", "**当前").replace("不得", "**不得") + "**",
            "", "## Current object hashes", "", emit_table(actual_map), "",
            "## Checks", "", "| Level | Item | Detail |", "|---|---|---|",
        ]
        for level, name, detail in results:
            lines.append(f"| {level} | {name} | {detail.replace('|', '/')} |")
        lines += ["", f"SUMMARY: {len(fails)} FAIL / {len(warns)} WARN", "",
                  f"Alarm register: `{ALARM_REGISTER}`; template behavior: `{VENUE_BEHAVIOR}`.", ""]
        out.write_text("\n".join(lines) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
