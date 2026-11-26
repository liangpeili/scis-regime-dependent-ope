# SCIS reproducibility snapshot — Regime-Dependent OPE

Public, signed snapshot for the SCIS paper:

> **Identifiability and Coverage Boundaries of Off-Policy Evaluation under
> Regime-Dependent Observation**

This snapshot contains the manuscript sources, SCIS figure/table generators,
curated aggregate result JSON, and the final main/supplement PDFs. It excludes
internal review records, logs, caches, credentials, third-party raw data, and
event-level real-path tables that cannot be redistributed.

## Layout

- `paper/main-scis.md`, `paper/related-work-scis.md`: Markdown sources.
- `paper/latex/scis/`: SCIS LaTeX sources, tables, class/bst, and `refs.bib`.
- `paper/figures/scis/`: delivered figure PDFs/PNGs and provenance sidecars.
- `paper/tables/`: Markdown table sources, `number-sources.md`, `m3-derived.json`.
- `paper/audit-m5.md`, `paper/citation-map-m4.md`: evidence/citation maps.
- `code/`: synthetic generators, estimators, table/figure generators, audit.
- `results/`: curated aggregate JSON; `real_data_replay*.json` are aggregate-only
  redactions with event rows and input paths removed.
- `pdf/main-scis.pdf`, `pdf/main-scis-supp.pdf`: final signed PDFs.
- `MANIFEST.sha256`: SHA256 of every file in this snapshot.
- `artifact-metadata.json`: file count, bytes, package versions, audit status.

## Environment

Python 3.10+ and TeX Live with `latexmk`/pdfLaTeX. No GPU or deep-learning
framework is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Core smoke pipeline

```bash
.venv/bin/python code/generate_dataset.py --n-seeds 32 --seed-start 42 --windows W1 W2 W3 W4
.venv/bin/python code/coverage_eval.py
.venv/bin/python code/general_pomdp_check.py
.venv/bin/python code/build_tables.py
.venv/bin/python code/make_scis_figures.py
.venv/bin/python code/make_scis_boundary.py
.venv/bin/python code/latex_convert_scis.py
.venv/bin/python code/latex_postprocess_scis.py
.venv/bin/python code/audit_numbers.py
```

The public audit is expected to report **200/200** rather than the internal
201/201: the omitted check is the event-level leave-one-episode-out
calibration-support reconstruction, which requires the restricted real-path
event table. All aggregate real-path checks remain.

## Manuscript rebuild

```bash
cd paper/latex/scis
latexmk -pdf -interaction=nonstopmode -outdir=../../../results/build/main main-scis.tex
grep -v '\newlabel{LastPage}' ../../../results/build/main/main-scis.aux   > ../../../results/build/supp/main-ref.aux
latexmk -pdf -interaction=nonstopmode -outdir=../../../results/build/supp main-scis-supp.tex
```

Expected page counts: main **16**, supplement **38**.

## Clean-room verification (2026-11-26)

The snapshot was copied to a clean `/tmp` directory, `code/audit_numbers.py`
reported **checked 200 ok 200 bad 0**, and both PDFs were rebuilt from the
snapshot sources (main 16 pages, supplement 38 pages; 0 mechanical log gates).
`pdftotext -layout` outputs were byte-identical to the delivered PDFs.

## Data boundary

Raw Ethereum/Chainlink responses, Yahoo Finance caches, and derived real-path
event tables are not redistributed. The extended real-data pipeline is
documented by `code/fetch_pool_swaps.py`, `code/build_real_slice.py`, and
`code/real_data_replay.py`; users must supply lawful data and credentials
locally. No credentials are included. The aggregate `real_slice_*_summary.json`
and `real_data_replay*.json` redactions support the reported aggregate tables.

## License

Code is released under the MIT License (`LICENSE`). The paper text, figures and
tables are released under CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`). Third-party raw
data and event-level real-path tables are not included and remain subject to
their original terms.
