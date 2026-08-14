# Reproducibility guide

## 1. Environment

Use Python 3.11 or newer from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Acquire source data

Automated helpers:

```bash
python scripts/data/download_fifa_squads.py
python scripts/data/download_statsbomb_wc2022.py
python scripts/data/download_transfermarkt_main.py
```

Optional archival FIFA post-match reports:

```bash
python scripts/data/download_fifa_reports.py
```

Place the injury-history and Understat source files as documented in `DATA_SOURCES.md`, then run:

```bash
python scripts/data/check_raw_inputs.py
```

## 3. Build auditable intermediate data

```bash
python scripts/etl/build_interim.py
```

Expected metadata/QA outputs include:

- `data/interim/manifest.csv`
- `data/interim/qa_summary.csv`

The row-level interim tables are intentionally ignored by Git.

## 4. Build the analytical panel

```bash
python scripts/analysis/build_analytic_panel.py
```

This creates the player–club-match grid and locally regenerates the processed datasets described in `data/processed/README.md`.

## 5. Frozen design and weighting

```bash
python scripts/analysis/build_design_phase.py
python scripts/analysis/build_weighting_phase.py
```

The settings are stored in `config/analysis.yml` and `config/weighting.yml`. These files use JSON syntax despite the `.yml` extension, which is intentional and allows parsing without a YAML dependency.

## 6. Confirmatory models

```bash
python scripts/analysis/build_confirmatory_phase.py
```

The aggregate outputs in `reports/confirmatory/` are included in the public repository and can be compared directly against a rebuilt run.

## 7. Robustness and sensitivity

```bash
python scripts/analysis/build_robustness_phase.py
python scripts/analysis/build_baseline90_sensitivity.py
```

## 8. Seasonal falsification

For the falsification reported in the paper:

```bash
python scripts/data/download_transfermarkt_placebo_2021_22.py
python scripts/analysis/run_placebo_2021_22.py
```

The 2023/24 placebo is an additional diagnostic:

```bash
python scripts/data/download_transfermarkt_placebo_2023_24.py
python scripts/analysis/run_placebo_2023_24.py
```

## 9. Paper

The final edited FAME 2026 submission source is **not regenerated automatically from prose templates**. It lives in `reports/submission_fame2026/`. Analysis scripts generate statistical outputs and draft reporting artifacts; the submission LaTeX is the authoritative manuscript source.

The current rendered PDF is `output/pdf/FAME2026.pdf`.

## 10. Release QA

Before publishing a GitHub release:

```bash
python scripts/qa/check_public_release.py
python -m compileall -q scripts
```

The release checker fails if known row-level datasets, build logs, nested archives or common credential patterns appear in the public tree.
