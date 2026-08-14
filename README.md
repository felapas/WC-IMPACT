# WC-IMPACT

Reproducible research repository for **“Da Copa ao clube: convocação para Qatar 2022 e utilização de jogadores no retorno às cinco grandes ligas europeias”**, prepared for FAME 2026 (UFMG/SALAB).

## Research question

Did players selected for the 2022 FIFA World Cup receive less playing time from their clubs immediately after returning to Europe's Big Five leagues?

The main analysis uses a player–club-match panel for the 2022/23 season, a qualified-national-team risk set, calibrated overlap weights, player fixed effects, club–match fixed effects, event-study diagnostics, sensitivity analyses and a seasonal falsification using 2021/22.

### Main result

The adjusted acute association for games 1–4 after the World Cup is **0.0233 in minutes share** (95% CI −0.0260 to 0.0726; p = 0.355), equivalent to approximately **+2.10 minutes per club match**. The study therefore does not detect a robust generalized acute decline in club utilization. Later positive estimates are treated cautiously because they are sensitive to trend specifications and falsification evidence.

The current submission PDF is available at [`output/pdf/FAME2026.pdf`](output/pdf/FAME2026.pdf). The LaTeX source is in [`reports/submission_fame2026/`](reports/submission_fame2026/).

## Repository structure

```text
config/                       Frozen design, weighting and robustness settings
data/                         Data documentation and local-data skeleton
  raw/                        Third-party source data (not tracked)
  interim/                    Rebuilt linkage/intermediate data (row-level files not tracked)
  processed/                  Rebuilt analysis data (row-level files not tracked)
docs/                         Analysis protocol and reproducibility notes
scripts/
  data/                       Source acquisition helpers
  etl/                        Raw-to-interim ETL
  analysis/                   Design, weighting, models, robustness and reporting
  qa/                         Public-release checks
reports/                      Aggregate diagnostics and model outputs
output/pdf/FAME2026.pdf       Current paper PDF
```

## Public-release policy

This repository intentionally **does not redistribute third-party raw data or row-level derived datasets** from Transfermarkt/Understat and related sources. The public package contains original code, configurations, documentation, aggregate model outputs, figures, QA summaries and manifests. Row-level datasets are reconstructed locally from the original sources.

See [`DATA_SOURCES.md`](DATA_SOURCES.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before running the pipeline.

## Quick start

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the sources that can be acquired automatically:

```bash
python scripts/data/download_fifa_squads.py
python scripts/data/download_statsbomb_wc2022.py
python scripts/data/download_transfermarkt_main.py
```

The Understat snapshot and injury-history file must be placed locally in the paths documented in [`DATA_SOURCES.md`](DATA_SOURCES.md). Then verify inputs:

```bash
python scripts/data/check_raw_inputs.py
```

Run the main pipeline from the repository root:

```bash
python scripts/etl/build_interim.py
python scripts/analysis/build_analytic_panel.py
python scripts/analysis/build_design_phase.py
python scripts/analysis/build_weighting_phase.py
python scripts/analysis/build_confirmatory_phase.py
python scripts/analysis/build_robustness_phase.py
python scripts/analysis/build_baseline90_sensitivity.py
python scripts/analysis/run_placebo_2021_22.py
```

The 2023/24 placebo is diagnostic and not part of the paper's main falsification:

```bash
python scripts/data/download_transfermarkt_placebo_2023_24.py
python scripts/analysis/run_placebo_2023_24.py
```

## Reproducibility safeguards

The project uses frozen JSON-compatible configuration files under `config/`, deterministic random seeds, explicit QA gates and SHA-256 manifests. The public bundle preserves aggregate outputs so that rebuilt results can be compared against the reported values without publishing row-level third-party data.

Run the repository-level QA before creating a public release:

```bash
python scripts/qa/check_public_release.py
python -m compileall -q scripts
```

## Paper build

The FAME/SBC submission source is under `reports/submission_fame2026/`.

With Tectonic installed:

```bash
cd reports/submission_fame2026
./build.sh
```

On Windows PowerShell:

```powershell
cd reports/submission_fame2026
.\build.ps1
```

Overleaf users can upload `main.tex`, `references.bib`, `sbc-template.sty`, `sbc.bst` and the `figures/` directory.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). When publishing the GitHub repository, create an immutable release such as `fame2026-v1`; if you archive that release with Zenodo, add the resulting DOI to both `CITATION.cff` and the paper's data/code availability statement.

## License and third-party material

Original project code is released under the MIT License; see [`LICENSE`](LICENSE). This license does **not** relicense third-party datasets, journal/conference templates or source materials. Their own terms continue to apply. See [`DATA_SOURCES.md`](DATA_SOURCES.md).
