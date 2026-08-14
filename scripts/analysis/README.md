# Analysis pipeline

Run all commands from the repository root after installing `requirements.txt` and preparing the raw inputs described in `DATA_SOURCES.md`.

## 1. Analytical panel

```bash
python scripts/analysis/build_analytic_panel.py
```

Builds the player–club-match grid, baseline-only sample and processed analytical datasets. Critical QA failures stop execution.

## 2. Design

```bash
python scripts/analysis/build_design_phase.py
```

Reads `config/analysis.yml`, constructs the qualified-national-team risk set and produces design diagnostics without fitting confirmatory post-World-Cup coefficients.

## 3. Weighting and pre-trends

```bash
python scripts/analysis/build_weighting_phase.py
```

Reads `config/weighting.yml`, fits ridge-logistic propensity models within league-position strata, computes overlap weights and calibrates them to baseline covariates, strata and game-level pre-treatment histories. Games −14 to −9 are held out from calibration for diagnostics.

## 4. Confirmatory models

```bash
python scripts/analysis/build_confirmatory_phase.py
```

Uses the frozen Phase 5 weights and `config/confirmatory.yml`. The primary specification uses player and club-match fixed effects with player-clustered standard errors; additional outputs include double clustering, external fixed effects, differential trends, event studies, non-inferiority diagnostics, PPML and the pre-Copa holdout placebo.

## 5. Robustness

```bash
python scripts/analysis/build_robustness_phase.py
python scripts/analysis/build_baseline90_sensitivity.py
```

Runs the window, weighting, sample, leave-one-out, trend and confounding sensitivities configured in `config/robustness.yml`. The 90-minute baseline sensitivity fully re-estimates the risk set and weights.

## 6. Seasonal falsifications

```bash
python scripts/data/download_transfermarkt_placebo_2021_22.py
python scripts/analysis/run_placebo_2021_22.py
```

The 2021/22 November international break is the seasonal falsification reported in the paper. An additional 2023/24 diagnostic is available through the corresponding downloader and analysis script.

## 7. Reporting artifacts

```bash
python scripts/analysis/build_manuscript_phase.py
```

Generates analytical tables/figures and draft reporting artifacts from audited outputs. The **authoritative edited FAME 2026 manuscript** is the LaTeX source in `reports/submission_fame2026/`; it is deliberately not overwritten by this script.
