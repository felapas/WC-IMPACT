# Project data layout

The public repository does not contain third-party row-level data. Data are rebuilt locally under three layers:

- `raw/`: immutable source snapshots acquired from the original providers;
- `interim/`: linkage tables and source-harmonized intermediate data;
- `processed/`: analytical panels, weights and design datasets.

The public package keeps only metadata, QA summaries and the processed data dictionary. Row-level CSVs are ignored by Git and intentionally omitted from public ZIP releases.

## Main source coverage used by the study

| Source | Main use | Validated study coverage |
|---|---|---|
| FIFA squad list | World Cup selection, club, position, birth date | 831 selected players, 32 national teams |
| StatsBomb Open Data | World Cup events, lineups and 360 data | 64 matches |
| transfermarkt-datasets | Big Five matches, appearances, lineups, transfers, market values | 1,826 club matches in 2022/23 |
| football-datasets injury histories | Baseline absence/injury-history covariates | source audit recorded in `processed/injury_source_audit.csv` |
| Understat-derived database | shots, npxG and xA proxy | 1,826 club matches in 2022/23 |

Exact local paths and acquisition helpers are documented in [`../DATA_SOURCES.md`](../DATA_SOURCES.md).

## Rebuilding

After acquiring the raw sources:

```bash
python scripts/data/check_raw_inputs.py
python scripts/etl/build_interim.py
python scripts/analysis/build_analytic_panel.py
```

Do not manually edit generated row-level CSVs. The pipeline writes SHA-256 manifests and QA summaries for verification.
