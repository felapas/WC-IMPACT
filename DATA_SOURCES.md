# Data sources and redistribution policy

This public repository is designed to reproduce the analysis **without republishing third-party row-level data**. Source files are stored locally under `data/raw/` and are ignored by Git.

## Inputs used by the main pipeline

| Source | Local path expected by the code | Acquisition in this repo | Public bundle |
|---|---|---|---|
| FIFA final squad list | `data/raw/fifa/wc_2022_squads/SquadLists-English.pdf` | `scripts/data/download_fifa_squads.py` | Not redistributed |
| StatsBomb Open Data, World Cup 2022 | `data/raw/statsbomb/wc_2022/` | `scripts/data/download_statsbomb_wc2022.py` | Not duplicated; downloaded from source |
| transfermarkt-datasets | `data/raw/transfermarkt/dcaribou/transfermarkt-datasets.duckdb` | `scripts/data/download_transfermarkt_main.py` | Not redistributed |
| football-datasets injury histories | `data/raw/transfermarkt/salimt/player_injuries.csv` | Manual acquisition from the cited source | Not redistributed |
| Understat-derived database | `data/raw/understat/extracted/understats/` | Manual acquisition from the cited source | Not redistributed |

The Understat directory must contain these files for each of `EPL`, `La_Liga`, `Bundesliga`, `Serie_A` and `Ligue_1`:

```text
shot_data.csv
match_info.csv
```

The analysis code filters those source files to season 2022.

## Seasonal placebo inputs

The 2021/22 and 2023/24 Transfermarkt placebo snapshots can be reconstructed with:

```bash
python scripts/data/download_transfermarkt_placebo_2021_22.py
python scripts/data/download_transfermarkt_placebo_2023_24.py
```

Those source files are also ignored by Git.

## Why row-level derived files are absent

Files in `data/interim/`, `data/processed/` and selected row-level report outputs can contain names, identifiers or values derived from third-party databases. The public release therefore keeps only documentation, manifests, QA summaries and aggregate/statistical outputs. This is a distribution choice for the repository, not a statement that every upstream source has identical licensing terms.

Before redistributing any upstream or derived dataset yourself, review the **current** terms of the original source. The bibliography in `reports/submission_fame2026/references.bib` records the source citations used by the manuscript.

## Verifying local inputs

After acquiring the inputs, run:

```bash
python scripts/data/check_raw_inputs.py
```

The script checks the exact paths required by the current ETL and reports missing items before a long pipeline run starts.

## Snapshot reproducibility note

The manuscript records the main source retrieval date as 10 July 2026. The provided project archive did not contain the ignored raw-source files themselves, so this public package cannot manufacture their original byte-level hashes. The included manifests preserve hashes for generated intermediate/analysis artifacts, while the download helpers record fresh source hashes when a user reconstructs the raw inputs. Because upstream datasets may be updated, a future download can differ from the snapshot used for the paper; compare coverage/QA outputs and archive any matching source snapshot used for a formal replication.
