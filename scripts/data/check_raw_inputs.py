"""Check whether all local raw inputs required by the main WC-IMPACT pipeline exist."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

REQUIRED_FILES = [
    RAW / "fifa" / "wc_2022_squads" / "SquadLists-English.pdf",
    RAW / "transfermarkt" / "dcaribou" / "transfermarkt-datasets.duckdb",
    RAW / "transfermarkt" / "salimt" / "player_injuries.csv",
    RAW / "statsbomb" / "wc_2022" / "data" / "competitions.json",
    RAW / "statsbomb" / "wc_2022" / "data" / "matches" / "43" / "106.json",
]

UNDERSTAT_ROOT = RAW / "understat" / "extracted" / "understats"
for league in ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]:
    REQUIRED_FILES.extend(
        [
            UNDERSTAT_ROOT / league / "shot_data.csv",
            UNDERSTAT_ROOT / league / "match_info.csv",
        ]
    )


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not path.exists()]
    if missing:
        print("Missing required raw inputs:")
        for path in missing:
            print(f"  - {path.relative_to(ROOT)}")
        print("\nSee DATA_SOURCES.md for acquisition and placement instructions.")
        return 1

    print(f"All {len(REQUIRED_FILES)} required raw input paths are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
