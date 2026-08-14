"""Fail fast if a WC-IMPACT public release contains known private/row-level artifacts."""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_EXACT = {
    "data/interim/big5_player_match_2022_23.csv",
    "data/interim/fifa_player_crosswalk.csv",
    "data/interim/fifa_squads_2022.csv",
    "data/interim/linkage_review.csv",
    "data/interim/understat_player_match_2022_23.csv",
    "data/interim/understat_transfermarkt_game_crosswalk.csv",
    "data/interim/understat_transfermarkt_player_crosswalk.csv",
    "data/interim/wc_player_match_2022.csv",
    "data/interim/wc_player_tournament_2022.csv",
    "data/processed/analysis_sample_main.csv",
    "data/processed/analysis_sample_baseline_only.csv",
    "data/processed/analytic_player_match.csv",
    "data/processed/design_player_match.csv",
    "data/processed/analysis_weights.csv",
    "reports/design/baseline_assignment_review.csv",
    "reports/design/national_eligibility_review.csv",
    "reports/placebo_2021_22/analysis_weights.csv",
    "reports/placebo_2023_24/analysis_weights.csv",
}

FORBIDDEN_SUFFIXES = {".zip.partial", ".env"}
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|password|secret)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".tex", ".bib", ".csv", ".ps1", ".sh", ".cff"}


def main() -> int:
    issues: list[str] = []
    for relative in sorted(FORBIDDEN_EXACT):
        if (ROOT / relative).exists():
            issues.append(f"forbidden row-level artifact present: {relative}")

    safe_data_csv = {
        "data/interim/manifest.csv",
        "data/interim/qa_summary.csv",
        "data/processed/data_dictionary.csv",
        "data/processed/manifest.csv",
        "data/processed/panel_qa_summary.csv",
        "data/processed/injury_source_audit.csv",
    }

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("data/raw/") and path.name not in {".gitkeep", "manifest.csv"}:
            issues.append(f"raw source file present: {relative}")
        if (relative.startswith("data/interim/") or relative.startswith("data/processed/")) and path.suffix.lower() == ".csv" and relative not in safe_data_csv:
            issues.append(f"row-level data CSV present: {relative}")
        if any(path.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            issues.append(f"forbidden file type: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 5_000_000:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    issues.append(f"possible credential in: {relative}")
                    break

    if issues:
        print("Public-release QA FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("Public-release QA passed: no known row-level artifacts or credential patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
