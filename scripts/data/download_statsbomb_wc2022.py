"""Baixa apenas os dados abertos da Copa do Mundo masculina de 2022."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master"
COMPETITION_ID = 43
SEASON_ID = 106
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "statsbomb" / "wc_2022"
USER_AGENT = "FAME-2026-research/1.0 (open-data downloader)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(relative: str, force: bool = False, optional: bool = False) -> dict[str, object]:
    destination = OUT / relative
    url = f"{BASE}/{relative}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "cached"
    if force or not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                json.loads(payload)
                destination.write_bytes(payload)
                status = "downloaded"
                break
            except urllib.error.HTTPError as exc:
                if optional and exc.code == 404:
                    return {
                        "source_url": url,
                        "relative_path": relative,
                        "bytes": 0,
                        "sha256": "",
                        "status": "not_available",
                    }
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
            except (OSError, json.JSONDecodeError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    return {
        "source_url": url,
        "relative_path": relative,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": status,
    }


def fetch_binary(relative: str, force: bool = False) -> dict[str, object]:
    destination = OUT / relative
    url = f"{BASE}/{relative}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "cached"
    if force or not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
        status = "downloaded"
    return {
        "source_url": url,
        "relative_path": relative,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="baixa novamente arquivos existentes")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    records.append(fetch("data/competitions.json", args.force))
    matches_rel = f"data/matches/{COMPETITION_ID}/{SEASON_ID}.json"
    records.append(fetch(matches_rel, args.force))
    matches = json.loads((OUT / matches_rel).read_text(encoding="utf-8"))

    competition = next(
        item
        for item in json.loads((OUT / "data/competitions.json").read_text(encoding="utf-8"))
        if item["competition_id"] == COMPETITION_ID and item["season_id"] == SEASON_ID
    )
    if competition["competition_name"] != "FIFA World Cup" or competition["season_name"] != "2022":
        raise RuntimeError(f"IDs inesperados na fonte: {competition}")

    for index, match in enumerate(matches, start=1):
        match_id = match["match_id"]
        records.append(fetch(f"data/events/{match_id}.json", args.force))
        records.append(fetch(f"data/lineups/{match_id}.json", args.force))
        records.append(fetch(f"data/three-sixty/{match_id}.json", args.force, optional=True))
        print(f"[{index:02d}/{len(matches)}] jogo {match_id}")

    records.append(fetch_binary("LICENSE.pdf", args.force))
    manifest = OUT / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    available_360 = sum(row["status"] != "not_available" for row in records if "three-sixty" in str(row["relative_path"]))
    print(f"Concluído: {len(matches)} jogos, {available_360} arquivos 360, manifest em {manifest}")


if __name__ == "__main__":
    main()

