"""Baixa o recorte público do Transfermarkt usado pelo placebo 2023/24.

Os arquivos brutos são mantidos fora do Git. Este script registra URL, tamanho e
SHA-256 em um manifesto local para permitir reproduzir o placebo sazonal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
FILES = [
    "games.csv.gz",
    "appearances.csv.gz",
    "game_lineups.csv.gz",
    "players.csv.gz",
    "player_valuations.csv.gz",
    "transfers.csv.gz",
]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "transfermarkt" / "dcaribou_2023_24"
USER_AGENT = "WC-IMPACT-research/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(filename: str, force: bool) -> dict[str, object]:
    destination = OUT / filename
    url = BASE_URL + filename
    status = "cached"
    if force or not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(destination)
        status = "downloaded"
    if destination.stat().st_size < 2 or destination.read_bytes()[:2] != b"\x1f\x8b":
        raise RuntimeError(f"Arquivo não parece gzip válido: {destination}")
    return {
        "source_url": url,
        "relative_path": filename,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="baixa novamente arquivos existentes")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = [download(filename, args.force) for filename in FILES]
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Concluído: {len(records)} arquivos em {OUT}")


if __name__ == "__main__":
    main()
