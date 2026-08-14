"""Build the local transfermarkt-datasets DuckDB input used by the main analysis.

The script downloads the same public compressed CSV endpoints used by the
seasonal-placebo helpers, records SHA-256 hashes, and imports the required tables
into data/raw/transfermarkt/dcaribou/transfermarkt-datasets.duckdb.
Raw files and the DuckDB database remain outside Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "transfermarkt" / "dcaribou"
SOURCE = OUT / "source"
DATABASE = OUT / "transfermarkt-datasets.duckdb"
BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
FILES = [
    "games.csv.gz",
    "appearances.csv.gz",
    "game_lineups.csv.gz",
    "players.csv.gz",
    "player_valuations.csv.gz",
    "transfers.csv.gz",
]
USER_AGENT = "WC-IMPACT-academic-research/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(filename: str, force: bool) -> dict[str, object]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    destination = SOURCE / filename
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
        raise RuntimeError(f"Downloaded file is not valid gzip: {destination}")
    return {
        "source_url": url,
        "relative_path": str(destination.relative_to(OUT)),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": status,
    }


def build_database() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if DATABASE.exists():
        DATABASE.unlink()
    con = duckdb.connect(str(DATABASE))
    try:
        for filename in FILES:
            table = filename.removesuffix(".csv.gz")
            path = (SOURCE / filename).resolve().as_posix().replace("'", "''")
            print(f"importing {table} ...")
            con.execute(
                f"CREATE TABLE {table} AS "
                f"SELECT * FROM read_csv_auto('{path}', header=true, sample_size=-1)"
            )
        con.execute("CHECKPOINT")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download source CSV files")
    parser.add_argument("--download-only", action="store_true", help="download files but do not rebuild DuckDB")
    args = parser.parse_args()

    rows = [download(filename, args.force) for filename in FILES]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if not args.download_only:
        build_database()
        print(f"database: {DATABASE.relative_to(ROOT)}")
        print(f"database_sha256={sha256(DATABASE)}")


if __name__ == "__main__":
    main()
