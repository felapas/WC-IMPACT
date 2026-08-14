"""Download the FIFA World Cup Qatar 2022 final squad-list PDF used by the ETL."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "fifa" / "wc_2022_squads"
URL = "https://fdp.fifa.org/assetspublic/ce44/pdf/SquadLists-English.pdf"
DESTINATION = OUT / "SquadLists-English.pdf"
USER_AGENT = "WC-IMPACT-academic-research/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="download even if the file already exists")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.force or not DESTINATION.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if not payload.startswith(b"%PDF"):
            raise RuntimeError("FIFA response is not a PDF")
        DESTINATION.write_bytes(payload)
        status = "downloaded"
    else:
        status = "cached"

    print(f"{status}: {DESTINATION.relative_to(ROOT)}")
    print(f"bytes={DESTINATION.stat().st_size} sha256={sha256(DESTINATION)}")


if __name__ == "__main__":
    main()
