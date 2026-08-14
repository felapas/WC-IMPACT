"""Baixa os 64 relatórios oficiais pós-jogo da Copa do Mundo de 2022."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


HUB = "https://www.fifatrainingcentre.com/en/fwc2022/post-match-summaries/post-match-summary-reports.php"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "fifa" / "wc_2022_match_reports"
USER_AGENT = "Mozilla/5.0 (compatible; FAME-2026-academic-research/1.0)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def report_urls(page: str) -> list[str]:
    hrefs = re.findall(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', page, flags=re.I)
    urls = sorted({urllib.parse.urljoin(HUB, html.unescape(href)) for href in hrefs})
    if len(urls) != 64:
        raise RuntimeError(f"Esperados 64 relatórios no hub da FIFA; encontrados {len(urls)}")
    return urls


def filename(url: str, sequence: int) -> str:
    original = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
    match = re.search(r"(?:^|\D)M(\d{1,2})(?:\D|$)", original, flags=re.I)
    number = int(match.group(1)) if match else sequence
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("_")
    return f"match_{number:02d}_{safe}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="baixa novamente arquivos existentes")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    hub_payload = read_url(HUB)
    (OUT / "source_hub.html").write_bytes(hub_payload)
    urls = report_urls(hub_payload.decode("utf-8", errors="replace"))
    records: list[dict[str, object]] = []
    for sequence, url in enumerate(urls, start=1):
        destination = OUT / filename(url, sequence)
        status = "cached"
        if args.force or not destination.exists():
            payload = read_url(url)
            if not payload.startswith(b"%PDF"):
                raise RuntimeError(f"Resposta não é PDF: {url}")
            destination.write_bytes(payload)
            status = "downloaded"
        records.append(
            {
                "source_url": url,
                "relative_path": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "status": status,
            }
        )
        print(f"[{sequence:02d}/64] {destination.name}")

    manifest = OUT / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Concluído: 64 relatórios, manifest em {manifest}")


if __name__ == "__main__":
    main()

