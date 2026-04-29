"""Batch-download Avalanche Canada ATES KMZ files.

Enumerates numeric IDs via HEAD requests (the API has no list endpoint).
Area names are read from the Content-Disposition header of each hit.
Each KMZ is saved to data/validation/<slug>/layers.kmz.

Usage (from repo root):
    uv run python scripts/fetch_avcan_areas.py
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_KMZ_URL = "https://avcan-services-api.prod.avalanche.ca/ates/en/areas/{area_id}.kmz"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "ice-autoATES/0.1 (research; github.com/jacKlinc)"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _probe_id(area_id: int) -> dict | None:
    """Streaming GET: read headers only, return {id, name} on 200, else None."""
    url = _KMZ_URL.format(area_id=area_id)
    try:
        r = _SESSION.get(url, timeout=10, stream=True)
        r.close()
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None

    cd = r.headers.get("Content-Disposition", "")
    m = re.search(r'filename="([^"]+)\.kmz"', cd, re.IGNORECASE)
    name = m.group(1) if m else str(area_id)
    return {"id": area_id, "name": name}


def _enumerate_ids(max_id: int, workers: int) -> list[dict]:
    hits: list[dict] = []
    ids = list(range(1, max_id + 1))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe_id, i): i for i in ids}
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result:
                hits.append(result)
                print(f"  [{done:>{len(str(max_id))}}/{max_id}] hit  id={result['id']:<6} {result['name']}")
            elif done % 200 == 0:
                print(f"  [{done:>{len(str(max_id))}}/{max_id}] …")
    return sorted(hits, key=lambda a: a["id"])


def _download_kmz(area: dict, out_dir: Path) -> Path | None:
    slug = _slugify(area["name"]) or str(area["id"])
    dest = out_dir / slug / "layers.kmz"

    if dest.exists():
        print(f"  skip  {slug}  (already present)")
        return dest

    url = _KMZ_URL.format(area_id=area["id"])
    try:
        r = _SESSION.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ERROR {slug}: {exc}")
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    print(f"  saved {dest}  ({len(r.content) // 1024} KB)")
    return dest


MAX_ID = 2000
WORKERS = 8
OUT_DIR = Path("data/validation")


def main() -> None:
    print("=== Avalanche Canada ATES area fetcher ===")
    print(f"Probing IDs 1–{MAX_ID} with {WORKERS} workers …\n")

    t0 = time.monotonic()
    areas = _enumerate_ids(MAX_ID, WORKERS)
    elapsed = time.monotonic() - t0

    print(f"\nFound {len(areas)} area(s) in {elapsed:.1f}s.")

    if not areas:
        print("Try increasing MAX_ID.")
        return

    print(f"\nDownloading KMZ files → {OUT_DIR} …")
    t1 = time.monotonic()
    saved = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_download_kmz, a, OUT_DIR): a for a in areas}
        for fut in as_completed(futures):
            if fut.result():
                saved += 1

    print(f"\nDone. {saved}/{len(areas)} saved in {time.monotonic() - t1:.1f}s.")


if __name__ == "__main__":
    main()
