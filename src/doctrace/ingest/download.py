import httpx

from doctrace.config import settings
from doctrace.ingest.sources import SOURCES


def download_all(force: bool = False) -> None:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        path = settings.raw_dir / f"{src.doc_id}.pdf"
        if path.exists() and not force:
            print(f"{src.doc_id}: already downloaded ({path.stat().st_size // 1024} KB)")
            continue
        try:
            r = httpx.get(
                src.url,
                follow_redirects=True,
                timeout=120,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            ok = r.status_code == 200 and r.content.startswith(b"%PDF")
        except httpx.HTTPError as e:
            ok, r = False, e
        if not ok:
            print(f"{src.doc_id}: automatic download failed.")
            print(f"  Open this link in a browser and save the PDF as {path}")
            print(f"  {src.url}")
            continue
        path.write_bytes(r.content)
        print(f"{src.doc_id}: saved {len(r.content) // 1024} KB")
