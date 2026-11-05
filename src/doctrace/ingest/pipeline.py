import json
from collections import Counter

from doctrace.config import settings
from doctrace.ingest.chunk import Chunk, build_chunks
from doctrace.ingest.pdf import extract_lines
from doctrace.ingest.sources import SOURCES
from doctrace.ingest.structure import block_texts, parse_blocks


def chunks_path(max_chars: int):
    return settings.processed_dir / f"chunks_{max_chars}.jsonl"


def parse_all(max_chars: int) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for src in SOURCES:
        pdf = settings.raw_dir / f"{src.doc_id}.pdf"
        if not pdf.exists():
            print(f"{src.doc_id}: {pdf} not found, run `uv run doctrace download` first")
            continue
        lines = extract_lines(pdf)
        blocks = parse_blocks(lines)
        chunks = build_chunks(src, block_texts(blocks), max_chars)
        all_chunks.extend(chunks)
        report(src, blocks, chunks)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out = chunks_path(max_chars)
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_chunks)} chunks to {out}")
    return all_chunks


def _roman(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = values[a]
        total += -v if values.get(b, 0) > v else v
    return total


def report(src, blocks, chunks) -> None:
    recitals = {b.recital for b in blocks if b.kind == "recital"}
    articles = {b.article for b in blocks if b.kind == "article"}
    annexes = sorted({b.annex for b in blocks if b.kind == "annex"}, key=_roman)
    kinds = Counter(c.kind for c in chunks)
    lengths = [len(c.text) for c in chunks]
    untitled = sorted(b.article for b in blocks if b.kind == "article" and not b.article_title)

    def mark(found, expected):
        return "ok" if found == expected else f"MISMATCH, expected {expected}"

    print(f"\n== {src.short} ==")
    print(f"recitals: {len(recitals)} ({mark(len(recitals), src.expected_recitals)})")
    print(f"articles: {len(articles)} ({mark(len(articles), src.expected_articles)})")
    print(f"annexes:  {len(annexes)} {' '.join(annexes)}")
    print(f"chunks:   {len(chunks)} {dict(kinds)}")
    if lengths:
        print(f"chars per chunk: avg {sum(lengths) // len(lengths)}, max {max(lengths)}")
    if untitled:
        print(f"articles without title: {untitled[:10]}")


def load_chunks(max_chars: int) -> list[dict]:
    path = chunks_path(max_chars)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def show(query: str, max_chars: int) -> None:
    q = query.lower()
    doc, _, ref = q.rpartition(":") if ":" in q and " " in q else ("", "", q)
    found = []
    for c in load_chunks(max_chars):
        r = c["ref"].lower()
        if doc and c["doc_id"] != doc:
            continue
        cid = c["id"].lower()
        if cid == q or cid.startswith((f"{q}(", f"{q}#")) or r == ref or r.startswith(f"{ref}("):
            found.append(c)
    for c in found[:5]:
        print(f"\n--- {c['id']} (page {c['page']}) ---")
        print(c["heading"])
        print()
        print(c["text"])
    if not found:
        print("nothing found")
    elif len(found) > 5:
        print(f"\n... and {len(found) - 5} more")
