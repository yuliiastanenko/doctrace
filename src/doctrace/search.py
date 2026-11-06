from dataclasses import dataclass

from doctrace.config import collection_name
from doctrace.db import connect
from doctrace.embed import embed_query

VECTOR_SQL = """
SELECT id, doc_id, ref, heading, text, page, 1 - (embedding <=> %(q)s) AS score
FROM chunks
WHERE collection = %(collection)s
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


@dataclass
class Hit:
    id: str
    doc_id: str
    ref: str
    heading: str
    text: str
    page: int
    score: float


def vector_search(query: str, model: str, max_chars: int, k: int = 5) -> list[Hit]:
    q = embed_query(query, model)
    with connect() as conn:
        rows = conn.execute(
            VECTOR_SQL,
            {"q": q, "collection": collection_name(model, max_chars), "k": k},
        ).fetchall()
    return [Hit(*row) for row in rows]


def print_hits(hits: list[Hit], preview: int = 300) -> None:
    if not hits:
        print("no results, did you run `uv run doctrace index`?")
    for i, h in enumerate(hits, 1):
        text = h.text.replace("\n", " ")
        print(f"\n{i}. [{h.score:.3f}] {h.id}  (page {h.page})")
        print(f"   {h.heading}")
        print(f"   {text[:preview]}{'...' if len(text) > preview else ''}")
