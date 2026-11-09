import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from doctrace.config import collection_name, settings
from doctrace.db import connect
from doctrace.embed import embed_query

MODES = ("vector", "fulltext", "hybrid", "rerank")
RRF_K = 60
WEIGHTS = {"vector": 1.0, "fulltext": 1.0, "ref": 3.0}

COLUMNS = "id, doc_id, ref, heading, text, page"

VECTOR_SQL = f"""
SELECT {COLUMNS}, 1 - (embedding <=> %(q)s) AS score
FROM chunks
WHERE collection = %(collection)s
ORDER BY embedding <=> %(q)s
LIMIT %(n)s
"""

FULLTEXT_SQL = f"""
WITH query AS (
    SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS any_terms,
           to_tsquery('simple', string_agg(quote_literal(lexeme), ' & ')) AS all_terms
    FROM unnest(to_tsvector('english', %(q)s))
)
SELECT {COLUMNS}, (tsv @@ query.all_terms)::int + ts_rank_cd(tsv, query.any_terms, 32) AS score
FROM chunks, query
WHERE collection = %(collection)s AND query.any_terms IS NOT NULL AND tsv @@ query.any_terms
ORDER BY score DESC
LIMIT %(n)s
"""

REF_SQL = f"""
SELECT {COLUMNS}, coalesce(%(para)s::int = ANY(paragraphs), false)::int AS score
FROM chunks
WHERE collection = %(collection)s
  AND kind = %(kind)s
  AND (%(doc)s::text IS NULL OR doc_id = %(doc)s)
  AND CASE kind
        WHEN 'article' THEN article = %(num)s::int
        WHEN 'annex' THEN annex = %(roman)s
        ELSE ref = 'Recital ' || %(num)s
      END
ORDER BY score DESC, page, coalesce(paragraphs[1], 0), id
LIMIT 12
"""

REF_PATTERN = re.compile(
    r"\b(articles?|art\.|annex(?:es)?|recitals?|стать\w*|приложени\w*|преамбул\w*)"
    r"\s*(\d{1,3}|[IVXLC]{1,6})\b(?:\s*\((\d{1,2})\))?",
    re.IGNORECASE,
)
KIND_BY_WORD = {"art": "article", "ста": "article", "ann": "annex", "при": "annex", "rec": "recital", "пре": "recital"}
DOC_HINTS = {
    "gdpr": re.compile(r"gdpr|general data protection|2016/679", re.IGNORECASE),
    "ai_act": re.compile(r"ai act|artificial intelligence act|2024/1689|акт\w* об ии|закон\w* об ии", re.IGNORECASE),
}


@dataclass
class Hit:
    id: str
    doc_id: str
    ref: str
    heading: str
    text: str
    page: int
    ranks: dict = field(default_factory=dict)
    fused: float = 0.0
    rerank: float | None = None
    pinned: bool = False


@dataclass
class Result:
    hits: list[Hit]
    timings: dict
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Reference:
    kind: str
    num: str
    para: int | None


def _int_to_roman(n: int) -> str:
    out = ""
    for value, sym in [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= value:
            out += sym
            n -= value
    return out


def parse_references(query: str) -> list[Reference]:
    refs = []
    for word, num, para in REF_PATTERN.findall(query):
        kind = KIND_BY_WORD[word[:3].lower()]
        if kind == "annex":
            num = num.upper() if not num.isdigit() else _int_to_roman(int(num))
        elif not num.isdigit():
            continue
        ref = Reference(kind, num, int(para) if para else None)
        if ref not in refs:
            refs.append(ref)
    return refs


def is_english(query: str) -> bool:
    letters = [c for c in query if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.7


def doc_hint(query: str) -> str | None:
    found = [doc for doc, pattern in DOC_HINTS.items() if pattern.search(query)]
    return found[0] if len(found) == 1 else None


def _rows_to_hits(rows) -> list[tuple[Hit, float]]:
    return [(Hit(*row[:6]), row[6]) for row in rows]


def vector_candidates(conn, query: str, model: str, collection: str, n: int):
    q = embed_query(query, model)
    return _rows_to_hits(conn.execute(VECTOR_SQL, {"q": q, "collection": collection, "n": n}).fetchall())


def fulltext_candidates(conn, query: str, collection: str, n: int):
    return _rows_to_hits(conn.execute(FULLTEXT_SQL, {"q": query, "collection": collection, "n": n}).fetchall())


def reference_candidates(conn, query: str, collection: str):
    doc = doc_hint(query)
    hits = []
    for ref in parse_references(query):
        params = {
            "collection": collection,
            "kind": ref.kind,
            "doc": doc,
            "num": ref.num if ref.kind != "annex" else None,
            "roman": ref.num if ref.kind == "annex" else None,
            "para": ref.para,
        }
        hits.extend(_rows_to_hits(conn.execute(REF_SQL, params).fetchall()))
    return hits


def fuse(lists: dict[str, list[tuple[Hit, float]]]) -> list[Hit]:
    merged: dict[str, Hit] = {}
    for source, items in lists.items():
        for rank, (hit, _) in enumerate(items, 1):
            h = merged.setdefault(hit.id, hit)
            h.ranks[source] = rank
            h.fused += WEIGHTS[source] / (RRF_K + rank)
            if source == "ref":
                h.pinned = True
    return sorted(merged.values(), key=lambda h: h.fused, reverse=True)


@lru_cache
def _reranker(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return TextCrossEncoder(name, cache_dir=str(settings.models_dir))


def rerank(query: str, hits: list[Hit], model: str) -> list[Hit]:
    if not hits:
        return hits
    scores = list(_reranker(model).rerank(query, [f"{h.heading}\n{h.text}" for h in hits]))
    for h, s in zip(hits, scores):
        h.rerank = float(s)
    return sorted(hits, key=lambda h: (h.pinned, h.rerank), reverse=True)


def retrieve(
    query: str,
    mode: str = "rerank",
    k: int = 5,
    model: str | None = None,
    max_chars: int | None = None,
    candidates: int = 30,
    rerank_pool: int = 20,
    reranker: str | None = None,
    rerank_any_language: bool = False,
) -> Result:
    model = model or settings.embedding_model
    max_chars = max_chars or settings.chunk_max_chars
    reranker = reranker or settings.reranker_model
    collection = collection_name(model, max_chars)
    timings = {}

    with connect() as conn:
        lists = {}
        if mode != "fulltext":
            start = time.perf_counter()
            lists["vector"] = vector_candidates(conn, query, model, collection, candidates)
            timings["vector"] = time.perf_counter() - start
        if mode != "vector":
            start = time.perf_counter()
            lists["fulltext"] = fulltext_candidates(conn, query, collection, candidates)
            timings["fulltext"] = time.perf_counter() - start
        if mode in ("hybrid", "rerank"):
            start = time.perf_counter()
            lists["ref"] = reference_candidates(conn, query, collection)
            timings["ref"] = time.perf_counter() - start

    hits = fuse(lists)
    notes = []
    if mode == "rerank" and (rerank_any_language or is_english(query)):
        start = time.perf_counter()
        hits = rerank(query, hits[:rerank_pool], reranker)
        timings["rerank"] = time.perf_counter() - start
    elif mode in ("hybrid", "rerank"):
        if mode == "rerank":
            notes.append("rerank skipped: the reranker is English-only, using hybrid order")
        hits.sort(key=lambda h: (h.pinned, h.fused), reverse=True)

    return Result(hits=hits[:k], timings=timings, notes=notes)


def print_result(result: Result, preview: int = 220) -> None:
    for note in result.notes:
        print(f"note: {note}")
    if not result.hits:
        print("no results")
    for i, h in enumerate(result.hits, 1):
        parts = [f"{src} #{rank}" for src, rank in h.ranks.items()]
        if h.rerank is not None:
            parts.append(f"rerank {h.rerank:.2f}")
        text = h.text.replace("\n", " ")
        print(f"\n{i}. {h.id}  (page {h.page})  [{' · '.join(parts)}]")
        print(f"   {h.heading}")
        print(f"   {text[:preview]}{'...' if len(text) > preview else ''}")
    total = sum(result.timings.values())
    detail = ", ".join(f"{name} {sec:.2f}s" for name, sec in result.timings.items())
    print(f"\ntime: {total:.2f}s ({detail})")
