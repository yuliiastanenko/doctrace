import re
from dataclasses import asdict, dataclass

from doctrace.ingest.sources import Source
from doctrace.ingest.structure import Block


@dataclass
class Chunk:
    id: str
    doc_id: str
    kind: str
    ref: str
    heading: str
    text: str
    page: int
    chapter: str | None
    article: int | None
    article_title: str
    paragraphs: list[int]
    annex: str | None

    def to_dict(self) -> dict:
        return asdict(self)


SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(‘“])")


def _pack(pieces: list[str], sep: str, max_chars: int) -> list[str]:
    parts, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + len(sep) > max_chars:
            parts.append(buf)
            buf = piece
        else:
            buf = f"{buf}{sep}{piece}" if buf else piece
    if buf:
        parts.append(buf)
    return parts


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    result = []
    for part in _pack(text.split("\n"), "\n", max_chars):
        if len(part) <= max_chars:
            result.append(part)
        else:
            result.extend(_pack(SENTENCE_END.split(part), " ", max_chars))
    return result


def _para_ref(paras: list[int]) -> str:
    if not paras:
        return ""
    if len(paras) == 1:
        return f"({paras[0]})"
    return f"({paras[0]})-({paras[-1]})"


def build_chunks(src: Source, items: list[tuple[Block, str]], max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}

    def add(kind, ref, heading, text, block: Block, paras: list[int]):
        base = f"{src.doc_id}:{ref.lower().replace(' ', '_')}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        cid = base if n == 0 else f"{base}#{n + 1}"
        chunks.append(
            Chunk(
                id=cid,
                doc_id=src.doc_id,
                kind=kind,
                ref=ref,
                heading=heading,
                text=text,
                page=block.page,
                chapter=block.chapter,
                article=block.article,
                article_title=block.article_title,
                paragraphs=paras,
                annex=block.annex,
            )
        )

    for block, text in items:
        if block.kind == "recital":
            ref = f"Recital {block.recital}"
            for part in _split_long(text, max_chars):
                add("recital", ref, f"{src.short}, {ref}", part, block, [])

    articles: dict[int, list[tuple[Block, str]]] = {}
    for block, text in items:
        if block.kind == "article":
            articles.setdefault(block.article, []).append((block, text))

    for num, parts in articles.items():
        first = parts[0][0]
        ctx = f"{src.short}"
        if first.chapter:
            ctx += f", Chapter {first.chapter} {first.chapter_title}"
        if first.section:
            ctx += f", Section {first.section} {first.section_title}"
        title = f"Article {num}: {first.article_title}"
        group: list[tuple[Block, str]] = []

        def flush():
            if not group:
                return
            paras = [b.para for b, _ in group if b.para]
            ref = f"Article {num}{_para_ref(paras)}"
            body = "\n".join(t if not b.para else f"{b.para}. {t}" for b, t in group)
            for piece in _split_long(body, max_chars):
                add("article", ref, f"{ctx}, {title}", piece, group[0][0], paras)
            group.clear()

        size = 0
        for block, text in parts:
            if group and size + len(text) > max_chars:
                flush()
                size = 0
            group.append((block, text))
            size += len(text)
        flush()

    for block, text in items:
        if block.kind == "annex":
            ref = f"Annex {block.annex}"
            heading = f"{src.short}, {ref}: {block.annex_title}"
            for part in _split_long(text, max_chars):
                add("annex", ref, heading, part, block, [])

    return chunks
