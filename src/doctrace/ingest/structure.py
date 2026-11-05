import re
from dataclasses import dataclass, field

from doctrace.ingest.pdf import Line

ADOPTED = re.compile(r"HAVE ADOPTED THIS REGULATION")
DONE_AT = re.compile(r"^Done at ")
CHAPTER = re.compile(r"^CHAPTER ([IVXLC]+)$")
SECTION = re.compile(r"^SECTION (\d+)$")
ARTICLE = re.compile(r"^Article (\d+)$")
ANNEX = re.compile(r"^ANNEX ([IVXLC]+)$")
RECITAL = re.compile(r"^\((\d+)\)\s*(.*)$")
PARA = re.compile(r"^(\d+)\.\s+(.*)$")
ITEM = re.compile(r"^(\([a-z]{1,5}\)|\(\d+\)|\d+\.|—|-|•)\s")


@dataclass
class Block:
    kind: str
    page: int
    recital: int | None = None
    chapter: str | None = None
    chapter_title: str = ""
    section: str | None = None
    section_title: str = ""
    article: int | None = None
    article_title: str = ""
    para: int | None = None
    annex: str | None = None
    annex_title: str = ""
    lines: list[str] = field(default_factory=list)


def parse_blocks(lines: list[Line]) -> list[Block]:
    blocks: list[Block] = []
    mode = "preamble"
    current: Block | None = None
    next_recital = 1
    next_article = 1
    next_para = 1
    pending: str | None = None
    ctx = dict(chapter=None, chapter_title="", section=None, section_title="")
    article_title = ""
    annex = None
    annex_title = ""

    def start(**kw) -> Block:
        b = Block(**kw)
        blocks.append(b)
        return b

    for line in lines:
        t = line.text

        if mode == "preamble":
            if ADOPTED.search(t):
                mode, current = "body", None
                continue
            m = RECITAL.match(t)
            if m and int(m.group(1)) == next_recital:
                current = start(kind="recital", page=line.page, recital=next_recital)
                next_recital += 1
                if m.group(2):
                    current.lines.append(m.group(2))
            elif current is not None:
                current.lines.append(t)
            continue

        if mode == "body":
            if DONE_AT.match(t):
                mode, current, pending = "tail", None, None
                continue
            if m := CHAPTER.match(t):
                ctx.update(chapter=m.group(1), chapter_title="", section=None, section_title="")
                pending, current = "chapter", None
                continue
            if m := SECTION.match(t):
                ctx.update(section=m.group(1), section_title="")
                pending, current = "section", None
                continue
            m = ARTICLE.match(t)
            if m and int(m.group(1)) == next_article:
                next_article += 1
                next_para = 1
                article_title = ""
                pending = "article"
                current = start(kind="article", page=line.page, article=next_article - 1, **ctx)
                continue
            if pending == "chapter":
                if t.isupper() or not ctx["chapter_title"]:
                    ctx["chapter_title"] = f"{ctx['chapter_title']} {t}".strip()
                    continue
                pending = None
            if pending == "section":
                if line.bold or not ctx["section_title"]:
                    ctx["section_title"] = f"{ctx['section_title']} {t}".strip()
                    continue
                pending = None
            if pending == "article":
                if not article_title or (line.bold and not PARA.match(t)):
                    article_title = f"{article_title} {t}".strip()
                    current.article_title = article_title
                    continue
                pending = None
            if current is None:
                continue
            m = PARA.match(t)
            if m and int(m.group(1)) == next_para:
                if next_para == 1 and not current.lines:
                    current.para = 1
                    current.page = line.page
                    current.lines.append(m.group(2))
                else:
                    current = start(
                        kind="article",
                        page=line.page,
                        article=current.article,
                        article_title=article_title,
                        para=next_para,
                        **ctx,
                    )
                    current.lines.append(m.group(2))
                next_para += 1
                continue
            current.lines.append(t)
            continue

        if mode == "tail":
            if not ANNEX.match(t):
                continue
            mode = "annex"

        if mode == "annex":
            if m := ANNEX.match(t):
                annex, annex_title = m.group(1), ""
                pending = "annex"
                current = start(kind="annex", page=line.page, annex=annex)
                continue
            if pending == "annex":
                if not annex_title or line.bold:
                    annex_title = f"{annex_title} {t}".strip()
                    current.annex_title = annex_title
                    continue
                pending = None
            if current is not None:
                current.lines.append(t)

    return blocks


def _hyphen_words(blocks: list[Block]) -> set[str]:
    words = set()
    for b in blocks:
        for ln in b.lines:
            for w in re.findall(r"\b[\w]+(?:-[\w]+)+\b", ln[:-1] if ln.endswith("-") else ln):
                words.add(w.lower())
    return words


def join_lines(lines: list[str], hyphenated: set[str]) -> str:
    out = ""
    for ln in lines:
        if not out:
            out = ln
            continue
        if ITEM.match(ln):
            out = f"{out}\n{ln}"
            continue
        if out.endswith("-") and ln[:1].islower():
            head = re.search(r"([\w-]+)-$", out).group(1) if re.search(r"([\w-]+)-$", out) else ""
            tail = re.match(r"[\w-]+", ln).group(0)
            if f"{head}-{tail}".lower() in hyphenated:
                out = f"{out}{ln}"
            else:
                out = f"{out[:-1]}{ln}"
            continue
        out = f"{out} {ln}"
    return out.strip()


def block_texts(blocks: list[Block]) -> list[tuple[Block, str]]:
    hyphenated = _hyphen_words(blocks)
    return [(b, join_lines(b.lines, hyphenated)) for b in blocks if b.lines]
