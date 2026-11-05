import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

NOISE = re.compile(
    r"^("
    r"EN|L series|Official Journal|"
    r"OJ L,? .*|ELI: .*|\d+/\d+|L \d+/\d+|"
    r".*Official Journal of the European Union.*|"
    r"\d{4}/\d+\s+\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\.\d{1,2}\.\d{4}"
    r")$"
)

MARGIN = 0.055
FOOTNOTE = re.compile(r"^\(\s*\d*\s*\)\s")
EMPTY_REF = re.compile(r"\s?\(\s*\)")
LEADING_ITEM = re.compile(r"^(\((?:\d{1,3}|[a-z]{1,5})\))(?=[^\s,.;:)])")


@dataclass
class Line:
    text: str
    page: int
    bold: bool


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("­", "").replace("‑", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return LEADING_ITEM.sub(r"\1 ", text)


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "Bold" in span["font"]


def _body_size(doc: pymupdf.Document) -> float:
    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0]


def extract_lines(path: Path) -> list[Line]:
    doc = pymupdf.open(path)
    body = _body_size(doc)
    min_size = body - 1.2
    result: list[Line] = []
    for page in doc:
        height = page.rect.height
        raw = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                x0, y0, _, y1 = line["bbox"]
                if y1 < height * MARGIN or y0 > height * (1 - MARGIN):
                    continue
                spans = [s for s in line["spans"] if s["size"] >= min_size and s["text"].strip()]
                if spans:
                    raw.append((y0, x0, spans))
        raw.sort(key=lambda r: (round(r[0]), r[1]))
        merged: list[tuple[float, list[tuple[float, list[dict]]]]] = []
        for y0, x0, spans in raw:
            if merged and abs(merged[-1][0] - y0) < 2.5:
                merged[-1][1].append((x0, spans))
            else:
                merged.append((y0, [(x0, spans)]))
        for _, groups in merged:
            groups.sort(key=lambda g: g[0])
            spans = [s for _, g in groups for s in g]
            text = _clean(" ".join("".join(s["text"] for s in g) for _, g in groups))
            if FOOTNOTE.match(text) and max(s["size"] for s in spans) < body - 0.5:
                break
            text = EMPTY_REF.sub("", text)
            if not text or NOISE.match(text):
                continue
            result.append(Line(text=text, page=page.number + 1, bold=all(_is_bold(s) for s in spans)))
    return result
