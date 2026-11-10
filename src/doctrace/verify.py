import json
import re
import unicodedata
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

MIN_QUOTE_WORDS = 4
ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
TRANSLATE = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", " ": " "})


class Citation(BaseModel):
    chunk_id: str
    quote: str


class ModelAnswer(BaseModel):
    found: bool
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = "low"


@dataclass
class CitationCheck:
    chunk_id: str
    quote: str
    ok: bool
    reason: str = ""


@dataclass
class Verification:
    ok: bool
    checks: list[CitationCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[CitationCheck]:
        return [c for c in self.checks if not c.ok]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(TRANSLATE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _strip_edges(quote: str) -> str:
    return quote.strip().strip("\"'").strip().strip(".,;:").strip()


def quote_in_text(quote: str, text: str) -> bool:
    haystack = normalize(text)
    position = 0
    parts = [normalize(_strip_edges(p)) for p in ELLIPSIS.split(quote)]
    parts = [p for p in parts if p]
    if not parts:
        return False
    for part in parts:
        found = haystack.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def parse_answer(raw: str) -> ModelAnswer:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the model output")
        text = text[start : end + 1]
    try:
        return ModelAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"invalid JSON answer: {e}") from e


def verify(answer: ModelAnswer, passages: dict[str, str]) -> Verification:
    if not answer.found:
        return Verification(ok=True)
    errors = []
    if not answer.answer.strip():
        errors.append("found is true but the answer is empty")
    if not answer.citations:
        errors.append("found is true but there are no citations")
    checks = []
    for c in answer.citations:
        if c.chunk_id not in passages:
            checks.append(CitationCheck(c.chunk_id, c.quote, False, "chunk_id is not one of the provided passages"))
        elif len(_strip_edges(ELLIPSIS.sub(" ", c.quote)).split()) < MIN_QUOTE_WORDS:
            checks.append(CitationCheck(c.chunk_id, c.quote, False, f"quote is shorter than {MIN_QUOTE_WORDS} words"))
        elif not quote_in_text(c.quote, passages[c.chunk_id]):
            checks.append(CitationCheck(c.chunk_id, c.quote, False, "quote is not found verbatim in the passage"))
        else:
            checks.append(CitationCheck(c.chunk_id, c.quote, True))
    ok = not errors and all(c.ok for c in checks)
    return Verification(ok=ok, checks=checks, errors=errors)
