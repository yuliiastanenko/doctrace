import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from doctrace.config import ROOT
from doctrace.verify import normalize

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
RUNS_DIR = ROOT / "eval" / "runs"
REF = re.compile(r"^(ai_act|gdpr):(article|recital|annex)_([0-9]+|[ivxlc]+)(?:\((\d+)\))?$")
KS = (1, 3, 5, 10)
DEPTH = 20


@dataclass
class Expected:
    ref: str
    contains: str = ""

    def __post_init__(self):
        m = REF.match(self.ref)
        if not m:
            raise ValueError(f"bad reference {self.ref!r}, use e.g. ai_act:article_99(3), gdpr:recital_65, ai_act:annex_iii")
        self.doc, self.kind, num, para = m.groups()
        self.num = num.upper() if self.kind == "annex" else int(num)
        self.para = int(para) if para else None

    def matches(self, chunk) -> bool:
        get = chunk.get if isinstance(chunk, dict) else lambda k: getattr(chunk, k)
        if get("doc_id") != self.doc or get("kind") != self.kind:
            return False
        if self.kind == "article":
            if get("article") != self.num:
                return False
            if self.para is not None and get("paragraphs") and self.para not in get("paragraphs"):
                return False
        elif self.kind == "annex" and get("annex") != self.num:
            return False
        elif self.kind == "recital" and get("ref") != f"Recital {self.num}":
            return False
        return not self.contains or normalize(self.contains) in normalize(get("text"))


@dataclass
class Question:
    id: str
    question: str
    type: str
    lang: str = "en"
    expected: list[Expected] = field(default_factory=list)
    answer_note: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.expected)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = []
    for item in raw:
        expected = [Expected(**e) if isinstance(e, dict) else Expected(ref=e) for e in item.pop("expected", [])]
        questions.append(Question(expected=expected, **item))
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return questions


def first_match_rank(hits, expected: list[Expected]) -> int | None:
    for rank, h in enumerate(hits, 1):
        if any(e.matches(h) for e in expected):
            return rank
    return None


def coverage(hits, expected: list[Expected]) -> float:
    return sum(any(e.matches(h) for h in hits) for e in expected) / len(expected)


def check_labels(chunks: list[dict], questions: list[Question]) -> list[str]:
    problems = []
    for q in questions:
        for e in q.expected:
            found = [c for c in chunks if e.matches(c)]
            if not found:
                problems.append(f"{q.id}: no chunk matches {e.ref} contains={e.contains!r}")
    return problems


def write_review(chunks: list[dict], questions: list[Question], path: Path) -> None:
    lines = ["# DocTrace eval questions: review sheet", "", "Check every question against the passage. Tick it when the expected reference really answers it.", ""]
    for q in questions:
        lines += [f"## {q.id} · {q.type} · {q.lang}", "", f"- [ ] **{q.question}**", ""]
        if not q.answerable:
            lines += ["Expected: not found in the documents.", ""]
            continue
        if q.answer_note:
            lines += [f"Expected answer: {q.answer_note}", ""]
        for e in q.expected:
            match = next((c for c in chunks if e.matches(c)), None)
            lines.append(f"**{e.ref}**" + (f" (contains: “{e.contains}”)" if e.contains else ""))
            if match:
                text = match["text"]
                if e.contains:
                    i = normalize(text).find(normalize(e.contains))
                    text = text[max(0, i - 300) : i + len(e.contains) + 300]
                lines += ["", f"> {match['heading']}, page {match['page']}", ">", "> " + text.replace("\n", "\n> "), ""]
            else:
                lines += ["", "> NO MATCHING CHUNK", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _pct(values) -> float:
    values = list(values)
    return round(100 * sum(values) / len(values), 1) if values else 0.0


def retrieval_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    out = {"questions": len(answerable)}
    for k in KS:
        out[f"hit@{k}"] = _pct(r["rank"] is not None and r["rank"] <= k for r in answerable)
    out["coverage@5"] = round(100 * statistics.mean(r["coverage@5"] for r in answerable), 1) if answerable else 0.0
    out["mrr"] = round(statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in answerable), 3) if answerable else 0.0
    out["latency_p50"] = round(statistics.median(r["retrieve_s"] for r in rows), 2)
    return out


def answer_metrics(rows: list[dict]) -> dict:
    rows = [r for r in rows if "status" in r]
    if not rows:
        return {}
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    first = [c for r in rows for c in r["first_attempt_citations"]]
    return {
        "answered": _pct(r["status"] == "answered" for r in answerable),
        "cites_expected": _pct(r["cites_expected"] for r in answerable),
        "false_refusal": _pct(r["status"] != "answered" for r in answerable),
        "correct_refusal": _pct(r["status"] == "not_found" for r in unanswerable),
        "wrongly_answered": _pct(r["status"] == "answered" for r in unanswerable),
        "first_try_citation_valid": _pct(first),
        "retried": _pct(r["attempts"] > 1 for r in rows),
        "rejected": _pct(r["status"] == "rejected" for r in rows),
        "llm_p50_s": round(statistics.median(r["llm_s"] for r in rows), 2),
        "tokens_avg": round(statistics.mean(r["tokens"] for r in rows)),
    }


def run(
    name: str,
    mode: str = "rerank",
    k: int = 5,
    with_answers: bool = False,
    only: list[str] | None = None,
    delay: float = 0.0,
    model: str | None = None,
    max_chars: int | None = None,
    rerank_any_language: bool = False,
) -> dict:
    from doctrace.answer import ask
    from doctrace.retrieve import retrieve

    questions = load_questions()
    if only:
        questions = [q for q in questions if q.id in only]
    rows = []
    for n, q in enumerate(questions, 1):
        start = time.perf_counter()
        result = retrieve(q.question, mode=mode, k=DEPTH, model=model, max_chars=max_chars, rerank_any_language=rerank_any_language)
        row = {
            "id": q.id,
            "type": q.type,
            "lang": q.lang,
            "answerable": q.answerable,
            "retrieved": [h.id for h in result.hits[:k]],
            "retrieve_s": round(time.perf_counter() - start, 3),
            "rank": first_match_rank(result.hits, q.expected) if q.answerable else None,
            "coverage@5": coverage(result.hits[:5], q.expected) if q.answerable else None,
        }
        if with_answers:
            a = ask(q.question, hits=result.hits[:k])
            first = a.attempts[0]
            row.update(
                status=a.status,
                attempts=len(a.attempts),
                answer=a.answer,
                citations=a.citations,
                first_attempt_citations=[c.ok for c in first.verification.checks] if first.verification else [False],
                cites_expected=a.status == "answered" and any(e.matches(h) for e in q.expected for h in result.hits[:k] if h.id in {c["chunk_id"] for c in a.citations}),
                llm_s=round(a.timings["llm"], 2),
                tokens=a.input_tokens + a.output_tokens,
            )
            if delay:
                time.sleep(delay)
        rows.append(row)
        status = f" {row['status']}" if with_answers else ""
        rank = row["rank"] if q.answerable else "-"
        print(f"[{n}/{len(questions)}] {q.id} rank={rank}{status}", flush=True)

    report = {
        "name": name,
        "config": {"mode": mode, "k": k, "model": model, "max_chars": max_chars, "rerank_any_language": rerank_any_language, "with_answers": with_answers},
        "retrieval": retrieval_metrics(rows),
        "answers": answer_metrics(rows),
        "rows": rows,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_report(report: dict, questions: list[Question] | None = None) -> None:
    by_id = {q.id: q for q in (questions or load_questions())}
    print(f"\n=== {report['name']} ===")
    print("config:", ", ".join(f"{k}={v}" for k, v in report["config"].items() if v is not None))
    print("\nretrieval:")
    for k, v in report["retrieval"].items():
        print(f"  {k:<26}{v}")
    if report["answers"]:
        print("\nanswers:")
        for k, v in report["answers"].items():
            print(f"  {k:<26}{v}")
    misses = [r for r in report["rows"] if r["answerable"] and (r["rank"] is None or r["rank"] > 5)]
    if misses:
        print("\nexpected passage not in top 5:")
        for r in misses:
            where = f"rank {r['rank']}" if r["rank"] else f"not in top {DEPTH}"
            print(f"  {r['id']} ({where}): {by_id[r['id']].question}")
    wrong = [r for r in report["rows"] if r.get("status") and (r["answerable"] != (r["status"] == "answered"))]
    if wrong:
        print("\nwrong answer status:")
        for r in wrong:
            expected = "answer" if r["answerable"] else "not_found"
            print(f"  {r['id']} expected {expected}, got {r['status']}: {by_id[r['id']].question}")


def compare(names: list[str]) -> None:
    reports = [json.loads((RUNS_DIR / f"{n}.json").read_text(encoding="utf-8")) for n in names]
    keys = ["hit@1", "hit@5", "hit@10", "coverage@5", "mrr", "latency_p50"]
    answer_keys = [k for k in ["cites_expected", "false_refusal", "correct_refusal", "first_try_citation_valid"] if all(k in r["answers"] for r in reports)]
    header = ["run", *keys, *answer_keys]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in reports:
        values = [r["name"], *(str(r["retrieval"][k]) for k in keys), *(str(r["answers"][k]) for k in answer_keys)]
        print("| " + " | ".join(values) + " |")
