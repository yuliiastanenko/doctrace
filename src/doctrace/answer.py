import json
import time
from dataclasses import dataclass, field

from doctrace.llm import Completion, complete
from doctrace.retrieve import Hit, retrieve
from doctrace.verify import ModelAnswer, Verification, parse_answer, verify

SYSTEM = """You answer questions about EU regulations using ONLY the passages provided by the user.

Reply with a single JSON object and nothing else:
{
  "found": true or false,
  "answer": "the answer, in the same language as the question",
  "citations": [{"chunk_id": "id of a passage", "quote": "exact words copied from that passage"}],
  "confidence": "high", "medium" or "low"
}

Rules:
- Every statement in the answer must be supported by at least one citation.
- A quote must be copied character for character from the passage text, in English, as one continuous span of 5 to 40 words. Never paraphrase, translate or shorten a quote.
- chunk_id must be exactly one of the ids in square brackets.
- If the passages do not contain the answer, reply {"found": false, "answer": "", "citations": [], "confidence": "low"}. Do not use outside knowledge.
"""

RETRY = """Some citations failed verification:
{problems}

Reply again with the full JSON object. Copy every quote exactly from the passage text, or remove the statement it supports. If the passages do not answer the question, set "found" to false."""


@dataclass
class Attempt:
    raw: str
    completion: Completion
    answer: ModelAnswer | None
    verification: Verification | None
    error: str = ""


@dataclass
class AnswerResult:
    question: str
    status: str
    answer: str
    citations: list[dict]
    confidence: str
    hits: list[Hit]
    attempts: list[Attempt]
    timings: dict = field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        return sum(a.completion.input_tokens for a in self.attempts)

    @property
    def output_tokens(self) -> int:
        return sum(a.completion.output_tokens for a in self.attempts)


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.id}] {h.heading}\n{h.text}" for h in hits)


def _problems(attempt: Attempt) -> str:
    if attempt.error:
        return f"- {attempt.error}"
    lines = [f"- {e}" for e in attempt.verification.errors]
    lines += [f"- [{c.chunk_id}] \"{c.quote}\": {c.reason}" for c in attempt.verification.failed]
    return "\n".join(lines)


def _attempt(messages: list[dict], passages: dict[str, str], provider, model) -> Attempt:
    completion = complete(messages, provider=provider, model=model)
    try:
        answer = parse_answer(completion.text)
    except ValueError as e:
        return Attempt(completion.text, completion, None, None, str(e))
    return Attempt(completion.text, completion, answer, verify(answer, passages))


def ask(
    question: str,
    mode: str = "rerank",
    k: int = 5,
    provider: str | None = None,
    model: str | None = None,
    max_retries: int = 1,
    hits: list[Hit] | None = None,
) -> AnswerResult:
    timings = {}
    if hits is None:
        start = time.perf_counter()
        result = retrieve(question, mode=mode, k=k)
        hits = result.hits
        timings["retrieve"] = time.perf_counter() - start

    passages = {h.id: h.text for h in hits}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"},
    ]

    attempts: list[Attempt] = []
    start = time.perf_counter()
    for _ in range(max_retries + 1):
        attempt = _attempt(messages, passages, provider, model)
        attempts.append(attempt)
        if attempt.verification and attempt.verification.ok:
            break
        messages += [
            {"role": "assistant", "content": attempt.raw},
            {"role": "user", "content": RETRY.format(problems=_problems(attempt))},
        ]
    timings["llm"] = time.perf_counter() - start

    final = attempts[-1]
    if final.verification is None or not final.verification.ok:
        return AnswerResult(question, "rejected", "", [], "low", hits, attempts, timings)
    if not final.answer.found:
        return AnswerResult(question, "not_found", "", [], final.answer.confidence, hits, attempts, timings)
    citations = [c.model_dump() for c in final.answer.citations]
    return AnswerResult(question, "answered", final.answer.answer, citations, final.answer.confidence, hits, attempts, timings)


MESSAGES = {
    "not_found": "Not found in the documents.",
    "rejected": "Could not produce an answer with verified citations.",
}


def print_answer(result: AnswerResult, verbose: bool = False) -> None:
    refs = {h.id: (h.heading, h.page) for h in result.hits}
    print(f"status: {result.status}")
    if result.status == "answered":
        print(f"confidence: {result.confidence}\n")
        print(result.answer)
        print("\nsources:")
        for i, c in enumerate(result.citations, 1):
            heading, page = refs.get(c["chunk_id"], ("", "?"))
            print(f"  [{i}] {heading} (page {page})")
            print(f"      \"{c['quote']}\"")
    else:
        print(f"\n{MESSAGES[result.status]}")

    for n, a in enumerate(result.attempts, 1):
        if a.error:
            print(f"\nattempt {n}: unreadable output, {a.error}")
        elif a.verification and not a.verification.ok:
            print(f"\nattempt {n}: verification failed")
            for e in a.verification.errors:
                print(f"  - {e}")
            for c in a.verification.failed:
                print(f"  - [{c.chunk_id}] {c.reason}: \"{c.quote}\"")
        if verbose:
            print(f"\nattempt {n} raw output:\n{a.raw}")

    model = result.attempts[-1].completion.model
    detail = ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items())
    print(f"\nmodel: {model}, attempts: {len(result.attempts)}, tokens: {result.input_tokens} in / {result.output_tokens} out")
    print(f"time: {sum(result.timings.values()):.2f}s ({detail})")


def to_json(result: AnswerResult) -> str:
    return json.dumps(
        {
            "question": result.question,
            "status": result.status,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "retrieved": [h.id for h in result.hits],
            "attempts": len(result.attempts),
            "tokens": {"input": result.input_tokens, "output": result.output_tokens},
            "timings": result.timings,
        },
        ensure_ascii=False,
        indent=2,
    )
