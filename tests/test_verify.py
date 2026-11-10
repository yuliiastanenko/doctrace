import pytest

from doctrace.verify import ModelAnswer, parse_answer, quote_in_text, verify

PASSAGE = (
    "1. Providers of high-risk AI systems shall:\n"
    "(a) ensure that their high-risk AI systems are compliant with the requirements set out in Section 2;\n"
    "(b) indicate on the high-risk AI system their name, registered trade name or registered trade mark;"
)
PASSAGES = {"ai_act:article_16": PASSAGE}


def answer(*citations, found=True, text="Providers must ensure compliance."):
    return ModelAnswer(
        found=found,
        answer=text,
        citations=[{"chunk_id": c, "quote": q} for c, q in citations],
    )


def test_exact_quote_passes():
    assert quote_in_text("ensure that their high-risk AI systems are compliant", PASSAGE)


def test_whitespace_case_and_curly_quotes_are_normalized():
    text = "the ‘real-time’ remote\nbiometric identification system"
    assert quote_in_text("The 'real-time'   remote biometric identification", text)


def test_paraphrase_fails():
    assert not quote_in_text("ensure their AI systems comply with Section 2", PASSAGE)


def test_ellipsis_parts_must_appear_in_order():
    assert quote_in_text("Providers of high-risk AI systems shall ... indicate on the high-risk AI system", PASSAGE)
    assert not quote_in_text("indicate on the high-risk AI system ... Providers of high-risk AI systems", PASSAGE)


def test_verified_answer():
    result = verify(answer(("ai_act:article_16", "are compliant with the requirements set out in Section 2")), PASSAGES)
    assert result.ok


def test_unknown_chunk_is_rejected():
    result = verify(answer(("ai_act:article_99", "are compliant with the requirements")), PASSAGES)
    assert not result.ok
    assert "not one of the provided passages" in result.failed[0].reason


def test_invented_quote_is_rejected():
    result = verify(answer(("ai_act:article_16", "providers must pay a fine of EUR 35 000 000")), PASSAGES)
    assert not result.ok


def test_too_short_quote_is_rejected():
    result = verify(answer(("ai_act:article_16", "shall")), PASSAGES)
    assert not result.ok


def test_answer_without_citations_is_rejected():
    result = verify(answer(), PASSAGES)
    assert not result.ok


def test_one_bad_citation_rejects_the_answer():
    result = verify(
        answer(
            ("ai_act:article_16", "are compliant with the requirements set out in Section 2"),
            ("ai_act:article_16", "must register in the EU database"),
        ),
        PASSAGES,
    )
    assert not result.ok
    assert len(result.failed) == 1


def test_not_found_needs_no_citations():
    assert verify(answer(found=False, text=""), PASSAGES).ok


@pytest.mark.parametrize(
    "raw",
    [
        '{"found": false, "answer": "", "citations": [], "confidence": "low"}',
        'Here is the answer:\n```json\n{"found": false, "citations": []}\n```',
        'Sure! {"found": false} Hope this helps.',
    ],
)
def test_parse_tolerates_wrapping(raw):
    assert parse_answer(raw).found is False


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_answer("I cannot answer that.")
