from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    doc_id: str
    title: str
    short: str
    url: str
    expected_articles: int
    expected_recitals: int


SOURCES = [
    Source(
        doc_id="ai_act",
        title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        short="AI Act",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
        expected_articles=113,
        expected_recitals=180,
    ),
    Source(
        doc_id="gdpr",
        title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        short="GDPR",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
        expected_articles=99,
        expected_recitals=173,
    ),
]

BY_ID = {s.doc_id: s for s in SOURCES}
