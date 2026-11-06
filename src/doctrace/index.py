import time

from doctrace.config import collection_name
from doctrace.db import connect, init_schema
from doctrace.embed import embed_passages
from doctrace.ingest.pipeline import chunks_path, load_chunks

INSERT = """
INSERT INTO chunks (collection, id, doc_id, kind, ref, heading, text, page,
                    chapter, article, article_title, paragraphs, annex, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def embedding_input(chunk: dict) -> str:
    return f"{chunk['heading']}\n{chunk['text']}"


def build_index(model: str, max_chars: int) -> None:
    if not chunks_path(max_chars).exists():
        raise SystemExit(f"{chunks_path(max_chars)} not found, run `uv run doctrace parse --max-chars {max_chars}` first")
    chunks = load_chunks(max_chars)
    collection = collection_name(model, max_chars)
    print(f"collection: {collection}")
    print(f"model: {model} (first run downloads it, this can take a few minutes)")

    start = time.time()
    vectors = embed_passages([embedding_input(c) for c in chunks], model)
    print(f"embedding: {len(chunks)} chunks, dim {vectors.shape[1]}, {time.time() - start:.0f}s")

    start = time.time()
    with connect() as conn:
        init_schema(conn)
        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE collection = %s", (collection,))
            with conn.cursor() as cur:
                cur.executemany(
                    INSERT,
                    [
                        (
                            collection,
                            c["id"],
                            c["doc_id"],
                            c["kind"],
                            c["ref"],
                            c["heading"],
                            c["text"],
                            c["page"],
                            c["chapter"],
                            c["article"],
                            c["article_title"],
                            c["paragraphs"],
                            c["annex"],
                            v,
                        )
                        for c, v in zip(chunks, vectors)
                    ],
                )
        total = conn.execute("SELECT count(*) FROM chunks WHERE collection = %s", (collection,)).fetchone()[0]
    print(f"database: {total} rows in collection {collection}, {time.time() - start:.0f}s")
