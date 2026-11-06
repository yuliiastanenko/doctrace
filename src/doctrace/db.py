import psycopg
from pgvector.psycopg import register_vector

from doctrace.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    collection text NOT NULL,
    id text NOT NULL,
    doc_id text NOT NULL,
    kind text NOT NULL,
    ref text NOT NULL,
    heading text NOT NULL,
    text text NOT NULL,
    page int NOT NULL,
    chapter text,
    article int,
    article_title text,
    paragraphs int[] NOT NULL DEFAULT '{}',
    annex text,
    embedding vector NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || text)) STORED,
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
"""


def connect() -> psycopg.Connection:
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is empty, check your .env file")
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
