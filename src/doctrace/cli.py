import argparse
import sys

from doctrace.config import settings


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="doctrace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def with_chunks(p):
        p.add_argument("--max-chars", type=int, default=settings.chunk_max_chars)
        return p

    def with_model(p):
        with_chunks(p)
        p.add_argument("--model", default=settings.embedding_model)
        return p

    p = sub.add_parser("download")
    p.add_argument("--force", action="store_true")

    with_chunks(sub.add_parser("parse"))

    p = with_chunks(sub.add_parser("show"))
    p.add_argument("query")

    with_model(sub.add_parser("index"))

    p = with_model(sub.add_parser("search"))
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "download":
        from doctrace.ingest.download import download_all

        download_all(force=args.force)
    elif args.cmd == "parse":
        from doctrace.ingest.pipeline import parse_all

        parse_all(args.max_chars)
    elif args.cmd == "show":
        from doctrace.ingest.pipeline import show

        show(args.query, args.max_chars)
    elif args.cmd == "index":
        from doctrace.index import build_index

        build_index(args.model, args.max_chars)
    elif args.cmd == "search":
        from doctrace.search import print_hits, vector_search

        print_hits(vector_search(args.query, args.model, args.max_chars, args.k))
