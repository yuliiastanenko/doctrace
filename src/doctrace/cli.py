import argparse
import sys

from doctrace.config import settings


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="doctrace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("parse")
    p.add_argument("--max-chars", type=int, default=settings.chunk_max_chars)

    p = sub.add_parser("show")
    p.add_argument("query")
    p.add_argument("--max-chars", type=int, default=settings.chunk_max_chars)

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
