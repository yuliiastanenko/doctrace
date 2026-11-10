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
    p.add_argument("--mode", choices=["vector", "fulltext", "hybrid", "rerank", "all"], default="rerank")

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["vector", "fulltext", "hybrid", "rerank"], default="rerank")
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("models")
    p.add_argument("--provider")

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
        from doctrace.retrieve import MODES, print_result, retrieve

        modes = MODES if args.mode == "all" else (args.mode,)
        for mode in modes:
            if len(modes) > 1:
                print(f"\n########## {mode} ##########")
            print_result(retrieve(args.query, mode=mode, k=args.k, model=args.model, max_chars=args.max_chars))
    elif args.cmd == "ask":
        from doctrace.answer import ask, print_answer, to_json

        result = ask(args.question, mode=args.mode, k=args.k, provider=args.provider, model=args.model)
        print(to_json(result) if args.json else "", end="")
        if not args.json:
            print_answer(result, verbose=args.verbose)
    elif args.cmd == "models":
        from doctrace.llm import list_models

        print("\n".join(list_models(args.provider)))
