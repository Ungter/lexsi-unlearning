import argparse
import json
from pathlib import Path

from . import book, curate, math_gen
from .pipeline import build_data, build_variants, check_leakage, freeze
from .splits import verify_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lfm3u.data")
    parser.add_argument("command", choices=("build", "math", "book", "retain", "variants", "check", "freeze", "verify"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--dev", type=int, default=300)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--tokenizer", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--no-hendrycks", action="store_true")
    parser.add_argument("--no-book", action="store_true")
    parser.add_argument("--no-retain", action="store_true")
    parser.add_argument("--privleak", action="store_true")
    parser.add_argument("--remote-curation", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--retain-count", type=int, default=2000)
    args = parser.parse_args(argv)
    if Path(".env").is_file():
        from dotenv import load_dotenv

        load_dotenv()
    root = args.root.resolve()
    sizes = {"train": args.train, "dev": args.dev, "test": args.test}
    tok = None
    if args.command == "book" or (args.command == "build" and not args.no_book):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.command == "math":
        result = math_gen.build_math(root, args.seed, sizes, include_hendrycks=not args.no_hendrycks)
    elif args.command == "book":
        result = book.build_book(root, tok, args.max_len, args.seed, include_privleak=args.privleak)
    elif args.command == "retain":
        result = curate.build_general_retain(root, args.retain_count, args.seed)
    elif args.command == "variants":
        result = build_variants(root, seed=args.seed, remote=args.remote_curation, translate=args.translate)
    elif args.command == "check":
        result = str(check_leakage(root))
    elif args.command == "freeze":
        result = str(freeze(root))
    elif args.command == "verify":
        result = verify_manifest(root)
    else:
        result = build_data(root, tok, seed=args.seed, sizes=sizes, include_hendrycks=not args.no_hendrycks,
                            include_book=not args.no_book, include_retain=not args.no_retain,
                            include_privleak=args.privleak, remote_curation=args.remote_curation,
                            translate=args.translate, freeze_data=not args.no_freeze,
                            max_len=args.max_len, retain_count=args.retain_count)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
