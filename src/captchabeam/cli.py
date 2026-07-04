"""Command-line interface: decode, eval, ablation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CHARSET, DecodeConfig
from .engine import CaptchaBeam


def _decode_config(args) -> DecodeConfig:
    return DecodeConfig(
        charset=args.charset,
        length=args.length,
        beam_size=args.beam_size,
        top_chars=args.top_chars,
    )


def _cmd_decode(args) -> int:
    cb = CaptchaBeam(
        variants=args.variants,
        decoder=args.decoder,
        decode_config=_decode_config(args),
    )
    result = cb.decode(args.image)
    print(f"{result.text}\tconf={result.confidence:.4f}\tvariant={result.variant_name}")
    return 0


def _cmd_eval(args) -> int:
    from .eval import build_samples, evaluate

    store = build_samples(
        args.data,
        variant_count=args.variants,
        decoder=args.decoder,
        decode_config=_decode_config(args),
        cache_path=args.cache,
        refresh=args.refresh,
    )
    if not store.samples:
        print("[error] no labeled samples found (need labels.csv in each dir)", file=sys.stderr)
        return 2
    metrics = evaluate(store, target_length=args.length)
    print(f"samples={metrics.total} variants={len(store.variant_names)} decoder={args.decoder}")
    print(metrics.format("all variants"))
    return 0


def _cmd_ablation(args) -> int:
    from .eval import build_samples, leave_one_out

    store = build_samples(
        args.data,
        variant_count=args.variants,
        decoder=args.decoder,
        decode_config=_decode_config(args),
        cache_path=args.cache,
        refresh=args.refresh,
    )
    if not store.samples:
        print("[error] no labeled samples found", file=sys.stderr)
        return 2
    baseline, rows = leave_one_out(store, target_length=args.length)
    print(baseline.format("all variants"))
    print("\n[leave one out]")
    for row in rows:
        print(row.metrics.format(f"drop {row.dropped}") + f"  delta_exact={row.delta_exact:+d}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variants", type=int, default=18)
    parser.add_argument("--decoder", choices=["beam", "native"], default="beam")
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--top-chars", type=int, default=8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="captchabeam", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_decode = sub.add_parser("decode", help="decode a single image")
    p_decode.add_argument("image", type=Path)
    _add_common(p_decode)
    p_decode.set_defaults(func=_cmd_decode)

    p_eval = sub.add_parser("eval", help="score labeled datasets")
    p_eval.add_argument("--data", nargs="+", type=Path, required=True)
    p_eval.add_argument("--cache", type=Path, default=None)
    p_eval.add_argument("--refresh", action="store_true")
    _add_common(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

    p_ab = sub.add_parser("ablation", help="leave-one-out variant ablation")
    p_ab.add_argument("--data", nargs="+", type=Path, required=True)
    p_ab.add_argument("--cache", type=Path, default=None)
    p_ab.add_argument("--refresh", action="store_true")
    _add_common(p_ab)
    p_ab.set_defaults(func=_cmd_ablation)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
