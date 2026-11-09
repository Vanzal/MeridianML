"""Text erzeugen ohne App – Bedienung angelehnt an llama-cli aus llama.cpp.

Beispiele:
    python generate_cli.py --list
    python generate_cli.py -m lauf-1 -p "Once upon a time" -n 120
    python generate_cli.py -m lauf-1 -p "Es war einmal" --temp 0.7 --top-k 40 --min-p 0.05 --repeat-penalty 1.1 -s 42
"""
from __future__ import annotations

import argparse
import sys
import time

from magicllm.generate import stream_generate
from magicllm.trainer import list_runs, load_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="Name des Laufs (Ordner unter runs/). Standard: der neueste")
    ap.add_argument("-p", "--prompt", default="", help="Textanfang")
    ap.add_argument("-n", "--n-predict", type=int, default=128, help="maximale Anzahl neuer Tokens")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40, help="0 = aus")
    ap.add_argument("--top-p", type=float, default=1.0, help="1.0 = aus")
    ap.add_argument("--min-p", type=float, default=0.0, help="0 = aus")
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="1.0 = aus")
    ap.add_argument("--repeat-last-n", type=int, default=64)
    ap.add_argument("-s", "--seed", type=int, default=-1, help="-1 = Zufall")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | xpu | mps")
    ap.add_argument("--ignore-eos", action="store_true", help="nicht beim End-Token aufhören")
    ap.add_argument("--list", action="store_true", help="gespeicherte Läufe auflisten")
    args = ap.parse_args(argv)

    runs = list_runs()
    if args.list:
        for r in runs:
            val = f"{r['best_val']:.3f}" if r.get("best_val") is not None else "-"
            print(f"{r['name']:<24} {r.get('n_params', 0):>12,} Param.  val {val:<6}  {r.get('data', '?')}")
        return 0
    if not runs:
        print("Keine trainierten Modelle gefunden. Zuerst trainieren (App oder train_cli.py).", file=sys.stderr)
        return 1
    name = args.model or runs[0]["name"]
    if name not in {r["name"] for r in runs}:
        print(f"Lauf '{name}' nicht gefunden. Verfügbar: {', '.join(r['name'] for r in runs)}", file=sys.stderr)
        return 1

    from magicllm import device as dev
    model, tok, ckpt = load_model(name, dev.pick_device(args.device))
    print(f"magic-llm: Modell {name} · {model.num_params():,} Param. · Schritt {ckpt.get('step', '?')} · "
          f"{next(model.parameters()).device}", file=sys.stderr)

    sys.stdout.write(args.prompt)
    sys.stdout.flush()
    n, t0 = 0, time.perf_counter()
    for tid, piece, _, _ in stream_generate(
            model, tok, args.prompt, args.n_predict, temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
            seed=None if args.seed < 0 else args.seed, min_p=args.min_p, repeat_penalty=args.repeat_penalty,
            repeat_last_n=args.repeat_last_n):
        n += 1
        if tid == tok.eot_id:
            sys.stdout.write(piece)
            if not args.ignore_eos:
                break
            sys.stdout.write("\n\n")
            continue
        sys.stdout.write(piece)
        sys.stdout.flush()
    dt = time.perf_counter() - t0
    print(file=sys.stdout)
    print(f"\nmagic-llm: {n} Tokens in {dt:.2f} s ({n / dt if dt > 0 else 0:.1f} tok/s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
