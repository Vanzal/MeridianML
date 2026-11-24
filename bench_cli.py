"""Tempo messen ohne App – angelehnt an llama-bench aus llama.cpp.

Beispiele:
    python bench_cli.py -m lauf-1                          # beste Einstellung messen, Tabelle + Obergrenzen
    python bench_cli.py -m lauf-1 --set use_cache=false    # eine Einstellung ändern
    python bench_cli.py -m lauf-1 --ab use_cache=false use_cache=true --target decode_tok_s
                                                           # Kontrolle – Kandidat – Kontrolle, mit Urteil

Jede Messung landet als Zeile in runs/<modell>/bench.jsonl (oder --ledger PFAD).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from magicllm import bench
from magicllm import trainer
from magicllm.trainer import list_runs, load_model


def _parse_set(pairs: list[str], base: bench.BenchConfig) -> bench.BenchConfig:
    d = base.to_dict()
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"Einstellung '{p}' muss die Form name=wert haben")
        k, v = p.split("=", 1)
        if k not in d:
            raise SystemExit(f"Unbekannte Einstellung '{k}'. Möglich: {', '.join(d)}")
        cur = d[k]
        if isinstance(cur, bool):
            d[k] = v.lower() in ("1", "true", "ja", "yes", "on")
        elif isinstance(cur, int):
            d[k] = int(v)
        else:
            d[k] = v
    d["name"] = ",".join(pairs) or "best"
    return bench.BenchConfig(**d)


def _table(rows: list[tuple[str, bench.BenchResult]]) -> str:
    keys = list(rows[0][1].metrics)
    head = f"| {'config':<28} | " + " | ".join(f"{k:>16}" for k in keys) + " |"
    sep = "|" + "-" * 30 + "|" + "|".join("-" * 18 for _ in keys) + "|"
    lines = [head, sep]
    for name, r in rows:
        lines.append(f"| {name[:28]:<28} | " + " | ".join(f"{r.metrics[k]:>16,.2f}" for k in keys) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="Lauf (Ordner unter runs/). Standard: der neueste")
    ap.add_argument("-n", "--n-gen", type=int, default=96, help="erzeugte Tokens pro Messung")
    ap.add_argument("-p", "--n-prompt", type=int, default=16, help="Prompt-Länge in Tokens")
    ap.add_argument("--prefill", type=int, nargs="+", default=[32, 64, 128], help="Prefill-Längen")
    ap.add_argument("-r", "--samples", type=int, default=bench.MIN_SAMPLES, help="Messungen pro Kennzahl (min. 6)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=WERT", help="Einstellung ändern")
    ap.add_argument("--ab", nargs=2, metavar=("KONTROLLE", "KANDIDAT"),
                    help="A/B-Test, je eine Einstellung name=wert (Kommas für mehrere)")
    ap.add_argument("--target", default="decode_tok_s", help="Kennzahl, die besser werden soll")
    ap.add_argument("--ledger", type=Path, help="Ledger-Datei (Standard: runs/<modell>/bench.jsonl)")
    args = ap.parse_args(argv)

    runs = list_runs()
    if not runs:
        print("Keine trainierten Modelle gefunden.", file=sys.stderr)
        return 1
    name = args.model or runs[0]["name"]
    if name not in {r["name"] for r in runs}:
        print(f"Lauf '{name}' nicht gefunden.", file=sys.stderr)
        return 1
    from magicllm import device as dev
    model, tok, _ = load_model(name, dev.pick_device(args.device))
    ledger = args.ledger or (trainer.RUNS_DIR / name / "bench.jsonl")
    kw = dict(n_tokens=args.n_gen, prompt_len=args.n_prompt, prefill_lengths=tuple(args.prefill), samples=args.samples)
    device = next(model.parameters()).device
    print(f"magic-bench: {name} · {model.num_params():,} Param. · {device} · Median aus {max(6, args.samples)} "
          "Messungen", file=sys.stderr)

    if args.ab:
        control = _parse_set(args.ab[0].split(","), bench.BenchConfig())
        cand = _parse_set(args.ab[1].split(","), bench.BenchConfig())
        verdict, (ra, rc, rb) = bench.ab_test(model, tok, control, cand, target=args.target, **kw)
        print(_table([("A " + control.name, ra), ("B " + cand.name, rc), ("A " + control.name, rb)]))
        print(f"\nUrteil: {'ANGENOMMEN' if verdict.accepted else 'VERWORFEN'} – {verdict.reason}")
        for r in (ra, rb):
            bench.append_ledger(ledger, name, r)
        bench.append_ledger(ledger, name, rc, verdict)
        return 0

    cfg = _parse_set(args.set, bench.BenchConfig())
    r = bench.run(model, tok, cfg, **kw)
    print(_table([(cfg.name, r)]))
    c = r.ceilings
    print(f"\nObergrenzen: decode {c['decode_tok_s']:,.0f} tok/s (gemessen {c['decode_pct']:.1f} %) · "
          f"prefill {c['prefill_tok_s']:,.0f} tok/s · Bandbreite {c['bandwidth_gbs']} GB/s · {c['gflops']} GFLOP/s")
    if not r.ok:
        print(f"WARNUNG: {r.problem}", file=sys.stderr)
    bench.append_ledger(ledger, name, r)
    print(f"Ledger: {ledger}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
