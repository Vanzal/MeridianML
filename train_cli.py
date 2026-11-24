"""Meridian – Training ohne App (Kommandozeile).

Beispiel:
    python train_cli.py --datasets tinystories=2000 shakespeare=200 --steps 500 --name test
    python train_cli.py --datasets tinystories=2000 --experts 8 --top-k 2 --name moe     # Mixture of Experts
"""
from __future__ import annotations

import argparse
import time

from meridianlm.catalog import BY_KEY
from meridianlm.data import load_selection, prepare, prepared_path, safe_name
from meridianlm.generate import generate_text
from meridianlm.model import GPTConfig
from meridianlm.trainer import TrainConfig, Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["tinystories=2000"],
                    help="key=Anzahl_Dokumente, z.B. tinystories=2000 (Schlüssel: " + ", ".join(BY_KEY) + ")")
    ap.add_argument("--name", default="cli")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--experts", type=int, default=0, help="Mixture of Experts: Anzahl Experten (0 = aus, sonst ab 2)")
    ap.add_argument("--top-k", type=int, default=2, help="MoE: an so viele Experten geht jedes Token")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=0, help="0 = automatisch (GPU: so gross wie möglich)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | xpu | mps")
    ap.add_argument("--no-amp", action="store_true", help="Mixed Precision ausschalten")
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    args.name = safe_name(args.name)

    selection = {k: int(n) for k, n in (s.split("=") for s in args.datasets)}
    print("Lade Datensätze:", selection)
    docs = load_selection(selection, lambda k, i, n: print(f"  {k}: {i}/{n}", end="\r"))
    print("\nTrainiere Tokenizer und kodiere Texte ...")
    meta = prepare(args.name, docs, args.vocab)
    print(f"  {meta['tokens_train']:,} Trainings-Tokens, {meta['chars_per_token']:.2f} Zeichen pro Token")

    try:
        cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block, n_layer=args.layers,
                        n_head=args.heads, n_embd=args.embd, n_experts=args.experts,
                        top_k_experts=min(args.top_k, args.experts) if args.experts else args.top_k)
    except ValueError as e:
        raise SystemExit(f"Ungültige Modell-Einstellung: {e}")
    tr = Trainer(args.name, prepared_path(args.name), cfg,
                 TrainConfig(max_steps=args.steps, batch_size=args.batch, lr=args.lr, device=args.device, amp=not args.no_amp,
                             eval_interval=max(10, args.steps // 10), sample_interval=max(50, args.steps // 4)))
    tr.start()
    time.sleep(3)
    print("Gerät:", tr.device_info.get("name", "?"), "| Batch:", tr.cfg.batch_size)
    while tr.is_running:
        time.sleep(2)
        if tr.metrics:
            m = tr.metrics[-1]
            bal = f"  balance {m['balance']:.2f}" if "balance" in m else ""
            print(f"Schritt {tr.step}/{args.steps}  loss {m['loss']:.3f}{bal}  {m['tok_s']:.0f} Tok/s", end="\r")
    print()
    if tr.status == "fehler":
        raise SystemExit(tr.error)
    print("Fertig:", tr.status, "| bester Val-Loss:", round(tr.best_val, 3))
    print("Beispiel:", generate_text(tr.model, tr.tok, "Once upon a time", max_new_tokens=60, stop_at_eot=True))
    print(f"Weiter mit:  python generate_cli.py -m {args.name} -p \"Once upon a time\"")


if __name__ == "__main__":
    main()
