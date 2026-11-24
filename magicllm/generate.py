"""Text erzeugen: Token für Token das nächste Wort aus der Wahrscheinlichkeitsverteilung ziehen."""
from __future__ import annotations

import codecs
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import EOT, BPETokenizer


def apply_repeat_penalty(logits: torch.Tensor, recent: Iterable[int], penalty: float) -> torch.Tensor:
    """Wiederholungs-Strafe wie in llama.cpp (``--repeat-penalty``): Tokens, die gerade schon vorkamen, werden
    unwahrscheinlicher. Positive Logits werden durch ``penalty`` geteilt, negative damit multipliziert."""
    if penalty == 1.0:
        return logits
    ids = torch.tensor(sorted(set(int(i) for i in recent)), dtype=torch.long)
    if ids.numel() == 0:
        return logits
    logits = logits.clone()
    vals = logits[ids]
    logits[ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def filter_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                  min_p: float = 0.0) -> torch.Tensor:
    """Temperatur, Top-k, Top-p (Nucleus) und Min-p anwenden. Gibt Wahrscheinlichkeiten zurück.

    Min-p (wie in llama.cpp): Es bleiben nur Tokens, deren Wahrscheinlichkeit mindestens ``min_p`` mal so gross
    ist wie die des besten Tokens. Das passt sich automatisch an: bei einem sicheren Modell bleiben wenige übrig,
    bei einem unsicheren viele."""
    logits = logits / max(temperature, 1e-4)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        remove = cum - sorted_p > top_p
        sorted_p = sorted_p.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_i, sorted_p)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if 0.0 < min_p < 1.0:
        keep = probs >= min_p * probs.max(dim=-1, keepdim=True).values
        probs = probs * keep
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


@torch.inference_mode()        # schneller als no_grad (gemessen: docs/perf/PLAN.md, Experiment 5 + 7)
def stream_generate(model: GPT, tok: BPETokenizer, prompt: str, max_new_tokens: int = 100,
                    temperature: float = 0.8, top_k: int = 40, top_p: float = 1.0,
                    seed: int | None = None, min_p: float = 0.0, repeat_penalty: float = 1.0,
                    repeat_last_n: int = 64, use_cache: bool = True) -> Iterator[tuple[int, str, torch.Tensor, torch.Tensor]]:
    """Erzeugt Tokens einzeln. Liefert (token_id, text, probs, raw_probs) pro Schritt.

    ``probs`` = Verteilung nach Filter, ``raw_probs`` = Verteilung ohne Temperatur/Filter.

    Der Text wird mit einem inkrementellen UTF-8-Decoder erzeugt: Ein Umlaut wie „ä“ besteht aus zwei Bytes und
    kann auf zwei Tokens verteilt sein. Dann liefert das erste Token "" und das zweite das ganze Zeichen
    (statt zweimal „�“). Das End-Token liefert immer "".

    ``use_cache``: KV-Cache benutzen (Standard). Dann wird pro Schritt nur das neue Token gerechnet. Ist der
    Kontext voll, wird – genau wie ohne Cache – das Fenster der letzten ``block_size`` Tokens neu gerechnet. Das
    Ergebnis ist mit und ohne Cache dasselbe; nur die Geschwindigkeit unterscheidet sich.
    """
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    device = next(model.parameters()).device
    ids = tok.encode(prompt) or [tok.eot_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    history = list(ids)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    block = model.cfg.block_size
    cache = model.new_cache() if use_cache and hasattr(model, "new_cache") else None
    try:
        for step in range(max_new_tokens):
            if cache is None:
                logits, _ = model(x[:, -block:])
            elif step > 0 and cache.length < block:      # nur das neue Token rechnen
                logits, _ = model(x[:, -1:], cache=cache)
            else:                                         # Start, oder Kontext voll: Fenster neu füllen
                cache.reset()
                logits, _ = model(x[:, -block:], cache=cache)
            last = logits[0, -1].float().cpu()          # Würfeln passiert auf der CPU (nur ein Vokabular-Vektor)
            raw = F.softmax(last, dim=-1)
            if repeat_penalty != 1.0 and repeat_last_n > 0:
                last = apply_repeat_penalty(last, history[-repeat_last_n:], repeat_penalty)
            probs = filter_logits(last, temperature, top_k, top_p, min_p)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
            history.append(nxt)
            if nxt == tok.eot_id:
                piece = decoder.decode(b"", final=True)      # angefangene Zeichen abschliessen
                decoder.reset()
            else:                                             # letzter Schritt: angefangene Zeichen abschliessen
                piece = decoder.decode(tok.token_bytes(nxt), final=step == max_new_tokens - 1)
            yield nxt, piece, probs, raw
    finally:
        model.train(was_training)


def generate_text(model: GPT, tok: BPETokenizer, prompt: str = "", stop_at_eot: bool = False, **kw) -> str:
    """Ganzen Text erzeugen. Das End-Token erscheint als ``<|endoftext|>`` (oder beendet den Text, wenn
    ``stop_at_eot`` gesetzt ist)."""
    out = []
    for tid, piece, _, _ in stream_generate(model, tok, prompt, **kw):
        if tid == tok.eot_id:
            out.append(piece)
            if stop_at_eot:
                break
            out.append(EOT)
            continue
        out.append(piece)
    return prompt + "".join(out)


@torch.no_grad()
def next_token_table(model: GPT, tok: BPETokenizer, prompt: str, k: int = 10) -> list[tuple[str, float]]:
    """Die k wahrscheinlichsten nächsten Tokens (für das Balkendiagramm)."""
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt) or [tok.eot_id]
    x = torch.tensor([ids[-model.cfg.block_size:]], device=next(model.parameters()).device)
    logits, _ = model(x)
    probs = F.softmax(logits[0, -1].float().cpu(), dim=-1)
    top = torch.topk(probs, k)
    model.train(was_training)
    return [(tok.token_str(int(i)), float(p)) for p, i in zip(top.values, top.indices)]


@torch.no_grad()
def next_logits(model: GPT, tok: BPETokenizer, prompt: str) -> torch.Tensor:
    """Rohe Logits für das nächste Token (vor Softmax)."""
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt) or [tok.eot_id]
    logits, _ = model(torch.tensor([ids[-model.cfg.block_size:]], device=next(model.parameters()).device))
    model.train(was_training)
    return logits[0, -1].float().cpu()


@torch.no_grad()
def attention_maps(model: GPT, tok: BPETokenizer, text: str):
    """Aufmerksamkeits-Matrizen für einen Text: Liste je Schicht mit Form (Köpfe, T, T)."""
    was_training = model.training
    model.eval()
    ids = tok.encode(text)[-model.cfg.block_size:]
    x = torch.tensor([ids], device=next(model.parameters()).device)
    _, _, atts = model(x, return_attn=True)
    model.train(was_training)
    return [tok.token_str(i) for i in ids], [a[0].float().cpu().numpy() for a in atts]
