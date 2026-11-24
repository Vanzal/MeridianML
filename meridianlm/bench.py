"""Tempo messen – nach der Methode des „inference-optimisation“-Skills (sybil-solutions):

1. Grundlinie messen.            2. Theoretische Obergrenze ausrechnen.
3. Genau eine Sache ändern.      4. Gegen Kontrollen messen (Kontrolle – Kandidat – Kontrolle).
5. Behalten oder verwerfen und den Grund aufschreiben (Ledger).

Gemessen werden:
  * decode  – Tokens pro Sekunde beim Erzeugen (ein Token nach dem anderen, „c1“)
  * prefill – Tokens pro Sekunde beim Einlesen eines Prompts (mehrere Längen)
  * ttft    – Zeit bis zum ersten erzeugten Token
Jede Zahl ist der Median aus mindestens 6 Messungen, weil einzelne Läufe um ±5–10 % schwanken.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .generate import stream_generate
from .model import GPT
from .tokenizer import BPETokenizer

MIN_SAMPLES = 6
REGRESSION_TOLERANCE = 0.03       # andere Kennzahlen dürfen höchstens 3 % schlechter werden


# --------------------------------------------------------------------------- Einstellungen
@dataclass
class BenchConfig:
    """Eine Konfiguration = ein Experiment. Immer nur eine Sache gegenüber der besten ändern."""
    name: str = "baseline"
    use_cache: bool = True
    threads: int = 0                 # 0 = PyTorch-Standard
    grad_mode: str = "inference_mode"  # inference_mode (Standard seit Experiment 7) | no_grad
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    config: dict
    metrics: dict                    # Median-Werte
    samples: dict                    # alle Einzelwerte
    ceilings: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    ok: bool = True
    problem: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


@contextmanager
def _applied(cfg: BenchConfig):
    old_threads = torch.get_num_threads()
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    mode = torch.inference_mode() if cfg.grad_mode == "inference_mode" else torch.no_grad()
    try:
        with mode:
            yield
    finally:
        torch.set_num_threads(old_threads)


def environment(device: torch.device) -> dict:
    env = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
           "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
           "threads": torch.get_num_threads()}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


# --------------------------------------------------------------------------- Maschine messen
def measure_bandwidth(device: torch.device, mb: int = 256, repeats: int = 8) -> float:
    """Echte Speicher-Bandbreite in GB/s (Kopie eines grossen Tensors: lesen + schreiben)."""
    n = mb * 1024 * 1024 // 4
    a = torch.empty(n, dtype=torch.float32, device=device).uniform_()
    b = torch.empty_like(a)
    b.copy_(a)
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        b.copy_(a)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return 2 * a.numel() * 4 / statistics.median(times) / 1e9


def measure_matmul(device: torch.device, n: int | None = None, repeats: int = 6) -> float:
    """Echter Rechen-Durchsatz in GFLOP/s (grosse Matrix-Multiplikation in float32)."""
    n = n or (4096 if device.type != "cpu" else 1024)
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        a @ b
        _sync(device)
        times.append(time.perf_counter() - t0)
    return 2 * n ** 3 / statistics.median(times) / 1e9


def ceilings(model: GPT, bandwidth_gbs: float, gflops: float, ctx: int) -> dict:
    """Theoretische Obergrenzen (siehe docs/perf/THEORY.md).

    decode: pro Token müssen alle Gewichte einmal aus dem Speicher gelesen werden (das Token-Embedding dient
            gleichzeitig als Ausgabe-Schicht), dazu der KV-Cache bis zur aktuellen Länge. → bandbreiten-begrenzt.
    prefill: ~2 FLOPs pro Parameter und Token plus Attention; praktisch erreichbar sind ~50 % der Spitze.
    """
    c = model.cfg
    elem = next(model.parameters()).element_size()
    weight_bytes = model.num_params() * elem
    kv_bytes = 2 * c.n_layer * ctx * c.n_embd * elem
    decode_bytes = weight_bytes + kv_bytes
    flops_tok = 2 * model.num_params() + 4 * c.n_layer * c.n_embd * ctx
    return {
        "bytes_per_token": decode_bytes,
        "decode_tok_s": bandwidth_gbs * 1e9 / decode_bytes,
        "flops_per_token": flops_tok,
        "prefill_tok_s": 0.5 * gflops * 1e9 / flops_tok,
        "bandwidth_gbs": round(bandwidth_gbs, 2),
        "gflops": round(gflops, 1),
    }


# --------------------------------------------------------------------------- Modell messen
def _prompt_ids(tok: BPETokenizer, length: int) -> list[int]:
    base = tok.encode("Once upon a time, there was a little dog named Max. ") or [tok.eot_id]
    ids = (base * (length // len(base) + 1))[:length]
    return ids


def _decode_once(model: GPT, tok: BPETokenizer, prompt: str, n_tokens: int, cfg: BenchConfig,
                 device: torch.device) -> tuple[float, float, list[int]]:
    """Ein Durchlauf: (decode tok/s ohne erstes Token, TTFT in s, erzeugte Token-IDs)."""
    ids, t_first = [], None
    _sync(device)
    t0 = time.perf_counter()
    for tid, *_ in stream_generate(model, tok, prompt, n_tokens, temperature=1.0, top_k=1, seed=0,
                                   use_cache=cfg.use_cache):
        ids.append(tid)
        if t_first is None:
            _sync(device)
            t_first = time.perf_counter()
    _sync(device)
    t_end = time.perf_counter()
    rest = max(1, len(ids) - 1)
    return rest / max(1e-9, t_end - t_first), t_first - t0, ids


def _prefill_once(model: GPT, ids: list[int], device: torch.device, cache: bool, min_seconds: float = 0.03) -> float:
    """Eine Prefill-Messung. Kurze Prompts dauern nur ~1 ms – da misst man vor allem den Timer. Deshalb wird der
    Durchlauf so oft wiederholt, bis mindestens ``min_seconds`` vergangen sind (Cache wird vorher angelegt)."""
    x = torch.tensor([ids], device=device)
    kv = model.new_cache() if cache else None
    reps, elapsed = 0, 0.0
    _sync(device)
    t0 = time.perf_counter()
    while elapsed < min_seconds:
        if kv is not None:
            kv.reset()
        model(x, cache=kv)
        reps += 1
        _sync(device)
        elapsed = time.perf_counter() - t0
    return reps * len(ids) / elapsed


def degenerate(ids: list[int]) -> bool:
    """Grobe Plausibilitätsprüfung: ein verdächtig schneller Lauf liefert oft kaputte Ausgabe (Skill-Regel 8)."""
    if not ids:
        return True
    return len(set(ids)) == 1 and len(ids) > 8


def run(model: GPT, tok: BPETokenizer, cfg: BenchConfig, n_tokens: int = 96, prompt_len: int = 16,
        prefill_lengths: tuple[int, ...] = (32, 64, 128), samples: int = MIN_SAMPLES,
        machine: dict | None = None) -> BenchResult:
    """Misst eine Konfiguration. ``machine`` = {"bandwidth_gbs", "gflops"} (sonst wird gemessen)."""
    samples = max(MIN_SAMPLES, samples)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    block = model.cfg.block_size
    prefill_lengths = tuple(sorted({min(block, L) for L in prefill_lengths}))
    prompt = tok.decode(_prompt_ids(tok, min(prompt_len, block - 1)))
    out: dict[str, list[float]] = {"decode_tok_s": [], "ttft_ms": []}
    for L in prefill_lengths:
        out[f"prefill{L}_tok_s"] = []
    ids_seen: list[int] = []
    try:
        with _applied(cfg):
            _decode_once(model, tok, prompt, 8, cfg, device)              # Aufwärmen (zählt nicht)
            for _ in range(samples):
                d, ttft, ids_seen = _decode_once(model, tok, prompt, n_tokens, cfg, device)
                out["decode_tok_s"].append(d)
                out["ttft_ms"].append(ttft * 1000)
            for L in prefill_lengths:
                pid = _prompt_ids(tok, L)
                _prefill_once(model, pid, device, cfg.use_cache)
                for _ in range(samples):
                    out[f"prefill{L}_tok_s"].append(_prefill_once(model, pid, device, cfg.use_cache))
    finally:
        model.train(was_training)
    metrics = {k: round(statistics.median(v), 2) for k, v in out.items()}
    m = machine or {"bandwidth_gbs": measure_bandwidth(device), "gflops": measure_matmul(device)}
    ceil = ceilings(model, m["bandwidth_gbs"], m["gflops"], ctx=min(block, prompt_len + n_tokens))
    ceil["decode_pct"] = round(100 * metrics["decode_tok_s"] / ceil["decode_tok_s"], 2)
    res = BenchResult(cfg.to_dict(), metrics, {k: [round(x, 3) for x in v] for k, v in out.items()}, ceil,
                      environment(device))
    if degenerate(ids_seen):
        res.ok, res.problem = False, "Ausgabe besteht nur aus einem Token – Messung ungültig"
    return res


# --------------------------------------------------------------------------- A/B und Gleichheit
def greedy_equal(model: GPT, tok: BPETokenizer, prompts: list[str], n_tokens: int = 64) -> bool:
    """Prüft, ob Cache und Nicht-Cache bei gieriger Auswahl dieselben Tokens liefern (verlustfrei?)."""
    for p in prompts:
        a = [t for t, *_ in stream_generate(model, tok, p, n_tokens, top_k=1, seed=0, use_cache=True)]
        b = [t for t, *_ in stream_generate(model, tok, p, n_tokens, top_k=1, seed=0, use_cache=False)]
        if a != b:
            return False
    return True


@dataclass
class ABVerdict:
    accepted: bool
    target: str
    control_a: float
    candidate: float
    control_b: float
    speedup: float
    regressions: dict
    reason: str


def ab_test(model: GPT, tok: BPETokenizer, control: BenchConfig, candidate: BenchConfig,
            target: str = "decode_tok_s", **run_kw) -> tuple[ABVerdict, list[BenchResult]]:
    """Kontrolle – Kandidat – Kontrolle. Angenommen nur, wenn der Kandidat beide Kontrollen bei ``target`` schlägt
    und keine andere Kennzahl um mehr als 3 % schlechter ist als *beide* Kontrollen."""
    machine = {"bandwidth_gbs": measure_bandwidth(next(model.parameters()).device),
               "gflops": measure_matmul(next(model.parameters()).device)}
    ra = run(model, tok, control, machine=machine, **run_kw)
    rc = run(model, tok, candidate, machine=machine, **run_kw)
    rb = run(model, tok, control, machine=machine, **run_kw)
    higher_better = not target.startswith("ttft")
    def better(x, y):
        return x > y if higher_better else x < y
    a, c, b = ra.metrics[target], rc.metrics[target], rb.metrics[target]
    beats = better(c, a) and better(c, b)
    regressions = {}
    for k, v in rc.metrics.items():
        if k == target:
            continue
        hb = not k.startswith("ttft")
        # Verglichen wird mit der *schlechteren* Kontrolle: Was innerhalb der Streuung der beiden Kontrollen liegt,
        # ist Rauschen und keine Verschlechterung.
        ref = min(ra.metrics[k], rb.metrics[k]) if hb else max(ra.metrics[k], rb.metrics[k])
        change = (v - ref) / ref if ref else 0.0
        if (hb and change < -REGRESSION_TOLERANCE) or (not hb and change > REGRESSION_TOLERANCE):
            regressions[k] = round(100 * change, 1)
    ok_outputs = rc.ok and ra.ok and rb.ok
    accepted = beats and not regressions and ok_outputs
    ref = (a + b) / 2
    speedup = (c / ref if higher_better else ref / c) if ref and c else 0.0
    if not ok_outputs:
        reason = "ungültige Ausgabe"
    elif not beats:
        reason = "schlägt nicht beide Kontrollen (innerhalb Rauschen)"
    elif regressions:
        reason = "andere Kennzahl schlechter: " + ", ".join(f"{k} {v:+.1f} %" for k, v in regressions.items())
    else:
        reason = f"{speedup:.2f}× bei {target}, keine Verschlechterung"
    return ABVerdict(accepted, target, a, c, b, round(speedup, 3), regressions, reason), [ra, rc, rb]


def append_ledger(path: Path, run_name: str, result: BenchResult, verdict: ABVerdict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": run_name, **result.to_dict()}
    if verdict is not None:
        row["verdict"] = asdict(verdict)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
