"""Tests für die Neuerungen von Magic LLM: Sampling (Min-p, Wiederholungs-Strafe), UTF-8-Streaming,
atomare Checkpoints, robuste Lauf-Liste, Kommandozeile und eine emoji-freie Oberfläche."""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
import torch

from magicllm.generate import apply_repeat_penalty, filter_logits, generate_text, stream_generate
from magicllm.model import GPT, GPTConfig
from magicllm.tokenizer import EOT, BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
GERMAN = ["Grüezi mitenand! Äpfel, Öl und Übermut. Schöne Grüsse aus Zürich.\n" * 3,
          "Die Bäume blühen, die Vögel zwitschern – überall ist Frühling.\n" * 3]


# ------------------------------------------------------------------------- Sampling
def test_min_p_keeps_only_tokens_close_to_the_best():
    logits = torch.log(torch.tensor([0.50, 0.30, 0.15, 0.05]))
    p = filter_logits(logits, 1.0, 0, 1.0, min_p=0.25)       # Schwelle: 0.25 * 0.5 = 0.125
    assert (p > 0).tolist() == [True, True, True, False]
    assert abs(p.sum().item() - 1) < 1e-6
    assert torch.equal(filter_logits(logits, 1.0, 0, 1.0, min_p=0.0), filter_logits(logits, 1.0, 0, 1.0))


def test_min_p_never_removes_the_best_token():
    logits = torch.randn(1000)
    p = filter_logits(logits, 0.7, 0, 1.0, min_p=0.99)
    assert p[logits.argmax()] > 0 and abs(p.sum().item() - 1) < 1e-6


def test_repeat_penalty_matches_llama_cpp_rule():
    logits = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_repeat_penalty(logits, [0, 1, 1], 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0, 0.5]                 # positiv: geteilt, negativ: multipliziert
    assert logits.tolist() == [2.0, -2.0, 1.0, 0.5]              # Eingabe bleibt unverändert
    assert torch.equal(apply_repeat_penalty(logits, [0], 1.0), logits)
    assert torch.equal(apply_repeat_penalty(logits, [], 1.5), logits)


def test_repeat_penalty_reduces_repetition():
    tok = BPETokenizer.train(["ab " * 50], 270)
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=32))
    def distinct(rp):
        ids = [t for t, *_ in stream_generate(m, tok, "ab", 60, temperature=0.3, top_k=0, seed=3, repeat_penalty=rp)]
        return len(set(ids))
    assert distinct(1.8) >= distinct(1.0)


# ------------------------------------------------------------------- UTF-8 streaming
def _tiny(tok):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=32))


def test_streamed_pieces_equal_full_decode():
    """Die gestreamten Textstücke müssen zusammen genau dem dekodierten Gesamttext entsprechen – auch wenn ein
    Umlaut auf zwei Tokens verteilt ist (früher entstanden dabei zwei „�“)."""
    tok = BPETokenizer.train(GERMAN, 300)
    m = _tiny(tok)
    for seed in range(5):
        ids, pieces = [], []
        for tid, piece, *_ in stream_generate(m, tok, "Gr", 80, temperature=1.5, top_k=0, seed=seed):
            ids.append(tid)
            pieces.append(piece + (EOT if tid == tok.eot_id else ""))
        assert "".join(pieces) == tok.decode(ids)


def test_multibyte_char_split_over_tokens_is_not_garbled():
    tok = BPETokenizer()                                   # nur Byte-Tokens: „ä“ = 2 Tokens
    ids = tok.encode("ä")
    assert len(ids) == 2

    class Fixed(torch.nn.Module):                          # „Modell“, das genau diese Bytes ausgibt
        cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8)
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.i = 0
        def forward(self, x):
            logits = torch.full((1, x.shape[1], tok.vocab_size), -1e9)
            logits[0, -1, ids[self.i % 2]] = 0.0
            self.i += 1
            return logits, None

    pieces = [p for _, p, *_ in stream_generate(Fixed(), tok, "", 4, temperature=1.0, top_k=0)]
    assert pieces == ["", "ä", "", "ä"]


def test_unfinished_char_is_flushed_on_last_step():
    tok = BPETokenizer()
    first_byte = tok.encode("ä")[0]

    class One(torch.nn.Module):
        cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8)
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            logits = torch.full((1, x.shape[1], tok.vocab_size), -1e9)
            logits[0, -1, first_byte] = 0.0
            return logits, None

    out = "".join(p for _, p, *_ in stream_generate(One(), tok, "", 1, top_k=0))
    assert out == "�"                                   # halbes Zeichen wird sichtbar statt verschluckt


def test_generate_text_can_stop_at_end_token():
    tok = BPETokenizer.train(GERMAN, 300)
    m = _tiny(tok)
    m.lm_head = torch.nn.Linear(32, tok.vocab_size, bias=True)   # Modell zwingen, immer das End-Token zu wählen
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.bias.fill_(-1e4)
        m.lm_head.bias[tok.eot_id] = 0
    assert generate_text(m, tok, "Hallo", stop_at_eot=True, max_new_tokens=5, top_k=0) == "Hallo"
    assert generate_text(m, tok, "Hallo", max_new_tokens=2, top_k=0) == "Hallo" + EOT + EOT


# ------------------------------------------------------------- Läufe und Checkpoints
@pytest.fixture
def trained_run(tmp_path, monkeypatch):
    from magicllm import trainer as trainer_mod
    from magicllm.data import prepare
    monkeypatch.setattr(trainer_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("magicllm.data.PREPARED_DIR", tmp_path / "prepared")
    meta = prepare("de", {"x": GERMAN * 10}, 300)
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=16, n_layer=1, n_head=2, n_embd=32)
    tr = trainer_mod.Trainer("demo", tmp_path / "prepared" / "de", cfg,
                             trainer_mod.TrainConfig(batch_size=8, max_steps=20, eval_interval=10, sample_interval=10,
                                                     eval_iters=2, device="cpu"))
    tr.start()
    while tr.is_running:
        time.sleep(0.1)
    assert tr.status == "fertig", tr.error
    return trainer_mod, tmp_path / "runs"


def test_checkpoint_writes_leave_no_temp_files(trained_run):
    _, runs = trained_run
    files = sorted(p.name for p in (runs / "demo").iterdir())
    assert "model.pt" in files and "run.json" in files and "metrics.jsonl" in files
    assert not [f for f in files if f.endswith(".tmp")]


def test_list_runs_skips_broken_run_folders(trained_run):
    trainer_mod, runs = trained_run
    bad = runs / "kaputt"
    bad.mkdir()
    (bad / "model.pt").write_bytes(b"x")
    (bad / "run.json").write_text("{ halb geschrieb", encoding="utf-8")
    (runs / "datei.txt").write_text("keine Lauf-Mappe", encoding="utf-8")
    assert [r["name"] for r in trainer_mod.list_runs()] == ["demo"]


def test_loaded_model_reproduces_trained_weights(trained_run):
    trainer_mod, _ = trained_run
    model, tok, ckpt = trainer_mod.load_model("demo", torch.device("cpu"))
    assert ckpt["step"] == 20 and tok.vocab_size == model.cfg.vocab_size
    text = generate_text(model, tok, "Grü", max_new_tokens=20, seed=1)
    assert text.startswith("Grü")


def test_generate_cli_runs(trained_run, capsys):
    import generate_cli
    assert generate_cli.main(["--list"]) == 0
    assert "demo" in capsys.readouterr().out
    assert generate_cli.main(["-m", "demo", "-p", "Grüezi", "-n", "12", "-s", "7", "--min-p", "0.05",
                              "--repeat-penalty", "1.1", "--device", "cpu"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("Grüezi") and "tok/s" in captured.err
    assert generate_cli.main(["-m", "gibtsnicht"]) == 1


def test_prepare_rejects_empty_data(tmp_path, monkeypatch):
    from magicllm.data import prepare
    monkeypatch.setattr("magicllm.data.PREPARED_DIR", tmp_path / "prepared")
    with pytest.raises(ValueError):
        prepare("leer", {"x": []}, 300)


def test_meta_device_param_count_matches_real_model():
    cfg = GPTConfig(vocab_size=1000, block_size=64, n_layer=3, n_head=4, n_embd=64)
    with torch.device("meta"):
        meta_model = GPT(cfg)
    real = GPT(cfg)
    assert meta_model.num_params() == real.num_params() == GPT.estimate_params(cfg)
    assert meta_model.param_breakdown() == real.param_breakdown()


# --------------------------------------------------------------------- Oberfläche
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF️‍"
    "⌚⌛⏩-⏺Ⓜ⤴⤵〰〽㊗㊙]")


@pytest.mark.parametrize("path", sorted(str(p.relative_to(ROOT)) for p in
                                        [ROOT / "app.py", ROOT / "generate_cli.py", ROOT / "train_cli.py",
                                         ROOT / "README.md", ROOT / "docs" / "DATASETS.md",
                                         *(ROOT / "ui").glob("*.py"), *(ROOT / "magicllm").glob("*.py")]))
def test_no_emojis_in_user_facing_files(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    hits = sorted({c for c in EMOJI.findall(text)})
    assert not hits, f"Emojis in {path}: {hits}"


def test_branding_is_magic_llm():
    for p in [ROOT / "app.py", *(ROOT / "ui").glob("*.py"), ROOT / "README.md"]:
        text = p.read_text(encoding="utf-8")
        assert "MiniLLM Studio" not in text.replace("auf MiniLLM Studio", "").replace("[MiniLLM Studio]", ""), p
    assert 'page_title="Magic LLM"' in (ROOT / "app.py").read_text(encoding="utf-8")


def test_merge_counts_are_recorded_at_merge_time():
    tok = BPETokenizer.train(["aa ab aa ab aa"], 260)
    assert len(tok.merge_counts) == len(tok.merges) and all(c >= 2 for c in tok.merge_counts)
    assert tok.merge_counts == sorted(tok.merge_counts, reverse=True)
