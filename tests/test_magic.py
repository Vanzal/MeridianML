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


# ------------------------------------------------------------------------- KV-Cache & Tempo
def test_kv_cache_gives_same_logits_as_full_forward():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=300, block_size=24, n_layer=2, n_head=2, n_embd=32)
    m = GPT(cfg).eval()
    x = torch.randint(0, 300, (2, 20))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache(batch=2)
        a, _ = m(x[:, :7], cache=cache)                  # Prefill
        b, _ = m(x[:, 7:10], cache=cache)                # mehrere neue Tokens hinter dem Cache (Maske)
        steps = [m(x[:, i:i + 1], cache=cache)[0] for i in range(10, 20)]   # einzeln
    inc = torch.cat([a, b, *steps], dim=1)
    assert cache.length == 20
    assert torch.allclose(full, inc, atol=1e-5)


def test_kv_cache_refuses_to_overflow_context():
    m = GPT(GPTConfig(vocab_size=300, block_size=8, n_layer=1, n_head=2, n_embd=16)).eval()
    cache = m.new_cache()
    with torch.no_grad():
        m(torch.zeros(1, 8, dtype=torch.long), cache=cache)
        with pytest.raises(AssertionError):
            m(torch.zeros(1, 1, dtype=torch.long), cache=cache)


def test_greedy_output_identical_with_and_without_cache_past_window():
    from magicllm.bench import greedy_equal
    tok = BPETokenizer.train(GERMAN, 300)
    torch.manual_seed(1)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32))
    assert greedy_equal(m, tok, ["Grüezi", "", "Die Bäume"], n_tokens=40)    # 40 > Kontext 16: Fenster wandert


def test_sampled_output_identical_with_and_without_cache():
    tok = BPETokenizer.train(GERMAN, 300)
    m = _tiny(tok)
    kw = dict(temperature=0.9, top_k=20, top_p=0.95, min_p=0.02, repeat_penalty=1.1, seed=5)
    a = [t for t, *_ in stream_generate(m, tok, "Grü", 50, use_cache=True, **kw)]
    b = [t for t, *_ in stream_generate(m, tok, "Grü", 50, use_cache=False, **kw)]
    assert a == b


def test_bench_run_ceilings_and_ab(tmp_path):
    from magicllm import bench
    tok = BPETokenizer.train(GERMAN, 300)
    m = _tiny(tok)
    machine = {"bandwidth_gbs": 10.0, "gflops": 100.0}
    r = bench.run(m, tok, bench.BenchConfig(), n_tokens=12, prompt_len=4, prefill_lengths=(8, 64), machine=machine)
    assert r.ok and set(r.metrics) == {"decode_tok_s", "ttft_ms", "prefill8_tok_s", "prefill32_tok_s"}
    assert all(len(v) >= bench.MIN_SAMPLES for v in r.samples.values())
    c = bench.ceilings(m, 10.0, 100.0, ctx=16)
    assert c["bytes_per_token"] == m.num_params() * 4 + 2 * 1 * 16 * 32 * 4
    assert abs(c["decode_tok_s"] - 10e9 / c["bytes_per_token"]) < 1e-6
    ledger = tmp_path / "l.jsonl"
    bench.append_ledger(ledger, "x", r)
    assert bench.read_ledger(ledger)[0]["metrics"] == r.metrics
    verdict, results = bench.ab_test(m, tok, bench.BenchConfig(use_cache=False), bench.BenchConfig(),
                                     n_tokens=12, prompt_len=4, prefill_lengths=(8,))
    assert len(results) == 3 and verdict.target == "decode_tok_s" and isinstance(verdict.accepted, bool)


def test_bench_flags_degenerate_output():
    from magicllm.bench import degenerate
    assert degenerate([5] * 20) and degenerate([]) and not degenerate([1, 2, 3] * 5)


def test_bench_cli_runs(trained_run, capsys):
    import bench_cli
    trainer_mod, runs = trained_run
    assert bench_cli.main(["-m", "demo", "-n", "8", "-p", "4", "--prefill", "8", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "decode_tok_s" in out and "Obergrenzen" in out
    assert bench_cli.main(["-m", "demo", "-n", "8", "-p", "4", "--prefill", "8", "--device", "cpu",
                           "--ab", "use_cache=false", "use_cache=true"]) == 0
    assert "Urteil" in capsys.readouterr().out
    assert len((runs / "demo" / "bench.jsonl").read_text().splitlines()) == 4


# ------------------------------------------------------------------------- Themes
@pytest.fixture
def settings(tmp_path, monkeypatch):
    from ui import theme as th
    monkeypatch.setattr(th, "SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(th, "THEME_FILE", tmp_path / "theme.json")
    monkeypatch.setenv("MAGICLLM_SETTINGS_DIR", str(tmp_path))
    return th


def test_presets_are_valid_and_readable(settings):
    th = settings
    assert {"Dunkel", "Hell", "Terminal"} <= set(th.PRESETS)
    for t in th.PRESETS.values():
        t.validate()
        k = th.tokens(t)
        assert th.contrast(t.fg, t.bg) >= 7, t.name                     # Text gut lesbar
        assert th.contrast(k["on_primary"], t.primary) >= 4.5, t.name   # Text auf Hauptknöpfen lesbar
        assert th.contrast(t.muted, t.bg) >= 3, t.name


def test_light_preset_is_light(settings):
    th = settings
    assert th.PRESETS["Hell"].base == "light" and th.luminance(th.PRESETS["Hell"].bg) > 0.9
    assert th.tokens(th.PRESETS["Hell"])["on_primary"] == "#FFFFFF"     # dunkler Knopf -> weisse Schrift
    assert th.tokens(th.PRESETS["Dunkel"])["on_primary"] == "#0A0A0A"


def test_custom_theme_roundtrip_and_delete(settings):
    th = settings
    assert th.active_theme().name == "Dunkel"
    t = th.Theme.from_dict({**th.PRESETS["Hell"].to_dict(), "name": "Mein Blau", "primary": "#2563EB"})
    assert th.save_custom(t) == "Mein Blau"
    assert th.active_theme().primary == "#2563EB" and "Mein Blau" in th.all_themes()
    th.set_active("Terminal")
    assert th.active_theme().name == "Terminal"
    th.delete_custom("Mein Blau")
    assert "Mein Blau" not in th.all_themes()
    th.set_active("gibts nicht")
    assert th.active_theme().name == "Dunkel"


def test_saving_under_preset_name_does_not_overwrite_preset(settings):
    th = settings
    t = th.Theme.from_dict({**th.PRESETS["Dunkel"].to_dict(), "primary": "#FF0000"})
    assert th.save_custom(t) == "Dunkel (eigenes)"
    assert th.PRESETS["Dunkel"].primary == "#FAFAFA"


def test_invalid_theme_rejected(settings):
    th = settings
    with pytest.raises(ValueError):
        th.Theme.from_dict({"name": "x", "bg": "rot"})
    with pytest.raises(ValueError):
        th.Theme.from_dict({"name": "x", "base": "pink"})
    (settings.THEME_FILE).write_text("{kaputt", encoding="utf-8")
    assert th.active_theme().name == "Dunkel"                        # kaputte Datei -> Standard


def test_streamlit_options_cover_light_mode(settings):
    o = settings.streamlit_options(settings.PRESETS["Hell"])
    assert o["theme.base"] == "light" and o["theme.backgroundColor"] == "#FFFFFF"
    assert o["theme.chartCategoricalColors"] == settings.PRESETS["Hell"].chart


def test_themed_dot_recolors_in_one_pass(settings):
    from ui import common
    common.apply_theme(settings.PRESETS["Hell"])
    out = common.themed_dot('fillcolor="#141414", color="#333333", fontcolor="#E5E5E5" c="#60A5FA" x="#123456"')
    assert "#141414" not in out and common.T["surface"] in out and common.T["fg"] in out
    assert settings.PRESETS["Hell"].chart[1] in out and "#123456" in out
    common.apply_theme(settings.PRESETS["Dunkel"])


# ------------------------------------------------------------------------- Hugging-Face-Token
@pytest.fixture
def hf(tmp_path, monkeypatch):
    from magicllm import hf_auth
    monkeypatch.setenv("MAGICLLM_SETTINGS_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv(hf_auth._ENV_FROM_FILE, raising=False)
    return hf_auth


FAKE = "hf_" + "A1b2C3d4" * 4 + "xy"


def test_hf_token_save_apply_remove(hf, tmp_path):
    import os
    import stat
    assert hf.source() == "" and hf.active_token() is None
    hf.save(FAKE)
    f = tmp_path / "hf_token"
    assert f.read_text() == FAKE
    if os.name == "posix":
        assert stat.S_IMODE(f.stat().st_mode) == 0o600              # nur für den Besitzer lesbar
    assert os.environ["HF_TOKEN"] == FAKE and hf.source() == "datei"
    hf.remove()
    assert not f.exists() and "HF_TOKEN" not in os.environ and hf.source() == ""


def test_hf_env_token_has_priority(hf, monkeypatch):
    import os
    monkeypatch.setenv("HF_TOKEN", "hf_" + "Z" * 34)
    hf.save(FAKE)
    assert os.environ["HF_TOKEN"] == "hf_" + "Z" * 34 and hf.source() == "umgebung"


def test_hf_token_validation_and_mask(hf):
    assert hf.looks_valid(FAKE) and not hf.looks_valid("abc") and not hf.looks_valid("hf_kurz")
    with pytest.raises(ValueError):
        hf.save("kein-token")
    assert hf.mask(FAKE) == "hf_…" + FAKE[-4:] and FAKE not in hf.mask(FAKE) and hf.mask(None) == "–"


def test_hf_verify_reports_role(hf, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami",
                        lambda self, token=None: {"name": "vanzal", "auth": {"accessToken": {"role": "read"}}})
    r = hf.verify(FAKE)
    assert r.ok and r.user == "vanzal" and r.read_only
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami",
                        lambda self, token=None: {"name": "v", "auth": {"accessToken": {"role": "write"}}})
    assert not hf.verify(FAKE).read_only
    def boom(self, token=None):
        raise RuntimeError("401 Client Error: Unauthorized")
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", boom)
    r = hf.verify(FAKE)
    assert not r.ok and "kennt diesen Token nicht" in r.message
