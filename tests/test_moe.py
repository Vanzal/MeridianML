"""Tests für Mixture of Experts (MoE), Prüfung der Modell-Einstellungen und Checkpoint-Format."""
from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F

from meridianlm.model import GPT, MoE, GPTConfig

TEXT = ["Grüezi mitenand! Äpfel, Öl und Übermut. Schöne Grüsse aus Zürich.\n" * 3,
        "Der Hund läuft über die Strasse. Die Katze schläft in der Sonne.\n" * 3]


def moe_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=200, block_size=32, n_layer=2, n_head=2, n_embd=32, n_experts=4, top_k_experts=2)
    return GPTConfig(**{**base, **kw})


# ------------------------------------------------------------- Einstellungen prüfen
@pytest.mark.parametrize("bad, msg", [
    (dict(n_head=3), "teilbar"),
    (dict(n_layer=0), "n_layer"),
    (dict(block_size=0), "block_size"),
    (dict(vocab_size=1), "vocab_size"),
    (dict(dropout=1.0), "dropout"),
    (dict(dropout=-0.1), "dropout"),
    (dict(n_embd=32.5), "ganze Zahl"),
    (dict(n_layer=True), "ganze Zahl"),
    (dict(n_experts=1), "mindestens 2"),
    (dict(n_experts=-2), "n_experts"),
    (dict(n_experts=65), "n_experts"),
    (dict(n_experts=4, top_k_experts=5), "top_k_experts"),
    (dict(n_experts=4, top_k_experts=0), "top_k_experts"),
    (dict(moe_aux_weight=-1.0), "moe_aux_weight"),
])
def test_invalid_config_gives_clear_error(bad, msg):
    base = dict(vocab_size=200, block_size=32, n_layer=2, n_head=2, n_embd=32)
    with pytest.raises(ValueError, match=msg):
        GPTConfig(**{**base, **bad})


def test_dense_default_has_no_experts_and_same_weight_names():
    cfg = GPTConfig()
    assert cfg.n_experts == 0 and not cfg.is_moe
    keys = set(GPT(GPTConfig(vocab_size=100, block_size=8, n_layer=1, n_head=2, n_embd=16)).state_dict())
    assert "blocks.0.mlp.fc.weight" in keys and not any("router" in k or "experts" in k for k in keys)


def test_old_config_dict_without_moe_fields_still_loads():
    old = dict(vocab_size=300, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    assert GPTConfig.from_dict(old).n_experts == 0


# ------------------------------------------------------------- MoE-Schicht
def test_moe_param_counts_match_estimates():
    for kw in (dict(), dict(n_experts=8, top_k_experts=1), dict(n_experts=2, top_k_experts=2)):
        cfg = moe_cfg(**kw)
        m = GPT(cfg)
        assert m.num_params() == GPT.estimate_params(cfg)
        assert m.active_params() == GPT.estimate_active_params(cfg)
    dense = GPT(moe_cfg(n_experts=0))
    assert dense.active_params() == dense.num_params()
    # top-k = alle Experten: aktiv = alles ausser dem Router
    full = GPT(moe_cfg(n_experts=2, top_k_experts=2))
    assert full.active_params() == full.num_params()


def test_moe_matches_per_token_reference():
    """Die gebündelte Verteilung auf die Experten muss genau dasselbe liefern wie Token für Token gerechnet."""
    torch.manual_seed(0)
    moe = MoE(moe_cfg(n_experts=4, top_k_experts=2)).eval()
    x = torch.randn(2, 7, 32)
    y, _ = moe(x)
    ref = torch.zeros_like(x)
    for b in range(2):
        for t in range(7):
            p = F.softmax(moe.router(x[b, t]), dim=-1)
            w, idx = p.topk(2)
            w = w / w.sum()
            ref[b, t] = sum(w[j] * moe.experts[int(idx[j])](x[b, t]) for j in range(2))
    assert torch.allclose(y, ref, atol=1e-6)


def test_balance_loss_is_one_when_even_and_higher_when_skewed():
    torch.manual_seed(0)
    moe = MoE(moe_cfg(n_experts=4, top_k_experts=1)).eval()
    x = torch.randn(4, 64, 32)
    with torch.no_grad():
        moe.router.weight.zero_()                      # alle gleich wahrscheinlich
        _, even = moe(x)
        moe.router.weight[0].fill_(10.0)               # Experte 0 bekommt fast alles
        _, skewed = moe(x.abs())
    assert even.item() == pytest.approx(1.0, abs=0.3)
    assert skewed.item() > 2.0


def test_moe_gradients_reach_router_and_all_experts():
    torch.manual_seed(0)
    m = GPT(moe_cfg())
    x = torch.randint(0, 200, (8, 32))
    _, loss = m(x, x)
    loss.backward()
    for b in m.blocks:
        assert b.mlp.router.weight.grad.abs().sum() > 0
        for e in b.mlp.experts:
            assert e.fc.weight.grad is not None and e.fc.weight.grad.abs().sum() > 0


def test_aux_loss_only_added_in_training():
    torch.manual_seed(0)
    m = GPT(moe_cfg(moe_aux_weight=0.5))
    x = torch.randint(0, 200, (4, 32))
    m.train()
    _, total, aux = m(x, x, return_aux=True)
    m.eval()
    with torch.no_grad():
        logits, ce = m(x, x)
    ce_ref = F.cross_entropy(logits.view(-1, 200), x.view(-1))
    assert torch.allclose(ce, ce_ref)
    assert aux.item() > 0
    # im Training: CE + Gewicht x Balance (gleiche Gewichte, kein Dropout -> gleiche CE)
    assert total.item() == pytest.approx(ce.item() + 0.5 * aux.item(), rel=1e-5)


def test_dense_model_returns_zero_aux():
    m = GPT(moe_cfg(n_experts=0))
    _, _, aux = m(torch.zeros(1, 4, dtype=torch.long), return_aux=True)
    assert aux.item() == 0.0


def test_moe_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(moe_cfg()).eval()
    x = torch.randint(0, 200, (1, 32))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        parts = [m(x[:, :10], cache=cache)[0]]
        parts += [m(x[:, t:t + 1], cache=cache)[0] for t in range(10, 32)]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5)


def test_moe_generation_identical_with_and_without_cache():
    from meridianlm.bench import greedy_equal
    from meridianlm.tokenizer import BPETokenizer
    tok = BPETokenizer.train(TEXT, 300)
    torch.manual_seed(1)
    m = GPT(moe_cfg(vocab_size=tok.vocab_size, block_size=16)).eval()
    assert greedy_equal(m, tok, ["Der Hund", "Die Katze schläft"], 40)


def test_expert_usage_sums_to_one_and_resets():
    torch.manual_seed(0)
    m = GPT(moe_cfg()).train()
    m(torch.randint(0, 200, (4, 32)))
    usage = m.expert_usage(reset=True)
    assert len(usage) == 2 and all(len(u) == 4 for u in usage)
    assert all(sum(u) == pytest.approx(1.0) for u in usage)
    assert all(sum(u) == 0 for u in m.expert_usage())
    m.eval()                                           # beim Erzeugen wird nicht gezählt
    with torch.no_grad():
        m(torch.randint(0, 200, (1, 8)))
    assert all(sum(u) == 0 for u in m.expert_usage())
    assert GPT(moe_cfg(n_experts=0)).expert_usage() == []


def test_moe_ceilings_count_only_active_experts():
    from meridianlm.bench import ceilings
    m = GPT(moe_cfg(n_experts=8, top_k_experts=1))
    c = ceilings(m, bandwidth_gbs=10.0, gflops=100.0, ctx=0)
    assert c["bytes_per_token"] == m.active_params() * 4 < m.num_params() * 4


# ------------------------------------------------------------- Training + Checkpoints
@pytest.fixture
def moe_run(tmp_path, monkeypatch):
    from meridianlm import trainer as trainer_mod
    from meridianlm.data import prepare
    monkeypatch.setattr(trainer_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("meridianlm.data.PREPARED_DIR", tmp_path / "prepared")
    meta = prepare("de", {"x": TEXT * 10}, 300)
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=16, n_layer=2, n_head=2, n_embd=32,
                    n_experts=4, top_k_experts=2)
    tr = trainer_mod.Trainer("moe", tmp_path / "prepared" / "de", cfg,
                             trainer_mod.TrainConfig(batch_size=8, max_steps=20, eval_interval=10, sample_interval=10,
                                                     eval_iters=2, device="cpu"))
    tr.start()
    while tr.is_running:
        time.sleep(0.1)
    assert tr.status == "fertig", tr.error
    return trainer_mod, tr


def test_moe_training_logs_balance_and_expert_usage(moe_run):
    _, tr = moe_run
    assert all("balance" in m for m in tr.metrics)
    usage = [e["experts"] for e in tr.evals if "experts" in e]
    assert usage and len(usage[-1]) == 2 and sum(usage[-1][0]) == pytest.approx(1.0, abs=1e-3)
    # angezeigter Loss ist die reine Cross-Entropy (ohne Balance-Anteil)
    assert all(m["loss"] < 10 for m in tr.metrics)


def test_moe_checkpoint_roundtrip(moe_run):
    trainer_mod, tr = moe_run
    model, tok, ckpt = trainer_mod.load_model("moe", torch.device("cpu"))
    assert ckpt["format"] == trainer_mod.CHECKPOINT_FORMAT
    assert model.cfg.n_experts == 4 and model.cfg.top_k_experts == 2
    x = torch.randint(0, model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(x)[0], tr.model.cpu().eval()(x)[0], atol=1e-5)


def _ckpt(**over):
    cfg = GPTConfig(vocab_size=300, block_size=16, n_layer=1, n_head=2, n_embd=32)
    base = {"format": 2, "model": GPT(cfg).state_dict(), "model_cfg": cfg.to_dict(), "step": 1}
    base.update(over)
    return base


def test_check_checkpoint_accepts_current_and_old_format():
    from meridianlm.trainer import check_checkpoint
    assert check_checkpoint(_ckpt()).n_layer == 1
    old = _ckpt()
    del old["format"]                                  # Checkpoints bis 2.1 haben kein Format-Feld
    assert check_checkpoint(old).n_experts == 0


@pytest.mark.parametrize("over, msg", [
    (dict(format=99), "neueren Meridian-Version"),
    (dict(format="x"), "Unbekanntes"),
    (dict(model_cfg={"vocab_size": 300, "n_embd": 30, "n_head": 4}), "ungültig"),
])
def test_check_checkpoint_rejects_bad_files(over, msg):
    from meridianlm.trainer import CheckpointError, check_checkpoint
    with pytest.raises(CheckpointError, match=msg):
        check_checkpoint(_ckpt(**over))


def test_check_checkpoint_rejects_non_checkpoint_and_tokenizer_mismatch():
    from meridianlm.tokenizer import BPETokenizer
    from meridianlm.trainer import CheckpointError, check_checkpoint
    with pytest.raises(CheckpointError, match="kein Meridian-Checkpoint"):
        check_checkpoint({"weights": 1})
    big_tok = BPETokenizer.train(TEXT * 20, 400)
    small = _ckpt(model_cfg=GPTConfig(vocab_size=big_tok.vocab_size - 5, block_size=16, n_layer=1, n_head=2,
                                      n_embd=32).to_dict())
    with pytest.raises(CheckpointError, match="passen"):
        check_checkpoint(small, big_tok)


def test_load_model_reports_mismatched_weights(moe_run):
    trainer_mod, _ = moe_run
    path = trainer_mod.RUNS_DIR / "moe" / "model.pt"
    ckpt = torch.load(path, weights_only=True)
    ckpt["model_cfg"]["n_experts"] = 8                 # passt nicht mehr zu den gespeicherten Gewichten
    torch.save(ckpt, path)
    with pytest.raises(trainer_mod.CheckpointError, match="passen nicht"):
        trainer_mod.load_model("moe", torch.device("cpu"))


def test_load_model_without_checkpoint_has_clear_error(tmp_path, monkeypatch):
    from meridianlm import trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "RUNS_DIR", tmp_path)
    (tmp_path / "leer").mkdir()
    with pytest.raises(trainer_mod.CheckpointError, match="keinen Checkpoint"):
        trainer_mod.load_model("leer")
