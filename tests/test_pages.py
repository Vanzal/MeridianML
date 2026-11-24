"""Rauchtest: Die Seiten Modell und Training laufen ohne Fehler durch – mit und ohne MoE.

Streamlit führt die Seite wie in der echten App aus (``AppTest``), nur ohne Browser.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

TEXT = ["Der Hund läuft über die Strasse. Die Katze schläft in der Sonne.\n" * 3,
        "Grüezi mitenand! Äpfel, Öl und Übermut. Schöne Grüsse aus Zürich.\n" * 3]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from meridianlm import data
    from ui import common
    monkeypatch.setattr(data, "PREPARED_DIR", tmp_path / "prepared")
    monkeypatch.setattr(common, "PREPARED_DIR", tmp_path / "prepared")
    data.prepare("smoke", {"x": TEXT * 20}, 300)
    return "smoke"


def _page(module: str, preset: str) -> str:
    return (f"import importlib\n"
            f"from ui import common, theme\n"
            f"common.apply_theme(theme.PRESETS[{preset!r}])\n"
            f"importlib.import_module({module!r}).render()\n")


def _run(module: str, state: dict, preset: str = "Dunkel") -> AppTest:
    at = AppTest.from_string(_page(module, preset), default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("preset", ["Dunkel", "Hell"])
@pytest.mark.parametrize("moe", [False, True])
def test_model_page_renders(prepared, moe, preset):
    at = _run("ui.page_model", {"active_data": prepared, "m_moe": moe, "m_experts": 4, "m_topk": 2}, preset)
    labels = [m.label for m in at.metric]
    assert "Parameter" in labels and "Daten-Tokens pro Parameter" in labels
    params = next(m for m in at.metric if m.label == "Parameter")
    assert ("aktiv pro Token" in (params.delta or "")) is moe


@pytest.mark.parametrize("moe", [False, True])
def test_train_page_renders(prepared, moe):
    at = _run("ui.page_train", {"active_data": prepared, "m_moe": moe, "m_experts": 4, "m_topk": 2})
    caption = " ".join(c.value for c in at.caption)
    assert ("Experten" in caption) is moe
