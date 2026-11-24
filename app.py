"""Magic LLM – Hauptprogramm der App.   Start:  streamlit run app.py"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st

from magicllm import __version__
from magicllm import device as dev
from ui import (page_bench, page_data, page_generate, page_model, page_runs, page_settings, page_start, page_tokenizer,
                page_train, theme)
from ui.common import PAGES, STATUS_DOT, dot, fmt_time, prepared_names, setup_page, store

ASSETS = Path(__file__).resolve().parent / "assets"

# Aktives Theme auf Streamlits eigene Farben anwenden. Hat sich etwas geändert, einmal neu laden, damit alle
# Eingabefelder, Tabellen und Regler die neuen Farben bekommen (nur einmal pro Theme, keine Endlosschleife).
_theme = theme.active_theme()
_sig = json.dumps(_theme.to_dict(), sort_keys=True)
if st.session_state.get("_theme_sig") != _sig:
    st.session_state["_theme_sig"] = _sig
    if theme.apply_to_streamlit(_theme):
        st.rerun()


def _logo_files(t: theme.Theme) -> tuple[str, str]:
    """Logo und Tab-Icon in den Farben des Themes (Vorlagen: assets/magic-llm-logo.svg und magic-llm-icon.svg)."""
    k = theme.tokens(t)
    logo = (ASSETS / "magic-llm-logo.svg").read_text(encoding="utf-8").replace('fill="#0A0A0A"', f'fill="{t.fg}"')
    icon = ((ASSETS / "magic-llm-icon.svg").read_text(encoding="utf-8")
            .replace('fill="#0A0A0A"', 'fill="__TILE__"').replace('fill="#FAFAFA"', f'fill="{k["on_primary"]}"')
            .replace("__TILE__", t.primary))
    out = theme.SETTINGS_DIR / "logo"
    try:
        out.mkdir(parents=True, exist_ok=True)
        tag = f"v2{t.fg}{t.primary}{k['on_primary']}".replace("#", "")
        lp, ip = out / f"logo-{tag}.svg", out / f"icon-{tag}.svg"
        if not lp.exists() or not ip.exists():
            lp.write_text(logo, encoding="utf-8")
            ip.write_text(icon, encoding="utf-8")
        return str(lp), str(ip)
    except OSError:                       # schreibgeschützt: mitgelieferte Dateien nehmen
        dark = t.base == "dark"
        return str(ASSETS / ("magic-llm-logo-dark.svg" if dark else "magic-llm-logo.svg")), str(ASSETS / "magic-llm-icon.svg")


_logo, _icon = _logo_files(_theme)
st.set_page_config(page_title="Magic LLM", page_icon=_icon, layout="wide", initial_sidebar_state="expanded")
st.logo(_logo, icon_image=_icon, size="large")
setup_page()

# Ein frisch vorbereiteter Datensatz wird automatisch ausgewählt (muss vor dem Widget passieren).
if pending := st.session_state.pop("pending_active", None):
    st.session_state["active_data"] = pending

pages = [
    st.Page(page_start.render, title="Start", icon=":material/home:", url_path="start", default=True),
    st.Page(page_data.render, title="Daten", icon=":material/database:", url_path="daten"),
    st.Page(page_tokenizer.render, title="Tokenizer", icon=":material/data_array:", url_path="tokenizer"),
    st.Page(page_model.render, title="Modell", icon=":material/account_tree:", url_path="modell"),
    st.Page(page_train.render, title="Training", icon=":material/monitoring:", url_path="training"),
    st.Page(page_generate.render, title="Generieren", icon=":material/terminal:", url_path="generieren"),
    st.Page(page_runs.render, title="Läufe", icon=":material/stacked_line_chart:", url_path="laeufe"),
    st.Page(page_bench.render, title="Tempo", icon=":material/speed:", url_path="tempo"),
    st.Page(page_settings.render, title="Einstellungen", icon=":material/settings:", url_path="einstellungen"),
]
PAGES.update(dict(zip(["start", "data", "tokenizer", "model", "train", "generate", "runs", "bench", "settings"], pages)))
nav = st.navigation(pages)

with st.sidebar:
    names = prepared_names()
    if names:
        if st.session_state.get("active_data") not in names:
            st.session_state["active_data"] = names[-1]
        st.selectbox("Aktiver Datensatz", names, key="active_data")
    else:
        st.caption("Noch kein Datensatz vorbereitet.")

    di = dev.describe()
    st.markdown('<div class="ml-side-label">Gerät</div>', unsafe_allow_html=True)
    if di["type"] != "cpu":
        mem = f" · {di['memory_gb']:.0f} GB" if di.get("memory_gb") else ""
        st.markdown(f'<div class="ml-side-row">{dot("ok")}{di["name"]}{mem}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ml-side-row">{dot("idle")}CPU</div>', unsafe_allow_html=True)
        st.caption(di["hint"])

    t = store()["trainer"]
    if t is not None:
        st.markdown('<div class="ml-side-label">Training</div>', unsafe_allow_html=True)
        state = STATUS_DOT.get(t.status, "idle")
        st.markdown(f'<div class="ml-side-row">{dot(state)}{t.run_name} · {t.status}</div>', unsafe_allow_html=True)
        st.progress(min(1.0, t.step / max(1, t.cfg.max_steps)), text=f"{t.step}/{t.cfg.max_steps}")
        if t.evals:
            st.caption(f"val {t.evals[-1]['val']:.3f} · {fmt_time(t.elapsed)}")
    st.divider()
    light = st.toggle("Heller Modus", value=_theme.base == "light",
                      help="Schnell zwischen den Vorlagen Hell und Dunkel wechseln. Eigene Themes: Einstellungen.")
    if light != (_theme.base == "light"):
        theme.set_active("Hell" if light else "Dunkel")
        st.session_state.pop("_theme_sig", None)
        st.session_state.pop("set_theme_pick", None)
        st.rerun()
    st.markdown(f'<div class="ml-side-row">magic-llm <span class="ml-brand-ver">v{__version__}</span></div>',
                unsafe_allow_html=True)
    st.caption("GPT von Grund auf in PyTorch · Tokenizer, Modell, Training und Sampling selbst gebaut")

nav.run()
