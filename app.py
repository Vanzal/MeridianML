"""Magic LLM – Hauptprogramm der App.   Start:  streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st

from magicllm import __version__
from magicllm import device as dev
from ui import page_data, page_generate, page_model, page_runs, page_start, page_tokenizer, page_train
from ui.common import PAGES, STATUS_DOT, dot, fmt_time, prepared_names, setup_page, store

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="Magic LLM", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
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
]
PAGES.update(dict(zip(["start", "data", "tokenizer", "model", "train", "generate", "runs"], pages)))
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
    st.markdown(f'<div class="ml-side-row">magic-llm <span class="ml-brand-ver">v{__version__}</span></div>',
                unsafe_allow_html=True)
    st.caption("GPT von Grund auf in PyTorch · Tokenizer, Modell, Training und Sampling selbst gebaut")

nav.run()
