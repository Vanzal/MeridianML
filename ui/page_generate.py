"""Seite 5: Mit dem trainierten Modell Text erzeugen und verstehen, wie es entscheidet."""
from __future__ import annotations

import html
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from magicllm.generate import attention_maps, filter_logits, next_logits, stream_generate
from magicllm.trainer import RUNS_DIR, list_runs, load_model
from ui.common import PALETTE, T, fmt_int, heat_scale, hero, next_step, restore_state, save_state, store, style_fig

DEFAULTS = dict(g_prompt="Once upon a time", g_tokens=120, g_temp=0.8, g_topk=40, g_topp=1.0, g_minp=0.0, g_rep=1.0,
                g_seed=-1, g_stop=True)


@st.cache_resource(show_spinner="Lade Modell …")
def _load(run: str, stamp: float):
    return load_model(run)


def _render_output(prompt: str, text: str, cursor: bool = False) -> str:
    """Prompt grau, erzeugter Text hell – wie im Terminal von llama.cpp."""
    cur = '<span class="cursor"></span>' if cursor else ""
    return (f'<div class="sample">{html.escape(prompt)}<span class="gen">{html.escape(text)}</span>{cur}</div>')


def _stats_line(n_tokens: int, seconds: float, prompt_tokens: int, stop: str) -> str:
    tps = n_tokens / seconds if seconds > 0 else 0.0
    return (f'<div class="ml-stats">prompt <b>{prompt_tokens}</b> tok · erzeugt <b>{n_tokens}</b> tok · '
            f'<b>{tps:,.1f}</b> tok/s · <b>{seconds:.2f}</b> s · stop: <b>{stop}</b></div>')


def render() -> None:
    hero("Generieren", "Gib einen Textanfang ein – das Modell schreibt weiter. Und du siehst, wie es sich entscheidet.")
    restore_state("gen", DEFAULTS)
    runs = list_runs()
    if not runs:
        st.info("Es gibt noch kein trainiertes Modell. Starte zuerst ein Training unter **Training**.",
                icon=":material/info:")
        next_step("train", "Zum Training")
        st.stop()

    live = store()["trainer"]
    names = [r["name"] for r in runs]
    default_idx = names.index(live.run_name) if live and live.run_name in names else 0
    by_name = {r["name"]: r for r in runs}
    c = st.columns([0.45, 0.55], vertical_alignment="bottom")
    run = c[0].selectbox("Modell", names, index=default_idx, key="g_run",
                         format_func=lambda n: f"{n}  ·  {fmt_int(by_name[n].get('n_params', 0))} Param.")
    info = by_name[run]
    meta_line = f"daten <b>{html.escape(str(info.get('data', '?')))}</b> · step <b>{info.get('step', 0)}/{info.get('max_steps', 0)}</b>"
    if info.get("best_val") is not None:
        meta_line += f" · val <b>{info['best_val']:.3f}</b>"
    c[1].markdown(f'<div class="ml-stats">{meta_line}</div>', unsafe_allow_html=True)
    if live and live.is_running and live.run_name == run:
        st.warning("Dieses Modell wird gerade noch trainiert – du siehst den letzten gespeicherten Zwischenstand.",
                   icon=":material/hourglass_top:")

    ckpt = RUNS_DIR / run / "model.pt"
    try:
        model, tok, _ = _load(run, ckpt.stat().st_mtime)
    except Exception as e:      # z. B. Lauf gerade gelöscht oder Datei beschädigt
        st.error(f"Modell **{run}** konnte nicht geladen werden: {type(e).__name__}: {e}")
        st.stop()

    left, right = st.columns([0.55, 0.45], gap="large")
    with left:
        st.text_area("Textanfang (Prompt)", key="g_prompt", height=130)
        b1, b2 = st.columns([1, 2], vertical_alignment="center")
        gen_clicked = b1.button("Erzeugen", type="primary", width="stretch", icon=":material/play_arrow:")
        b2.checkbox("Am Text-Ende stoppen", key="g_stop",
                    help="Das Modell hat ein eigenes End-Token (<|endoftext|>). Ist das an, endet die Ausgabe dort.")
        st.caption("Tipp: Prompt leer lassen, damit das Modell frei erfindet.")
    with right:
        st.slider("Maximale Länge (Tokens)", 10, 1000, key="g_tokens")
        a, b = st.columns(2)
        a.slider("Temperatur", 0.1, 2.0, step=0.05, key="g_temp",
                 help="Niedrig = vorsichtig & wiederholend, hoch = kreativ & chaotisch.")
        b.slider("Top-k", 0, 200, key="g_topk", help="Nur die k wahrscheinlichsten Tokens kommen in Frage (0 = aus).")
        a.slider("Top-p", 0.1, 1.0, step=0.05, key="g_topp",
                 help="Nur die kleinste Token-Menge, die zusammen p Wahrscheinlichkeit hat (1.0 = aus).")
        b.slider("Min-p", 0.0, 0.5, step=0.01, key="g_minp",
                 help="Wie in llama.cpp: Nur Tokens mit mindestens min-p × (Wahrscheinlichkeit des besten Tokens) "
                      "bleiben übrig (0 = aus). Passt sich der Sicherheit des Modells an.")
        a.slider("Wiederholungs-Strafe", 1.0, 2.0, step=0.05, key="g_rep",
                 help="Wie llama.cpp --repeat-penalty: Tokens aus den letzten 64 werden unwahrscheinlicher (1.0 = aus).")
        b.number_input("Seed (-1 = Zufall)", -1, 10**6, key="g_seed")
    save_state("gen")
    s = st.session_state
    prompt = s["g_prompt"]
    sampler = dict(temperature=s["g_temp"], top_k=s["g_topk"], top_p=s["g_topp"], min_p=s["g_minp"],
                   repeat_penalty=s["g_rep"])

    if gen_clicked:
        box, stats = st.empty(), st.empty()
        out, rows, stop = "", [], "Länge"
        seed = None if s["g_seed"] < 0 else int(s["g_seed"])
        n_prompt = len(tok.encode(prompt))
        t0 = time.perf_counter()
        for tid, piece, probs, raw in stream_generate(model, tok, prompt, s["g_tokens"], seed=seed, **sampler):
            rows.append((tok.token_str(tid), float(probs[tid]), float(raw[tid])))
            if tid == tok.eot_id:
                out += piece
                if s["g_stop"]:
                    stop = "Text-Ende"
                    break
                out += "\n\n"
                continue
            out += piece
            box.markdown(_render_output(prompt, out, cursor=True), unsafe_allow_html=True)
        secs = time.perf_counter() - t0
        box.markdown(_render_output(prompt, out), unsafe_allow_html=True)
        s["g_result"] = {"prompt": prompt, "text": out, "rows": rows, "run": run,
                         "stats": (len(rows), secs, n_prompt, stop)}
        stats.markdown(_stats_line(*s["g_result"]["stats"]), unsafe_allow_html=True)

    res = s.get("g_result")
    if res and not gen_clicked:
        st.markdown(_render_output(res["prompt"], res["text"]), unsafe_allow_html=True)
        if res.get("stats"):
            st.markdown(_stats_line(*res["stats"]), unsafe_allow_html=True)
    if res:
        st.download_button("Text herunterladen", res["prompt"] + res["text"], file_name=f"{res['run']}-ausgabe.txt",
                           mime="text/plain", icon=":material/download:")

    tabs = st.tabs(["Wie sicher war das Modell?", "Nächstes Token", "Temperatur-Effekt", "Aufmerksamkeit"])

    with tabs[0]:
        if res and res["run"] == run and res["rows"]:
            df = pd.DataFrame(res["rows"], columns=["Token", "p (gewählt)", "p (Modell)"])
            fig = go.Figure(go.Bar(x=list(range(len(df))), y=df["p (Modell)"], text=df["Token"],
                                   marker=dict(color=df["p (Modell)"], colorscale=[[0, T["border_strong"]], [1, T["fg"]]], cmin=0, cmax=1),
                                   textposition="none", hovertemplate="%{text}<br>p = %{y:.1%}<extra></extra>"))
            fig.update_layout(title="Wahrscheinlichkeit, die das Modell dem gewählten Token gab",
                              xaxis_title="Position im erzeugten Text", yaxis_title="Wahrscheinlichkeit", yaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig, 320, legend=False), width="stretch", key="g_conf")
            st.caption(f"Mittlere Sicherheit: **{df['p (Modell)'].mean():.0%}**. Niedrige Balken = das Modell hat geraten; "
                       "oft an Satz-Anfängen oder bei Namen.")
        else:
            st.info("Erzeuge zuerst einen Text.")

    logits = next_logits(model, tok, prompt)
    with tabs[1]:
        probs = filter_logits(logits, s["g_temp"], s["g_topk"], s["g_topp"], s["g_minp"])
        raw = torch.softmax(logits, -1)
        top = torch.topk(raw, 15)
        labels = [tok.token_str(int(i)) for i in top.indices]
        fig = go.Figure()
        fig.add_bar(x=labels, y=top.values.numpy(), name="Modell (roh)", marker_color=PALETTE["fg"])
        fig.add_bar(x=labels, y=probs[top.indices].numpy(), name="nach Temperatur / Top-k / Top-p / Min-p",
                    marker_color=PALETTE["blue"])
        fig.update_layout(barmode="group", title="Die 15 wahrscheinlichsten nächsten Tokens nach dem Prompt",
                          yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 340), width="stretch", key="g_next")
        ent = float(-(raw * torch.log(raw + 1e-12)).sum())
        st.caption(f"Entropie: **{ent:.2f} nats** (0 = sicher, {np.log(len(raw)):.1f} = völlig unsicher). "
                   "Tokens, die durch Top-k / Top-p / Min-p herausgefiltert wurden, haben nach dem Filter 0 %. "
                   "Die Wiederholungs-Strafe wirkt erst beim Erzeugen (sie hängt vom bisherigen Text ab).")

    with tabs[2]:
        fig = go.Figure()
        for t, col in zip((0.3, 0.8, 1.5), (PALETTE["blue"], PALETTE["fg"], PALETTE["amber"])):
            p = torch.softmax(logits / t, -1)
            vals = torch.sort(p, descending=True).values[:25].numpy()
            fig.add_scatter(x=list(range(1, 26)), y=vals, mode="lines+markers", name=f"T = {t}", line=dict(color=col, width=2))
        fig.update_layout(title="Verteilung der Top-25 Tokens bei verschiedenen Temperaturen",
                          xaxis_title="Rang des Tokens", yaxis_title="Wahrscheinlichkeit", yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 340), width="stretch", key="g_temp_plot")
        st.caption("Niedrige Temperatur macht die Kurve spitz (das Top-Token gewinnt fast immer), hohe Temperatur flach "
                   "(auch seltsame Tokens bekommen eine Chance).")

    with tabs[3]:
        toks, maps = attention_maps(model, tok, prompt or " ")
        cc = st.columns(2)
        layer = cc[0].selectbox("Schicht", range(len(maps)), format_func=lambda i: f"Schicht {i + 1}", key="g_layer")
        head = cc[1].selectbox("Kopf", range(maps[0].shape[0]), format_func=lambda i: f"Kopf {i + 1}", key="g_head")
        labels = [f"{i}:{t}" for i, t in enumerate(toks)]
        fig = go.Figure(go.Heatmap(z=maps[layer][head], x=labels, y=labels, colorscale=heat_scale(), zmin=0, zmax=1))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 480, legend=False), width="stretch", key="g_att")
