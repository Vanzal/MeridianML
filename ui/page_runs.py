"""Seite 6: Trainings-Läufe vergleichen."""
from __future__ import annotations

import math
import shutil

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from magicllm.trainer import RUNS_DIR, list_runs, load_run_metrics
from ui.common import COLORS, fmt_time, hero, next_step, store, style_fig


def render() -> None:
    hero("Läufe vergleichen", "Alle trainierten Modelle auf einen Blick – welche Einstellung war die beste?",
         crumb="läufe")
    runs = list_runs()
    if not runs:
        st.info("Noch keine abgeschlossenen oder laufenden Trainings gespeichert.", icon=":material/info:")
        next_step("train", "Zum Training")
        st.stop()

    df = pd.DataFrame([{
        "Lauf": r["name"], "Daten": r.get("data"), "Parameter": r.get("n_params"),
        "Schichten": r.get("model_cfg", {}).get("n_layer"), "Breite": r.get("model_cfg", {}).get("n_embd"),
        "Kontext": r.get("model_cfg", {}).get("block_size"), "Schritte": f"{r.get('step', 0)}/{r.get('max_steps', 0)}",
        "Lernrate": r.get("train_cfg", {}).get("lr"), "Batch": r.get("train_cfg", {}).get("batch_size"),
        "Val-Loss": r.get("best_val"),
        "Perplexity": round(math.exp(min(20.0, r["best_val"])), 1) if r.get("best_val") is not None else None,
        "Dauer": fmt_time(r.get("elapsed", 0)), "Gerät": r.get("device", "–"), "Status": r.get("status"),
        "Stand": r.get("finished"),
    } for r in runs])
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"Val-Loss": st.column_config.NumberColumn(format="%.3f"),
                                "Parameter": st.column_config.NumberColumn(format="%d")})

    names = [r["name"] for r in runs]
    pick = st.multiselect("Läufe im Diagramm vergleichen", names, default=names[:4])
    if pick:
        a, b = st.columns(2)
        fig_val, fig_tr = go.Figure(), go.Figure()
        for i, n in enumerate(pick):
            steps, evals = load_run_metrics(n)
            col = COLORS[i % len(COLORS)]
            if evals:
                fig_val.add_scatter(x=[e["step"] for e in evals], y=[e["val"] for e in evals], name=n,
                                    mode="lines+markers", line=dict(color=col, width=3))
            if steps:
                s = pd.Series([x["loss"] for x in steps]).ewm(alpha=0.05).mean()
                fig_tr.add_scatter(x=[x["step"] for x in steps], y=s, name=n, mode="lines", line=dict(color=col, width=2))
        fig_val.update_layout(title="Validierungs-Loss", xaxis_title="Schritt")
        fig_tr.update_layout(title="Trainings-Loss (geglättet)", xaxis_title="Schritt")
        a.plotly_chart(style_fig(fig_val, 340), width="stretch", key="r_val")
        b.plotly_chart(style_fig(fig_tr, 340), width="stretch", key="r_tr")

        scored = [r for r in runs if r["name"] in pick and r.get("best_val") is not None and r.get("n_params")]
        if scored:
            fig = go.Figure(go.Scatter(
                x=[r["n_params"] for r in scored], y=[r["best_val"] for r in scored], mode="markers+text",
                text=[r["name"] for r in scored], textposition="top center",
                marker=dict(size=[min(60, 12 + r.get("elapsed", 0) ** 0.5) for r in scored], color=COLORS[1], opacity=0.8)))
            fig.update_layout(title="Parameter vs. Val-Loss (Punktgrösse = Trainingsdauer)",
                              xaxis_title="Parameter", yaxis_title="Val-Loss", xaxis_type="log")
            st.plotly_chart(style_fig(fig, 340, legend=False), width="stretch", key="r_scatter")

    st.subheader("Details, Export & Aufräumen")
    sel = st.selectbox("Lauf", names)
    run_dir = RUNS_DIR / sel
    st.json(next(r for r in runs if r["name"] == sel), expanded=False)

    live = store()["trainer"]
    busy = bool(live and live.is_running and live.run_name == sel)
    c1, c2, c3 = st.columns(3)
    ckpt = run_dir / "model.pt"
    if ckpt.exists():
        c1.download_button("Checkpoint (.pt)", data=lambda p=ckpt: p.read_bytes(), file_name=f"{sel}.pt",
                           mime="application/octet-stream", icon=":material/download:", width="stretch",
                           help=f"Gewichte + Konfiguration ({ckpt.stat().st_size / 1e6:.1f} MB). Laden mit "
                                "magicllm.trainer.load_model oder generate_cli.py.")
    tokf = run_dir / "tokenizer.json"
    if tokf.exists():
        c2.download_button("Tokenizer (.json)", data=lambda p=tokf: p.read_bytes(), file_name=f"{sel}-tokenizer.json",
                           mime="application/json", icon=":material/download:", width="stretch")
    if busy:
        c3.button("Löschen", disabled=True, width="stretch", icon=":material/delete:",
                  help="Dieser Lauf trainiert gerade. Stoppe ihn zuerst auf der Seite Training.")
    else:
        with c3.popover("Löschen", icon=":material/delete:", width="stretch"):
            st.write(f"Lauf **{sel}** endgültig löschen? Das kann nicht rückgängig gemacht werden.")
            if st.button("Ja, löschen", type="primary", key=f"del_{sel}"):
                shutil.rmtree(run_dir, ignore_errors=True)
                st.rerun()
